//! [`LlamaLeaf`]: ignis's [`StepLeaf`] over one llama.cpp context (ADR 0052).
//!
//! Everything above the step is ignis's own -- the scheduler, admission, the
//! KV page ledger, prompt reuse, the media cache, `/v1/decide`. This file
//! answers the step calls with llama.cpp:
//!
//! - a **sequence, prefix or checkpoint is a llama sequence id** in one
//!   unified KV buffer. Publishing a prefix tags its cells with one more id
//!   (`llama_memory_seq_cp`): the claimant shares them in place, as the CUDA
//!   leaf shares pages.
//! - the **pending token** is the CUDA leaf's: a prefill draws the span's
//!   successor and keeps it, a decode round emits it, feeds it and draws the
//!   next. A constrained run's first token is the prefill's draw.
//! - **sampling runs on the host** ([`crate::sampling`]): on unified memory
//!   the logits are already there.
//! - **positions** are ignis's: a text token at `cache index + rope delta`,
//!   an image token at the processor's `[t, h, w]`, the model contract's
//!   M-RoPE either way.
//! - an **attention readout** is not read (`read` stays false): the question
//!   fails above, it is never answered from anything else (ADR 0038).

use std::sync::Mutex;

use ignis_core::vision::MediaItem;
use ignis_core::{DecodeParams, TokenId};
use ignis_runtime::{
    AttentionRead, DecodeLane, DrawRedirect, LaneRun, MultimodalSpan, RuntimeStats, StepLeaf, KV_PAGE_TOKENS,
};

use crate::ffi;
use crate::sampling::{self, Rng};
use crate::unpatch;

/// What a load asks of llama.cpp.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub gguf: std::path::PathBuf,
    pub mmproj: Option<std::path::PathBuf>,
    /// KV cells shared by every sequence.
    pub n_ctx: u32,
    /// Sequence ids: live sequences, published prefixes and checkpoints.
    pub n_seq_max: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: i32,
    /// -1 auto, 0 off, 1 on.
    pub flash_attn: i32,
    /// `ggml_type` of the K and V cache, -1 for the default (f16).
    pub type_k: i32,
    pub type_v: i32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            gguf: Default::default(),
            mmproj: None,
            n_ctx: 16_384,
            n_seq_max: 64,
            n_batch: 2048,
            n_ubatch: 512,
            n_threads: 8,
            flash_attn: 1,
            type_k: -1,
            type_v: -1,
        }
    }
}

pub struct LlamaLeaf {
    config: LlamaConfig,
}

impl LlamaLeaf {
    pub fn new(config: LlamaConfig) -> Self {
        Self { config }
    }
}

struct Raw(*mut ffi::IlModel);

// SAFETY: the context is only ever touched under `LlamaModel::raw`'s lock.
unsafe impl Send for Raw {}

pub struct LlamaModel {
    raw: Mutex<Raw>,
    free_ids: Mutex<Vec<i32>>,
    vocab: u32,
    embd_inp: usize,
    n_ctx: u32,
    cell_bytes: u64,
    model_bytes: u64,
}

impl LlamaModel {
    fn with<R>(&self, f: impl FnOnce(*mut ffi::IlModel) -> R) -> R {
        let raw = self.raw.lock().expect("llama context lock");
        f(raw.0)
    }

    fn take_id(&self) -> Result<i32, i32> {
        self.free_ids.lock().expect("id pool").pop().ok_or_else(|| {
            tracing::warn!(name: "ignis.llama.no_sequence_id", "every llama sequence id is in use");
            -2
        })
    }

    fn give_id(&self, id: i32) {
        self.with(|m| unsafe { ffi::il_seq_rm(m, id, -1, -1) });
        self.free_ids.lock().expect("id pool").push(id);
    }

    /// A fresh id holding `src`'s cells below `p1` (all of them for -1).
    fn share(&self, src: i32, p1: i32) -> Result<i32, i32> {
        let id = self.take_id()?;
        self.with(|m| unsafe { ffi::il_seq_cp(m, src, id, 0, p1) });
        Ok(id)
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        let raw = self.raw.get_mut().expect("llama context lock");
        unsafe { ffi::il_free(raw.0) };
    }
}

/// What the scheduler calls a sequence: a llama id plus the host side of its
/// state, which the CUDA leaf keeps on the device.
pub struct LlamaSequence {
    id: i32,
    /// Tokens in the cache (the scheduler's positions).
    len: u32,
    /// Each cached token's KV position (its temporal M-RoPE axis).
    pos_of: Vec<i32>,
    rope_delta: i32,
    pending: Option<TokenId>,
    history: Vec<TokenId>,
    rng: Option<Rng>,
}

impl LlamaSequence {
    fn new(id: i32) -> Self {
        Self { id, len: 0, pos_of: Vec::new(), rope_delta: 0, pending: None, history: Vec::new(), rng: None }
    }

    fn standing_on(id: i32, base: &Stood) -> Self {
        Self {
            id,
            len: base.len,
            pos_of: base.pos_of.clone(),
            rope_delta: base.rope_delta,
            pending: None,
            history: base.history.clone(),
            rng: None,
        }
    }

    /// The KV position below which the first `tokens` tokens lie, when that
    /// cut separates them from the rest (an image's tokens share one).
    fn cut(&self, tokens: u32) -> Result<i32, i32> {
        let k = tokens as usize;
        if k == 0 || k > self.pos_of.len() {
            return Err(-3);
        }
        let end = self.pos_of[k - 1] + 1;
        if self.pos_of.get(k).is_some_and(|&next| next < end) {
            return Err(-4);
        }
        Ok(end)
    }

    fn stood(&self, id: i32, tokens: u32) -> Stood {
        let k = tokens as usize;
        Stood {
            id,
            len: tokens,
            pos_of: self.pos_of[..k].to_vec(),
            rope_delta: self.rope_delta,
            history: self.history.clone(),
        }
    }
}

/// A published prefix or a prompt checkpoint: cells under its own id and the
/// host state a sequence standing on it starts from.
pub struct Stood {
    id: i32,
    len: u32,
    pos_of: Vec<i32>,
    rope_delta: i32,
    history: Vec<TokenId>,
}

pub type LlamaPrefix = Stood;
pub type LlamaCheckpoint = Stood;

/// One image's encoder output, rows of the text model's input width.
pub struct LlamaMedia {
    rows: Vec<f32>,
    tokens: usize,
}

fn leaf_error(what: &str) -> i32 {
    let why = ffi::last_error();
    tracing::error!(name: "ignis.llama.leaf_error", step = what, error = %why, "llama leaf call failed");
    -1
}

/// The draw a span or a round makes, redirected inside an open reasoning
/// block (a stop id becomes the close).
fn draw_successor(
    logits: &[f32],
    params: &DecodeParams,
    permitted: &[TokenId],
    seq: &mut LlamaSequence,
    stop_ids: &[TokenId],
    close: Option<TokenId>,
) -> (TokenId, f32, bool) {
    let rng = seq.rng.get_or_insert_with(|| Rng::new(params.seed));
    let drawn = sampling::draw(logits, params, permitted, &seq.history, rng);
    match close {
        Some(close) if stop_ids.contains(&drawn.token) => (close, drawn.probability, true),
        _ => (drawn.token, drawn.probability, false),
    }
}

impl LlamaLeaf {
    #[allow(clippy::too_many_arguments)]
    fn span(
        &self,
        model: &LlamaModel,
        seq: &mut LlamaSequence,
        tokens: &[i32],
        pos4: &[i32],
        embd: Option<&[f32]>,
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        redirect: Option<&mut DrawRedirect<'_>>,
        out_logits: Option<&mut [f32]>,
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        if start_position > seq.len {
            tracing::error!(name: "ignis.llama.span_gap", start_position, cached = seq.len, "a span past the cache end");
            return Err(-5);
        }
        if start_position < seq.len {
            let from = seq.pos_of[start_position as usize];
            model.with(|m| unsafe { ffi::il_seq_rm(m, seq.id, from, -1) });
            seq.pos_of.truncate(start_position as usize);
            seq.len = start_position;
        }
        let mut logits = vec![0.0f32; model.vocab as usize];
        let rc = model.with(|m| unsafe {
            ffi::il_decode_span(
                m,
                seq.id,
                tokens.len() as i32,
                tokens.as_ptr(),
                pos4.as_ptr(),
                embd.map_or(std::ptr::null(), |e| e.as_ptr()),
                logits.as_mut_ptr(),
            )
        });
        if rc != 0 {
            return Err(leaf_error("prefill"));
        }
        seq.pos_of.extend(pos4.chunks_exact(4).map(|p| p[0]));
        seq.len += tokens.len() as u32;
        if let Some(out) = out_logits {
            out.copy_from_slice(&logits);
        }
        if let Some(attention) = attention {
            attention.read = false;
        }
        let (stop_ids, close) = match redirect.as_deref() {
            Some(r) => (r.stop_ids, Some(r.close_id)),
            None => (&[][..], None),
        };
        let (token, probability, redirected) = draw_successor(&logits, &params, permitted, seq, stop_ids, close);
        if let Some(r) = redirect {
            r.redirected = redirected;
        }
        seq.pending = Some(token);
        Ok(probability)
    }

    fn snapshot_header(seq: &LlamaSequence) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(b"IGLL");
        h.extend_from_slice(&1u32.to_le_bytes());
        h.extend_from_slice(&seq.len.to_le_bytes());
        h.extend_from_slice(&seq.rope_delta.to_le_bytes());
        h.extend_from_slice(&seq.pending.map_or(u32::MAX, |t| t).to_le_bytes());
        h.extend_from_slice(&seq.rng.map_or(0u64, |r| r.state()).to_le_bytes());
        h.push(seq.rng.is_some() as u8);
        h.extend_from_slice(&(seq.pos_of.len() as u32).to_le_bytes());
        h.extend_from_slice(&(seq.history.len() as u32).to_le_bytes());
        for p in &seq.pos_of {
            h.extend_from_slice(&p.to_le_bytes());
        }
        for t in &seq.history {
            h.extend_from_slice(&t.to_le_bytes());
        }
        h
    }
}

fn read_u32(src: &[u8], at: &mut usize) -> Result<u32, i32> {
    let b = src.get(*at..*at + 4).ok_or(-6)?;
    *at += 4;
    Ok(u32::from_le_bytes(b.try_into().expect("4 bytes")))
}

impl StepLeaf for LlamaLeaf {
    type Model = LlamaModel;
    type Sequence = LlamaSequence;
    type Prefix = LlamaPrefix;
    type SnapshotBuf = Vec<u8>;
    type Media = LlamaMedia;
    type Checkpoint = LlamaCheckpoint;

    fn load_model(&self) -> Result<Self::Model, i32> {
        use std::ffi::CString;
        let c = &self.config;
        let gguf = CString::new(c.gguf.to_string_lossy().as_bytes()).map_err(|_| -1)?;
        let mmproj = c.mmproj.as_ref().map(|p| CString::new(p.to_string_lossy().as_bytes())).transpose().map_err(|_| -1)?;
        let params = ffi::IlParams {
            n_ctx: c.n_ctx,
            n_seq_max: c.n_seq_max,
            n_batch: c.n_batch,
            n_ubatch: c.n_ubatch,
            n_gpu_layers: -1,
            n_threads: c.n_threads,
            flash_attn: c.flash_attn,
            type_k: c.type_k,
            type_v: c.type_v,
        };
        let raw = unsafe { ffi::il_load(gguf.as_ptr(), mmproj.as_ref().map_or(std::ptr::null(), |m| m.as_ptr()), &params) };
        if raw.is_null() {
            return Err(leaf_error("load"));
        }
        let (vocab, embd_inp, n_ctx, n_seq, cell_bytes, model_bytes) = unsafe {
            (
                ffi::il_n_vocab(raw) as u32,
                ffi::il_n_embd_inp(raw) as usize,
                ffi::il_n_ctx(raw),
                ffi::il_n_seq_max(raw),
                ffi::il_kv_bytes_per_cell(raw),
                ffi::il_model_bytes(raw),
            )
        };
        tracing::info!(
            name: "ignis.llama.loaded",
            gguf = %c.gguf.display(),
            vision = c.mmproj.is_some(),
            vocab, embd_inp, n_ctx, n_seq, cell_bytes, model_bytes,
            "llama.cpp model loaded"
        );
        Ok(LlamaModel {
            raw: Mutex::new(Raw(raw)),
            free_ids: Mutex::new((0..n_seq as i32).rev().collect()),
            vocab,
            embd_inp,
            n_ctx,
            cell_bytes,
            model_bytes,
        })
    }

    fn release_model(&self, model: Self::Model) {
        drop(model);
    }

    fn stats(&self, model: &Self::Model) -> Result<RuntimeStats, i32> {
        let pages = model.n_ctx / KV_PAGE_TOKENS;
        let page_bytes = model.cell_bytes * KV_PAGE_TOKENS as u64;
        Ok(RuntimeStats {
            vram_bytes: model.model_bytes + page_bytes * pages as u64,
            kv_page_tokens: KV_PAGE_TOKENS,
            kv_page_bytes: page_bytes,
            kv_page_count: pages,
            last_step_micros: model.with(|m| unsafe { ffi::il_last_step_micros(m) }),
            free_vram_bytes: mem_available().unwrap_or(0),
            ..RuntimeStats::default()
        })
    }

    fn vocab(&self, model: &Self::Model) -> u32 {
        model.vocab
    }

    fn allocate_sequence(&self, model: &Self::Model, _context_tokens: u32) -> Result<Self::Sequence, i32> {
        Ok(LlamaSequence::new(model.take_id()?))
    }

    fn release_sequence(&self, model: &Self::Model, sequence: Self::Sequence) {
        model.give_id(sequence.id);
    }

    fn allocate_sequence_shared(
        &self,
        model: &Self::Model,
        _context_tokens: u32,
        prefix: &Self::Prefix,
    ) -> Result<Self::Sequence, i32> {
        let id = model.share(prefix.id, -1)?;
        Ok(LlamaSequence::standing_on(id, prefix))
    }

    fn publish_prefix(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        prefix_tokens: u32,
        _retained_slot: u32,
    ) -> Result<Self::Prefix, i32> {
        let end = sequence.cut(prefix_tokens)?;
        let id = model.share(sequence.id, end)?;
        Ok(sequence.stood(id, prefix_tokens))
    }

    fn release_prefix(&self, model: &Self::Model, prefix: Self::Prefix) {
        model.give_id(prefix.id);
    }

    fn capture_checkpoint(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        opener_tokens: u32,
        _retained_slot: u32,
    ) -> Result<Self::Checkpoint, i32> {
        let end = sequence.cut(opener_tokens)?;
        let id = model.share(sequence.id, end)?;
        Ok(sequence.stood(id, opener_tokens))
    }

    fn allocate_sequence_from_checkpoint(
        &self,
        model: &Self::Model,
        _context_tokens: u32,
        checkpoint: &Self::Checkpoint,
    ) -> Result<(Self::Sequence, u64), i32> {
        let t0 = std::time::Instant::now();
        let id = model.share(checkpoint.id, -1)?;
        Ok((LlamaSequence::standing_on(id, checkpoint), t0.elapsed().as_micros() as u64))
    }

    fn release_checkpoint(&self, model: &Self::Model, checkpoint: Self::Checkpoint) {
        model.give_id(checkpoint.id);
    }

    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        redirect: Option<&mut DrawRedirect<'_>>,
        out_logits: Option<&mut [f32]>,
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let base = start_position as i32 + sequence.rope_delta;
        let pos4: Vec<i32> = (0..ids.len() as i32).flat_map(|i| [base + i, base + i, base + i, 0]).collect();
        self.span(model, sequence, &ids, &pos4, None, start_position, params, permitted, redirect, out_logits, attention)
    }

    fn encode_media(&self, model: &Self::Model, item: &MediaItem) -> Result<Self::Media, i32> {
        let (gh, gw) = (item.grid.h as usize, item.grid.w as usize);
        let rgb = unpatch::unpatch(&item.patches, gh, gw).ok_or_else(|| {
            tracing::error!(name: "ignis.llama.media_grid", gh, gw, rows = item.patches.len(), "patch rows do not fill the grid");
            -1
        })?;
        let (nx, ny) = ((gw * unpatch::PATCH) as i32, (gh * unpatch::PATCH) as i32);
        let tokens = model.with(|m| unsafe { ffi::il_image_tokens(m, nx, ny) }) as usize;
        if tokens as u64 != item.grid.vision_tokens() {
            tracing::error!(name: "ignis.llama.media_tokens", encoder = tokens, prompt = item.grid.vision_tokens(),
                            "the encoder would emit a different number of tokens than the prompt holds placeholders");
            return Err(-1);
        }
        let mut rows = vec![0.0f32; tokens * model.embd_inp];
        let got = model.with(|m| unsafe { ffi::il_encode_image(m, rgb.as_ptr(), nx, ny, rows.as_mut_ptr(), rows.len()) });
        if got < 0 {
            return Err(leaf_error("encode media"));
        }
        Ok(LlamaMedia { rows, tokens })
    }

    fn prefill_multimodal(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
        permitted: &[TokenId],
        redirect: Option<&mut DrawRedirect<'_>>,
        span: MultimodalSpan<'_, Self::Media>,
        out_logits: Option<&mut [f32]>,
        attention: Option<&mut AttentionRead>,
    ) -> Result<f32, i32> {
        let n = tokens.len();
        if span.positions.len() != 3 * n {
            tracing::error!(name: "ignis.llama.span_positions", tokens = n, positions = span.positions.len(), "positions are not [3, tokens]");
            return Err(-1);
        }
        let mut ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut embd: Option<&[f32]> = None;
        if let Some(media) = span.media {
            let first = media.first_column as usize;
            let count = media.scatter_indices.len();
            if first + count > media.embedding.tokens {
                tracing::error!(name: "ignis.llama.media_columns", first, count, have = media.embedding.tokens, "columns past the embedding");
                return Err(-1);
            }
            for &i in media.scatter_indices {
                ids[i as usize] = -1;
            }
            let w = model.embd_inp;
            embd = Some(&media.embedding.rows[first * w..(first + count) * w]);
        }
        let pos4: Vec<i32> = (0..n)
            .flat_map(|i| [span.positions[i], span.positions[n + i], span.positions[2 * n + i], 0])
            .collect();
        sequence.rope_delta = span.rope_delta;
        self.span(model, sequence, &ids, &pos4, embd, start_position, params, permitted, redirect, out_logits, attention)
    }

    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        lanes: &[DecodeLane<'_>],
    ) -> Result<Vec<LaneRun>, i32> {
        if sequences.len() != lanes.len() {
            return Err(-1);
        }
        let mut seqs = Vec::with_capacity(lanes.len());
        let mut tokens = Vec::with_capacity(lanes.len());
        let mut pos = Vec::with_capacity(lanes.len());
        for s in sequences.iter() {
            let Some(pending) = s.pending else {
                tracing::error!(name: "ignis.llama.no_pending", "a decode round over a sequence nothing was drawn for");
                return Err(-1);
            };
            seqs.push(s.id);
            tokens.push(pending as i32);
            pos.push(s.len as i32 + s.rope_delta);
        }
        let vocab = model.vocab as usize;
        let mut logits = vec![0.0f32; vocab * lanes.len()];
        let rc = model.with(|m| unsafe {
            ffi::il_decode_round(m, lanes.len() as i32, seqs.as_ptr(), tokens.as_ptr(), pos.as_ptr(), logits.as_mut_ptr())
        });
        if rc != 0 {
            return Err(leaf_error("decode"));
        }
        let mut runs = Vec::with_capacity(lanes.len());
        for (i, (seq, lane)) in sequences.iter_mut().zip(lanes).enumerate() {
            let emitted = tokens[i] as TokenId;
            seq.pos_of.push(pos[i]);
            seq.len += 1;
            seq.history.push(emitted);
            let (next, probability, redirected) = draw_successor(
                &logits[i * vocab..(i + 1) * vocab],
                &lane.params,
                lane.permitted,
                seq,
                lane.stop_ids,
                lane.reasoning_close,
            );
            seq.pending = Some(next);
            runs.push(LaneRun {
                redirected,
                ..LaneRun::drawn(emitted, (!lane.permitted.is_empty()).then_some(probability))
            });
        }
        Ok(runs)
    }

    fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32> {
        Ok(vec![0u8; bytes as usize])
    }

    fn snapshot_bytes(&self, model: &Self::Model, sequence: &Self::Sequence) -> Result<u64, i32> {
        let state = model.with(|m| unsafe { ffi::il_state_size(m, sequence.id) });
        Ok((Self::snapshot_header(sequence).len() + 8 + state) as u64)
    }

    fn snapshot_into(&self, model: &Self::Model, sequence: &Self::Sequence, dst: &mut [u8]) -> Result<(), i32> {
        let header = Self::snapshot_header(sequence);
        let h = header.len();
        if dst.len() < h + 8 {
            return Err(-6);
        }
        dst[..h].copy_from_slice(&header);
        let room = dst.len() - h - 8;
        let wrote = model.with(|m| unsafe { ffi::il_state_get(m, sequence.id, dst[h + 8..].as_mut_ptr(), room) });
        if wrote == 0 {
            return Err(leaf_error("snapshot"));
        }
        dst[h..h + 8].copy_from_slice(&(wrote as u64).to_le_bytes());
        Ok(())
    }

    fn restore_sequence(&self, model: &Self::Model, sequence: &mut Self::Sequence, src: &[u8]) -> Result<(), i32> {
        if src.get(..4) != Some(b"IGLL") {
            return Err(-6);
        }
        let mut at = 4;
        let _version = read_u32(src, &mut at)?;
        let len = read_u32(src, &mut at)?;
        let rope_delta = read_u32(src, &mut at)? as i32;
        let pending = read_u32(src, &mut at)?;
        let rng = u64::from_le_bytes(src.get(at..at + 8).ok_or(-6)?.try_into().expect("8 bytes"));
        at += 8;
        let has_rng = *src.get(at).ok_or(-6)? != 0;
        at += 1;
        let n_pos = read_u32(src, &mut at)? as usize;
        let n_hist = read_u32(src, &mut at)? as usize;
        let mut pos_of = Vec::with_capacity(n_pos);
        for _ in 0..n_pos {
            pos_of.push(read_u32(src, &mut at)? as i32);
        }
        let mut history = Vec::with_capacity(n_hist);
        for _ in 0..n_hist {
            history.push(read_u32(src, &mut at)?);
        }
        let state = u64::from_le_bytes(src.get(at..at + 8).ok_or(-6)?.try_into().expect("8 bytes")) as usize;
        at += 8;
        let blob = src.get(at..at + state).ok_or(-6)?;
        model.with(|m| unsafe { ffi::il_seq_rm(m, sequence.id, -1, -1) });
        if model.with(|m| unsafe { ffi::il_state_set(m, sequence.id, blob.as_ptr(), blob.len()) }) == 0 {
            return Err(leaf_error("restore"));
        }
        sequence.len = len;
        sequence.rope_delta = rope_delta;
        sequence.pending = (pending != u32::MAX).then_some(pending);
        sequence.rng = has_rng.then(|| Rng::from_state(rng));
        sequence.pos_of = pos_of;
        sequence.history = history;
        Ok(())
    }
}

/// What the board's unified memory has free (`MemAvailable`).
fn mem_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}
