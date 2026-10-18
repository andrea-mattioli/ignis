//! `csrc/ignis_llama.h`, 1:1.

use std::ffi::{c_char, CStr};

#[repr(C)]
pub struct IlModel {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IlParams {
    pub n_ctx: u32,
    pub n_seq_max: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub flash_attn: i32,
    pub type_k: i32,
    pub type_v: i32,
}

unsafe extern "C" {
    pub fn il_load(gguf: *const c_char, mmproj: *const c_char, params: *const IlParams) -> *mut IlModel;
    pub fn il_free(m: *mut IlModel);
    pub fn il_last_error() -> *const c_char;
    pub fn il_n_vocab(m: *const IlModel) -> i32;
    pub fn il_n_embd_inp(m: *const IlModel) -> i32;
    pub fn il_n_ctx(m: *const IlModel) -> u32;
    pub fn il_n_seq_max(m: *const IlModel) -> u32;
    pub fn il_kv_bytes_per_cell(m: *const IlModel) -> u64;
    pub fn il_model_bytes(m: *const IlModel) -> u64;
    pub fn il_last_step_micros(m: *const IlModel) -> u64;
    pub fn il_seq_rm(m: *mut IlModel, seq: i32, p0: i32, p1: i32) -> i32;
    pub fn il_seq_cp(m: *mut IlModel, src: i32, dst: i32, p0: i32, p1: i32) -> i32;
    pub fn il_decode_span(
        m: *mut IlModel,
        seq: i32,
        n: i32,
        tokens: *const i32,
        pos4: *const i32,
        embd: *const f32,
        out_logits: *mut f32,
    ) -> i32;
    pub fn il_decode_round(
        m: *mut IlModel,
        n: i32,
        seqs: *const i32,
        tokens: *const i32,
        pos: *const i32,
        out_logits: *mut f32,
    ) -> i32;
    pub fn il_state_size(m: *mut IlModel, seq: i32) -> usize;
    pub fn il_state_get(m: *mut IlModel, seq: i32, dst: *mut u8, size: usize) -> usize;
    pub fn il_state_set(m: *mut IlModel, seq: i32, src: *const u8, size: usize) -> usize;
    pub fn il_image_tokens(m: *mut IlModel, nx: i32, ny: i32) -> i32;
    pub fn il_encode_image(m: *mut IlModel, rgb: *const f32, nx: i32, ny: i32, out: *mut f32, out_floats: usize) -> i32;
}

/// The shim's message for its last failure on this thread.
pub fn last_error() -> String {
    // SAFETY: the shim returns a pointer to a thread-local NUL-terminated string.
    unsafe { CStr::from_ptr(il_last_error()) }.to_string_lossy().into_owned()
}
