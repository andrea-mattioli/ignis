# ADR 0052 — a step leaf over llama.cpp, for the cards the CUDA leaf does not target

## Status

Proposed (2026-10-18). First built and run on a Jetson Orin NX 16 GB
(JetPack 6, CUDA 12.6, Ampere `sm_87`, unified memory) serving
Qwen3-VL-4B-Instruct. Adds a fourth model family (`qwen3-vl-4b`).

## Context

ignis is one model family on one class of card (`CONTEXT.md`): the leaf's
forward pass is the Qwen3.8 program in NVFP4 on `sm_120a` (block-scaled FP4
MMA, TMA, `setmaxnreg`). An Orin has none of those instructions and 16 GB of
memory shared with everything else on the board, so neither the leaf nor the
27B can run there. What does run there is everything above the step: the Rust
core builds for `aarch64` unchanged and its CPU test suites pass on the board.

The workload that wants ignis on such a board is a robot's: closed decisions
over camera frames (`/v1/decide`), a few at a time, with the decode-bound
generation rare. The value it needs is the core's -- readouts with nothing
generated, prompt reuse, the media cache, admission and lanes -- more than the
leaf's kernels. llama.cpp already has an Ampere CUDA backend, GGUF weights of
small vision models, a unified-KV context with per-sequence ids, and its own
vision encoder.

## Decision

**A second implementation of `StepLeaf`, `ignis-llama`, whose step is a
llama.cpp context.** The core does not change; the leaf is selected by the
artifact, a model directory instead of a `.ninfer` container.

- **The boundary is the step ABI's, cut down.** A flat C shim
  (`crates/llama/csrc/ignis_llama.h`) over `libllama` and the encoder in
  `libmtmd` exposes load, span prefill, a batched decode round, sequence copy
  and removal, per-sequence state blobs and image encode. Rust owns the rest,
  as ADR 0009 has it.
- **A sequence, a published prefix and a prompt checkpoint are llama sequence
  ids in one unified KV buffer.** Publishing tags the prefix's cells with one
  more id (`llama_memory_seq_cp`): a claimant shares them in place, the way
  the CUDA leaf shares pages. The scheduler's page ledger is the buffer's
  cells in `KV_PAGE_TOKENS` pages, read back from the leaf at load.
- **The pending token is the CUDA leaf's.** A prefill draws the span's
  successor and keeps it; a round emits it, feeds it and draws the next. A
  constrained run's first token is the prefill's draw, so `number`, `point`
  and `box` by chain work unchanged.
- **The draw runs on the host.** ADR 0034 kept constrained sampling in the
  leaf to keep 970 KB of logits off PCIe; on unified memory llama.cpp has
  already copied them to host memory, so penalties, the permitted set,
  temperature, top-k, top-p and the in-set probability are applied there.
  Seeded per sequence, not bit-identical to the CUDA epilogue.
- **Images keep the ignis processor.** Qwen3-VL has the Qwen3.8 vision
  contract (16 px patches, the frame doubled, 2×2 merge, M-RoPE). The leaf
  inverts the processor's patch packing exactly and gives llama.cpp's encoder
  the normalized image, at the grid the prompt's placeholders were counted
  for; an encoder that would emit a different count is a failed encode. The
  placeholder ids are the tokenizer's (`VisionProcessor::with_tokenizer_pads`).
- **Positions are ignis's.** A text token goes in at `cache index + rope
  delta`, an image token at the processor's `[t, h, w]`.
- **An attention readout is not read.** No head is calibrated for this
  model, and the leaf cannot yet read scores: `point` and `box` answer by
  chain, `locate` is refused at load, and a head readout asked anyway fails
  its question (ADR 0038), never answers from anything else.
- **A template that never writes `<think>` never starts a generation in
  reasoning.** Qwen3-VL-Instruct's has none; with the old default its answer
  was read as reasoning and its EOS redirected to a close it had never opened.

The model directory holds `ignis-llama.json` (model id, GGUF, mmproj,
expected sizes), the GGUF files and the six frontend resources of the model's
Hugging Face repository; `FrontendSet::from_resources` reads them as the
container's. A build without `--features llama` refuses such a directory by
name rather than serving it on the mock.

## Measured (Orin NX 16 GB, MAXN, aircraft services stopped, 2026-10-18)

Qwen3-VL-4B-Instruct Q4_K_M + mmproj Q8_0, 16,384 KV cells, f16 KV, flash
attention, 48 VisDrone crop pairs (24 persons, 24 vehicles), two closed
questions per pair (is it a person, what class is it):

| | both questions |
|---|---|
| ignis `/v1/decide`, one request, new image | 0.74 s p50 (0.75 p95) |
| ignis `/v1/decide`, the same image again (media cache) | 0.61 s |
| llama-server, two one-token requests from the client, the second on its prefix cache | 0.52 + 0.15 = 0.66 s |
| text-only `choice` over 103 tokens, three options | 0.26 s |
| `point` by chain at 960 px | ~2.5 s |

Answers agree with the llama-server readout of the same model (21/24 persons
kept, 6/24 vehicles let through at P ≥ 0.95 and P ≥ 0.9). Point accuracy was
not measured.

## Consequences

- The two questions of one request each prefill the whole state: siblings with
  different instructions share nothing (ignis #271). On this board that is the
  0.08 s the client-side trick still wins by; sharing the state's prefix
  across a request's questions is the next lever, and it is engine-wide.
- A head-calibrated `point`/`box` needs the attention scores out of
  llama.cpp (its evaluation callback) and a calibration of this model on the
  pointing scenes. Until then, the chain.
- KV-disk and the windowed transfer are off for this family; the KV-RAM tier
  works through llama.cpp's state blobs.
- The family is not a config scope in `FAMILIES`: it takes the shared values.
  The vision and media fields now apply to it as to the 27B.
