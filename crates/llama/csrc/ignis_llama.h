/* The llama.cpp leaf's flat C ABI (crates/llama, ADR 0052).
 *
 * The CUDA leaf's step ABI (ADR 0009) cut to what a llama.cpp context can
 * do: one model, one context with a unified KV buffer, a sequence id per
 * Ignis sequence, prefix, checkpoint and snapshot target, and the vision
 * encoder of the model's mmproj. Everything above the step stays in Rust:
 * sampling, permitted sets, the reasoning redirect, the pending token, the
 * readout gather. Integer returns: 0 ok, negative a failure the Rust side
 * names; the message of the last failure is il_last_error(). */
#ifndef IGNIS_LLAMA_H
#define IGNIS_LLAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct il_model il_model;

typedef struct il_params {
    uint32_t n_ctx;        /* KV cells shared by every sequence (kv_unified) */
    uint32_t n_seq_max;    /* sequence ids: live sequences + prefixes + checkpoints + one scratch */
    uint32_t n_batch;      /* largest span one il_decode_span call submits at a time */
    uint32_t n_ubatch;
    int32_t n_gpu_layers;  /* -1: all */
    int32_t n_threads;
    int32_t flash_attn;    /* -1 auto, 0 off, 1 on */
    int32_t type_k;        /* ggml_type, -1 model default (f16) */
    int32_t type_v;
} il_params;

/* Load the text model and, when mmproj is not NULL, its vision encoder. */
il_model *il_load(const char *gguf, const char *mmproj, const il_params *params);
void il_free(il_model *m);
const char *il_last_error(void);

int32_t il_n_vocab(const il_model *m);
int32_t il_n_embd_inp(const il_model *m);   /* width of one embedding row a span takes */
uint32_t il_n_ctx(const il_model *m);
uint32_t il_n_seq_max(const il_model *m);
uint64_t il_kv_bytes_per_cell(const il_model *m);
uint64_t il_model_bytes(const il_model *m);
uint64_t il_last_step_micros(const il_model *m);

/* KV positions of `seq` in [p0, p1) (p1 < 0: to the end). */
int32_t il_seq_rm(il_model *m, int32_t seq, int32_t p0, int32_t p1);
/* `dst` also holds `src`'s cells in [p0, p1): shared in place, nothing copied. */
int32_t il_seq_cp(il_model *m, int32_t src, int32_t dst, int32_t p0, int32_t p1);

/* One span of one sequence. Entry i is text token tokens[i] at position
 * pos4[4*i] when tokens[i] >= 0, or the next row of `embd` (il_n_embd_inp
 * floats) at M-RoPE position pos4[4*i..4*i+4] = [t, y, x, z] when
 * tokens[i] < 0. The last entry's logits are copied to out_logits
 * (il_n_vocab floats) when it is not NULL. */
int32_t il_decode_span(il_model *m, int32_t seq, int32_t n, const int32_t *tokens, const int32_t *pos4,
                       const float *embd, float *out_logits);

/* One round: lane i feeds tokens[i] at pos[i] into seqs[i]; out_logits holds
 * n * il_n_vocab floats, lane-major. */
int32_t il_decode_round(il_model *m, int32_t n, const int32_t *seqs, const int32_t *tokens, const int32_t *pos,
                        float *out_logits);

/* A sequence's state as a host blob (llama_state_seq_*). */
size_t il_state_size(il_model *m, int32_t seq);
size_t il_state_get(il_model *m, int32_t seq, uint8_t *dst, size_t size);
size_t il_state_set(il_model *m, int32_t seq, const uint8_t *src, size_t size);

/* Vision: merged tokens for an nx x ny image, and its encode. `rgb` is the
 * image as the processor normalized it, row-major interleaved RGB floats;
 * `out` takes tokens * il_n_embd_inp floats in raster order of the merged
 * grid. 0 tokens when the model has no vision encoder. */
int32_t il_image_tokens(il_model *m, int32_t nx, int32_t ny);
int32_t il_encode_image(il_model *m, const float *rgb, int32_t nx, int32_t ny, float *out, size_t out_floats);

#ifdef __cplusplus
}
#endif

#endif
