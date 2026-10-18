// The llama.cpp leaf (ignis_llama.h). One llama_context with a unified KV
// buffer: a sequence id per Ignis sequence, prefix and checkpoint, so a
// shared prefix is the same cells tagged with one more id, never a copy.
#include "ignis_llama.h"

#include "llama.h"
#include "ggml.h"
#include "clip.h"
#include "clip-impl.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

thread_local std::string g_error;

int32_t fail(const std::string &why, int32_t code = -1) {
    g_error = why;
    return code;
}

void quiet_log(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int64_t meta_int(const llama_model *model, const std::string &key, int64_t fallback) {
    char buf[64];
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof buf) < 0) {
        return fallback;
    }
    return std::strtoll(buf, nullptr, 10);
}

}  // namespace

struct il_model {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    clip_ctx *vision = nullptr;
    llama_batch_ext *batch = nullptr;
    int32_t n_vocab = 0;
    int32_t n_embd_inp = 0;
    uint32_t n_batch = 0;
    uint64_t kv_bytes_per_cell = 0;
    uint64_t last_step_us = 0;
    int32_t n_threads = 4;
};

extern "C" {

const char *il_last_error(void) { return g_error.c_str(); }

il_model *il_load(const char *gguf, const char *mmproj, const il_params *p) {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(quiet_log, nullptr);
        llama_backend_init();
    });
    auto *m = new il_model();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p->n_gpu_layers < 0 ? 999 : p->n_gpu_layers;
    m->model = llama_model_load_from_file(gguf, mp);
    if (!m->model) {
        fail(std::string("cannot load the model ") + gguf);
        delete m;
        return nullptr;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = p->n_ctx;
    cp.n_batch = p->n_batch;
    cp.n_ubatch = p->n_ubatch;
    cp.n_seq_max = p->n_seq_max;
    cp.n_threads = p->n_threads;
    cp.n_threads_batch = p->n_threads;
    cp.kv_unified = true;
    cp.flash_attn_type = (llama_flash_attn_type) p->flash_attn;
    if (p->type_k >= 0) cp.type_k = (ggml_type) p->type_k;
    if (p->type_v >= 0) cp.type_v = (ggml_type) p->type_v;
    cp.no_perf = true;
    m->ctx = llama_init_from_model(m->model, cp);
    if (!m->ctx) {
        fail("cannot create the llama context (n_ctx, n_seq_max or memory)");
        il_free(m);
        return nullptr;
    }
    m->batch = llama_batch_ext_init(m->ctx);
    m->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m->model));
    m->n_embd_inp = llama_model_n_embd_inp(m->model);
    m->n_batch = p->n_batch;
    m->n_threads = p->n_threads;

    char arch[64] = {0};
    llama_model_meta_val_str(m->model, "general.architecture", arch, sizeof arch);
    const std::string a(arch);
    const int64_t layers = meta_int(m->model, a + ".block_count", 0);
    const int64_t kv_heads = meta_int(m->model, a + ".attention.head_count_kv", 0);
    const int64_t k_len = meta_int(m->model, a + ".attention.key_length", 0);
    const int64_t v_len = meta_int(m->model, a + ".attention.value_length", k_len);
    const ggml_type tk = p->type_k >= 0 ? (ggml_type) p->type_k : GGML_TYPE_F16;
    const ggml_type tv = p->type_v >= 0 ? (ggml_type) p->type_v : GGML_TYPE_F16;
    m->kv_bytes_per_cell = (uint64_t) layers * (ggml_row_size(tk, kv_heads * k_len) + ggml_row_size(tv, kv_heads * v_len));

    if (mmproj) {
        clip_context_params vp{};
        vp.use_gpu = true;
        vp.device = nullptr;
        vp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
        vp.image_min_tokens = -1;
        vp.image_max_tokens = -1;
        vp.warmup = true;
        clip_init_result r = clip_init(mmproj, vp);
        m->vision = r.ctx_v;
        if (r.ctx_a) clip_free(r.ctx_a);
        if (r.ctx_gen_a) clip_free(r.ctx_gen_a);
        if (!m->vision) {
            fail(std::string("cannot load the vision encoder ") + mmproj);
            il_free(m);
            return nullptr;
        }
        if (clip_n_mmproj_embd(m->vision) != m->n_embd_inp) {
            fail("the vision encoder's embedding width is not the text model's input width");
            il_free(m);
            return nullptr;
        }
    }
    return m;
}

void il_free(il_model *m) {
    if (!m) return;
    if (m->batch) llama_batch_ext_free(m->batch);
    if (m->vision) clip_free(m->vision);
    if (m->ctx) llama_free(m->ctx);
    if (m->model) llama_model_free(m->model);
    delete m;
}

int32_t il_n_vocab(const il_model *m) { return m->n_vocab; }
int32_t il_n_embd_inp(const il_model *m) { return m->n_embd_inp; }
uint32_t il_n_ctx(const il_model *m) { return llama_n_ctx(m->ctx); }
uint32_t il_n_seq_max(const il_model *m) { return llama_n_seq_max(m->ctx); }
uint64_t il_kv_bytes_per_cell(const il_model *m) { return m->kv_bytes_per_cell; }
uint64_t il_model_bytes(const il_model *m) { return llama_model_size(m->model); }
uint64_t il_last_step_micros(const il_model *m) { return m->last_step_us; }

int32_t il_seq_rm(il_model *m, int32_t seq, int32_t p0, int32_t p1) {
    return llama_memory_seq_rm(llama_get_memory(m->ctx), seq, p0, p1) ? 0 : fail("seq_rm refused");
}

int32_t il_seq_cp(il_model *m, int32_t src, int32_t dst, int32_t p0, int32_t p1) {
    llama_memory_seq_cp(llama_get_memory(m->ctx), src, dst, p0, p1);
    return 0;
}

int32_t il_decode_span(il_model *m, int32_t seq, int32_t n, const int32_t *tokens, const int32_t *pos4,
                       const float *embd, float *out_logits) {
    if (n <= 0) return fail("empty span");
    const auto t0 = std::chrono::steady_clock::now();
    size_t row = 0;
    for (int32_t begin = 0; begin < n; begin += (int32_t) m->n_batch) {
        const int32_t end = std::min(n, begin + (int32_t) m->n_batch);
        llama_batch_ext_clear(m->batch);
        for (int32_t i = begin; i < end; ++i) {
            int32_t idx;
            if (tokens[i] >= 0) {
                idx = llama_batch_ext_add_token(m->batch, seq, tokens[i]);
            } else {
                const llama_embd e = {embd + row * (size_t) m->n_embd_inp, 1, (size_t) m->n_embd_inp};
                ++row;
                idx = llama_batch_ext_add_embd(m->batch, seq, e);
            }
            if (idx < 0) return fail("batch add refused", idx);
            const llama_pos pos[4] = {pos4[4 * i], pos4[4 * i + 1], pos4[4 * i + 2], pos4[4 * i + 3]};
            llama_batch_ext_set_pos(m->batch, idx, pos);
            if (i == n - 1 && out_logits) llama_batch_ext_set_output_logits(m->batch, idx, true);
        }
        const int32_t r = llama_process(m->ctx, LLAMA_PROCESS_TYPE_DECODE, m->batch);
        if (r != 0) return fail("llama_process failed on a span (" + std::to_string(r) + ")", r < 0 ? r : -r - 10);
    }
    if (out_logits) {
        const float *l = llama_get_logits_ith(m->ctx, -1);
        if (!l) return fail("no logits for the span's last position");
        std::memcpy(out_logits, l, sizeof(float) * (size_t) m->n_vocab);
    }
    m->last_step_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - t0).count();
    return 0;
}

int32_t il_decode_round(il_model *m, int32_t n, const int32_t *seqs, const int32_t *tokens, const int32_t *pos,
                        float *out_logits) {
    if (n <= 0) return fail("empty round");
    const auto t0 = std::chrono::steady_clock::now();
    llama_batch_ext_clear(m->batch);
    for (int32_t i = 0; i < n; ++i) {
        const int32_t idx = llama_batch_ext_add_token(m->batch, seqs[i], tokens[i]);
        if (idx < 0) return fail("batch add refused", idx);
        const llama_pos p[4] = {pos[i], pos[i], pos[i], 0};
        llama_batch_ext_set_pos(m->batch, idx, p);
        llama_batch_ext_set_output_logits(m->batch, idx, true);
    }
    const int32_t r = llama_process(m->ctx, LLAMA_PROCESS_TYPE_DECODE, m->batch);
    if (r != 0) return fail("llama_process failed on a round (" + std::to_string(r) + ")", r < 0 ? r : -r - 10);
    for (int32_t i = 0; i < n; ++i) {
        const float *l = llama_get_logits_ith(m->ctx, i);
        if (!l) return fail("no logits for lane " + std::to_string(i));
        std::memcpy(out_logits + (size_t) i * m->n_vocab, l, sizeof(float) * (size_t) m->n_vocab);
    }
    m->last_step_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - t0).count();
    return 0;
}

size_t il_state_size(il_model *m, int32_t seq) { return llama_state_seq_get_size(m->ctx, seq); }

size_t il_state_get(il_model *m, int32_t seq, uint8_t *dst, size_t size) {
    return llama_state_seq_get_data(m->ctx, dst, size, seq);
}

size_t il_state_set(il_model *m, int32_t seq, const uint8_t *src, size_t size) {
    return llama_state_seq_set_data(m->ctx, src, size, seq);
}

int32_t il_image_tokens(il_model *m, int32_t nx, int32_t ny) {
    if (!m->vision) return 0;
    clip_image_f32 img;
    img.set_size({nx, ny}, true, false);
    return clip_n_output_tokens(m->vision, &img);
}

int32_t il_encode_image(il_model *m, const float *rgb, int32_t nx, int32_t ny, float *out, size_t out_floats) {
    if (!m->vision) return fail("this load has no vision encoder");
    clip_image_f32 img;
    img.set_size({nx, ny}, false, false);
    img.cpy_buf(std::vector<float>(rgb, rgb + (size_t) nx * ny * 3));
    const int32_t tokens = clip_n_output_tokens(m->vision, &img);
    if ((size_t) tokens * m->n_embd_inp > out_floats) return fail("the embedding buffer is too small for the image");
    // the encoder writes into the caller's vector, sized beforehand (as mtmd sizes it)
    std::vector<float> vec((size_t) tokens * m->n_embd_inp);
    if (!clip_image_encode(m->vision, m->n_threads, &img, vec)) return fail("the vision encoder failed");
    if (vec.size() != (size_t) tokens * m->n_embd_inp) {
        return fail("the vision encoder returned " + std::to_string(vec.size()) + " floats, not " +
                    std::to_string(tokens) + " tokens x " + std::to_string(m->n_embd_inp) + " (image " +
                    std::to_string(nx) + "x" + std::to_string(ny) + ")");
    }
    std::memcpy(out, vec.data(), vec.size() * sizeof(float));
    return tokens;
}

}  // extern "C"
