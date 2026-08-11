/*
 * bench_m1.c — synthetic-tensor kernel micro-benchmark for H3 shapes.
 *
 * Purpose: measure the portable (non-M5) Metal/MPSGraph path on real H3
 * tensor shapes WITHOUT the 498 GB model. This is the measurement substrate
 * for the M1 Max autoresearch loop.
 *
 * Build (from repo root, after make -j8):
 *   clang -O2 -I. -o tests/bench_m1 tests/bench_m1.c h3_gpu.m \
 *       h3_metal.m h3_safetensors.c h3_dit.c h3_weights.c h3_dit_schedule.c \
 *       h3_video_vae.c h3_audio_vae.c h3_text_encoder.c h3_vision_encoder.c \
 *       h3_multimodal.c h3_ffmpeg.c h3_terminal.c h3_cli.c h3.c main.c \
 *       -framework Metal -framework MetalPerformanceShadersGraph \
 *       -framework MetalPerformanceShaders -framework Foundation \
 *       -framework CoreVideo -framework AVFoundation -framework CoreMedia \
 *       -lobjc -lm
 *
 * Simpler: add a target to the Makefile. See Makefile for the exact link line
 * used by the other test binaries.
 *
 * Usage: ./tests/bench_m1 [rows] [iterations]
 *   rows defaults to 2048 (max H3 working set), iterations to 10.
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- H3 real shapes (from h3_dit.c: FFN=14336, H3_DIT_HIDDEN=5376) ---- */
#define H3_WIDTH      5376u   /* H3_DIT_HIDDEN */
#define FFN           14336u  /* h3_dit.c FFN */
#define FC1_OUT       28672u  /* FFN * 2 (SwiGLU gate+up) */
#define QKV_OUT       7168u   /* 56 heads x 128 head_dim (h3_gpu.m:3799) */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void fill_rand(float *v, size_t n) {
    /* Deterministic but non-trivial: avoids RNG cost dominating tiny benches. */
    for (size_t i = 0; i < n; i++)
        v[i] = (float)((i * 2654435761u) % 1000u) / 500.0f - 1.0f;
}

static void fail(const char *msg) {
    fprintf(stderr, "bench_m1: %s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    unsigned rows = argc > 1 ? (unsigned)atoi(argv[1]) : 2048;
    unsigned iters = argc > 2 ? (unsigned)atoi(argv[2]) : 10;
    char error[512];

    printf("# bench_m1: H3-shape synthetic kernel micro-benchmark\n");
    printf("# rows=%u iterations=%u\n", rows, iters);

    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);

    printf("# device: %s | is_m5=%d | has_int8_mlp=%d | has_nax_mlp=%d\n",
           gpu ? "metal" : "?", h3_gpu_is_m5(gpu),
           h3_gpu_has_int8_mlp(gpu), h3_gpu_has_nax_mlp(gpu));

    /* ---- allocate synthetic tensors ---- */
    size_t in_el = (size_t)rows * H3_WIDTH;
    size_t fc1_el = (size_t)FC1_OUT * H3_WIDTH;   /* 28672 x 5376 */
    size_t fc2_el = (size_t)H3_WIDTH * FFN;       /* 5376 x 14336 */
    size_t qkv_el = (size_t)H3_WIDTH * QKV_OUT;
    size_t attn_el = (size_t)QKV_OUT * H3_WIDTH;

    float *buf = malloc(sizeof(float) *
        (in_el + fc1_el + fc2_el + qkv_el + attn_el + 4 * in_el + 2 * fc2_el + 3 * (size_t)rows * QKV_OUT + 2 * (size_t)rows * H3_WIDTH));
    if (!buf) fail("malloc");
    float *p = buf;

    float *input = p; p += in_el;
    float *fc1_w = p; p += fc1_el;
    float *fc2_w = p; p += fc2_el;
    float *qkv_w = p; p += qkv_el;
    float *attn_w = p; p += attn_el;
    float *bias_5376 = p; p += H3_WIDTH;
    float *bias_21504 = p; p += FC1_OUT;
    float *bias_7168 = p; p += QKV_OUT;
    float *qkv_buf = p; p += 3 * (size_t)rows * QKV_OUT;
    float *out1 = p; p += (size_t)rows * FC1_OUT;
    float *out2 = p; p += (size_t)rows * H3_WIDTH;
    float *attn_out = p; p += (size_t)rows * QKV_OUT;
    float *tmp1 = p; p += (size_t)rows * FC1_OUT;
    float *tmp2 = p; p += (size_t)rows * FFN;    /* swiglu output: rows x 14336 */

    fill_rand(input, in_el);
    fill_rand(fc1_w, fc1_el);
    fill_rand(fc2_w, fc2_el);
    fill_rand(qkv_w, qkv_el);
    fill_rand(attn_w, attn_el);
    memset(bias_5376, 0, H3_WIDTH * sizeof(float));
    memset(bias_21504, 0, FC1_OUT * sizeof(float));
    memset(bias_7168, 0, QKV_OUT * sizeof(float));
    memset(qkv_buf, 0, 3 * (size_t)rows * QKV_OUT * sizeof(float));
    memset(out1, 0, (size_t)rows * FC1_OUT * sizeof(float));
    memset(out2, 0, (size_t)rows * H3_WIDTH * sizeof(float));
    memset(attn_out, 0, (size_t)rows * QKV_OUT * sizeof(float));
    memset(tmp1, 0, (size_t)rows * FC1_OUT * sizeof(float));
    memset(tmp2, 0, (size_t)rows * FFN * sizeof(float));

    /* BF16 buffers for the real portable MLP path (after fill_rand) */
    size_t bf16_in_el = in_el, bf16_fc1_el = fc1_el, bf16_fc2_el = fc2_el;
    size_t bf16_out_el = (size_t)rows * H3_WIDTH;
    size_t bf16_tmp_el = (size_t)rows * FC1_OUT;
    uint16_t *bf16_buf = malloc(sizeof(uint16_t) *
        (bf16_in_el + bf16_fc1_el + bf16_fc2_el + bf16_out_el + bf16_tmp_el + bf16_tmp_el));
    if (!bf16_buf) fail("bf16 malloc");
    uint16_t *bp = bf16_buf;
    uint16_t *b_in = bp; bp += bf16_in_el;
    uint16_t *b_fc1 = bp; bp += bf16_fc1_el;
    uint16_t *b_fc2 = bp; bp += bf16_fc2_el;
    uint16_t *b_out = bp; bp += bf16_out_el;
    uint16_t *b_tmp1 = bp; bp += bf16_tmp_el;
    uint16_t *b_tmp2 = bp; bp += bf16_tmp_el;
    for (size_t i = 0; i < bf16_in_el; i++) b_in[i] = f32_to_bf16(input[i]);
    for (size_t i = 0; i < bf16_fc1_el; i++) b_fc1[i] = f32_to_bf16(fc1_w[i]);
    for (size_t i = 0; i < bf16_fc2_el; i++) b_fc2[i] = f32_to_bf16(fc2_w[i]);
    memset(b_out, 0, bf16_out_el * sizeof(uint16_t));
    memset(b_tmp1, 0, bf16_tmp_el * sizeof(uint16_t));
    memset(b_tmp2, 0, bf16_tmp_el * sizeof(uint16_t));

    /* ---- GPU tensors ---- */
    h3_gpu_tensor *t_in = h3_gpu_tensor_from_f32(gpu, input, in_el);
    h3_gpu_tensor *t_fc1 = h3_gpu_tensor_from_f32(gpu, fc1_w, fc1_el);
    h3_gpu_tensor *t_fc2 = h3_gpu_tensor_from_f32(gpu, fc2_w, fc2_el);
    h3_gpu_tensor *t_qkv = h3_gpu_tensor_from_f32(gpu, qkv_w, qkv_el);
    h3_gpu_tensor *t_attn = h3_gpu_tensor_from_f32(gpu, attn_w, attn_el);
    h3_gpu_tensor *t_b5376 = h3_gpu_tensor_from_f32(gpu, bias_5376, H3_WIDTH);
    h3_gpu_tensor *t_b21504 = h3_gpu_tensor_from_f32(gpu, bias_21504, FC1_OUT);
    h3_gpu_tensor *t_b7168 = h3_gpu_tensor_from_f32(gpu, bias_7168, QKV_OUT);
    h3_gpu_tensor *t_qkv_buf = h3_gpu_tensor_new_f32(gpu, 3 * (size_t)rows * QKV_OUT);
    h3_gpu_tensor *t_out1 = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FC1_OUT);
    h3_gpu_tensor *t_out2 = h3_gpu_tensor_new_f32(gpu, (size_t)rows * H3_WIDTH);
    h3_gpu_tensor *t_attn_out = h3_gpu_tensor_new_f32(gpu, (size_t)rows * QKV_OUT);
    h3_gpu_tensor *t_tmp1 = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FC1_OUT);
    h3_gpu_tensor *t_tmp2 = h3_gpu_tensor_new_f32(gpu, (size_t)rows * FFN);

    /* BF16 tensors (real portable path) */
    h3_gpu_tensor *tb_in = h3_gpu_tensor_from_bf16(gpu, b_in, bf16_in_el);
    h3_gpu_tensor *tb_fc1 = h3_gpu_tensor_from_bf16(gpu, b_fc1, bf16_fc1_el);
    h3_gpu_tensor *tb_fc2 = h3_gpu_tensor_from_bf16(gpu, b_fc2, bf16_fc2_el);
    h3_gpu_tensor *tb_out = h3_gpu_tensor_new_bf16(gpu, bf16_out_el);
    h3_gpu_tensor *tb_tmp1 = h3_gpu_tensor_new_bf16(gpu, bf16_tmp_el);
    h3_gpu_tensor *tb_tmp2 = h3_gpu_tensor_new_bf16(gpu, bf16_tmp_el);

    if (!t_in || !t_fc1 || !t_fc2 || !t_qkv || !t_attn ||
        !t_b5376 || !t_b21504 || !t_b7168 || !t_qkv_buf ||
        !t_out1 || !t_out2 || !t_attn_out || !t_tmp1 || !t_tmp2 ||
        !tb_in || !tb_fc1 || !tb_fc2 || !tb_out || !tb_tmp1 || !tb_tmp2)
        fail("tensor allocation");

    /* ---- measurement helpers ---- */
    struct bench {
        const char *name;
        double best_ms;
        double gpu_ms;
        double wait_ms;      /* full command turnaround incl. MPSGraph children */
        uint64_t mps_dispatches;
        uint64_t direct_dispatches;
    } results[16];
    int n_results = 0;

    /* ---- run one op: warmup then timed loop ---- */
#define RUN_OP(NAME, BODY) do {                                             \
        /* warmup */                                                        \
        h3_gpu_begin(gpu);                                                  \
        for (int w = 0; w < 3; w++) { BODY; }                               \
        h3_gpu_submit(gpu);                                                 \
        h3_gpu_stats before, after;                                         \
        h3_gpu_get_stats(gpu, &before);                                     \
        h3_gpu_begin(gpu);                                                  \
        double t0 = now_seconds();                                          \
        int ok = 1;                                                         \
        for (unsigned i = 0; i < iters; i++) {                              \
            BODY;                                                           \
        }                                                                   \
        h3_gpu_submit(gpu);                                                 \
        double t1 = now_seconds();                                          \
        h3_gpu_get_stats(gpu, &after);                                      \
        double wall_ms = (t1 - t0) * 1000.0 / iters;                        \
        double gpu_s = (after.gpu_seconds - before.gpu_seconds) / iters;    \
        double wait_s = (after.command_wait_seconds - \
                         before.command_wait_seconds) / iters;              \
        (void)ok;                                                           \
        if (wall_ms < 0.01 &&                                                \
            after.direct_dispatches == before.direct_dispatches &&          \
            after.mps_linear_dispatches == before.mps_linear_dispatches) {  \
            printf("WARN: no GPU work observed for %s (op likely failed "    \
                   "silently: %s)\n", NAME, h3_gpu_error(gpu));             \
        }                                                                   \
        results[n_results++] = (struct bench){NAME, wall_ms, gpu_s * 1000.0,\
            wait_s * 1000.0,                                                 \
            after.mps_linear_dispatches - before.mps_linear_dispatches,     \
            after.direct_dispatches - before.direct_dispatches};            \
    } while (0)

    /* FC1: linear 5376 -> 28672 (F32 path) */
    RUN_OP("fc1_linear_f32_5376x28672",
        h3_gpu_linear_f32(gpu, t_out1, t_in, t_fc1, t_b21504, rows, H3_WIDTH, FC1_OUT));

    /* FC1 -> SwiGLU -> FC2 as separate ops (F32) */
    RUN_OP("fc1_swiglu_fc2_f32",
        h3_gpu_linear_f32(gpu, t_tmp1, t_in, t_fc1, t_b21504, rows, H3_WIDTH, FC1_OUT);
        h3_gpu_swiglu_f32(gpu, t_tmp2, t_tmp1, rows, FFN);
        h3_gpu_linear_f32(gpu, t_out2, t_tmp2, t_fc2, t_b5376, rows, FFN, H3_WIDTH));

    /* Portable BF16 MLP (the real DiT block MLP on non-M5 machines) */
    RUN_OP("mlp_bf16_fused",
        h3_gpu_mlp_bf16(gpu, tb_out, tb_in, tb_fc1, tb_fc2, rows, H3_WIDTH, FFN, H3_WIDTH));

    /* BF16 split: linear fc1 + swiglu + linear fc2 (portable path components) */
    RUN_OP("fc1_swiglu_fc2_bf16_split",
        h3_gpu_linear_bf16(gpu, tb_tmp1, tb_in, tb_fc1, NULL, rows, H3_WIDTH, FC1_OUT);
        h3_gpu_swiglu_bf16(gpu, tb_tmp2, tb_tmp1, rows, FFN);
        h3_gpu_linear_bf16(gpu, tb_out, tb_tmp2, tb_fc2, NULL, rows, FFN, H3_WIDTH));

    /* QKV projection 5376 -> 7168 */
    RUN_OP("qkv_linear_f32_5376x7168",
        h3_gpu_linear_f32(gpu, t_qkv_buf, t_in, t_qkv, t_b7168, rows, H3_WIDTH, QKV_OUT));

    /* Attention output projection 7168 -> 5376 */
    RUN_OP("attn_out_linear_f32_7168x5376",
        h3_gpu_linear_f32(gpu, t_out2, t_attn_out, t_attn, t_b5376, rows, QKV_OUT, H3_WIDTH));

    /* SDPA (56 heads, head_dim 128, seq=rows) */
    RUN_OP("sdpa_f32_56h_128d",
        h3_gpu_sdpa_f32(gpu, t_attn_out, t_qkv_buf, t_qkv_buf, t_qkv_buf,
                        rows, 56, 128, 0.088388f));

    /* ---- report ---- */
    puts("op,wall_ms,perf_gflops,bandwidth_GBps,gpu_ms,wait_ms,mps_dispatch,direct_dispatch");
    for (int i = 0; i < n_results; i++) {
        struct bench *b = &results[i];
        /* crude FLOPs + bytes estimate per op for context */
        double flops = 0, bytes = 0;
        if (!strcmp(b->name, "fc1_linear_f32_5376x28672")) {
            flops = 2.0 * rows * H3_WIDTH * FC1_OUT;
            bytes = ((double)rows * H3_WIDTH + (double)H3_WIDTH * FC1_OUT + (double)rows * FC1_OUT) * 4;
        } else if (!strcmp(b->name, "fc1_swiglu_fc2_f32")) {
            flops = 2.0 * rows * H3_WIDTH * FC1_OUT + 2.0 * rows * FFN * H3_WIDTH;
            bytes = ((double)rows * H3_WIDTH + 2.0 * (double)H3_WIDTH * FC1_OUT + 2.0 * (double)rows * FC1_OUT) * 4;
        } else if (!strcmp(b->name, "mlp_bf16_fused") ||
                   !strcmp(b->name, "fc1_swiglu_fc2_bf16_split")) {
            flops = 2.0 * rows * H3_WIDTH * FC1_OUT + 2.0 * rows * FFN * H3_WIDTH;
            bytes = ((double)rows * H3_WIDTH + 2.0 * (double)H3_WIDTH * FC1_OUT + 2.0 * (double)rows * FC1_OUT) * 2;
        } else if (!strcmp(b->name, "qkv_linear_f32_5376x7168")) {
            flops = 2.0 * rows * H3_WIDTH * QKV_OUT;
            bytes = ((double)rows * H3_WIDTH + (double)H3_WIDTH * QKV_OUT + (double)rows * QKV_OUT) * 4;
        } else if (!strcmp(b->name, "attn_out_linear_f32_7168x5376")) {
            flops = 2.0 * rows * QKV_OUT * H3_WIDTH;
            bytes = ((double)rows * QKV_OUT + (double)QKV_OUT * H3_WIDTH + (double)rows * H3_WIDTH) * 4;
        } else if (!strcmp(b->name, "sdpa_f32_56h_128d")) {
            flops = 2.0 * (double)rows * rows * 56 * 128;
            bytes = 3.0 * (double)rows * 56 * 128 * 4;
        }
        double perf = flops / (b->best_ms / 1000.0) / 1e9;
        double bw = bytes / (b->best_ms / 1000.0) / 1e9;
        printf("%s,%.2f,%.0f,%.0f,%.2f,%.2f,%llu,%llu\n",
               b->name, b->best_ms, perf, bw, b->gpu_ms, b->wait_ms,
               (unsigned long long)b->mps_dispatches,
               (unsigned long long)b->direct_dispatches);
    }

    /* ---- int8 probe: does the API-level int8 MLP work on Metal 3? ---- */
    printf("\n# int8 probe (does API int8 path dispatch on non-M5?)\n");
    if (h3_gpu_has_int8_mlp(gpu)) {
        printf("int8_mlp: API reports available\n");
    } else {
        printf("int8_mlp: API reports UNAVAILABLE on this device (expected on Metal 3)\n");
    }

    h3_gpu_free(gpu);
    free(buf);
    printf("\n# done\n");
    return 0;
}
