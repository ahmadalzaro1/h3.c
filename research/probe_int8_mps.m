// probe_int8_mps.m — raw MPSGraph int8 vs BF16 matmul throughput at H3 shapes.
//
// Question: does MPSGraph int8 matrix multiplication beat its BF16 path on
// Metal 3 (M1 Max)? The repo's int8 kernels are M5 TensorOps-gated, but
// MPSGraph supports MPSDataTypeInt8 matmul on all Metal GPUs. If raw int8
// matmul is ~2x faster (bandwidth argument: half the bytes), a portable
// int8 MLP via MPSGraph is worth building. If not, the thesis dies cheaply.
//
// Build: clang -fobjc-arc -framework Foundation -framework Metal \
//        -framework MetalPerformanceShadersGraph probe_int8_mps.m -o probe_int8_mps
// Run:   ./probe_int8_mps [rows]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Proper f32 -> fp16 (round-to-nearest-even) so FP16 buffers hold VALID data */
static uint16_t f32_to_fp16(float f) {
    union { float f; uint32_t u; } in = {f};
    uint32_t sign = (in.u >> 16) & 0x8000u;
    uint32_t exp = (in.u >> 23) & 0xffu;
    uint32_t mant = in.u & 0x7fffffu;
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    uint32_t e = exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);  /* overflow -> inf */
    if (e <= 0) {
        /* subnormal: we don't need them for benchmark values; flush to 0 */
        return (uint16_t)sign;
    }
    uint32_t m = mant >> 13;
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) m++;  /* round-half-even */
    if (m == 0x400u) { m = 0; e++; }
    return (uint16_t)(sign | (e << 10) | m);
}

int main(int argc, char **argv) {
    uint32_t rows = argc > 1 ? (uint32_t)atoi(argv[1]) : 2048;
    uint32_t input_dim = 5376, output_dim = 28672; /* H3 fc1 shape */
    size_t in_el = (size_t)rows * input_dim;
    size_t wt_el = (size_t)output_dim * input_dim;
    size_t out_el = (size_t)rows * output_dim;
    int iters = 5;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "no Metal device\n"); return 1; }
    NSLog(@"device: %@", device.name);
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* ---------- bf16 graph: matmul -> bf16 (repo-style 3D + transpose) ---------- */
    MPSGraph *g16 = [[MPSGraph alloc] init];
    MPSGraphTensor *b16_in = [g16 placeholderWithShape:@[@(1), @(rows), @(input_dim)]
                                              dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *b16_wt = [g16 placeholderWithShape:@[@(1), @(output_dim), @(input_dim)]
                                              dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *wt_t = [g16 transposeTensor:b16_wt dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *b16_mm = [g16 matrixMultiplicationWithPrimaryTensor:b16_in
                                                        secondaryTensor:wt_t name:nil];

    /* ---------- fp16 graph: matmul -> fp16 (M1 native FP16 ALU) ---------- */
    MPSGraph *gfp = [[MPSGraph alloc] init];
    MPSGraphTensor *fp_in = [gfp placeholderWithShape:@[@(1), @(rows), @(input_dim)]
                                             dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *fp_wt = [gfp placeholderWithShape:@[@(1), @(output_dim), @(input_dim)]
                                             dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *fp_wt_t = [gfp transposeTensor:fp_wt dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *fp_mm = [gfp matrixMultiplicationWithPrimaryTensor:fp_in
                                                       secondaryTensor:fp_wt_t name:nil];

    /* ---------- full MLP in BF16 (fc1 -> swiglu -> fc2) ---------- */
    uint32_t hidden = 14336; /* H3 FFN */
    MPSGraph *gm16 = [[MPSGraph alloc] init];
    MPSGraphTensor *m16_in = [gm16 placeholderWithShape:@[@(1), @(rows), @(input_dim)]
                                               dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *m16_fc1 = [gm16 placeholderWithShape:@[@(1), @(hidden * 2), @(input_dim)]
                                                dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *m16_fc2 = [gm16 placeholderWithShape:@[@(1), @(input_dim), @(hidden)]
                                                dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *m16_fc1_t = [gm16 transposeTensor:m16_fc1 dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *m16_fused = [gm16 matrixMultiplicationWithPrimaryTensor:m16_in
                                                            secondaryTensor:m16_fc1_t name:nil];
    NSArray *m16_halves = [gm16 splitTensor:m16_fused numSplits:2 axis:2 name:nil];
    MPSGraphTensor *m16_sig = [gm16 sigmoidWithTensor:m16_halves[0] name:nil];
    MPSGraphTensor *m16_silu = [gm16 multiplicationWithPrimaryTensor:m16_halves[0]
                                                     secondaryTensor:m16_sig name:nil];
    MPSGraphTensor *m16_act = [gm16 multiplicationWithPrimaryTensor:m16_silu
                                                    secondaryTensor:m16_halves[1] name:nil];
    MPSGraphTensor *m16_fc2_t = [gm16 transposeTensor:m16_fc2 dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *m16_out = [gm16 matrixMultiplicationWithPrimaryTensor:m16_act
                                                          secondaryTensor:m16_fc2_t name:nil];

    /* ---------- full MLP in FP16 ---------- */
    MPSGraph *gmfp = [[MPSGraph alloc] init];
    MPSGraphTensor *mfp_in = [gmfp placeholderWithShape:@[@(1), @(rows), @(input_dim)]
                                               dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *mfp_fc1 = [gmfp placeholderWithShape:@[@(1), @(hidden * 2), @(input_dim)]
                                                dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *mfp_fc2 = [gmfp placeholderWithShape:@[@(1), @(input_dim), @(hidden)]
                                                dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *mfp_fc1_t = [gmfp transposeTensor:mfp_fc1 dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *mfp_fused = [gmfp matrixMultiplicationWithPrimaryTensor:mfp_in
                                                            secondaryTensor:mfp_fc1_t name:nil];
    NSArray *mfp_halves = [gmfp splitTensor:mfp_fused numSplits:2 axis:2 name:nil];
    MPSGraphTensor *mfp_sig = [gmfp sigmoidWithTensor:mfp_halves[0] name:nil];
    MPSGraphTensor *mfp_silu = [gmfp multiplicationWithPrimaryTensor:mfp_halves[0]
                                                     secondaryTensor:mfp_sig name:nil];
    MPSGraphTensor *mfp_act = [gmfp multiplicationWithPrimaryTensor:mfp_silu
                                                    secondaryTensor:mfp_halves[1] name:nil];
    MPSGraphTensor *mfp_fc2_t = [gmfp transposeTensor:mfp_fc2 dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *mfp_out = [gmfp matrixMultiplicationWithPrimaryTensor:mfp_act
                                                          secondaryTensor:mfp_fc2_t name:nil];

    /* ---------- buffers ---------- */
    id<MTLBuffer> in8 = [device newBufferWithLength:in_el options:MTLResourceStorageModeShared];
    id<MTLBuffer> wt8 = [device newBufferWithLength:wt_el options:MTLResourceStorageModeShared];
    id<MTLBuffer> out32 = [device newBufferWithLength:out_el * 4 options:MTLResourceStorageModeShared];
    int8_t *in8_p = in8.contents, *wt8_p = wt8.contents;
    for (size_t i = 0; i < in_el; i++) in8_p[i] = (int8_t)((i * 1103515245u + 12345u) >> 16);
    for (size_t i = 0; i < wt_el; i++) wt8_p[i] = (int8_t)((i * 2654435761u + 99991u) >> 16);

    id<MTLBuffer> in16 = [device newBufferWithLength:in_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wt16 = [device newBufferWithLength:wt_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out16 = [device newBufferWithLength:out_el * 2 options:MTLResourceStorageModeShared];
    uint16_t *in16_p = in16.contents, *wt16_p = wt16.contents;
    for (size_t i = 0; i < in_el; i++) in16_p[i] = (uint16_t)(in8_p[i] << 8);
    for (size_t i = 0; i < wt_el; i++) wt16_p[i] = (uint16_t)(wt8_p[i] << 8);

    /* FP16 buffers — VALID fp16 values converted from f32, not BF16 reinterp */
    id<MTLBuffer> infp = [device newBufferWithLength:in_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wtfp = [device newBufferWithLength:wt_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outfp = [device newBufferWithLength:out_el * 2 options:MTLResourceStorageModeShared];
    uint16_t *infp_p = infp.contents, *wtfp_p = wtfp.contents;
    for (size_t i = 0; i < in_el; i++) infp_p[i] = f32_to_fp16((float)in8_p[i] * 0.01f);
    for (size_t i = 0; i < wt_el; i++) wtfp_p[i] = f32_to_fp16((float)wt8_p[i] * 0.001f);

    /* MLP weight buffers: fc1 [hidden*2 x input_dim], fc2 [input_dim x hidden] */
    size_t fc1_el = (size_t)(hidden * 2) * input_dim;
    size_t fc2_el = (size_t)input_dim * hidden;
    id<MTLBuffer> fc1w16 = [device newBufferWithLength:fc1_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> fc2w16 = [device newBufferWithLength:fc2_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> mlp_out16 = [device newBufferWithLength:out_el * 2 options:MTLResourceStorageModeShared];
    uint16_t *fc1w16_p = fc1w16.contents, *fc2w16_p = fc2w16.contents;
    for (size_t i = 0; i < fc1_el; i++) fc1w16_p[i] = (uint16_t)((i * 1103515245u + 12345u) >> 16);
    for (size_t i = 0; i < fc2_el; i++) fc2w16_p[i] = (uint16_t)((i * 2654435761u + 99991u) >> 16);

    /* FP16 MLP weights — valid conversion */
    id<MTLBuffer> fc1wfp = [device newBufferWithLength:fc1_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> fc2wfp = [device newBufferWithLength:fc2_el * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> mlp_outfp = [device newBufferWithLength:out_el * 2 options:MTLResourceStorageModeShared];
    uint16_t *fc1wfp_p = fc1wfp.contents, *fc2wfp_p = fc2wfp.contents;
    for (size_t i = 0; i < fc1_el; i++) fc1wfp_p[i] = f32_to_fp16((float)((i * 1103515245u + 12345u) >> 16) * 0.001f);
    for (size_t i = 0; i < fc2_el; i++) fc2wfp_p[i] = f32_to_fp16((float)((i * 2654435761u + 99991u) >> 16) * 0.001f);

    NSArray *in_shape = @[@(1), @(rows), @(input_dim)];
    NSArray *wt_shape = @[@(1), @(output_dim), @(input_dim)];
    NSArray *out_shape = @[@(1), @(rows), @(output_dim)];

    MPSGraphTensorData *b16_in_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:in16
                                                                           shape:in_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *b16_wt_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:wt16
                                                                           shape:wt_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *b16_out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:out16
                                                                            shape:out_shape dataType:MPSDataTypeBFloat16];
    NSDictionary *feeds16 = @{ b16_in: b16_in_d, b16_wt: b16_wt_d };
    NSDictionary *res16 = @{ b16_mm: b16_out_d };

    MPSGraphTensorData *fp_in_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:infp
                                                                          shape:in_shape dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *fp_wt_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:wt16
                                                                          shape:wt_shape dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *fp_out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:out16
                                                                           shape:out_shape dataType:MPSDataTypeFloat16];
    NSDictionary *feedsfp = @{ fp_in: fp_in_d, fp_wt: fp_wt_d };
    NSDictionary *resfp = @{ fp_mm: fp_out_d };

    /* MLP tensor data: BF16 and FP16 */
    NSArray *fc1_shape = @[@(1), @(hidden * 2), @(input_dim)];
    NSArray *fc2_shape = @[@(1), @(input_dim), @(hidden)];
    MPSGraphTensorData *m16_in_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:in16
                                                                           shape:in_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *m16_fc1_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:fc1w16
                                                                            shape:fc1_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *m16_fc2_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:fc2w16
                                                                            shape:fc2_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *m16_out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:mlp_out16
                                                                            shape:out_shape dataType:MPSDataTypeBFloat16];
    NSDictionary *m16_feeds = @{ m16_in: m16_in_d, m16_fc1: m16_fc1_d, m16_fc2: m16_fc2_d };
    NSDictionary *m16_res = @{ m16_out: m16_out_d };

    MPSGraphTensorData *mfp_in_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:infp
                                                                           shape:in_shape dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *mfp_fc1_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:fc1wfp
                                                                            shape:fc1_shape dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *mfp_fc2_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:fc2wfp
                                                                            shape:fc2_shape dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *mfp_out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:mlp_outfp
                                                                            shape:out_shape dataType:MPSDataTypeFloat16];
    NSDictionary *mfp_feeds = @{ mfp_in: mfp_in_d, mfp_fc1: mfp_fc1_d, mfp_fc2: mfp_fc2_d };
    NSDictionary *mfp_res = @{ mfp_out: mfp_out_d };

    /* ---------- helpers ---------- */
    void (^run)(MPSGraph *, NSDictionary *, NSDictionary *, int, double *) =
        ^(MPSGraph *graph, NSDictionary *feeds, NSDictionary *results, int n, double *ms) {
        for (int w = 0; w < 2; w++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            MPSCommandBuffer *mps = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
            [graph encodeToCommandBuffer:mps feeds:feeds targetOperations:nil
                       resultsDictionary:results executionDescriptor:nil];
            [mps commit];
            [cb waitUntilCompleted];
        }
        double t0 = now_seconds();
        for (int i = 0; i < n; i++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            MPSCommandBuffer *mps = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
            [graph encodeToCommandBuffer:mps feeds:feeds targetOperations:nil
                       resultsDictionary:results executionDescriptor:nil];
            [mps commit];
            [cb waitUntilCompleted];
        }
        double t1 = now_seconds();
        *ms = (t1 - t0) * 1000.0 / n;
    };

    double b16_ms = 0, fp_ms = 0, mlp16_ms = 0, mlpfp_ms = 0;
    run(g16, feeds16, res16, iters, &b16_ms);
    run(gfp, feedsfp, resfp, iters, &fp_ms);
    run(gm16, m16_feeds, m16_res, iters, &mlp16_ms);
    run(gmfp, mfp_feeds, mfp_res, iters, &mlpfp_ms);

    double flops = 2.0 * (double)rows * input_dim * output_dim;
    double mlp_flops = 2.0 * (double)rows * input_dim * (double)(hidden * 2) +
                       2.0 * (double)rows * (double)hidden * input_dim;
    printf("bf16_matmul rows=%u %ux%u: %.2f ms  %.0f GFLOPS\n",
           rows, input_dim, output_dim, b16_ms, flops / (b16_ms / 1000.0) / 1e9);
    printf("fp16_matmul rows=%u %ux%u: %.2f ms  %.0f GFLOPS\n",
           rows, input_dim, output_dim, fp_ms, flops / (fp_ms / 1000.0) / 1e9);
    printf("mlp_bf16_fused  rows=%u: %.2f ms  %.0f GFLOPS\n",
           rows, mlp16_ms, mlp_flops / (mlp16_ms / 1000.0) / 1e9);
    printf("mlp_fp16_fused  rows=%u: %.2f ms  %.0f GFLOPS\n",
           rows, mlpfp_ms, mlp_flops / (mlpfp_ms / 1000.0) / 1e9);
    printf("matmul fp16/bf16: %.2fx  |  MLP fp16/bf16: %.2fx\n",
           fp_ms / b16_ms, mlpfp_ms / mlp16_ms);
    return 0;
}
