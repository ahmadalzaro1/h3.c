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

    /* ---------- int8 graph: matmul -> int32 ---------- */
    MPSGraph *g8 = [[MPSGraph alloc] init];
    MPSGraphTensor *i8_in = [g8 placeholderWithShape:@[@(rows), @(input_dim)]
                                            dataType:MPSDataTypeInt8 name:nil];
    MPSGraphTensor *i8_wt = [g8 placeholderWithShape:@[@(output_dim), @(input_dim)]
                                            dataType:MPSDataTypeInt8 name:nil];
    MPSGraphTensor *i8_mm = [g8 matrixMultiplicationWithPrimaryTensor:i8_in
                                                      secondaryTensor:i8_wt name:nil];

    /* ---------- bf16 graph: matmul -> bf16 ---------- */
    MPSGraph *g16 = [[MPSGraph alloc] init];
    MPSGraphTensor *b16_in = [g16 placeholderWithShape:@[@(rows), @(input_dim)]
                                              dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *b16_wt = [g16 placeholderWithShape:@[@(output_dim), @(input_dim)]
                                              dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *b16_mm = [g16 matrixMultiplicationWithPrimaryTensor:b16_in
                                                        secondaryTensor:b16_wt name:nil];

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

    NSArray *in_shape = @[@(rows), @(input_dim)];
    NSArray *wt_shape = @[@(output_dim), @(input_dim)];
    NSArray *out_shape = @[@(rows), @(output_dim)];

    MPSGraphTensorData *i8_in_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:in8
                                                                          shape:in_shape dataType:MPSDataTypeInt8];
    MPSGraphTensorData *i8_wt_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:wt8
                                                                          shape:wt_shape dataType:MPSDataTypeInt8];
    MPSGraphTensorData *i8_out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:out32
                                                                           shape:out_shape dataType:MPSDataTypeInt32];
    NSDictionary *feeds8 = @{ i8_in: i8_in_d, i8_wt: i8_wt_d };
    NSDictionary *res8 = @{ i8_mm: i8_out_d };

    MPSGraphTensorData *b16_in_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:in16
                                                                           shape:in_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *b16_wt_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:wt16
                                                                           shape:wt_shape dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData *b16_out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:out16
                                                                            shape:out_shape dataType:MPSDataTypeBFloat16];
    NSDictionary *feeds16 = @{ b16_in: b16_in_d, b16_wt: b16_wt_d };
    NSDictionary *res16 = @{ b16_mm: b16_out_d };

    /* ---------- helpers ---------- */
    void (^run)(MPSGraph *, NSDictionary *, NSDictionary *, int, double *, double *) =
        ^(MPSGraph *graph, NSDictionary *feeds, NSDictionary *results,
          int n, double *ms, double *gbps) {
        double total_bytes = (double)in_el * 1 + (double)wt_el * 1; /* per-iter input bytes */
        (void)total_bytes;
        for (int w = 0; w < 2; w++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            MPSCommandBuffer *mps = [[MPSCommandBuffer alloc] initWithCommandBuffer:cb];
            [graph encodeToCommandBuffer:mps feeds:feeds targetOperations:nil
                       resultsDictionary:results executionDescriptor:nil];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double t0 = now_seconds();
        for (int i = 0; i < n; i++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            MPSCommandBuffer *mps = [[MPSCommandBuffer alloc] initWithCommandBuffer:cb];
            [graph encodeToCommandBuffer:mps feeds:feeds targetOperations:nil
                       resultsDictionary:results executionDescriptor:nil];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double t1 = now_seconds();
        *ms = (t1 - t0) * 1000.0 / n;
    };

    double i8_ms = 0, b16_ms = 0;
    run(g8, feeds8, res8, iters, &i8_ms, NULL);
    run(g16, feeds16, res16, iters, &b16_ms, NULL);

    double flops = 2.0 * (double)rows * input_dim * output_dim;
    printf("int8_matmul rows=%u %ux%u: %.2f ms  %.0f GFLOPS\n",
           rows, input_dim, output_dim, i8_ms, flops / (i8_ms / 1000.0) / 1e9);
    printf("bf16_matmul rows=%u %ux%u: %.2f ms  %.0f GFLOPS\n",
           rows, input_dim, output_dim, b16_ms, flops / (b16_ms / 1000.0) / 1e9);
    printf("int8/bf16 wall ratio: %.2fx\n", i8_ms / b16_ms);
    return 0;
}
