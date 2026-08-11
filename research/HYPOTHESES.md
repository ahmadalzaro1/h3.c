
## [2026-08-11T20:23] Apple M1 Max
HYPOTHESIS 1 (CORRECTED): Portable BF16 MLP on M1 Max is GPU-compute-bound at ~3.7 TFLOP/s (36% of 10.4 peak), NOT CPU-encode-bound. Evidence: wait_ms==wall_ms (256.7 vs 256.9) across all ops; root gpu_ms undercounts MPSGraph child buffers (documented README limitation). First reading (gpu_ms=0 -> 'encode-bound') was WRONG; corrected by adding command_wait_seconds to bench. NEXT: probe MPSGraph INT8 matmul (MPSDataTypeInt8) at H3 shapes - if int8 halves weight traffic and MPSGraph executes it faster than BF16 on Metal 3, portable int8 MLP is the contribution. Also probe matmul efficiency: is MPSGraph picking a slow M1 path?

## [2026-08-11T20:28] Apple M1 Max
EXPERIMENT 1 (PROBE, NEGATIVE): MPSGraph int8 matmul is IMPOSSIBLE on any Metal — MLIR compiler rejects si8 operands ('must be tensor of floating point values'). Portable-int8-via-MPSGraph thesis DEAD (killed in 30min instead of 3 weeks). M5 kernels need Metal 4 matmul2d/tensor types — hardware-locked, can't compile on M1. Remaining int8 option: hand-written Metal int8 dot-product kernel (llama.cpp style, plain threads, portable to Metal 3) — weeks, high effort.

## [2026-08-11T20:39] Apple M1 Max
EXPERIMENT 2 (A/B, NEGATIVE): H3_FORCE_DIRECT_LINEAR=1 routes big matmuls through repo's h3_linear_bf16 16x16-tiled kernel. Result: 10-66x SLOWER than MPSGraph (qkv 34->2232ms, attn 37->2361ms, MLP split 257->1159ms). MPSGraph is already optimal for big linears on M1. Direct kernel stays patch-only. Remaining options: (a) hand-written portable int8 Metal matmul [weeks], (b) FP16 MPSGraph test [cheap], (c) SSD-streaming preset [blocked: no weights]

## [2026-08-11T22:36] Apple M1 Max
EXPERIMENT 3 (BREAKTHROUGH): FP16 MPSGraph is ~2x faster than BF16 on M1 Max. Raw matmul 5376x28672: 41.6->23.1ms (1.8x). Fused MLP fc1->swiglu->fc2: 1698->842ms (2.0x) with VALID fp16 data. Why: M1 GPU has native FP16 ALU (2x FP32); BF16 has no hw on Metal 3 -> MPSGraph emulates it. FP16 = same 2 bytes (no memory change) + BETTER mantissa (10 vs 7 bits). Caveat: BF16-bytes-reinterpreted-as-FP16 gives NaN data and runs SLOWER (433ms) - must do real conversion. Implementation: H3_MPS_FP16=1 env in h3_gpu.m (graph dtype flip) + one-time BF16->FP16 weight conversion at load. This is THE portable pre-M5 optimization - works on all M1/M2/M3/M4.

## [2026-08-12T00:10] Apple M1 Max
EXPERIMENT 4 (phased FP16-resident MLP + custom h3_swiglu_fp16) — RESULT: HYPOTHESIS FALSIFIED at big rows. Full crossover: 256r BF16 32.89 vs FP16 19.47 (1.69x FASTER); 512r 66.41 vs 37.67 (1.76x FASTER); 1024r 128.61 vs 225.40 (1.75x SLOWER); 2048r 254.05 vs 435.79 (1.72x SLOWER). Custom half2 SwiGLU adds ~0 over MPSGraph fused split+sigmoid+multiply at every row size. Raw single-GEMM FP16 win (93 vs 173ms @2048) does NOT survive chaining — MPSGraph FP16 multi-op DAG at large rows is the bottleneck, not entry/exit casts. MoA '~150ms @2048' prediction falsified. Next: examine whether large-row loss is MPSGraph kernel selection (fp16 GEMM tiles) vs memory bandwidth of 117MB intermediate.

## [2026-08-12T00:15] Apple M1 Max
EXPERIMENT 4 follow-up (crossover localization): razor-sharp kernel boundary. 640r 1.75x, 672r 1.82x, 704r 1.77x WIN; 736r 0.55x, 768r 0.53x, 896r 0.59x LOSS. Boundary between 704 and 736 rows (22x32 vs 23x32 tiles) — MPSGraph switches FP16 GEMM strategy. Deterministic auto-select threshold: rows<=704 → FP16 phased MLP (1.8x), rows>704 → BF16 fused (1.9x better). This is the shippable contribution: env-toggle H3_MPS_FP16 stays default-off, but an auto-router can pick per-batch.
