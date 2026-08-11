
## [2026-08-11T20:23] Apple M1 Max
HYPOTHESIS 1 (CORRECTED): Portable BF16 MLP on M1 Max is GPU-compute-bound at ~3.7 TFLOP/s (36% of 10.4 peak), NOT CPU-encode-bound. Evidence: wait_ms==wall_ms (256.7 vs 256.9) across all ops; root gpu_ms undercounts MPSGraph child buffers (documented README limitation). First reading (gpu_ms=0 -> 'encode-bound') was WRONG; corrected by adding command_wait_seconds to bench. NEXT: probe MPSGraph INT8 matmul (MPSDataTypeInt8) at H3 shapes - if int8 halves weight traffic and MPSGraph executes it faster than BF16 on Metal 3, portable int8 MLP is the contribution. Also probe matmul efficiency: is MPSGraph picking a slow M1 path?

## [2026-08-11T20:28] Apple M1 Max
EXPERIMENT 1 (PROBE, NEGATIVE): MPSGraph int8 matmul is IMPOSSIBLE on any Metal — MLIR compiler rejects si8 operands ('must be tensor of floating point values'). Portable-int8-via-MPSGraph thesis DEAD (killed in 30min instead of 3 weeks). M5 kernels need Metal 4 matmul2d/tensor types — hardware-locked, can't compile on M1. Remaining int8 option: hand-written Metal int8 dot-product kernel (llama.cpp style, plain threads, portable to Metal 3) — weeks, high effort.
