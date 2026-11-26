# eigen-mcu

**An MCU-targeted fork of [Eigen 5.0.1](https://gitlab.com/libeigen/eigen) optimized for small-matrix (`<= 40x40`) linear algebra on ARM Cortex-M4F and Cortex-M7 processors.**

Designed for real-time flight estimation and control stacks ([sfwa/ukf](https://github.com/sfwa/ukf), [sfwa/nmpc](https://github.com/sfwa/nmpc), [sfwa/models](https://github.com/sfwa/models)), while remaining a drop-in, C++14/C++17-compatible replacement for upstream Eigen on desktop (`x86_64` / `arm64`) SITL builds.

## Key changes from upstream Eigen 5.0.1

- **Unpacked register-blocked small GEMM, SYRK, and triangular products** (`Eigen/src/Core/products/McuSmallGemm.h`):
  - Uses `4x4` register tiles for `float` (24 of 32 FP32 registers on Cortex-M4F/M7) and `2x4` tiles for `double` (14 of 16 FP64 registers on Cortex-M7).
  - Eliminates GotoBLAS packing passes and `gemm_blocking_space` stack buffers (saving 0.8–8.6 KB of stack) for matrices with `M, N <= EIGEN_MCU_SMALL_GEMM_THRESHOLD` (default `40`) and `K <= EIGEN_MCU_SMALL_GEMM_MAX_DEPTH` (default `80`).
  - Enabled by default on scalar Cortex-M targets (`EIGEN_MCU_SMALL_GEMM=EIGEN_MCU_ARCH_CORTEX_M`) and disabled by default on SIMD desktop targets (`x86_64` / `arm64`) to preserve desktop SIMD GEMM speed; define `-DEIGEN_MCU_SMALL_GEMM=1` on desktop to force identical scalar accumulation order when testing SITL/HITL parity.
- **Unblocked pre-inverted-diagonal small TRSM & direct TRSV** (`Eigen/src/Core/SolveTriangular.h`, `McuSmallGemm.h`):
  - For `N <= EIGEN_MCU_SMALL_TRSM_THRESHOLD` (default `32`), avoids `gemm_blocking_space` and replaces `(nrhs - 1) * N` divisions with guarded pre-inverted diagonal multiplies (`OnTheLeft` and `OnTheRight`, `Lower` and `Upper`, `UnitDiag` and non-unit, `ColMajor` and `RowMajor`).
- **Scalar dot-product unrolling and 4-way FMA accumulator** (`Eigen/src/Core/InnerProduct.h`, `Dot.h`):
  - Restores compile-time small-vector unrolling (`CompileSize * 4 <= EIGEN_UNROLLING_LIMIT`) and adds a 4-way scalar accumulator for dynamic/un-unrolled vectors (`size >= 8`) to hide Cortex-M4F/M7 3-cycle FMA latency.
- **Targeted `EIGEN_ALWAYS_INLINE` hygiene** (`Dot.h`, `InnerProduct.h`, `GeneralBlockPanelKernel.h`, `Quaternion.h`, `RotationBase.h`, `MathFunctions.h`, `Block.h`):
  - Prevents GCC `-Os` (and `-O2` wrapper) outlining collapse without perturbing `-O2` loop unrolling heuristics.
- **Zero-malloc fixed-size Cholesky and Householder paths** (`LLT.h`, `LDLT.h`, `Householder.h`, `HouseholderSequence.h`, `McuConfig.h`):
  - Replaces `LLT` `rankUpdate`'s dynamic `TempVectorType` with a bounded vector (`EIGEN_NO_MALLOC` clean).
  - Defers `m_l1_norm` computation from `LLT::compute()` / `LDLT::compute()` into `rcond()`.
  - Uses guarded reciprocal multiplies in `LLT`, `LDLT`, `Householder::makeHouseholder`, and real `normalize()`/`normalized()`.
  - Skips instantiating `apply_block_householder_on_the_left` (and its dynamic temporaries) when compile-time matrix dimensions are `< 48`.
  - Defines `EIGEN_ALLOCA __builtin_alloca` on bare-metal Cortex-M GCC so `HouseholderQR` and `GEMV` temporary buffers stay on the stack instead of falling back to heap `malloc`.
- **Bitwise IEEE-754 `allFinite()` and `hasNaN()`** (`Visitor.h`):
  - Processes 4 contiguous elements per iteration in integer ALU registers (zero `vmrs` FPU flag sync stalls, bit-exact IEEE-754 semantics).
- **Cortex-M7 FPv5 `vmaxnm` / `vminnm`** (`GenericPacketMath.h`, `McuConfig.h`):
  - Specializes scalar `pmin`/`pmax` (and `PropagateNumbers`) to `std::fmin`/`std::fmax` only when `EIGEN_MCU_HAS_VMAXNM` is enabled (`__ARM_ARCH_7EM__ && (__ARM_FP & 8)` or `__ARM_FEATURE_DIRECTED_ROUNDING`), avoiding slow libm traps on Cortex-M4F (`fpv4-sp-d16`).
- **Cortex-M defaults & flash pruning** (`McuConfig.h`, `GeneralMatrixVector.h`):
  - Defaults `EIGEN_MAX_STATIC_ALIGN_BYTES=0`, `EIGEN_DONT_PARALLELIZE`, `EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS=16`, and `EIGEN_HAS_SINGLE_INSTRUCTION_MADD=1` on scalar Cortex-M targets.
  - Prunes dead half/quarter SIMD and remainder loops in `GeneralMatrixVector.h` when `PacketSize == 1`.

## Feature macros

All configuration macros live in `Eigen/src/Core/util/McuConfig.h` and can be overridden before including `<Eigen/Core>`:

| Macro | Default | Meaning |
| --- | --- | --- |
| `EIGEN_MCU` | `1` | Identifies `eigen-mcu`. |
| `EIGEN_MCU_SMALL_GEMM` | `EIGEN_MCU_ARCH_CORTEX_M` | Enables small-matrix unpacked GEMM/SYRK/triangular-product/TRSM/TRSV dispatch (`1` on Cortex-M, `0` on desktop SIMD unless overridden). |
| `EIGEN_MCU_SMALL_GEMM_THRESHOLD` | `40` | Maximum `M` and `N` for unpacked small GEMM/SYRK. |
| `EIGEN_MCU_SMALL_GEMM_MAX_DEPTH` | `80` | Maximum `K` (inner dimension) for unpacked small GEMM/SYRK. |
| `EIGEN_MCU_SMALL_TRSM_THRESHOLD` | `32` | Maximum triangular dimension `N` for unblocked TRSM/TRSV. |
| `EIGEN_MCU_HAS_VMAXNM` | auto | `1` on Cortex-M7 FPv5-D16 / ARMv8 directed rounding, `0` on Cortex-M4F. |

## Verification

A standalone zero-allocation self-test lives in `test_mcu/test_mcu_kernels.cpp` covering GEMM, triangular products, SYRK, TRSM/TRSV, LLT/LDLT (`rankUpdate`, `rcond`), HouseholderQR, SelfAdjointEigenSolver, dot/normalize/quaternion, `allFinite`/`hasNaN`, and `cwiseMin`/`cwiseMax` in both `float` and `double`.

## License

Like upstream Eigen, `eigen-mcu` is primarily licensed under the **Mozilla Public License Version 2.0** (`LICENSE` / `COPYING.MPL2`). All upstream license and notice files (`LICENSE`, `COPYING.MPL2`, `COPYING.BSD`, `COPYING.APACHE`, `COPYING.MINPACK`, `COPYING.README`, `CHANGELOG.md`) are preserved unmodified. For upstream Eigen information, see https://libeigen.gitlab.io/docs/.
