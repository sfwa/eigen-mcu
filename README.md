# eigen-mcu

**An MCU-targeted fork of [Eigen 5.0.1](https://gitlab.com/libeigen/eigen) optimized for small-matrix (`<= 40x40`) linear algebra on ARM Cortex-M4F and Cortex-M7 processors.**

Designed for real-time flight estimation and control stacks ([sfwa/ukf](https://github.com/sfwa/ukf), [sfwa/nmpc](https://github.com/sfwa/nmpc), [sfwa/models](https://github.com/sfwa/models)), while remaining a drop-in, C++14/C++17-compatible replacement for upstream Eigen on desktop (`x86_64` / `arm64`) SITL builds.

## Key changes from upstream Eigen 5.0.1

- **Unpacked register-blocked small GEMM, SYRK, triangular products, and GEMV** (`Eigen/src/Core/products/McuSmallGemm.h`, `GeneralProduct.h`, `ProductEvaluators.h`):
  - Uses `4x4` register tiles for `float` (24 of 32 FP32 registers on Cortex-M4F/M7) and `2x4` tiles for `double` (14 of 16 FP64 registers on Cortex-M7), with `MRx2` column-tail tiling and spill-free FP64 diagonal tiles.
  - Threads `beta_zero` through `evalTo()` so assignments write directly without a preceding `setZero()` pass, and detects symmetric SYRK diagonal tiles to halve operand loads.
  - Adds a `4x1` register-blocked small `GEMV` kernel (`ColMajor`, `RowMajor`, arbitrary strides, `alpha_one` skip, `beta_zero` direct store) and removes `gemv_static_vector_if` alignment over-padding on scalar Cortex-M targets.
  - Eliminates GotoBLAS packing passes and `gemm_blocking_space` stack buffers (saving 0.8–8.6 KB of stack) for matrices with `M, N <= EIGEN_MCU_SMALL_GEMM_THRESHOLD` (default `40`) and `K <= EIGEN_MCU_SMALL_GEMM_MAX_DEPTH` (default `80`).
  - Enabled by default on scalar Cortex-M targets (`EIGEN_MCU_SMALL_GEMM=EIGEN_MCU_ARCH_CORTEX_M`) and disabled by default on SIMD desktop targets (`x86_64` / `arm64`) to preserve desktop SIMD GEMM speed; define `-DEIGEN_MCU_SMALL_GEMM=1` on desktop to force identical scalar accumulation order when testing SITL/HITL parity.
- **Unblocked pre-inverted-diagonal small TRSM & direct TRSV** (`Eigen/src/Core/SolveTriangular.h`, `McuSmallGemm.h`):
  - For `N <= EIGEN_MCU_SMALL_TRSM_THRESHOLD` (default `32`), avoids `gemm_blocking_space`, replaces `(nrhs - 1) * N` divisions with guarded pre-inverted diagonal multiplies, and blocks `OnTheLeft` TRSM across `4/2/1` RHS columns (`Lower`/`Upper`, `UnitDiag`/non-unit, `ColMajor`/`RowMajor`).
- **Direct small-matrix Cholesky & fused multi-RHS LLT solve** (`LLT.h`, `LDLT.h`, `McuSmallGemm.h`):
  - Adds direct unit-stride `ColMajor` Lower/Upper small Cholesky kernels (`MaxRowsAtCompileTime < 32`) and fused LLT solve kernels sharing one pre-inverted diagonal across forward and back substitution with `4/2/1`-column RHS blocking.
  - Replaces `LLT::rankUpdate`'s dynamic `TempVectorType` with a bounded vector (`EIGEN_NO_MALLOC` clean), defers `m_l1_norm` into `rcond()`, and uses guarded reciprocal multiplies in `LLT` and `LDLT`.
- **Fused Householder reflectors & unblocked small HouseholderQR** (`Householder.h`, `HouseholderSequence.h`, `HouseholderQR.h`, `McuConfig.h`):
  - Adds `4/2/1`-column-fused left and `4/1`-row-fused right Householder reflector kernels (with alias-safe workspace fallback) and dispatches `HouseholderQR::computeInPlace()` directly to unblocked QR when `MaxRowsAtCompileTime <= 48` or `MaxColsAtCompileTime <= 48`.
  - Defines `EIGEN_ALLOCA __builtin_alloca` on bare-metal Cortex-M GCC so `HouseholderQR` and `GEMV` temporary buffers stay on the stack instead of falling back to heap `malloc`.
- **Flat FMA dot unroller and 4-way scalar accumulator** (`Eigen/src/Core/InnerProduct.h`, `Dot.h`):
  - Replaces Eigen's recursive binary-tree compile-time dot unroller with flat interleaved 4-way (`float`, `Len >= 8`) and 2-way (`double` or `Len in [4, 7]`) FMA chains matching Cortex-M7 `vfma` latency, plus a 4-way loop for dynamic/un-unrolled vectors (`size >= 8`).
- **Targeted `-Os` evaluator & wrapper inlining** (`McuConfig.h`, `AssignEvaluator.h`, `CoreEvaluators.h`, `Block.h`, `VectorBlock.h`, `TriangularMatrix.h`, `Redux.h`, `Visitor.h`, `AssignmentFunctors.h`, `GenericPacketMath.h`, `Quaternion.h`, `MathFunctions.h`):
  - Applies `EIGEN_MCU_INLINE_SIZE` (`always_inline` at `-Os` on MCU targets) to per-element evaluators, views, and assignment functors to prevent GCC `-Os` wrapper outlining collapse without bloating multi-element loops.
- **Bitwise IEEE-754 `allFinite()` and `hasNaN()`** (`Visitor.h`):
  - Processes 4 contiguous elements per iteration in integer ALU registers (zero `vmrs` FPU flag sync stalls, bit-exact IEEE-754 semantics).
- **Cortex-M7 FPv5 `vmaxnm` / `vminnm`** (`GenericPacketMath.h`, `McuConfig.h`):
  - Specializes scalar `pmin`/`pmax` (and `PropagateNumbers`) to `std::fmin`/`std::fmax` only when `EIGEN_MCU_HAS_VMAXNM` is enabled (`__ARM_ARCH_7EM__ && (__ARM_FP & 8)` or `__ARM_FEATURE_DIRECTED_ROUNDING`), avoiding slow libm traps on Cortex-M4F (`fpv4-sp-d16`).
- **Cortex-M defaults & flash pruning** (`McuConfig.h`, `GeneralMatrixVector.h`):
  - Defaults `EIGEN_MAX_STATIC_ALIGN_BYTES=0`, `EIGEN_DONT_PARALLELIZE`, `EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS=16`, and `EIGEN_HAS_SINGLE_INSTRUCTION_MADD=1` on scalar Cortex-M targets, and prunes dead SIMD remainder loops when `PacketSize == 1`.

## Feature macros

All configuration macros live in `Eigen/src/Core/util/McuConfig.h` and can be overridden before including `<Eigen/Core>`:

| Macro | Default | Meaning |
| --- | --- | --- |
| `EIGEN_MCU` | `1` | Identifies `eigen-mcu`. |
| `EIGEN_MCU_SMALL_GEMM` | `EIGEN_MCU_ARCH_CORTEX_M` | Enables small-matrix unpacked GEMM/SYRK/triangular-product/GEMV/TRSM/TRSV/LLT/QR dispatch (`1` on Cortex-M, `0` on desktop SIMD unless overridden). |
| `EIGEN_MCU_SMALL_GEMM_THRESHOLD` | `40` | Maximum `M` and `N` for unpacked small GEMM/SYRK. |
| `EIGEN_MCU_SMALL_GEMM_MAX_DEPTH` | `80` | Maximum `K` (inner dimension) for unpacked small GEMM/SYRK. |
| `EIGEN_MCU_SMALL_TRSM_THRESHOLD` | `32` | Maximum triangular dimension `N` for unblocked TRSM/TRSV. |
| `EIGEN_MCU_HAS_VMAXNM` | auto | `1` on Cortex-M7 FPv5-D16 / ARMv8 directed rounding, `0` on Cortex-M4F. |

## Verification and CI

- **Correctness suite** (`test_mcu/test_mcu_kernels.cpp`): standalone zero-allocation (`EIGEN_NO_MALLOC`) self-test covering GEMM (`beta=0/1`, `MRx2` tails), symmetric/asymmetric triangular products, SYRK, small GEMV, TRSM/TRSV, LLT (all layouts, multi-RHS solve, `rankUpdate`, `rcond`), LDLT, HouseholderQR (`28x9`, `31x15`), SelfAdjointEigenSolver, flat dot/normalize/quaternion, `allFinite`/`hasNaN`, and `cwiseMin`/`cwiseMax` in `float` and `double`.
- **Cortex-M4F & M7 QEMU gate** (`test_mcu/check_mcu.py`, `test_mcu/bench_mcu_kernels.cpp`, `test_mcu/accepted.json`): builds and runs both correctness and kernel benchmarks under QEMU (`mps2-an500` Cortex-M7 and `mps2-an386` Cortex-M4F) at `-O2` and `-Os -DEIGEN_UNROLLING_LIMIT=0` with linker-wrapped heap traps (`--wrap=malloc/...`), strict zero-FP64 symbol enforcement on Cortex-M4F, compile-only Cortex-M33/M3 smoke builds, and performance regression gating against `test_mcu/accepted.json`.

```sh
python3 test_mcu/check_mcu.py --jobs 4
```

## Packaging a release

Use `scripts/package_release.sh` to verify version headers and license files and build a CMake-consumable tarball and SHA-256 checksum:

```sh
bash scripts/package_release.sh --out dist --verify
```

Downstream CMake projects (`sfwa/ukf`, `sfwa/nmpc`) can consume a release tarball directly via `UKF_EIGEN_VERSION=5.0.1` / `NMPC_EIGEN_VERSION=5.0.1` and `UKF_EIGEN_URL` / `NMPC_EIGEN_URL`.

## License

Like upstream Eigen, `eigen-mcu` is primarily licensed under the **Mozilla Public License Version 2.0** (`LICENSE` / `COPYING.MPL2`). All upstream license and notice files (`LICENSE`, `COPYING.MPL2`, `COPYING.BSD`, `COPYING.APACHE`, `COPYING.MINPACK`, `COPYING.README`, `CHANGELOG.md`) are preserved unmodified. For upstream Eigen information, see https://libeigen.gitlab.io/docs/.
