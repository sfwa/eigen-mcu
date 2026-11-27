// This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
//
// Copyright (C) 2026 SFWA contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MCU_CONFIG_H
#define EIGEN_MCU_CONFIG_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

#include <cstdint>

// Identify this MCU-targeted fork of Eigen 5.0.1 for downstream feature checks.
#ifndef EIGEN_MCU
#define EIGEN_MCU 1
#endif

#ifndef EIGEN_MCU_VERSION_MAJOR
#define EIGEN_MCU_VERSION_MAJOR 5
#define EIGEN_MCU_VERSION_MINOR 0
#define EIGEN_MCU_VERSION_PATCH 1
#endif

// Detect scalar ARM Cortex-M targets (M4F, M7, M3, M33 without NEON/MVE).
#ifndef EIGEN_MCU_ARCH_CORTEX_M
#if (defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_8M_MAIN__) || \
     defined(__ARM_ARCH_8M_BASE__) || defined(__ARM_ARCH_6M__)) &&                              \
    !defined(__ARM_NEON) && !defined(__ARM_FEATURE_MVE)
#define EIGEN_MCU_ARCH_CORTEX_M 1
#else
#define EIGEN_MCU_ARCH_CORTEX_M 0
#endif
#endif

// Small-matrix unpacked register-blocked GEMM, SYRK, and triangular products.
// Enabled by default on scalar Cortex-M targets (30-50% faster than packed
// GEMM with zero blocking stack). Disabled by default on SIMD desktop targets
// (x86_64 SSE/AVX, arm64 NEON) where scalar loops are >5% slower than Eigen's
// SIMD GEMM; define -DEIGEN_MCU_SMALL_GEMM=1 on desktop to force identical
// scalar accumulation order when debugging SITL/HITL parity.
#ifndef EIGEN_MCU_SMALL_GEMM
#define EIGEN_MCU_SMALL_GEMM EIGEN_MCU_ARCH_CORTEX_M
#endif

#ifndef EIGEN_MCU_SMALL_GEMM_THRESHOLD
#define EIGEN_MCU_SMALL_GEMM_THRESHOLD 40
#endif

#ifndef EIGEN_MCU_SMALL_GEMM_MAX_DEPTH
#define EIGEN_MCU_SMALL_GEMM_MAX_DEPTH (2 * EIGEN_MCU_SMALL_GEMM_THRESHOLD)
#endif

#ifndef EIGEN_MCU_SMALL_TRSM_THRESHOLD
#define EIGEN_MCU_SMALL_TRSM_THRESHOLD 32
#endif

// Cortex-M7 FPv5-D16 (and ARMv8/AArch64 directed rounding) have 1-cycle
// hardware vmaxnm/vminnm instructions matching std::fmax/std::fmin.
// Cortex-M4F (fpv4-sp-d16, __ARM_FP == 4) does NOT have vmaxnm/vminnm and
// would trap into slow libm fmaxf/fminf calls if enabled.
#ifndef EIGEN_MCU_HAS_VMAXNM
#if (defined(__ARM_ARCH_7EM__) && defined(__ARM_FP) && (__ARM_FP & 8)) || defined(__ARM_FEATURE_DIRECTED_ROUNDING)
#define EIGEN_MCU_HAS_VMAXNM 1
#else
#define EIGEN_MCU_HAS_VMAXNM 0
#endif
#endif

// Cortex-M defaults: no 16-byte static alignment padding (ARM EABI stack is
// 8-byte aligned and there is no SIMD), no OpenMP/threadpool parallelism, and
// accurate register-file / FMA configuration for fallback >40x40 kernels.
#if EIGEN_MCU_ARCH_CORTEX_M
#ifndef EIGEN_MAX_STATIC_ALIGN_BYTES
#define EIGEN_MAX_STATIC_ALIGN_BYTES 0
#endif
#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif
#if defined(__ARM_FP) && (__ARM_FP > 0)
#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 16
#endif
#endif
#if defined(__ARM_FEATURE_FMA) && !defined(EIGEN_HAS_SINGLE_INSTRUCTION_MADD)
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD 1
#endif
// Bare-metal ARM toolchains do not define EIGEN_OS_LINUX or include <alloca.h>,
// which causes Eigen's ei_declare_aligned_stack_constructed_variable (used by
// HouseholderQR and GEMV) to fall back unconditionally to heap malloc. Enable
// GCC's native stack alloca (avoiding Clang Thumb2 per Eigen's Memory.h note).
#if !defined(EIGEN_ALLOCA) && !defined(EIGEN_NO_ALLOCA) && EIGEN_COMP_GNUC && !defined(__clang__)
#define EIGEN_ALLOCA __builtin_alloca
#endif
#endif

// Force-inline tiny view constructors and per-element evaluator/assignment
// wrappers under -Os so GCC does not outline them, without disturbing -O2
// loop unrolling heuristics.
#ifndef EIGEN_MCU_INLINE_SIZE
#if defined(__OPTIMIZE_SIZE__)
#define EIGEN_MCU_INLINE_SIZE EIGEN_ALWAYS_INLINE
#else
#define EIGEN_MCU_INLINE_SIZE EIGEN_STRONG_INLINE
#endif
#endif

#endif  // EIGEN_MCU_CONFIG_H
