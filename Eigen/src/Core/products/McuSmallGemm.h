// This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
//
// Copyright (C) 2026 SFWA contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unpacked register-blocked small-matrix GEMM, SYRK, triangular product, GEMV,
// fused Householder reflection, direct Cholesky / fused LLT solve, and
// unblocked pre-inverted-diagonal TRSM/TRSV kernels for small matrices.
//
// Avoids GotoBLAS packing buffers (0.8-8.6 KB stack) and pack/unpack passes
// that have no cache benefit when operands fit in SRAM/DTCM/L1. Uses 4x4
// tiles for float (24 of 32 FP32 registers on Cortex-M4F/M7) and 2x4 tiles
// for double (14 of 16 FP64 registers on Cortex-M7).

#ifndef EIGEN_MCU_SMALL_GEMM_H
#define EIGEN_MCU_SMALL_GEMM_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// ============================================================================
// Shared MR x 4 and MR x 2 micro-tile helpers (MR = 4 for float, MR = 2 double)
// ============================================================================

template <typename Scalar, int LhsOrder, int RhsOrder, int MR>
EIGEN_ALWAYS_INLINE void mcu_small_gemm_tile_mr4(Index i, Index j, Index K, const Scalar* A, Index lda, Index ldb,
                                                 const Scalar* b_base, const Scalar* b0, const Scalar* b1,
                                                 const Scalar* b2, const Scalar* b3, Scalar* EIGEN_RESTRICT C,
                                                 Index ldc, Scalar alpha, bool alpha_one, bool beta_zero) {
  constexpr bool LhsCol = (LhsOrder == ColMajor);
  constexpr bool RhsCol = (RhsOrder == ColMajor);
  EIGEN_UNUSED_VARIABLE(ldb);
  EIGEN_UNUSED_VARIABLE(b_base);
  EIGEN_UNUSED_VARIABLE(b0);
  EIGEN_UNUSED_VARIABLE(b1);
  EIGEN_UNUSED_VARIABLE(b2);
  EIGEN_UNUSED_VARIABLE(b3);

  EIGEN_IF_CONSTEXPR(MR == 4) {
    Scalar c00 = Scalar(0), c10 = Scalar(0), c20 = Scalar(0), c30 = Scalar(0);
    Scalar c01 = Scalar(0), c11 = Scalar(0), c21 = Scalar(0), c31 = Scalar(0);
    Scalar c02 = Scalar(0), c12 = Scalar(0), c22 = Scalar(0), c32 = Scalar(0);
    Scalar c03 = Scalar(0), c13 = Scalar(0), c23 = Scalar(0), c33 = Scalar(0);

    const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
    const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
    const Scalar* a2 = LhsCol ? nullptr : (A + (i + 2) * lda);
    const Scalar* a3 = LhsCol ? nullptr : (A + (i + 3) * lda);
    const Scalar* bk = b_base;
    EIGEN_UNUSED_VARIABLE(a1);
    EIGEN_UNUSED_VARIABLE(a2);
    EIGEN_UNUSED_VARIABLE(a3);
    EIGEN_UNUSED_VARIABLE(bk);

    for (Index k = 0; k < K; ++k) {
      Scalar va0, va1, va2, va3;
      EIGEN_IF_CONSTEXPR(LhsCol) {
        va0 = ak[0];
        va1 = ak[1];
        va2 = ak[2];
        va3 = ak[3];
        ak += lda;
      } else {
        va0 = ak[k];
        va1 = a1[k];
        va2 = a2[k];
        va3 = a3[k];
      }
      Scalar vb0, vb1, vb2, vb3;
      EIGEN_IF_CONSTEXPR(RhsCol) {
        vb0 = b0[k];
        vb1 = b1[k];
        vb2 = b2[k];
        vb3 = b3[k];
      } else {
        vb0 = bk[0];
        vb1 = bk[1];
        vb2 = bk[2];
        vb3 = bk[3];
        bk += ldb;
      }
      c00 += va0 * vb0;
      c10 += va1 * vb0;
      c20 += va2 * vb0;
      c30 += va3 * vb0;
      c01 += va0 * vb1;
      c11 += va1 * vb1;
      c21 += va2 * vb1;
      c31 += va3 * vb1;
      c02 += va0 * vb2;
      c12 += va1 * vb2;
      c22 += va2 * vb2;
      c32 += va3 * vb2;
      c03 += va0 * vb3;
      c13 += va1 * vb3;
      c23 += va2 * vb3;
      c33 += va3 * vb3;
    }
    if (!alpha_one) {
      c00 *= alpha;
      c10 *= alpha;
      c20 *= alpha;
      c30 *= alpha;
      c01 *= alpha;
      c11 *= alpha;
      c21 *= alpha;
      c31 *= alpha;
      c02 *= alpha;
      c12 *= alpha;
      c22 *= alpha;
      c32 *= alpha;
      c03 *= alpha;
      c13 *= alpha;
      c23 *= alpha;
      c33 *= alpha;
    }
    Scalar* p0 = C + (j + 0) * ldc + i;
    Scalar* p1 = C + (j + 1) * ldc + i;
    Scalar* p2 = C + (j + 2) * ldc + i;
    Scalar* p3 = C + (j + 3) * ldc + i;
    if (beta_zero) {
      p0[0] = c00;
      p0[1] = c10;
      p0[2] = c20;
      p0[3] = c30;
      p1[0] = c01;
      p1[1] = c11;
      p1[2] = c21;
      p1[3] = c31;
      p2[0] = c02;
      p2[1] = c12;
      p2[2] = c22;
      p2[3] = c32;
      p3[0] = c03;
      p3[1] = c13;
      p3[2] = c23;
      p3[3] = c33;
    } else {
      p0[0] += c00;
      p0[1] += c10;
      p0[2] += c20;
      p0[3] += c30;
      p1[0] += c01;
      p1[1] += c11;
      p1[2] += c21;
      p1[3] += c31;
      p2[0] += c02;
      p2[1] += c12;
      p2[2] += c22;
      p2[3] += c32;
      p3[0] += c03;
      p3[1] += c13;
      p3[2] += c23;
      p3[3] += c33;
    }
  } else {
    // 2x4 microkernel for double (14 of 16 FP64 registers on Cortex-M7).
    Scalar c00 = Scalar(0), c10 = Scalar(0);
    Scalar c01 = Scalar(0), c11 = Scalar(0);
    Scalar c02 = Scalar(0), c12 = Scalar(0);
    Scalar c03 = Scalar(0), c13 = Scalar(0);

    const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
    const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
    const Scalar* bk = b_base;
    EIGEN_UNUSED_VARIABLE(a1);
    EIGEN_UNUSED_VARIABLE(bk);

    for (Index k = 0; k < K; ++k) {
      Scalar va0, va1;
      EIGEN_IF_CONSTEXPR(LhsCol) {
        va0 = ak[0];
        va1 = ak[1];
        ak += lda;
      } else {
        va0 = ak[k];
        va1 = a1[k];
      }
      Scalar vb0, vb1, vb2, vb3;
      EIGEN_IF_CONSTEXPR(RhsCol) {
        vb0 = b0[k];
        vb1 = b1[k];
        vb2 = b2[k];
        vb3 = b3[k];
      } else {
        vb0 = bk[0];
        vb1 = bk[1];
        vb2 = bk[2];
        vb3 = bk[3];
        bk += ldb;
      }
      c00 += va0 * vb0;
      c10 += va1 * vb0;
      c01 += va0 * vb1;
      c11 += va1 * vb1;
      c02 += va0 * vb2;
      c12 += va1 * vb2;
      c03 += va0 * vb3;
      c13 += va1 * vb3;
    }
    if (!alpha_one) {
      c00 *= alpha;
      c10 *= alpha;
      c01 *= alpha;
      c11 *= alpha;
      c02 *= alpha;
      c12 *= alpha;
      c03 *= alpha;
      c13 *= alpha;
    }
    Scalar* p0 = C + (j + 0) * ldc + i;
    Scalar* p1 = C + (j + 1) * ldc + i;
    Scalar* p2 = C + (j + 2) * ldc + i;
    Scalar* p3 = C + (j + 3) * ldc + i;
    if (beta_zero) {
      p0[0] = c00;
      p0[1] = c10;
      p1[0] = c01;
      p1[1] = c11;
      p2[0] = c02;
      p2[1] = c12;
      p3[0] = c03;
      p3[1] = c13;
    } else {
      p0[0] += c00;
      p0[1] += c10;
      p1[0] += c01;
      p1[1] += c11;
      p2[0] += c02;
      p2[1] += c12;
      p3[0] += c03;
      p3[1] += c13;
    }
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder, int MR>
EIGEN_ALWAYS_INLINE void mcu_small_gemm_tile_mr2(Index i, Index j, Index K, const Scalar* A, Index lda, Index ldb,
                                                 const Scalar* b_base, const Scalar* b0, const Scalar* b1,
                                                 Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha, bool alpha_one,
                                                 bool beta_zero) {
  constexpr bool LhsCol = (LhsOrder == ColMajor);
  constexpr bool RhsCol = (RhsOrder == ColMajor);
  EIGEN_UNUSED_VARIABLE(ldb);
  EIGEN_UNUSED_VARIABLE(b_base);
  EIGEN_UNUSED_VARIABLE(b0);
  EIGEN_UNUSED_VARIABLE(b1);

  EIGEN_IF_CONSTEXPR(MR == 4) {
    Scalar c00 = Scalar(0), c10 = Scalar(0), c20 = Scalar(0), c30 = Scalar(0);
    Scalar c01 = Scalar(0), c11 = Scalar(0), c21 = Scalar(0), c31 = Scalar(0);

    const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
    const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
    const Scalar* a2 = LhsCol ? nullptr : (A + (i + 2) * lda);
    const Scalar* a3 = LhsCol ? nullptr : (A + (i + 3) * lda);
    const Scalar* bk = b_base;
    EIGEN_UNUSED_VARIABLE(a1);
    EIGEN_UNUSED_VARIABLE(a2);
    EIGEN_UNUSED_VARIABLE(a3);
    EIGEN_UNUSED_VARIABLE(bk);

    for (Index k = 0; k < K; ++k) {
      Scalar va0, va1, va2, va3;
      EIGEN_IF_CONSTEXPR(LhsCol) {
        va0 = ak[0];
        va1 = ak[1];
        va2 = ak[2];
        va3 = ak[3];
        ak += lda;
      } else {
        va0 = ak[k];
        va1 = a1[k];
        va2 = a2[k];
        va3 = a3[k];
      }
      Scalar vb0, vb1;
      EIGEN_IF_CONSTEXPR(RhsCol) {
        vb0 = b0[k];
        vb1 = b1[k];
      } else {
        vb0 = bk[0];
        vb1 = bk[1];
        bk += ldb;
      }
      c00 += va0 * vb0;
      c10 += va1 * vb0;
      c20 += va2 * vb0;
      c30 += va3 * vb0;
      c01 += va0 * vb1;
      c11 += va1 * vb1;
      c21 += va2 * vb1;
      c31 += va3 * vb1;
    }
    if (!alpha_one) {
      c00 *= alpha;
      c10 *= alpha;
      c20 *= alpha;
      c30 *= alpha;
      c01 *= alpha;
      c11 *= alpha;
      c21 *= alpha;
      c31 *= alpha;
    }
    Scalar* p0 = C + (j + 0) * ldc + i;
    Scalar* p1 = C + (j + 1) * ldc + i;
    if (beta_zero) {
      p0[0] = c00;
      p0[1] = c10;
      p0[2] = c20;
      p0[3] = c30;
      p1[0] = c01;
      p1[1] = c11;
      p1[2] = c21;
      p1[3] = c31;
    } else {
      p0[0] += c00;
      p0[1] += c10;
      p0[2] += c20;
      p0[3] += c30;
      p1[0] += c01;
      p1[1] += c11;
      p1[2] += c21;
      p1[3] += c31;
    }
  } else {
    Scalar c00 = Scalar(0), c10 = Scalar(0);
    Scalar c01 = Scalar(0), c11 = Scalar(0);

    const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
    const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
    const Scalar* bk = b_base;
    EIGEN_UNUSED_VARIABLE(a1);
    EIGEN_UNUSED_VARIABLE(bk);

    for (Index k = 0; k < K; ++k) {
      Scalar va0, va1;
      EIGEN_IF_CONSTEXPR(LhsCol) {
        va0 = ak[0];
        va1 = ak[1];
        ak += lda;
      } else {
        va0 = ak[k];
        va1 = a1[k];
      }
      Scalar vb0, vb1;
      EIGEN_IF_CONSTEXPR(RhsCol) {
        vb0 = b0[k];
        vb1 = b1[k];
      } else {
        vb0 = bk[0];
        vb1 = bk[1];
        bk += ldb;
      }
      c00 += va0 * vb0;
      c10 += va1 * vb0;
      c01 += va0 * vb1;
      c11 += va1 * vb1;
    }
    if (!alpha_one) {
      c00 *= alpha;
      c10 *= alpha;
      c01 *= alpha;
      c11 *= alpha;
    }
    Scalar* p0 = C + (j + 0) * ldc + i;
    Scalar* p1 = C + (j + 1) * ldc + i;
    if (beta_zero) {
      p0[0] = c00;
      p0[1] = c10;
      p1[0] = c01;
      p1[1] = c11;
    } else {
      p0[0] += c00;
      p0[1] += c10;
      p1[0] += c01;
      p1[1] += c11;
    }
  }
}

// Helper for 2x2 diagonal triangle in triangular GEMM/SYRK.
template <typename Scalar, int LhsOrder, int RhsOrder, bool IsLower>
EIGEN_ALWAYS_INLINE void mcu_small_gemm_diag2(Index j, Index K, const Scalar* A, Index lda, const Scalar* B, Index ldb,
                                              Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha, bool alpha_one,
                                              bool beta_zero, bool is_sym) {
  constexpr bool LhsCol = (LhsOrder == ColMajor);
  constexpr bool RhsCol = (RhsOrder == ColMajor);
  Scalar d00 = Scalar(0), d10 = Scalar(0), d11 = Scalar(0);
  const Scalar* ak = LhsCol ? (A + j) : (A + j * lda);
  const Scalar* a1 = LhsCol ? nullptr : (A + (j + 1) * lda);
  const Scalar* b_base = RhsCol ? (B + j * ldb) : (B + j);
  const Scalar* b0 = RhsCol ? (b_base + 0 * ldb) : b_base;
  const Scalar* b1 = RhsCol ? (b_base + 1 * ldb) : b_base;
  const Scalar* bk = b_base;
  EIGEN_UNUSED_VARIABLE(a1);
  EIGEN_UNUSED_VARIABLE(b0);
  EIGEN_UNUSED_VARIABLE(b1);
  EIGEN_UNUSED_VARIABLE(bk);
  EIGEN_UNUSED_VARIABLE(is_sym);

  EIGEN_IF_CONSTEXPR(LhsCol != RhsCol) {
    if (is_sym) {
      for (Index k = 0; k < K; ++k) {
        Scalar v0, v1;
        EIGEN_IF_CONSTEXPR(LhsCol) {
          v0 = ak[0];
          v1 = ak[1];
          ak += lda;
        } else {
          v0 = ak[k];
          v1 = a1[k];
        }
        d00 += v0 * v0;
        d10 += v1 * v0;
        d11 += v1 * v1;
      }
      goto store_diag2;
    }
  }
  for (Index k = 0; k < K; ++k) {
    Scalar va0, va1;
    EIGEN_IF_CONSTEXPR(LhsCol) {
      va0 = ak[0];
      va1 = ak[1];
      ak += lda;
    } else {
      va0 = ak[k];
      va1 = a1[k];
    }
    Scalar vb0, vb1;
    EIGEN_IF_CONSTEXPR(RhsCol) {
      vb0 = b0[k];
      vb1 = b1[k];
    } else {
      vb0 = bk[0];
      vb1 = bk[1];
      bk += ldb;
    }
    EIGEN_IF_CONSTEXPR(IsLower) {
      d00 += va0 * vb0;
      d10 += va1 * vb0;
      d11 += va1 * vb1;
    } else {
      d00 += va0 * vb0;
      d10 += va0 * vb1;
      d11 += va1 * vb1;
    }
  }

store_diag2:
  if (!alpha_one) {
    d00 *= alpha;
    d10 *= alpha;
    d11 *= alpha;
  }
  EIGEN_IF_CONSTEXPR(IsLower) {
    Scalar* p0 = C + (j + 0) * ldc + j;
    Scalar* p1 = C + (j + 1) * ldc + j;
    if (beta_zero) {
      p0[0] = d00;
      p0[1] = d10;
      p1[1] = d11;
    } else {
      p0[0] += d00;
      p0[1] += d10;
      p1[1] += d11;
    }
  } else {
    Scalar* p0 = C + (j + 0) * ldc + j;
    Scalar* p1 = C + (j + 1) * ldc + j;
    if (beta_zero) {
      p0[0] = d00;
      p1[0] = d10;
      p1[1] = d11;
    } else {
      p0[0] += d00;
      p1[0] += d10;
      p1[1] += d11;
    }
  }
}

// ============================================================================
// Unpacked Register-Blocked Small GEMM: C = (beta_zero ? 0 : C) + alpha * A * B
// Note: A and B may alias each other (e.g. A * A), so only C is EIGEN_RESTRICT.
// ============================================================================

template <typename Scalar, int LhsOrder, int RhsOrder>
EIGEN_DONT_INLINE void mcu_small_gemm_colmajor(Index M, Index N, Index K, const Scalar* A, Index lda, const Scalar* B,
                                               Index ldb, Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha,
                                               bool beta_zero = false) {
  if (M <= 0 || N <= 0) return;
  if (K <= 0) {
    if (beta_zero) {
      for (Index j = 0; j < N; ++j) {
        Scalar* cj = C + j * ldc;
        for (Index i = 0; i < M; ++i) cj[i] = Scalar(0);
      }
    }
    return;
  }

  constexpr bool LhsCol = (LhsOrder == ColMajor);
  constexpr bool RhsCol = (RhsOrder == ColMajor);
  constexpr int MR = is_same<Scalar, float>::value ? 4 : 2;
  const bool alpha_one = (alpha == Scalar(1));

  Index j = 0;
  for (; j + 4 <= N; j += 4) {
    const Scalar* b_base = RhsCol ? (B + j * ldb) : (B + j);
    const Scalar* b0 = RhsCol ? (b_base + 0 * ldb) : b_base;
    const Scalar* b1 = RhsCol ? (b_base + 1 * ldb) : b_base;
    const Scalar* b2 = RhsCol ? (b_base + 2 * ldb) : b_base;
    const Scalar* b3 = RhsCol ? (b_base + 3 * ldb) : b_base;

    Index i = 0;
    for (; i + MR <= M; i += MR) {
      mcu_small_gemm_tile_mr4<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, b2, b3, C, ldc,
                                                              alpha, alpha_one, beta_zero);
    }
    // Tail rows (i < M) for 4 columns.
    for (; i < M; ++i) {
      Scalar c0 = Scalar(0), c1 = Scalar(0), c2 = Scalar(0), c3 = Scalar(0);
      const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
      const Scalar* bk = b_base;
      EIGEN_UNUSED_VARIABLE(bk);
      for (Index k = 0; k < K; ++k) {
        Scalar va;
        EIGEN_IF_CONSTEXPR(LhsCol) {
          va = *ak;
          ak += lda;
        } else {
          va = ak[k];
        }
        EIGEN_IF_CONSTEXPR(RhsCol) {
          c0 += va * b0[k];
          c1 += va * b1[k];
          c2 += va * b2[k];
          c3 += va * b3[k];
        } else {
          c0 += va * bk[0];
          c1 += va * bk[1];
          c2 += va * bk[2];
          c3 += va * bk[3];
          bk += ldb;
        }
      }
      if (!alpha_one) {
        c0 *= alpha;
        c1 *= alpha;
        c2 *= alpha;
        c3 *= alpha;
      }
      if (beta_zero) {
        C[(j + 0) * ldc + i] = c0;
        C[(j + 1) * ldc + i] = c1;
        C[(j + 2) * ldc + i] = c2;
        C[(j + 3) * ldc + i] = c3;
      } else {
        C[(j + 0) * ldc + i] += c0;
        C[(j + 1) * ldc + i] += c1;
        C[(j + 2) * ldc + i] += c2;
        C[(j + 3) * ldc + i] += c3;
      }
    }
  }

  // 2-column tail (j + 2 <= N): MRx2 tile avoids falling straight to 1-column loops.
  if (j + 2 <= N) {
    const Scalar* b_base = RhsCol ? (B + j * ldb) : (B + j);
    const Scalar* b0 = RhsCol ? (b_base + 0 * ldb) : b_base;
    const Scalar* b1 = RhsCol ? (b_base + 1 * ldb) : b_base;

    Index i = 0;
    for (; i + MR <= M; i += MR) {
      mcu_small_gemm_tile_mr2<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, C, ldc, alpha,
                                                              alpha_one, beta_zero);
    }
    for (; i < M; ++i) {
      Scalar c0 = Scalar(0), c1 = Scalar(0);
      const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
      const Scalar* bk = b_base;
      EIGEN_UNUSED_VARIABLE(bk);
      for (Index k = 0; k < K; ++k) {
        Scalar va;
        EIGEN_IF_CONSTEXPR(LhsCol) {
          va = *ak;
          ak += lda;
        } else {
          va = ak[k];
        }
        EIGEN_IF_CONSTEXPR(RhsCol) {
          c0 += va * b0[k];
          c1 += va * b1[k];
        } else {
          c0 += va * bk[0];
          c1 += va * bk[1];
          bk += ldb;
        }
      }
      if (!alpha_one) {
        c0 *= alpha;
        c1 *= alpha;
      }
      if (beta_zero) {
        C[(j + 0) * ldc + i] = c0;
        C[(j + 1) * ldc + i] = c1;
      } else {
        C[(j + 0) * ldc + i] += c0;
        C[(j + 1) * ldc + i] += c1;
      }
    }
    j += 2;
  }

  // Final 1-column tail (j < N).
  if (j < N) {
    const Scalar* bj = RhsCol ? (B + j * ldb) : (B + j);
    Index i = 0;
    for (; i + MR <= M; i += MR) {
      EIGEN_IF_CONSTEXPR(MR == 4) {
        Scalar c0 = Scalar(0), c1 = Scalar(0), c2 = Scalar(0), c3 = Scalar(0);
        const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
        const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
        const Scalar* a2 = LhsCol ? nullptr : (A + (i + 2) * lda);
        const Scalar* a3 = LhsCol ? nullptr : (A + (i + 3) * lda);
        const Scalar* bk = bj;
        EIGEN_UNUSED_VARIABLE(a1);
        EIGEN_UNUSED_VARIABLE(a2);
        EIGEN_UNUSED_VARIABLE(a3);
        EIGEN_UNUSED_VARIABLE(bk);
        for (Index k = 0; k < K; ++k) {
          Scalar vb;
          EIGEN_IF_CONSTEXPR(RhsCol) {
            vb = bj[k];
          } else {
            vb = *bk;
            bk += ldb;
          }
          EIGEN_IF_CONSTEXPR(LhsCol) {
            c0 += ak[0] * vb;
            c1 += ak[1] * vb;
            c2 += ak[2] * vb;
            c3 += ak[3] * vb;
            ak += lda;
          } else {
            c0 += ak[k] * vb;
            c1 += a1[k] * vb;
            c2 += a2[k] * vb;
            c3 += a3[k] * vb;
          }
        }
        if (!alpha_one) {
          c0 *= alpha;
          c1 *= alpha;
          c2 *= alpha;
          c3 *= alpha;
        }
        Scalar* p = C + j * ldc + i;
        if (beta_zero) {
          p[0] = c0;
          p[1] = c1;
          p[2] = c2;
          p[3] = c3;
        } else {
          p[0] += c0;
          p[1] += c1;
          p[2] += c2;
          p[3] += c3;
        }
      } else {
        Scalar c0 = Scalar(0), c1 = Scalar(0);
        const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
        const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
        const Scalar* bk = bj;
        EIGEN_UNUSED_VARIABLE(a1);
        EIGEN_UNUSED_VARIABLE(bk);
        for (Index k = 0; k < K; ++k) {
          Scalar vb;
          EIGEN_IF_CONSTEXPR(RhsCol) {
            vb = bj[k];
          } else {
            vb = *bk;
            bk += ldb;
          }
          EIGEN_IF_CONSTEXPR(LhsCol) {
            c0 += ak[0] * vb;
            c1 += ak[1] * vb;
            ak += lda;
          } else {
            c0 += ak[k] * vb;
            c1 += a1[k] * vb;
          }
        }
        if (!alpha_one) {
          c0 *= alpha;
          c1 *= alpha;
        }
        Scalar* p = C + j * ldc + i;
        if (beta_zero) {
          p[0] = c0;
          p[1] = c1;
        } else {
          p[0] += c0;
          p[1] += c1;
        }
      }
    }
    for (; i < M; ++i) {
      Scalar c0 = Scalar(0);
      const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
      const Scalar* bk = bj;
      EIGEN_UNUSED_VARIABLE(bk);
      for (Index k = 0; k < K; ++k) {
        Scalar va, vb;
        EIGEN_IF_CONSTEXPR(LhsCol) {
          va = *ak;
          ak += lda;
        } else {
          va = ak[k];
        }
        EIGEN_IF_CONSTEXPR(RhsCol) {
          vb = bj[k];
        } else {
          vb = *bk;
          bk += ldb;
        }
        c0 += va * vb;
      }
      if (!alpha_one) c0 *= alpha;
      if (beta_zero)
        C[j * ldc + i] = c0;
      else
        C[j * ldc + i] += c0;
    }
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder, int DstOrder>
EIGEN_ALWAYS_INLINE void mcu_small_gemm(Index M, Index N, Index K, const Scalar* A, Index lda, const Scalar* B,
                                        Index ldb, Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha,
                                        bool beta_zero = false) {
  EIGEN_IF_CONSTEXPR(DstOrder == ColMajor) {
    mcu_small_gemm_colmajor<Scalar, LhsOrder, RhsOrder>(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta_zero);
  } else {
    constexpr int FlippedRhs = (RhsOrder == RowMajor) ? ColMajor : RowMajor;
    constexpr int FlippedLhs = (LhsOrder == RowMajor) ? ColMajor : RowMajor;
    mcu_small_gemm_colmajor<Scalar, FlippedRhs, FlippedLhs>(N, M, K, B, ldb, A, lda, C, ldc, alpha, beta_zero);
  }
}

// ============================================================================
// Unpacked Register-Blocked Small Triangular Product & SYRK:
// C_uplo = (beta_zero ? 0 : C_uplo) + alpha * A * B (Dst is ColMajor).
// Detects symmetric SYRK (A == B with transposed layout) to halve diagonal-tile
// loads and keep FP64 diagonal tiles within 14 of 16 registers (zero spills).
// ============================================================================

template <typename Scalar, int LhsOrder, int RhsOrder, int UpLo>
EIGEN_DONT_INLINE void mcu_small_gemm_triangular_colmajor(Index size, Index K, const Scalar* A, Index lda,
                                                          const Scalar* B, Index ldb, Scalar* EIGEN_RESTRICT C,
                                                          Index ldc, Scalar alpha, bool beta_zero = false) {
  if (size <= 0) return;
  constexpr bool LhsCol = (LhsOrder == ColMajor);
  constexpr bool RhsCol = (RhsOrder == ColMajor);
  constexpr bool IsLower = (UpLo == Lower);
  constexpr int MR = is_same<Scalar, float>::value ? 4 : 2;

  if (K <= 0) {
    if (beta_zero) {
      for (Index j = 0; j < size; ++j) {
        const Index i0 = IsLower ? j : Index(0);
        const Index i1 = IsLower ? size : (j + 1);
        Scalar* cj = C + j * ldc;
        for (Index i = i0; i < i1; ++i) cj[i] = Scalar(0);
      }
    }
    return;
  }

  const bool alpha_one = (alpha == Scalar(1));
  const bool is_sym = (LhsCol != RhsCol) && (A == B) && (lda == ldb);

  Index j = 0;
  for (; j + 4 <= size; j += 4) {
    const Scalar* b_base = RhsCol ? (B + j * ldb) : (B + j);
    const Scalar* b0 = RhsCol ? (b_base + 0 * ldb) : b_base;
    const Scalar* b1 = RhsCol ? (b_base + 1 * ldb) : b_base;
    const Scalar* b2 = RhsCol ? (b_base + 2 * ldb) : b_base;
    const Scalar* b3 = RhsCol ? (b_base + 3 * ldb) : b_base;

    EIGEN_IF_CONSTEXPR(!IsLower) {
      // Strictly above diagonal: rows [0, j) (j is a multiple of 4, hence of MR).
      for (Index i = 0; i + MR <= j; i += MR) {
        mcu_small_gemm_tile_mr4<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, b2, b3, C, ldc,
                                                                alpha, alpha_one, beta_zero);
      }
    }

    // Diagonal 4x4 triangle at (j, j):
    // - For symmetric SYRK (A == B, LhsCol != RhsCol): single-load 4x4 triangle
    //   uses 10 accumulators + 4 operands = 14 FP regs (zero spills for float and double).
    // - For general double A*B (MR == 2, !is_sym): split into two 2x2 diagonal triangles
    //   plus one 2x2 off-diagonal tile (<= 8 FP64 regs, zero spills).
    // - For general float A*B (MR == 4): dual-load 4x4 triangle (18 of 32 FP32 regs).
    if (MR == 2 && !is_sym) {
      mcu_small_gemm_diag2<Scalar, LhsOrder, RhsOrder, IsLower>(j, K, A, lda, B, ldb, C, ldc, alpha, alpha_one,
                                                                beta_zero, false);
      EIGEN_IF_CONSTEXPR(IsLower) {
        const Scalar* tb0 = RhsCol ? (B + j * ldb) : (B + j);
        const Scalar* tb1 = RhsCol ? (B + (j + 1) * ldb) : tb0;
        mcu_small_gemm_tile_mr2<Scalar, LhsOrder, RhsOrder, 2>(j + 2, j, K, A, lda, ldb, tb0, tb0, tb1, C, ldc, alpha,
                                                               alpha_one, beta_zero);
      } else {
        const Scalar* tb0 = RhsCol ? (B + (j + 2) * ldb) : (B + j + 2);
        const Scalar* tb1 = RhsCol ? (B + (j + 3) * ldb) : tb0;
        mcu_small_gemm_tile_mr2<Scalar, LhsOrder, RhsOrder, 2>(j, j + 2, K, A, lda, ldb, tb0, tb0, tb1, C, ldc, alpha,
                                                               alpha_one, beta_zero);
      }
      mcu_small_gemm_diag2<Scalar, LhsOrder, RhsOrder, IsLower>(j + 2, K, A, lda, B, ldb, C, ldc, alpha, alpha_one,
                                                                beta_zero, false);
    } else {
      Scalar d00 = Scalar(0), d11 = Scalar(0), d22 = Scalar(0), d33 = Scalar(0);
      Scalar d10 = Scalar(0), d20 = Scalar(0), d30 = Scalar(0);
      Scalar d21 = Scalar(0), d31 = Scalar(0), d32 = Scalar(0);
      const Scalar* ak = LhsCol ? (A + j) : (A + j * lda);
      const Scalar* a1 = LhsCol ? nullptr : (A + (j + 1) * lda);
      const Scalar* a2 = LhsCol ? nullptr : (A + (j + 2) * lda);
      const Scalar* a3 = LhsCol ? nullptr : (A + (j + 3) * lda);
      const Scalar* bk = b_base;
      EIGEN_UNUSED_VARIABLE(a1);
      EIGEN_UNUSED_VARIABLE(a2);
      EIGEN_UNUSED_VARIABLE(a3);
      EIGEN_UNUSED_VARIABLE(bk);

      EIGEN_IF_CONSTEXPR(LhsCol != RhsCol) {
        if (is_sym) {
          for (Index k = 0; k < K; ++k) {
            Scalar v0, v1, v2, v3;
            EIGEN_IF_CONSTEXPR(LhsCol) {
              v0 = ak[0];
              v1 = ak[1];
              v2 = ak[2];
              v3 = ak[3];
              ak += lda;
            } else {
              v0 = ak[k];
              v1 = a1[k];
              v2 = a2[k];
              v3 = a3[k];
            }
            d00 += v0 * v0;
            d10 += v1 * v0;
            d20 += v2 * v0;
            d30 += v3 * v0;
            d11 += v1 * v1;
            d21 += v2 * v1;
            d31 += v3 * v1;
            d22 += v2 * v2;
            d32 += v3 * v2;
            d33 += v3 * v3;
          }
          goto store_diag4;
        }
      }
      for (Index k = 0; k < K; ++k) {
        Scalar va0, va1, va2, va3;
        EIGEN_IF_CONSTEXPR(LhsCol) {
          va0 = ak[0];
          va1 = ak[1];
          va2 = ak[2];
          va3 = ak[3];
          ak += lda;
        } else {
          va0 = ak[k];
          va1 = a1[k];
          va2 = a2[k];
          va3 = a3[k];
        }
        Scalar vb0, vb1, vb2, vb3;
        EIGEN_IF_CONSTEXPR(RhsCol) {
          vb0 = b0[k];
          vb1 = b1[k];
          vb2 = b2[k];
          vb3 = b3[k];
        } else {
          vb0 = bk[0];
          vb1 = bk[1];
          vb2 = bk[2];
          vb3 = bk[3];
          bk += ldb;
        }
        EIGEN_IF_CONSTEXPR(IsLower) {
          d00 += va0 * vb0;
          d10 += va1 * vb0;
          d20 += va2 * vb0;
          d30 += va3 * vb0;
          d11 += va1 * vb1;
          d21 += va2 * vb1;
          d31 += va3 * vb1;
          d22 += va2 * vb2;
          d32 += va3 * vb2;
          d33 += va3 * vb3;
        } else {
          d00 += va0 * vb0;
          d10 += va0 * vb1;
          d11 += va1 * vb1;
          d20 += va0 * vb2;
          d21 += va1 * vb2;
          d22 += va2 * vb2;
          d30 += va0 * vb3;
          d31 += va1 * vb3;
          d32 += va2 * vb3;
          d33 += va3 * vb3;
        }
      }

    store_diag4:
      if (!alpha_one) {
        d00 *= alpha;
        d10 *= alpha;
        d20 *= alpha;
        d30 *= alpha;
        d11 *= alpha;
        d21 *= alpha;
        d31 *= alpha;
        d22 *= alpha;
        d32 *= alpha;
        d33 *= alpha;
      }
      EIGEN_IF_CONSTEXPR(IsLower) {
        Scalar* p0 = C + (j + 0) * ldc + j;
        Scalar* p1 = C + (j + 1) * ldc + j;
        Scalar* p2 = C + (j + 2) * ldc + j;
        Scalar* p3 = C + (j + 3) * ldc + j;
        if (beta_zero) {
          p0[0] = d00;
          p0[1] = d10;
          p0[2] = d20;
          p0[3] = d30;
          p1[1] = d11;
          p1[2] = d21;
          p1[3] = d31;
          p2[2] = d22;
          p2[3] = d32;
          p3[3] = d33;
        } else {
          p0[0] += d00;
          p0[1] += d10;
          p0[2] += d20;
          p0[3] += d30;
          p1[1] += d11;
          p1[2] += d21;
          p1[3] += d31;
          p2[2] += d22;
          p2[3] += d32;
          p3[3] += d33;
        }
      } else {
        Scalar* p0 = C + (j + 0) * ldc + j;
        Scalar* p1 = C + (j + 1) * ldc + j;
        Scalar* p2 = C + (j + 2) * ldc + j;
        Scalar* p3 = C + (j + 3) * ldc + j;
        if (beta_zero) {
          p0[0] = d00;
          p1[0] = d10;
          p1[1] = d11;
          p2[0] = d20;
          p2[1] = d21;
          p2[2] = d22;
          p3[0] = d30;
          p3[1] = d31;
          p3[2] = d32;
          p3[3] = d33;
        } else {
          p0[0] += d00;
          p1[0] += d10;
          p1[1] += d11;
          p2[0] += d20;
          p2[1] += d21;
          p2[2] += d22;
          p3[0] += d30;
          p3[1] += d31;
          p3[2] += d32;
          p3[3] += d33;
        }
      }
    }

    EIGEN_IF_CONSTEXPR(IsLower) {
      // Strictly below diagonal: rows [j + 4, size).
      Index i = j + 4;
      for (; i + MR <= size; i += MR) {
        mcu_small_gemm_tile_mr4<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, b2, b3, C, ldc,
                                                                alpha, alpha_one, beta_zero);
      }
      for (; i < size; ++i) {
        Scalar r0 = Scalar(0), r1 = Scalar(0), r2 = Scalar(0), r3 = Scalar(0);
        const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
        const Scalar* bk = b_base;
        EIGEN_UNUSED_VARIABLE(bk);
        for (Index k = 0; k < K; ++k) {
          Scalar va;
          EIGEN_IF_CONSTEXPR(LhsCol) {
            va = *ak;
            ak += lda;
          } else {
            va = ak[k];
          }
          EIGEN_IF_CONSTEXPR(RhsCol) {
            r0 += va * b0[k];
            r1 += va * b1[k];
            r2 += va * b2[k];
            r3 += va * b3[k];
          } else {
            r0 += va * bk[0];
            r1 += va * bk[1];
            r2 += va * bk[2];
            r3 += va * bk[3];
            bk += ldb;
          }
        }
        if (!alpha_one) {
          r0 *= alpha;
          r1 *= alpha;
          r2 *= alpha;
          r3 *= alpha;
        }
        if (beta_zero) {
          C[(j + 0) * ldc + i] = r0;
          C[(j + 1) * ldc + i] = r1;
          C[(j + 2) * ldc + i] = r2;
          C[(j + 3) * ldc + i] = r3;
        } else {
          C[(j + 0) * ldc + i] += r0;
          C[(j + 1) * ldc + i] += r1;
          C[(j + 2) * ldc + i] += r2;
          C[(j + 3) * ldc + i] += r3;
        }
      }
    }
  }

  // 2-column tail (j + 2 <= size): MRx2 off-diagonal + 2x2 diagonal.
  if (j + 2 <= size) {
    const Scalar* b_base = RhsCol ? (B + j * ldb) : (B + j);
    const Scalar* b0 = RhsCol ? (b_base + 0 * ldb) : b_base;
    const Scalar* b1 = RhsCol ? (b_base + 1 * ldb) : b_base;

    EIGEN_IF_CONSTEXPR(!IsLower) {
      for (Index i = 0; i + MR <= j; i += MR) {
        mcu_small_gemm_tile_mr2<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, C, ldc, alpha,
                                                                alpha_one, beta_zero);
      }
    }
    mcu_small_gemm_diag2<Scalar, LhsOrder, RhsOrder, IsLower>(j, K, A, lda, B, ldb, C, ldc, alpha, alpha_one, beta_zero,
                                                              is_sym);
    EIGEN_IF_CONSTEXPR(IsLower) {
      Index i = j + 2;
      for (; i + MR <= size; i += MR) {
        mcu_small_gemm_tile_mr2<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, C, ldc, alpha,
                                                                alpha_one, beta_zero);
      }
      for (; i < size; ++i) {
        Scalar r0 = Scalar(0), r1 = Scalar(0);
        const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
        const Scalar* bk = b_base;
        EIGEN_UNUSED_VARIABLE(bk);
        for (Index k = 0; k < K; ++k) {
          Scalar va;
          EIGEN_IF_CONSTEXPR(LhsCol) {
            va = *ak;
            ak += lda;
          } else {
            va = ak[k];
          }
          EIGEN_IF_CONSTEXPR(RhsCol) {
            r0 += va * b0[k];
            r1 += va * b1[k];
          } else {
            r0 += va * bk[0];
            r1 += va * bk[1];
            bk += ldb;
          }
        }
        if (!alpha_one) {
          r0 *= alpha;
          r1 *= alpha;
        }
        if (beta_zero) {
          C[(j + 0) * ldc + i] = r0;
          C[(j + 1) * ldc + i] = r1;
        } else {
          C[(j + 0) * ldc + i] += r0;
          C[(j + 1) * ldc + i] += r1;
        }
      }
    }
    j += 2;
  }

  // Remaining 1-column tail (j < size).
  if (j < size) {
    const Index i_start = IsLower ? j : Index(0);
    const Index i_end = IsLower ? size : (j + 1);
    const Scalar* bj = RhsCol ? (B + j * ldb) : (B + j);
    Index i = i_start;
    for (; i + MR <= i_end; i += MR) {
      EIGEN_IF_CONSTEXPR(MR == 4) {
        Scalar c0 = Scalar(0), c1 = Scalar(0), c2 = Scalar(0), c3 = Scalar(0);
        const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
        const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
        const Scalar* a2 = LhsCol ? nullptr : (A + (i + 2) * lda);
        const Scalar* a3 = LhsCol ? nullptr : (A + (i + 3) * lda);
        const Scalar* bk = bj;
        EIGEN_UNUSED_VARIABLE(a1);
        EIGEN_UNUSED_VARIABLE(a2);
        EIGEN_UNUSED_VARIABLE(a3);
        EIGEN_UNUSED_VARIABLE(bk);
        for (Index k = 0; k < K; ++k) {
          Scalar vb;
          EIGEN_IF_CONSTEXPR(RhsCol) {
            vb = bj[k];
          } else {
            vb = *bk;
            bk += ldb;
          }
          EIGEN_IF_CONSTEXPR(LhsCol) {
            c0 += ak[0] * vb;
            c1 += ak[1] * vb;
            c2 += ak[2] * vb;
            c3 += ak[3] * vb;
            ak += lda;
          } else {
            c0 += ak[k] * vb;
            c1 += a1[k] * vb;
            c2 += a2[k] * vb;
            c3 += a3[k] * vb;
          }
        }
        if (!alpha_one) {
          c0 *= alpha;
          c1 *= alpha;
          c2 *= alpha;
          c3 *= alpha;
        }
        Scalar* p = C + j * ldc + i;
        if (beta_zero) {
          p[0] = c0;
          p[1] = c1;
          p[2] = c2;
          p[3] = c3;
        } else {
          p[0] += c0;
          p[1] += c1;
          p[2] += c2;
          p[3] += c3;
        }
      } else {
        Scalar c0 = Scalar(0), c1 = Scalar(0);
        const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
        const Scalar* a1 = LhsCol ? nullptr : (A + (i + 1) * lda);
        const Scalar* bk = bj;
        EIGEN_UNUSED_VARIABLE(a1);
        EIGEN_UNUSED_VARIABLE(bk);
        for (Index k = 0; k < K; ++k) {
          Scalar vb;
          EIGEN_IF_CONSTEXPR(RhsCol) {
            vb = bj[k];
          } else {
            vb = *bk;
            bk += ldb;
          }
          EIGEN_IF_CONSTEXPR(LhsCol) {
            c0 += ak[0] * vb;
            c1 += ak[1] * vb;
            ak += lda;
          } else {
            c0 += ak[k] * vb;
            c1 += a1[k] * vb;
          }
        }
        if (!alpha_one) {
          c0 *= alpha;
          c1 *= alpha;
        }
        Scalar* p = C + j * ldc + i;
        if (beta_zero) {
          p[0] = c0;
          p[1] = c1;
        } else {
          p[0] += c0;
          p[1] += c1;
        }
      }
    }
    for (; i < i_end; ++i) {
      Scalar s = Scalar(0);
      const Scalar* ak = LhsCol ? (A + i) : (A + i * lda);
      const Scalar* bk = bj;
      EIGEN_UNUSED_VARIABLE(bk);
      for (Index k = 0; k < K; ++k) {
        Scalar va, vb;
        EIGEN_IF_CONSTEXPR(LhsCol) {
          va = *ak;
          ak += lda;
        } else {
          va = ak[k];
        }
        EIGEN_IF_CONSTEXPR(RhsCol) {
          vb = bj[k];
        } else {
          vb = *bk;
          bk += ldb;
        }
        s += va * vb;
      }
      if (!alpha_one) s *= alpha;
      if (beta_zero)
        C[j * ldc + i] = s;
      else
        C[j * ldc + i] += s;
    }
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder, int DstOrder, int UpLo>
EIGEN_ALWAYS_INLINE void mcu_small_gemm_triangular(Index size, Index K, const Scalar* A, Index lda, const Scalar* B,
                                                   Index ldb, Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha,
                                                   bool beta_zero = false) {
  EIGEN_IF_CONSTEXPR(DstOrder == ColMajor) {
    mcu_small_gemm_triangular_colmajor<Scalar, LhsOrder, RhsOrder, UpLo>(size, K, A, lda, B, ldb, C, ldc, alpha,
                                                                         beta_zero);
  } else {
    constexpr int FlippedRhs = (RhsOrder == RowMajor) ? ColMajor : RowMajor;
    constexpr int FlippedLhs = (LhsOrder == RowMajor) ? ColMajor : RowMajor;
    constexpr int FlippedUpLo = (UpLo == Lower) ? Upper : Lower;
    mcu_small_gemm_triangular_colmajor<Scalar, FlippedRhs, FlippedLhs, FlippedUpLo>(size, K, B, ldb, A, lda, C, ldc,
                                                                                    alpha, beta_zero);
  }
}

// ============================================================================
// Unblocked Pre-Inverted-Diagonal Small TRSM (N <= 32, zero blocking stack)
// Hoists all_inv_ok out of the RHS loop and blocks OnTheLeft across 4/2 RHS cols.
// ============================================================================

template <typename Scalar, bool IsUnit>
EIGEN_ALWAYS_INLINE Scalar mcu_trsm_scale_diag_fast(Scalar v, Index k, const Scalar* inv_d) {
  EIGEN_IF_CONSTEXPR(IsUnit) {
    EIGEN_UNUSED_VARIABLE(k);
    EIGEN_UNUSED_VARIABLE(inv_d);
    return v;
  } else {
    return v * inv_d[k];
  }
}

template <typename Scalar, bool IsUnit>
EIGEN_ALWAYS_INLINE Scalar mcu_trsm_scale_diag_guarded(Scalar v, Index k, const Scalar* L, Index ldl,
                                                       const Scalar* inv_d) {
  EIGEN_IF_CONSTEXPR(IsUnit) {
    EIGEN_UNUSED_VARIABLE(k);
    EIGEN_UNUSED_VARIABLE(L);
    EIGEN_UNUSED_VARIABLE(ldl);
    EIGEN_UNUSED_VARIABLE(inv_d);
    return v;
  } else {
    const Scalar inv = inv_d[k];
    return (numext::isfinite(inv) && inv != Scalar(0)) ? (v * inv) : (v / L[k * ldl + k]);
  }
}

template <typename Scalar, int Side, int Mode, int LhsOrder>
EIGEN_DONT_INLINE void mcu_small_trsm_colmajor(Index size, Index othersize, const Scalar* L, Index ldl,
                                               Scalar* EIGEN_RESTRICT X, Index ldx) {
  if (size <= 0 || othersize <= 0) return;

  constexpr bool IsLower = ((Mode & Lower) != 0);
  constexpr bool IsUnit = ((Mode & UnitDiag) != 0);
  constexpr bool LhsRow = (LhsOrder == RowMajor);
  constexpr int InvBufSize = IsUnit ? 1 : EIGEN_MCU_SMALL_TRSM_THRESHOLD;

  Scalar inv_d[InvBufSize];
  bool all_inv_ok = true;
  EIGEN_IF_CONSTEXPR(!IsUnit) {
    for (Index i = 0; i < size; ++i) {
      const Scalar inv = Scalar(1) / L[i * ldl + i];
      inv_d[i] = inv;
      if (!(numext::isfinite(inv) && inv != Scalar(0))) all_inv_ok = false;
    }
  } else {
    inv_d[0] = Scalar(1);
  }

  EIGEN_IF_CONSTEXPR(Side == OnTheLeft) {
    if (IsUnit || all_inv_ok) {
      Index j = 0;
      for (; j + 4 <= othersize; j += 4) {
        Scalar* x0 = X + (j + 0) * ldx;
        Scalar* x1 = X + (j + 1) * ldx;
        Scalar* x2 = X + (j + 2) * ldx;
        Scalar* x3 = X + (j + 3) * ldx;
        EIGEN_IF_CONSTEXPR(IsLower) {
          for (Index k = 0; k < size; ++k) {
            EIGEN_IF_CONSTEXPR(LhsRow) {
              Scalar s0 = x0[k], s1 = x1[k], s2 = x2[k], s3 = x3[k];
              const Scalar* row_k = L + k * ldl;
              for (Index m = 0; m < k; ++m) {
                const Scalar lkm = row_k[m];
                s0 -= lkm * x0[m];
                s1 -= lkm * x1[m];
                s2 -= lkm * x2[m];
                s3 -= lkm * x3[m];
              }
              x0[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s0, k, inv_d);
              x1[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s1, k, inv_d);
              x2[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s2, k, inv_d);
              x3[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s3, k, inv_d);
            } else {
              const Scalar v0 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x0[k], k, inv_d);
              const Scalar v1 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x1[k], k, inv_d);
              const Scalar v2 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x2[k], k, inv_d);
              const Scalar v3 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x3[k], k, inv_d);
              x0[k] = v0;
              x1[k] = v1;
              x2[k] = v2;
              x3[k] = v3;
              const Scalar* col_k = L + k * ldl;
              for (Index i = k + 1; i < size; ++i) {
                const Scalar lik = col_k[i];
                x0[i] -= lik * v0;
                x1[i] -= lik * v1;
                x2[i] -= lik * v2;
                x3[i] -= lik * v3;
              }
            }
          }
        } else {
          for (Index k = size - 1; k >= 0; --k) {
            EIGEN_IF_CONSTEXPR(LhsRow) {
              Scalar s0 = x0[k], s1 = x1[k], s2 = x2[k], s3 = x3[k];
              const Scalar* row_k = L + k * ldl;
              for (Index m = k + 1; m < size; ++m) {
                const Scalar lkm = row_k[m];
                s0 -= lkm * x0[m];
                s1 -= lkm * x1[m];
                s2 -= lkm * x2[m];
                s3 -= lkm * x3[m];
              }
              x0[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s0, k, inv_d);
              x1[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s1, k, inv_d);
              x2[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s2, k, inv_d);
              x3[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s3, k, inv_d);
            } else {
              const Scalar v0 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x0[k], k, inv_d);
              const Scalar v1 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x1[k], k, inv_d);
              const Scalar v2 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x2[k], k, inv_d);
              const Scalar v3 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x3[k], k, inv_d);
              x0[k] = v0;
              x1[k] = v1;
              x2[k] = v2;
              x3[k] = v3;
              const Scalar* col_k = L + k * ldl;
              for (Index i = 0; i < k; ++i) {
                const Scalar lik = col_k[i];
                x0[i] -= lik * v0;
                x1[i] -= lik * v1;
                x2[i] -= lik * v2;
                x3[i] -= lik * v3;
              }
            }
          }
        }
      }
      if (j + 2 <= othersize) {
        Scalar* x0 = X + (j + 0) * ldx;
        Scalar* x1 = X + (j + 1) * ldx;
        EIGEN_IF_CONSTEXPR(IsLower) {
          for (Index k = 0; k < size; ++k) {
            EIGEN_IF_CONSTEXPR(LhsRow) {
              Scalar s0 = x0[k], s1 = x1[k];
              const Scalar* row_k = L + k * ldl;
              for (Index m = 0; m < k; ++m) {
                const Scalar lkm = row_k[m];
                s0 -= lkm * x0[m];
                s1 -= lkm * x1[m];
              }
              x0[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s0, k, inv_d);
              x1[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s1, k, inv_d);
            } else {
              const Scalar v0 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x0[k], k, inv_d);
              const Scalar v1 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x1[k], k, inv_d);
              x0[k] = v0;
              x1[k] = v1;
              const Scalar* col_k = L + k * ldl;
              for (Index i = k + 1; i < size; ++i) {
                const Scalar lik = col_k[i];
                x0[i] -= lik * v0;
                x1[i] -= lik * v1;
              }
            }
          }
        } else {
          for (Index k = size - 1; k >= 0; --k) {
            EIGEN_IF_CONSTEXPR(LhsRow) {
              Scalar s0 = x0[k], s1 = x1[k];
              const Scalar* row_k = L + k * ldl;
              for (Index m = k + 1; m < size; ++m) {
                const Scalar lkm = row_k[m];
                s0 -= lkm * x0[m];
                s1 -= lkm * x1[m];
              }
              x0[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s0, k, inv_d);
              x1[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s1, k, inv_d);
            } else {
              const Scalar v0 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x0[k], k, inv_d);
              const Scalar v1 = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x1[k], k, inv_d);
              x0[k] = v0;
              x1[k] = v1;
              const Scalar* col_k = L + k * ldl;
              for (Index i = 0; i < k; ++i) {
                const Scalar lik = col_k[i];
                x0[i] -= lik * v0;
                x1[i] -= lik * v1;
              }
            }
          }
        }
        j += 2;
      }
      if (j < othersize) {
        Scalar* x = X + j * ldx;
        EIGEN_IF_CONSTEXPR(IsLower) {
          for (Index k = 0; k < size; ++k) {
            EIGEN_IF_CONSTEXPR(LhsRow) {
              Scalar s = x[k];
              const Scalar* row_k = L + k * ldl;
              for (Index m = 0; m < k; ++m) s -= row_k[m] * x[m];
              x[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s, k, inv_d);
            } else {
              const Scalar xk = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x[k], k, inv_d);
              x[k] = xk;
              const Scalar* col_k = L + k * ldl;
              for (Index i = k + 1; i < size; ++i) x[i] -= col_k[i] * xk;
            }
          }
        } else {
          for (Index k = size - 1; k >= 0; --k) {
            EIGEN_IF_CONSTEXPR(LhsRow) {
              Scalar s = x[k];
              const Scalar* row_k = L + k * ldl;
              for (Index m = k + 1; m < size; ++m) s -= row_k[m] * x[m];
              x[k] = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(s, k, inv_d);
            } else {
              const Scalar xk = mcu_trsm_scale_diag_fast<Scalar, IsUnit>(x[k], k, inv_d);
              x[k] = xk;
              const Scalar* col_k = L + k * ldl;
              for (Index i = 0; i < k; ++i) x[i] -= col_k[i] * xk;
            }
          }
        }
      }
      return;
    }

    // Guarded fallback when any diagonal reciprocal underflowed/overflowed.
    for (Index j = 0; j < othersize; ++j) {
      Scalar* x = X + j * ldx;
      EIGEN_IF_CONSTEXPR(IsLower) {
        for (Index k = 0; k < size; ++k) {
          EIGEN_IF_CONSTEXPR(LhsRow) {
            Scalar s = x[k];
            const Scalar* row_k = L + k * ldl;
            for (Index m = 0; m < k; ++m) s -= row_k[m] * x[m];
            x[k] = mcu_trsm_scale_diag_guarded<Scalar, IsUnit>(s, k, L, ldl, inv_d);
          } else {
            const Scalar xk = mcu_trsm_scale_diag_guarded<Scalar, IsUnit>(x[k], k, L, ldl, inv_d);
            x[k] = xk;
            const Scalar* col_k = L + k * ldl;
            for (Index i = k + 1; i < size; ++i) x[i] -= col_k[i] * xk;
          }
        }
      } else {
        for (Index k = size - 1; k >= 0; --k) {
          EIGEN_IF_CONSTEXPR(LhsRow) {
            Scalar s = x[k];
            const Scalar* row_k = L + k * ldl;
            for (Index m = k + 1; m < size; ++m) s -= row_k[m] * x[m];
            x[k] = mcu_trsm_scale_diag_guarded<Scalar, IsUnit>(s, k, L, ldl, inv_d);
          } else {
            const Scalar xk = mcu_trsm_scale_diag_guarded<Scalar, IsUnit>(x[k], k, L, ldl, inv_d);
            x[k] = xk;
            const Scalar* col_k = L + k * ldl;
            for (Index i = 0; i < k; ++i) x[i] -= col_k[i] * xk;
          }
        }
      }
    }
  } else {
    // Side == OnTheRight: solve X * T = B in-place (X is othersize x size ColMajor).
    EIGEN_IF_CONSTEXPR(IsLower) {
      for (Index k = size - 1; k >= 0; --k) {
        Scalar* xk = X + k * ldx;
        for (Index j = k + 1; j < size; ++j) {
          const Scalar ljk = LhsRow ? L[j * ldl + k] : L[k * ldl + j];
          const Scalar* xj = X + j * ldx;
          for (Index i = 0; i < othersize; ++i) xk[i] -= xj[i] * ljk;
        }
        EIGEN_IF_CONSTEXPR(!IsUnit) {
          const Scalar inv = inv_d[k];
          if (numext::isfinite(inv) && inv != Scalar(0)) {
            for (Index i = 0; i < othersize; ++i) xk[i] *= inv;
          } else {
            const Scalar diag = L[k * ldl + k];
            for (Index i = 0; i < othersize; ++i) xk[i] /= diag;
          }
        }
      }
    } else {
      for (Index k = 0; k < size; ++k) {
        Scalar* xk = X + k * ldx;
        for (Index j = 0; j < k; ++j) {
          const Scalar ujk = LhsRow ? L[j * ldl + k] : L[k * ldl + j];
          const Scalar* xj = X + j * ldx;
          for (Index i = 0; i < othersize; ++i) xk[i] -= xj[i] * ujk;
        }
        EIGEN_IF_CONSTEXPR(!IsUnit) {
          const Scalar inv = inv_d[k];
          if (numext::isfinite(inv) && inv != Scalar(0)) {
            for (Index i = 0; i < othersize; ++i) xk[i] *= inv;
          } else {
            const Scalar diag = L[k * ldl + k];
            for (Index i = 0; i < othersize; ++i) xk[i] /= diag;
          }
        }
      }
    }
  }
}

// ============================================================================
// Direct Unblocked Small TRSV (N <= 32, avoids 8-panel GEMV setup overhead)
// ============================================================================

template <typename Scalar, int Mode, int LhsOrder>
EIGEN_DONT_INLINE void mcu_small_trsv(Index size, const Scalar* L, Index ldl, Scalar* EIGEN_RESTRICT x) {
  if (size <= 0) return;

  constexpr bool IsLower = ((Mode & Lower) != 0);
  constexpr bool IsUnit = ((Mode & UnitDiag) != 0);
  constexpr bool LhsRow = (LhsOrder == RowMajor);

  EIGEN_IF_CONSTEXPR(IsLower) {
    for (Index k = 0; k < size; ++k) {
      EIGEN_IF_CONSTEXPR(LhsRow) {
        Scalar s = x[k];
        const Scalar* row_k = L + k * ldl;
        for (Index m = 0; m < k; ++m) s -= row_k[m] * x[m];
        x[k] = IsUnit ? s : (s / row_k[k]);
      } else {
        const Scalar* col_k = L + k * ldl;
        const Scalar xk = IsUnit ? x[k] : (x[k] / col_k[k]);
        x[k] = xk;
        for (Index i = k + 1; i < size; ++i) x[i] -= col_k[i] * xk;
      }
    }
  } else {
    for (Index k = size - 1; k >= 0; --k) {
      EIGEN_IF_CONSTEXPR(LhsRow) {
        Scalar s = x[k];
        const Scalar* row_k = L + k * ldl;
        for (Index m = k + 1; m < size; ++m) s -= row_k[m] * x[m];
        x[k] = IsUnit ? s : (s / row_k[k]);
      } else {
        const Scalar* col_k = L + k * ldl;
        const Scalar xk = IsUnit ? x[k] : (x[k] / col_k[k]);
        x[k] = xk;
        for (Index i = 0; i < k; ++i) x[i] -= col_k[i] * xk;
      }
    }
  }
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_MCU_SMALL_GEMM_H
