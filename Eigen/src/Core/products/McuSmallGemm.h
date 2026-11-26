// This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
//
// Copyright (C) 2026 SFWA contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Unpacked register-blocked small-matrix GEMM, SYRK, triangular product, and
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
// Shared MR x 4 micro-tile helper (MR = 4 for float, MR = 2 for double)
// ============================================================================

template <typename Scalar, int LhsOrder, int RhsOrder, int MR>
EIGEN_ALWAYS_INLINE void mcu_small_gemm_tile_mr4(Index i, Index j, Index K, const Scalar* A, Index lda, Index ldb,
                                                 const Scalar* b_base, const Scalar* b0, const Scalar* b1,
                                                 const Scalar* b2, const Scalar* b3, Scalar* EIGEN_RESTRICT C,
                                                 Index ldc, Scalar alpha, bool alpha_one) {
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
    p0[0] += c00;
    p0[1] += c10;
    p0[2] += c20;
    p0[3] += c30;
    Scalar* p1 = C + (j + 1) * ldc + i;
    p1[0] += c01;
    p1[1] += c11;
    p1[2] += c21;
    p1[3] += c31;
    Scalar* p2 = C + (j + 2) * ldc + i;
    p2[0] += c02;
    p2[1] += c12;
    p2[2] += c22;
    p2[3] += c32;
    Scalar* p3 = C + (j + 3) * ldc + i;
    p3[0] += c03;
    p3[1] += c13;
    p3[2] += c23;
    p3[3] += c33;
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
    p0[0] += c00;
    p0[1] += c10;
    Scalar* p1 = C + (j + 1) * ldc + i;
    p1[0] += c01;
    p1[1] += c11;
    Scalar* p2 = C + (j + 2) * ldc + i;
    p2[0] += c02;
    p2[1] += c12;
    Scalar* p3 = C + (j + 3) * ldc + i;
    p3[0] += c03;
    p3[1] += c13;
  }
}

// ============================================================================
// Unpacked Register-Blocked Small GEMM: C += alpha * A * B (Dst is ColMajor)
// Note: A and B may alias each other (e.g. A * A), so only C is EIGEN_RESTRICT.
// ============================================================================

template <typename Scalar, int LhsOrder, int RhsOrder>
EIGEN_DONT_INLINE void mcu_small_gemm_colmajor(Index M, Index N, Index K, const Scalar* A, Index lda, const Scalar* B,
                                               Index ldb, Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha) {
  if (M <= 0 || N <= 0 || K <= 0) return;

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
                                                              alpha, alpha_one);
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
      C[(j + 0) * ldc + i] += alpha * c0;
      C[(j + 1) * ldc + i] += alpha * c1;
      C[(j + 2) * ldc + i] += alpha * c2;
      C[(j + 3) * ldc + i] += alpha * c3;
    }
  }

  // Tail columns (j < N).
  for (; j < N; ++j) {
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
        Scalar* p = C + j * ldc + i;
        p[0] += alpha * c0;
        p[1] += alpha * c1;
        p[2] += alpha * c2;
        p[3] += alpha * c3;
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
        Scalar* p = C + j * ldc + i;
        p[0] += alpha * c0;
        p[1] += alpha * c1;
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
      C[j * ldc + i] += alpha * c0;
    }
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder, int DstOrder>
EIGEN_ALWAYS_INLINE void mcu_small_gemm(Index M, Index N, Index K, const Scalar* A, Index lda, const Scalar* B,
                                        Index ldb, Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha) {
  EIGEN_IF_CONSTEXPR(DstOrder == ColMajor) {
    mcu_small_gemm_colmajor<Scalar, LhsOrder, RhsOrder>(M, N, K, A, lda, B, ldb, C, ldc, alpha);
  } else {
    constexpr int FlippedRhs = (RhsOrder == RowMajor) ? ColMajor : RowMajor;
    constexpr int FlippedLhs = (LhsOrder == RowMajor) ? ColMajor : RowMajor;
    mcu_small_gemm_colmajor<Scalar, FlippedRhs, FlippedLhs>(N, M, K, B, ldb, A, lda, C, ldc, alpha);
  }
}

// ============================================================================
// Unpacked Register-Blocked Small Triangular Product & SYRK:
// C_uplo += alpha * A * B (Dst is ColMajor, UpLo is Lower or Upper).
// Computes only the requested triangle (half FLOPs on diagonal tiles and zero
// work on the untouched triangle).
// ============================================================================

template <typename Scalar, int LhsOrder, int RhsOrder, int UpLo>
EIGEN_DONT_INLINE void mcu_small_gemm_triangular_colmajor(Index size, Index K, const Scalar* A, Index lda,
                                                          const Scalar* B, Index ldb, Scalar* EIGEN_RESTRICT C,
                                                          Index ldc, Scalar alpha) {
  if (size <= 0 || K <= 0) return;

  constexpr bool LhsCol = (LhsOrder == ColMajor);
  constexpr bool RhsCol = (RhsOrder == ColMajor);
  constexpr bool IsLower = (UpLo == Lower);
  constexpr int MR = is_same<Scalar, float>::value ? 4 : 2;
  const bool alpha_one = (alpha == Scalar(1));

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
                                                                alpha, alpha_one);
      }
    }

    // Diagonal 4x4 triangle at (j, j): only the 10 entries in UpLo.
    {
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
          // Upper triangle: (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3), (3,3).
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
        p0[0] += d00;
        p0[1] += d10;
        p0[2] += d20;
        p0[3] += d30;
        Scalar* p1 = C + (j + 1) * ldc + j;
        p1[1] += d11;
        p1[2] += d21;
        p1[3] += d31;
        Scalar* p2 = C + (j + 2) * ldc + j;
        p2[2] += d22;
        p2[3] += d32;
        C[(j + 3) * ldc + j + 3] += d33;
      } else {
        C[(j + 0) * ldc + j + 0] += d00;
        Scalar* p1 = C + (j + 1) * ldc + j;
        p1[0] += d10;
        p1[1] += d11;
        Scalar* p2 = C + (j + 2) * ldc + j;
        p2[0] += d20;
        p2[1] += d21;
        p2[2] += d22;
        Scalar* p3 = C + (j + 3) * ldc + j;
        p3[0] += d30;
        p3[1] += d31;
        p3[2] += d32;
        p3[3] += d33;
      }
    }

    EIGEN_IF_CONSTEXPR(IsLower) {
      // Strictly below diagonal: rows [j + 4, size).
      Index i = j + 4;
      for (; i + MR <= size; i += MR) {
        mcu_small_gemm_tile_mr4<Scalar, LhsOrder, RhsOrder, MR>(i, j, K, A, lda, ldb, b_base, b0, b1, b2, b3, C, ldc,
                                                                alpha, alpha_one);
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
        C[(j + 0) * ldc + i] += alpha * r0;
        C[(j + 1) * ldc + i] += alpha * r1;
        C[(j + 2) * ldc + i] += alpha * r2;
        C[(j + 3) * ldc + i] += alpha * r3;
      }
    }
  }

  // Remaining tail columns (j < size).
  for (; j < size; ++j) {
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
        Scalar* p = C + j * ldc + i;
        p[0] += alpha * c0;
        p[1] += alpha * c1;
        p[2] += alpha * c2;
        p[3] += alpha * c3;
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
        Scalar* p = C + j * ldc + i;
        p[0] += alpha * c0;
        p[1] += alpha * c1;
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
      C[j * ldc + i] += alpha * s;
    }
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder, int DstOrder, int UpLo>
EIGEN_ALWAYS_INLINE void mcu_small_gemm_triangular(Index size, Index K, const Scalar* A, Index lda, const Scalar* B,
                                                   Index ldb, Scalar* EIGEN_RESTRICT C, Index ldc, Scalar alpha) {
  EIGEN_IF_CONSTEXPR(DstOrder == ColMajor) {
    mcu_small_gemm_triangular_colmajor<Scalar, LhsOrder, RhsOrder, UpLo>(size, K, A, lda, B, ldb, C, ldc, alpha);
  } else {
    constexpr int FlippedRhs = (RhsOrder == RowMajor) ? ColMajor : RowMajor;
    constexpr int FlippedLhs = (LhsOrder == RowMajor) ? ColMajor : RowMajor;
    constexpr int FlippedUpLo = (UpLo == Lower) ? Upper : Lower;
    mcu_small_gemm_triangular_colmajor<Scalar, FlippedRhs, FlippedLhs, FlippedUpLo>(size, K, B, ldb, A, lda, C, ldc,
                                                                                    alpha);
  }
}

// ============================================================================
// Unblocked Pre-Inverted-Diagonal Small TRSM (N <= 32, zero blocking stack)
// Supports OnTheLeft and OnTheRight, Lower and Upper, Unit and Non-Unit,
// ColMajor and RowMajor LHS, ColMajor RHS (RowMajor RHS transposed at caller).
// ============================================================================

template <typename Scalar, bool IsUnit>
EIGEN_ALWAYS_INLINE Scalar mcu_trsm_scale_diag(Scalar v, Index k, const Scalar* L, Index ldl, const Scalar* inv_d,
                                               const bool* use_inv) {
  EIGEN_IF_CONSTEXPR(IsUnit) {
    EIGEN_UNUSED_VARIABLE(k);
    EIGEN_UNUSED_VARIABLE(L);
    EIGEN_UNUSED_VARIABLE(ldl);
    EIGEN_UNUSED_VARIABLE(inv_d);
    EIGEN_UNUSED_VARIABLE(use_inv);
    return v;
  } else {
    return use_inv[k] ? (v * inv_d[k]) : (v / L[k * ldl + k]);
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
  bool use_inv[InvBufSize];
  EIGEN_IF_CONSTEXPR(!IsUnit) {
    for (Index i = 0; i < size; ++i) {
      const Scalar diag = L[i * ldl + i];
      const Scalar inv = Scalar(1) / diag;
      inv_d[i] = inv;
      use_inv[i] = numext::isfinite(inv) && (inv != Scalar(0));
    }
  } else {
    inv_d[0] = Scalar(1);
    use_inv[0] = true;
  }

  EIGEN_IF_CONSTEXPR(Side == OnTheLeft) {
    for (Index j = 0; j < othersize; ++j) {
      Scalar* x = X + j * ldx;
      EIGEN_IF_CONSTEXPR(IsLower) {
        for (Index k = 0; k < size; ++k) {
          EIGEN_IF_CONSTEXPR(LhsRow) {
            Scalar s = x[k];
            const Scalar* row_k = L + k * ldl;
            for (Index m = 0; m < k; ++m) s -= row_k[m] * x[m];
            x[k] = mcu_trsm_scale_diag<Scalar, IsUnit>(s, k, L, ldl, inv_d, use_inv);
          } else {
            const Scalar xk = mcu_trsm_scale_diag<Scalar, IsUnit>(x[k], k, L, ldl, inv_d, use_inv);
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
            x[k] = mcu_trsm_scale_diag<Scalar, IsUnit>(s, k, L, ldl, inv_d, use_inv);
          } else {
            const Scalar xk = mcu_trsm_scale_diag<Scalar, IsUnit>(x[k], k, L, ldl, inv_d, use_inv);
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
          if (use_inv[k]) {
            const Scalar inv = inv_d[k];
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
          if (use_inv[k]) {
            const Scalar inv = inv_d[k];
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
