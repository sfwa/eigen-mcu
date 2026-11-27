// This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
//
// Copyright (C) 2026 SFWA contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Deterministic bare-metal Cortex-M4F/M7 kernel benchmark for eigen-mcu
// performance regression gating under QEMU (-icount shift=0).

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

extern "C" {
void print(const char* s);
void print_u(std::uint64_t v);
void timer_start(void);
std::uint64_t instructions(void);
void stack_paint(void);
std::uint32_t stack_used(void);
}

static int g_bad = 0;

template <typename T>
EIGEN_DONT_INLINE static void sink(const T& x) {
  asm volatile("" : : "g"(&x) : "memory");
}

template <typename T>
EIGEN_DONT_INLINE static void touch(T& x) {
  asm volatile("" : "+m"(x) : : "memory");
}

template <typename T>
static T pseudo(int idx) {
  std::uint32_t x = static_cast<std::uint32_t>(idx) * 2654435761u + 1013904223u;
  x ^= x >> 13;
  x *= 1274126177u;
  x ^= x >> 16;
  const int signed_v = static_cast<int>(x & 0xFFFFu) - 32768;
  return T(signed_v) * T(1.5f / 32768.0f);
}

template <typename T, int R, int C, int Order = Eigen::ColMajor>
static Eigen::Matrix<T, R, C, Order> make_mat(int seed) {
  Eigen::Matrix<T, R, C, Order> m;
  for (std::ptrdiff_t j = 0; j < C; ++j) {
    for (std::ptrdiff_t i = 0; i < R; ++i) {
      m(i, j) = pseudo<T>(seed + static_cast<int>(i * 37 + j * 131));
    }
  }
  return m;
}

template <typename T, int N>
static Eigen::Matrix<T, N, N> make_spd(int seed) {
  auto A = make_mat<T, N, N>(seed);
  Eigen::Matrix<T, N, N> S(A.transpose() * A);
  S += T(N) * Eigen::Matrix<T, N, N>::Identity();
  return S;
}

template <typename T, int N>
static Eigen::Matrix<T, N, N> make_tri_lower(int seed) {
  Eigen::Matrix<T, N, N> L = make_mat<T, N, N>(seed).template triangularView<Eigen::Lower>();
  for (std::ptrdiff_t i = 0; i < N; ++i) {
    L(i, i) = T(2.5f) + T(0.1f) * T(i);
  }
  return L;
}

template <typename F>
EIGEN_DONT_INLINE static void measure(const char* name, std::uint32_t reps, F&& f) {
  f();  // Warm up any static state before painting stack.
  stack_paint();
  const std::uint64_t t0 = instructions();
  for (std::uint32_t i = 0; i < reps; ++i) {
    f();
  }
  const std::uint64_t t1 = instructions();
  const std::uint32_t stk = stack_used();
  print(name);
  print(": ");
  print_u((t1 - t0) / reps);
  print(" instructions, stack ");
  print_u(stk);
  print(" bytes\n");
}

template <typename Mat>
static void check_finite(const Mat& m) {
  if (!m.allFinite()) {
    ++g_bad;
  }
}

// Non-inlined kernel wrappers ensure every benchmark exercises the real call
// frame and stack footprint of the corresponding Eigen operation.

template <typename T, int M, int N, int K, bool BetaZero>
EIGEN_DONT_INLINE static void run_gemm(const Eigen::Matrix<T, M, K>& A, const Eigen::Matrix<T, K, N>& B,
                                       Eigen::Matrix<T, M, N>& C) {
  if constexpr (BetaZero) {
    C.noalias() = A * B;
  } else {
    C.noalias() += A * B;
  }
}

template <typename T, int N, int K, int UpLo, bool BetaZero>
EIGEN_DONT_INLINE static void run_syrk_at_a(const Eigen::Matrix<T, K, N>& J, Eigen::Matrix<T, N, N>& S) {
  if constexpr (BetaZero) {
    S.template triangularView<UpLo>() = J.transpose() * J;
  } else {
    S.template triangularView<UpLo>() += J.transpose() * J;
  }
}

template <typename T, int N, int K, int UpLo, bool BetaZero>
EIGEN_DONT_INLINE static void run_syrk_a_at(const Eigen::Matrix<T, N, K>& A, Eigen::Matrix<T, N, N>& S) {
  if constexpr (BetaZero) {
    S.template triangularView<UpLo>() = A * A.transpose();
  } else {
    S.template triangularView<UpLo>() += A * A.transpose();
  }
}

template <typename T, int N, int K, int UpLo>
EIGEN_DONT_INLINE static void run_triprod_at_b(const Eigen::Matrix<T, K, N>& A, const Eigen::Matrix<T, K, N>& B,
                                               Eigen::Matrix<T, N, N>& C) {
  C.template triangularView<UpLo>() += A.transpose() * B;
}

template <typename T, int M, int K, int Order>
EIGEN_DONT_INLINE static void run_gemv(const Eigen::Matrix<T, M, K, Order>& A, const Eigen::Matrix<T, K, 1>& x,
                                       Eigen::Matrix<T, M, 1>& y) {
  y.noalias() = A * x;
}

template <typename T, int M, int K>
EIGEN_DONT_INLINE static void run_gemv_trans(const Eigen::Matrix<T, M, K>& A, const Eigen::Matrix<T, M, 1>& x,
                                             Eigen::Matrix<T, K, 1>& y) {
  y.noalias() = A.transpose() * x;
}

template <typename T, int N, int NRHS, int UpLo, int Side>
EIGEN_DONT_INLINE static void run_trsm(const Eigen::Matrix<T, N, N>& L, Eigen::Matrix<T, Side == Eigen::OnTheLeft ? N : NRHS,
                                                                                      Side == Eigen::OnTheLeft ? NRHS : N>& B) {
  L.template triangularView<UpLo>().template solveInPlace<Side>(B);
}

template <typename T, int N>
EIGEN_DONT_INLINE static void run_trsv(const Eigen::Matrix<T, N, N>& L, Eigen::Matrix<T, N, 1>& b) {
  L.template triangularView<Eigen::Lower>().solveInPlace(b);
}

template <typename T, int N, int NRHS>
EIGEN_DONT_INLINE static void run_llt_compute_solve(const Eigen::Matrix<T, N, N>& H, const Eigen::Matrix<T, N, NRHS>& B,
                                                    Eigen::Matrix<T, N, NRHS>& X) {
  Eigen::LLT<Eigen::Matrix<T, N, N>> llt(H);
  X = llt.solve(B);
}

template <typename T, int N>
EIGEN_DONT_INLINE static void run_llt_rank_update(Eigen::LLT<Eigen::Matrix<T, N, N>>& llt,
                                                  const Eigen::Matrix<T, N, 1>& v) {
  llt.rankUpdate(v, T(0.25f));
}

template <typename T, int N>
EIGEN_DONT_INLINE static void run_ldlt_compute_solve(const Eigen::Matrix<T, N, N>& H, const Eigen::Matrix<T, N, 1>& b,
                                                     Eigen::Matrix<T, N, 1>& x) {
  Eigen::LDLT<Eigen::Matrix<T, N, N>> ldlt(H);
  x = ldlt.solve(b);
}

template <typename T, int R, int C>
EIGEN_DONT_INLINE static void run_qr(const Eigen::Matrix<T, R, C>& A, Eigen::Matrix<T, C, C>& Rmat) {
  Eigen::HouseholderQR<Eigen::Matrix<T, R, C>> qr(A);
  Rmat = qr.matrixQR().template topRows<C>().template triangularView<Eigen::Upper>();
}

template <typename T, int N>
EIGEN_DONT_INLINE static T run_dot(const Eigen::Matrix<T, N, 1>& a, const Eigen::Matrix<T, N, 1>& b) {
  return a.dot(b);
}

template <typename T, int R, int C>
EIGEN_DONT_INLINE static bool run_allfinite(const Eigen::Matrix<T, R, C>& A) {
  return A.allFinite();
}

template <typename T, int N>
EIGEN_DONT_INLINE static void run_cwise_clamp(const Eigen::Matrix<T, N, 1>& x, const Eigen::Matrix<T, N, 1>& lo,
                                              const Eigen::Matrix<T, N, 1>& hi, Eigen::Matrix<T, N, 1>& y) {
  y = x.cwiseMax(lo).cwiseMin(hi);
}

EIGEN_DONT_INLINE static void run_quat_mul_norm(const Eigen::Quaternion<float>& q1, const Eigen::Quaternion<float>& q2,
                                                Eigen::Quaternion<float>& out) {
  out = (q1 * q2).normalized();
}

static void bench_float_suite() {
  // Fixture matrices live in static storage so stack_used() reflects each
  // kernel's actual call-stack footprint rather than the suite's locals.

  // 1. Small GEMM (beta=0 & beta=1, square, rectangular, odd & MRx2 tails).
  {
    static Eigen::Matrix<float, 12, 12> A, B, C;
    A = make_mat<float, 12, 12>(1);
    B = make_mat<float, 12, 12>(2);
    measure("gemm_f_12x12_b0", 20, [&] {
      touch(A);
      run_gemm<float, 12, 12, 12, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<float, 16, 16> A, B, C;
    A = make_mat<float, 16, 16>(3);
    B = make_mat<float, 16, 16>(4);
    measure("gemm_f_16x16_b0", 10, [&] {
      touch(A);
      run_gemm<float, 16, 16, 16, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<float, 17, 17> A, B, C;
    A = make_mat<float, 17, 17>(5);
    B = make_mat<float, 17, 17>(6);
    C.setZero();
    measure("gemm_f_17x17_b1", 10, [&] {
      C.setZero();
      touch(A);
      run_gemm<float, 17, 17, 17, false>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<float, 20, 20> A, B, C;
    A = make_mat<float, 20, 20>(7);
    B = make_mat<float, 20, 20>(8);
    measure("gemm_f_20x20_b0", 8, [&] {
      touch(A);
      run_gemm<float, 20, 20, 20, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<float, 32, 32> A, B, C;
    A = make_mat<float, 32, 32>(9);
    B = make_mat<float, 32, 32>(10);
    measure("gemm_f_32x32_b0", 4, [&] {
      touch(A);
      run_gemm<float, 32, 32, 32, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<float, 16, 16> A;
    static Eigen::Matrix<float, 16, 4> B, C;
    A = make_mat<float, 16, 16>(11);
    B = make_mat<float, 16, 4>(12);
    measure("gemm_f_16x16x4_b0", 20, [&] {
      touch(A);
      run_gemm<float, 16, 4, 16, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<float, 9, 19> A;
    static Eigen::Matrix<float, 19, 6> B;
    static Eigen::Matrix<float, 9, 6> C;
    A = make_mat<float, 9, 19>(13);
    B = make_mat<float, 19, 6>(14);
    measure("gemm_f_9x19x6_b0", 20, [&] {
      touch(A);
      run_gemm<float, 9, 6, 19, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }

  // 2. Triangular products & symmetric SYRK.
  {
    static Eigen::Matrix<float, 22, 17> J;
    static Eigen::Matrix<float, 17, 17> S;
    J = make_mat<float, 22, 17>(15);
    S.setZero();
    measure("syrk_f_u_17x22_b0", 10, [&] {
      touch(J);
      run_syrk_at_a<float, 17, 22, Eigen::Upper, true>(J, S);
      sink(S);
    });
    check_finite(S);
  }
  {
    static Eigen::Matrix<float, 15, 30> A;
    static Eigen::Matrix<float, 15, 15> S;
    A = make_mat<float, 15, 30>(16);
    S.setZero();
    measure("syrk_f_l_15x30_b0", 10, [&] {
      touch(A);
      run_syrk_a_at<float, 15, 30, Eigen::Lower, true>(A, S);
      sink(S);
    });
    check_finite(S);
  }
  {
    static Eigen::Matrix<float, 5, 17> A, B;
    static Eigen::Matrix<float, 17, 17> C;
    A = make_mat<float, 5, 17>(17);
    B = make_mat<float, 5, 17>(18);
    C.setZero();
    measure("triprod_f_u_17x5_b1", 20, [&] {
      C.setZero();
      touch(A);
      run_triprod_at_b<float, 17, 5, Eigen::Upper>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }

  // 3. Small unpacked GEMV (ColMajor, RowMajor, Transpose).
  {
    static Eigen::Matrix<float, 17, 22, Eigen::ColMajor> Ac;
    static Eigen::Matrix<float, 17, 22, Eigen::RowMajor> Ar;
    static Eigen::Matrix<float, 22, 1> x22, y22;
    static Eigen::Matrix<float, 17, 1> x17, y17;
    Ac = make_mat<float, 17, 22, Eigen::ColMajor>(19);
    Ar = make_mat<float, 17, 22, Eigen::RowMajor>(20);
    x22 = make_mat<float, 22, 1>(21);
    x17 = make_mat<float, 17, 1>(22);
    measure("gemv_f_col_17x22", 40, [&] {
      touch(x22);
      run_gemv<float, 17, 22, Eigen::ColMajor>(Ac, x22, y17);
      sink(y17);
    });
    check_finite(y17);
    measure("gemv_f_row_17x22", 40, [&] {
      touch(x22);
      run_gemv<float, 17, 22, Eigen::RowMajor>(Ar, x22, y17);
      sink(y17);
    });
    check_finite(y17);
    measure("gemv_f_trans_22x17", 40, [&] {
      touch(x17);
      run_gemv_trans<float, 17, 22>(Ac, x17, y22);
      sink(y22);
    });
    check_finite(y22);
  }

  // 4. Unblocked pre-inverted-diagonal TRSM and direct TRSV.
  {
    static Eigen::Matrix<float, 15, 15> L15, B15_0, B15;
    L15 = make_tri_lower<float, 15>(23);
    B15_0 = make_mat<float, 15, 15>(24);
    measure("trsm_f_ll_15x15", 10, [&] {
      B15 = B15_0;
      touch(L15);
      run_trsm<float, 15, 15, Eigen::Lower, Eigen::OnTheLeft>(L15, B15);
      sink(B15);
    });
    check_finite(B15);

    static Eigen::Matrix<float, 5, 5> U5;
    static Eigen::Matrix<float, 5, 17> B5_0, B5;
    U5 = make_tri_lower<float, 5>(25).transpose();
    B5_0 = make_mat<float, 5, 17>(26);
    measure("trsm_f_lu_5x17", 25, [&] {
      B5 = B5_0;
      touch(U5);
      run_trsm<float, 5, 17, Eigen::Upper, Eigen::OnTheLeft>(U5, B5);
      sink(B5);
    });
    check_finite(B5);

    static Eigen::Matrix<float, 8, 15> B15x8_0, B15x8;
    B15x8_0 = make_mat<float, 8, 15>(27);
    measure("trsm_f_rl_8x15", 15, [&] {
      B15x8 = B15x8_0;
      touch(L15);
      run_trsm<float, 15, 8, Eigen::Lower, Eigen::OnTheRight>(L15, B15x8);
      sink(B15x8);
    });
    check_finite(B15x8);

    static Eigen::Matrix<float, 15, 1> b15_0, b15;
    b15_0 = make_mat<float, 15, 1>(28);
    measure("trsv_f_l_15", 60, [&] {
      b15 = b15_0;
      touch(L15);
      run_trsv<float, 15>(L15, b15);
      sink(b15);
    });
    check_finite(b15);
  }

  // 5. Direct small LLT, fused multi-RHS LLT solve, rankUpdate, and LDLT.
  {
    static Eigen::Matrix<float, 5, 5> H5;
    static Eigen::Matrix<float, 5, 17> B5, X5;
    H5 = make_spd<float, 5>(29);
    B5 = make_mat<float, 5, 17>(30);
    measure("llt_f_5x5_solve17", 20, [&] {
      touch(H5);
      run_llt_compute_solve<float, 5, 17>(H5, B5, X5);
      sink(X5);
    });
    check_finite(X5);

    static Eigen::Matrix<float, 15, 15> H15;
    static Eigen::Matrix<float, 15, 1> b15, x15, v15;
    static Eigen::LLT<Eigen::Matrix<float, 15, 15>> llt15;
    H15 = make_spd<float, 15>(31);
    b15 = make_mat<float, 15, 1>(32);
    measure("llt_f_15x15_compute", 15, [&] {
      touch(H15);
      run_llt_compute_solve<float, 15, 1>(H15, b15, x15);
      sink(x15);
    });
    check_finite(x15);

    v15 = make_mat<float, 15, 1>(33);
    measure("llt_f_rank_update_15", 20, [&] {
      llt15.compute(H15);
      touch(v15);
      run_llt_rank_update<float, 15>(llt15, v15);
      sink(llt15);
    });

    static Eigen::Matrix<float, 12, 12> H12;
    static Eigen::Matrix<float, 12, 1> b12, x12;
    H12 = make_spd<float, 12>(34);
    b12 = make_mat<float, 12, 1>(35);
    measure("ldlt_f_12x12_compute", 15, [&] {
      touch(H12);
      run_ldlt_compute_solve<float, 12>(H12, b12, x12);
      sink(x12);
    });
    check_finite(x12);
  }

  // 6. Unblocked HouseholderQR with fused reflectors (SR-UKF shapes).
  {
    static Eigen::Matrix<float, 28, 9> Q28;
    static Eigen::Matrix<float, 9, 9> R9;
    Q28 = make_mat<float, 28, 9>(36);
    R9.setZero();
    measure("qr_f_28x9", 10, [&] {
      touch(Q28);
      run_qr<float, 28, 9>(Q28, R9);
      sink(R9);
    });
    check_finite(R9);

    static Eigen::Matrix<float, 31, 15> Q31;
    static Eigen::Matrix<float, 15, 15> R15;
    Q31 = make_mat<float, 31, 15>(37);
    R15.setZero();
    measure("qr_f_31x15", 5, [&] {
      touch(Q31);
      run_qr<float, 31, 15>(Q31, R15);
      sink(R15);
    });
    check_finite(R15);
  }

  // 7. Flat FMA dot, bitwise allFinite, cwise clamp (vmaxnm/vminnm), quaternion.
  {
    static Eigen::Matrix<float, 17, 1> a17, b17, lo, hi, clamped;
    a17 = make_mat<float, 17, 1>(38);
    b17 = make_mat<float, 17, 1>(39);
    volatile float acc17 = 0.0f;
    measure("dot_f_17", 200, [&] {
      touch(a17);
      acc17 = run_dot<float, 17>(a17, b17);
    });
    if (!std::isfinite(static_cast<float>(acc17))) ++g_bad;

    static Eigen::Matrix<float, 21, 1> a21, b21;
    a21 = make_mat<float, 21, 1>(40);
    b21 = make_mat<float, 21, 1>(41);
    volatile float acc21 = 0.0f;
    measure("dot_f_21", 200, [&] {
      touch(a21);
      acc21 = run_dot<float, 21>(a21, b21);
    });
    if (!std::isfinite(static_cast<float>(acc21))) ++g_bad;

    static Eigen::Matrix<float, 15, 31> F15;
    F15 = make_mat<float, 15, 31>(42);
    volatile bool ok = false;
    measure("allfinite_f_15x31", 40, [&] {
      touch(F15);
      ok = run_allfinite<float, 15, 31>(F15);
    });
    if (!ok) ++g_bad;

    lo.setConstant(-0.5f);
    hi.setConstant(0.5f);
    measure("cwise_clamp_f_17", 150, [&] {
      touch(a17);
      run_cwise_clamp<float, 17>(a17, lo, hi, clamped);
      sink(clamped);
    });
    check_finite(clamped);

    static Eigen::Quaternion<float> q1, q2, qout;
    q1 = Eigen::Quaternion<float>(0.9f, 0.2f, -0.3f, 0.25f).normalized();
    q2 = Eigen::Quaternion<float>(0.85f, -0.15f, 0.35f, -0.35f).normalized();
    measure("quat_f_mul_norm", 150, [&] {
      touch(q1);
      run_quat_mul_norm(q1, q2, qout);
      sink(qout);
    });
    check_finite(qout.coeffs());
  }
}

#ifndef BENCH_FLOAT_ONLY
static void bench_double_suite() {
  {
    static Eigen::Matrix<double, 16, 16> A, B, C;
    A = make_mat<double, 16, 16>(50);
    B = make_mat<double, 16, 16>(51);
    measure("gemm_d_16x16_b0", 8, [&] {
      touch(A);
      run_gemm<double, 16, 16, 16, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<double, 9, 13> A;
    static Eigen::Matrix<double, 13, 6> B;
    static Eigen::Matrix<double, 9, 6> C;
    A = make_mat<double, 9, 13>(52);
    B = make_mat<double, 13, 6>(53);
    measure("gemm_d_9x13x6_b0", 15, [&] {
      touch(A);
      run_gemm<double, 9, 6, 13, true>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<double, 15, 30> A;
    static Eigen::Matrix<double, 15, 15> S;
    A = make_mat<double, 15, 30>(54);
    S.setZero();
    measure("syrk_d_l_15x30_b0", 6, [&] {
      touch(A);
      run_syrk_a_at<double, 15, 30, Eigen::Lower, true>(A, S);
      sink(S);
    });
    check_finite(S);
  }
  {
    static Eigen::Matrix<double, 6, 12> A, B;
    static Eigen::Matrix<double, 12, 12> C;
    A = make_mat<double, 6, 12>(55);
    B = make_mat<double, 6, 12>(56);
    C.setZero();
    measure("triprod_d_u_12x6_b1", 15, [&] {
      C.setZero();
      touch(A);
      run_triprod_at_b<double, 12, 6, Eigen::Upper>(A, B, C);
      sink(C);
    });
    check_finite(C);
  }
  {
    static Eigen::Matrix<double, 16, 16> A;
    static Eigen::Matrix<double, 16, 1> x, y;
    A = make_mat<double, 16, 16>(57);
    x = make_mat<double, 16, 1>(58);
    measure("gemv_d_col_16x16", 30, [&] {
      touch(x);
      run_gemv<double, 16, 16, Eigen::ColMajor>(A, x, y);
      sink(y);
    });
    check_finite(y);
  }
  {
    static Eigen::Matrix<double, 12, 12> H;
    static Eigen::Matrix<double, 12, 8> B, X;
    static Eigen::Matrix<double, 12, 1> b, x;
    H = make_spd<double, 12>(59);
    B = make_mat<double, 12, 8>(60);
    measure("llt_d_12x12_solve8", 10, [&] {
      touch(H);
      run_llt_compute_solve<double, 12, 8>(H, B, X);
      sink(X);
    });
    check_finite(X);

    b = make_mat<double, 12, 1>(61);
    measure("ldlt_d_12x12_compute", 10, [&] {
      touch(H);
      run_ldlt_compute_solve<double, 12>(H, b, x);
      sink(x);
    });
    check_finite(x);
  }
  {
    static Eigen::Matrix<double, 25, 12> Q;
    static Eigen::Matrix<double, 12, 12> R;
    Q = make_mat<double, 25, 12>(62);
    R.setZero();
    measure("qr_d_25x12", 5, [&] {
      touch(Q);
      run_qr<double, 25, 12>(Q, R);
      sink(R);
    });
    check_finite(R);
  }
  {
    static Eigen::Matrix<double, 15, 31> F;
    F = make_mat<double, 15, 31>(63);
    volatile bool ok = false;
    measure("allfinite_d_15x31", 30, [&] {
      touch(F);
      ok = run_allfinite<double, 15, 31>(F);
    });
    if (!ok) ++g_bad;
  }
}
#endif

int main() {
  static_assert(EIGEN_MCU == 1, "EIGEN_MCU must be 1");
#if EIGEN_MCU_ARCH_CORTEX_M
  static_assert(EIGEN_MCU_SMALL_GEMM == 1, "EIGEN_MCU_SMALL_GEMM must be 1 on Cortex-M");
#endif

  timer_start();
  bench_float_suite();
#ifndef BENCH_FLOAT_ONLY
  bench_double_suite();
#endif

  if (g_bad == 0) {
    print("BENCH_OK\n");
    return 0;
  }
  print("BENCH_FAIL\n");
  return 1;
}
