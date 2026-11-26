// This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
//
// Copyright (C) 2026 SFWA contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Standalone correctness and zero-allocation test suite for eigen-mcu kernels.
// Runs on host (x64/arm64) and bare-metal Cortex-M4F/M7 (QEMU).

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include <cmath>
#include <cstdint>
#ifndef TEST_MCU_BAREMETAL
#include <cstdio>
#endif
#include <limits>
#include <type_traits>

#ifdef TEST_MCU_BAREMETAL
extern "C" {
void print(const char* s);
void print_u(std::uint64_t v);
}
#endif

static int g_failures = 0;

static void check(bool cond, const char* msg, int line) {
  if (!cond) {
#ifdef TEST_MCU_BAREMETAL
    print("FAIL line ");
    print_u(static_cast<std::uint64_t>(line));
    print(": ");
    print(msg);
    print("\n");
#else
    std::printf("FAIL line %d: %s\n", line, msg);
#endif
    ++g_failures;
  }
}

#define CHECK(c) check((c), #c, __LINE__)

// Deterministic pseudo-random scalar in [-1.5, 1.5].
template <typename T>
static T pseudo(int idx) {
  std::uint32_t x = static_cast<std::uint32_t>(idx) * 2654435761u + 1013904223u;
  x ^= x >> 13;
  x *= 1274126177u;
  x ^= x >> 16;
  const int signed_v = static_cast<int>(x & 0xFFFFu) - 32768;
  return T(signed_v) * T(1.5 / 32768.0);
}

template <typename T, int R, int C, int Order>
static Eigen::Matrix<T, R, C, Order> make_mat(int seed) {
  Eigen::Matrix<T, R, C, Order> m;
  for (std::ptrdiff_t j = 0; j < C; ++j) {
    for (std::ptrdiff_t i = 0; i < R; ++i) {
      m(i, j) = pseudo<T>(seed + static_cast<int>(i * 37 + j * 131));
    }
  }
  return m;
}

template <typename T>
static T tol_scale() {
  return std::is_same<T, float>::value ? T(1e-4f) : T(1e-11);
}

// Reference naive GEMM: C += alpha * A * B.
template <typename MatA, typename MatB, typename MatC>
static void ref_gemm(const MatA& A, const MatB& B, MatC& C, typename MatC::Scalar alpha) {
  using T = typename MatC::Scalar;
  for (std::ptrdiff_t j = 0; j < C.cols(); ++j) {
    for (std::ptrdiff_t i = 0; i < C.rows(); ++i) {
      T s = T(0);
      for (std::ptrdiff_t k = 0; k < A.cols(); ++k) {
        s += A(i, k) * B(k, j);
      }
      C(i, j) += alpha * s;
    }
  }
}

template <typename T, int M, int N, int K, int AO, int BO, int CO>
EIGEN_DONT_INLINE static void test_gemm_case(int seed, T alpha) {
  auto A = make_mat<T, M, K, AO>(seed);
  auto B = make_mat<T, K, N, BO>(seed + 1000);
  auto C0 = make_mat<T, M, N, CO>(seed + 2000);
  auto C_ref = C0;
  ref_gemm(A, B, C_ref, alpha);

  auto C_got = C0;
  C_got.noalias() += alpha * (A * B);

  const T err = (C_got - C_ref).cwiseAbs().maxCoeff();
  const T limit = tol_scale<T>() * T(K + 1);
  CHECK(err <= limit);
}

template <typename T, int N, int K, int AO, int BO, int CO, int UpLo>
EIGEN_DONT_INLINE static void test_tri_prod_case(int seed, T alpha) {
  auto A = make_mat<T, N, K, AO>(seed);
  auto B = make_mat<T, K, N, BO>(seed + 3000);
  auto C0 = make_mat<T, N, N, CO>(seed + 4000);
  auto C_ref = C0;
  auto Full = C0;
  ref_gemm(A, B, Full, alpha);
  for (std::ptrdiff_t j = 0; j < N; ++j) {
    for (std::ptrdiff_t i = 0; i < N; ++i) {
      const bool in_tri = (UpLo == Eigen::Lower) ? (i >= j)
                        : (UpLo == Eigen::Upper) ? (i <= j)
                        : (UpLo == Eigen::StrictlyLower) ? (i > j)
                                                         : (i < j);
      if (in_tri) C_ref(i, j) = Full(i, j);
    }
  }

  auto C_got = C0;
  C_got.template triangularView<UpLo>() += alpha * (A * B);
  const T err = (C_got - C_ref).cwiseAbs().maxCoeff();
  CHECK(err <= tol_scale<T>() * T(K + 1));

  // Also test assignment (beta = false) on Lower/Upper.
  if (UpLo == Eigen::Lower || UpLo == Eigen::Upper) {
    auto C_assign = C0;
    C_assign.template triangularView<UpLo>() = alpha * (A * B);
    auto Expected = C0;
    for (std::ptrdiff_t j = 0; j < N; ++j) {
      for (std::ptrdiff_t i = 0; i < N; ++i) {
        const bool in_tri = (UpLo == Eigen::Lower) ? (i >= j) : (i <= j);
        if (in_tri) Expected(i, j) = Full(i, j) - C0(i, j);
      }
    }
    CHECK((C_assign - Expected).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(K + 1));
  }
}

template <typename T, int N, int K, int AO, int CO, int UpLo>
EIGEN_DONT_INLINE static void test_syrk_case(int seed, T alpha) {
  auto A = make_mat<T, N, K, AO>(seed);
  auto C0 = make_mat<T, N, N, CO>(seed + 5000);
  auto Full = C0;
  ref_gemm(A, A.transpose().eval(), Full, alpha);
  auto C_ref = C0;
  for (std::ptrdiff_t j = 0; j < N; ++j) {
    for (std::ptrdiff_t i = 0; i < N; ++i) {
      const bool in_tri = (UpLo == Eigen::Lower) ? (i >= j) : (i <= j);
      if (in_tri) C_ref(i, j) = Full(i, j);
    }
  }

  auto C_got = C0;
  C_got.template selfadjointView<UpLo>().rankUpdate(A, alpha);
  CHECK((C_got - C_ref).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(K + 1));
}

template <int Side>
struct trsm_make_rhs {
  template <typename MatT, typename MatX, typename MatB>
  static void run(const MatT& Tmat, const MatX& X_true, MatB& B) {
    B = Tmat * X_true;
  }
};

template <>
struct trsm_make_rhs<Eigen::OnTheRight> {
  template <typename MatT, typename MatX, typename MatB>
  static void run(const MatT& Tmat, const MatX& X_true, MatB& B) {
    B = X_true * Tmat;
  }
};

template <typename T, int N, int M, int LOrder, int XOrder, int Mode, int Side>
EIGEN_DONT_INLINE static void test_trsm_case(int seed) {
  auto L = make_mat<T, N, N, LOrder>(seed);
  for (std::ptrdiff_t i = 0; i < N; ++i) {
    L(i, i) = (i % 2 == 0 ? T(1) : T(-1)) * (T(3) + std::abs(L(i, i)));
  }

  constexpr int XR = (Side == Eigen::OnTheLeft) ? N : M;
  constexpr int XC = (Side == Eigen::OnTheLeft) ? M : N;
  auto X_true = make_mat<T, XR, XC, XOrder>(seed + 6000);

  Eigen::Matrix<T, N, N> Tmat = L.template triangularView<Mode>();
  Eigen::Matrix<T, XR, XC, XOrder> B;
  trsm_make_rhs<Side>::run(Tmat, X_true, B);

  auto X_got = B;
  L.template triangularView<Mode>().template solveInPlace<Side>(X_got);
  const T err = (X_got - X_true).cwiseAbs().maxCoeff();
  CHECK(err <= tol_scale<T>() * T(N * 4));
}

template <typename T>
static void test_all_for_scalar() {
  // 1. GEMM: all 8 storage-order combos (multiple-of-4 and non-multiple-of-4 tails) + A*A aliasing.
  test_gemm_case<T, 12, 12, 12, Eigen::ColMajor, Eigen::ColMajor, Eigen::ColMajor>(1, T(1));
  test_gemm_case<T, 13, 11, 15, Eigen::ColMajor, Eigen::ColMajor, Eigen::ColMajor>(2, T(1));
  test_gemm_case<T, 13, 11, 15, Eigen::ColMajor, Eigen::RowMajor, Eigen::ColMajor>(3, T(-0.75));
  test_gemm_case<T, 13, 11, 15, Eigen::RowMajor, Eigen::ColMajor, Eigen::ColMajor>(4, T(2.25));
  test_gemm_case<T, 13, 11, 15, Eigen::RowMajor, Eigen::RowMajor, Eigen::ColMajor>(5, T(1));
  test_gemm_case<T, 13, 11, 15, Eigen::ColMajor, Eigen::ColMajor, Eigen::RowMajor>(6, T(1));
  test_gemm_case<T, 13, 11, 15, Eigen::ColMajor, Eigen::RowMajor, Eigen::RowMajor>(7, T(-1));
  test_gemm_case<T, 13, 11, 15, Eigen::RowMajor, Eigen::ColMajor, Eigen::RowMajor>(8, T(0.5));
  test_gemm_case<T, 13, 11, 15, Eigen::RowMajor, Eigen::RowMajor, Eigen::RowMajor>(9, T(1));

  // Self-aliasing A * A (verifies no restrict violation between A and B).
  {
    auto A = make_mat<T, 13, 13, Eigen::ColMajor>(99);
    Eigen::Matrix<T, 13, 13> C_ref = Eigen::Matrix<T, 13, 13>::Zero();
    ref_gemm(A, A, C_ref, T(1));
    Eigen::Matrix<T, 13, 13> C_got;
    C_got.noalias() = A * A;
    CHECK((C_got - C_ref).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(20));
  }

  // 2. Triangular product (Lower, Upper, StrictlyLower, StrictlyUpper, Col/RowMajor).
  test_tri_prod_case<T, 12, 15, Eigen::ColMajor, Eigen::ColMajor, Eigen::ColMajor, Eigen::Upper>(10, T(1));
  test_tri_prod_case<T, 13, 15, Eigen::RowMajor, Eigen::ColMajor, Eigen::ColMajor, Eigen::Upper>(11, T(1));
  test_tri_prod_case<T, 13, 15, Eigen::ColMajor, Eigen::RowMajor, Eigen::ColMajor, Eigen::Lower>(12, T(-1.5));
  test_tri_prod_case<T, 13, 15, Eigen::RowMajor, Eigen::RowMajor, Eigen::RowMajor, Eigen::Lower>(13, T(0.5));
  test_tri_prod_case<T, 13, 15, Eigen::ColMajor, Eigen::ColMajor, Eigen::RowMajor, Eigen::Upper>(14, T(1));
  test_tri_prod_case<T, 13, 15, Eigen::ColMajor, Eigen::ColMajor, Eigen::ColMajor, Eigen::StrictlyLower>(15, T(1));
  test_tri_prod_case<T, 13, 15, Eigen::ColMajor, Eigen::ColMajor, Eigen::ColMajor, Eigen::StrictlyUpper>(16, T(1));

  // 3. SYRK (Lower/Upper, Col/RowMajor, including 15x31 UKF sigma-point shape).
  test_syrk_case<T, 15, 31, Eigen::ColMajor, Eigen::ColMajor, Eigen::Lower>(20, T(1));
  test_syrk_case<T, 15, 31, Eigen::RowMajor, Eigen::ColMajor, Eigen::Upper>(21, T(-0.5));
  test_syrk_case<T, 13, 15, Eigen::ColMajor, Eigen::RowMajor, Eigen::Lower>(22, T(1));
  test_syrk_case<T, 13, 15, Eigen::RowMajor, Eigen::RowMajor, Eigen::Upper>(23, T(2));

  // 4. TRSM & TRSV: OnTheLeft / OnTheRight, Lower / Upper, Unit / NonUnit, Col / RowMajor.
  test_trsm_case<T, 13, 11, Eigen::ColMajor, Eigen::ColMajor, Eigen::Lower, Eigen::OnTheLeft>(30);
  test_trsm_case<T, 13, 11, Eigen::RowMajor, Eigen::ColMajor, Eigen::Lower, Eigen::OnTheLeft>(31);
  test_trsm_case<T, 13, 11, Eigen::ColMajor, Eigen::ColMajor, Eigen::Upper, Eigen::OnTheLeft>(32);
  test_trsm_case<T, 13, 11, Eigen::RowMajor, Eigen::ColMajor, Eigen::Upper, Eigen::OnTheLeft>(33);
  test_trsm_case<T, 13, 11, Eigen::ColMajor, Eigen::ColMajor, Eigen::UnitLower, Eigen::OnTheLeft>(34);
  test_trsm_case<T, 13, 11, Eigen::RowMajor, Eigen::ColMajor, Eigen::UnitUpper, Eigen::OnTheLeft>(35);
  test_trsm_case<T, 13, 11, Eigen::ColMajor, Eigen::ColMajor, Eigen::Lower, Eigen::OnTheRight>(36);
  test_trsm_case<T, 13, 11, Eigen::RowMajor, Eigen::ColMajor, Eigen::Upper, Eigen::OnTheRight>(37);
  test_trsm_case<T, 13, 11, Eigen::ColMajor, Eigen::RowMajor, Eigen::Lower, Eigen::OnTheLeft>(38);
  test_trsm_case<T, 13, 11, Eigen::RowMajor, Eigen::RowMajor, Eigen::Upper, Eigen::OnTheRight>(39);
  // Vector RHS (TRSV, N > 8 to hit mcu_small_trsv).
  test_trsm_case<T, 13, 1, Eigen::ColMajor, Eigen::ColMajor, Eigen::Lower, Eigen::OnTheLeft>(40);
  test_trsm_case<T, 13, 1, Eigen::RowMajor, Eigen::ColMajor, Eigen::Upper, Eigen::OnTheLeft>(41);
  test_trsm_case<T, 13, 1, Eigen::ColMajor, Eigen::ColMajor, Eigen::UnitLower, Eigen::OnTheLeft>(42);
  test_trsm_case<T, 13, 1, Eigen::RowMajor, Eigen::ColMajor, Eigen::UnitUpper, Eigen::OnTheLeft>(43);

#ifndef EIGEN_NO_MALLOC
  // Dynamic small (<=40) and dynamic large (>40 GotoBLAS fallback).
  for (int sz : {13, 45}) {
    using DynMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    DynMat A(sz, sz), B(sz, sz), C0(sz, sz);
    for (int j = 0; j < sz; ++j) {
      for (int i = 0; i < sz; ++i) {
        A(i, j) = pseudo<T>(500 + i * 19 + j * 41);
        B(i, j) = pseudo<T>(900 + i * 23 + j * 37);
        C0(i, j) = pseudo<T>(1300 + i * 17 + j * 29);
      }
    }
    DynMat C_ref = C0;
    ref_gemm(A, B, C_ref, T(1));
    DynMat C_got = C0;
    C_got.noalias() += A * B;
    CHECK((C_got - C_ref).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(sz + 1));

    // Dynamic TRSM (small and large fallback).
    DynMat L = A;
    for (int i = 0; i < sz; ++i) L(i, i) = T(10) + std::abs(L(i, i));
    DynMat Tmat = L.template triangularView<Eigen::Lower>();
    DynMat X_got = Tmat * B;
    L.template triangularView<Eigen::Lower>().solveInPlace(X_got);
    CHECK((X_got - B).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(sz * 8));
  }
#endif

  // 5. LLT & LDLT (compute, solve, rankUpdate zero-malloc, deferred rcond()).
  {
    auto M = make_mat<T, 15, 15, Eigen::ColMajor>(50);
    Eigen::Matrix<T, 15, 15> A = M * M.transpose() + T(5) * Eigen::Matrix<T, 15, 15>::Identity();
    auto b = make_mat<T, 15, 1, Eigen::ColMajor>(51);
    auto B = make_mat<T, 15, 5, Eigen::ColMajor>(52);

    Eigen::LLT<Eigen::Matrix<T, 15, 15>> llt(A);
    CHECK(llt.info() == Eigen::Success);
    Eigen::Matrix<T, 15, 1> x = llt.solve(b);
    CHECK((A * x - b).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(20));
    Eigen::Matrix<T, 15, 5> X = llt.solve(B);
    CHECK((A * X - B).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(20));
    const T rc = llt.rcond();
    CHECK(rc > T(0) && rc <= T(1));

    // Rank update with column vector and row vector (zero malloc).
    Eigen::Matrix<T, 15, 1> u = make_mat<T, 15, 1, Eigen::ColMajor>(53);
    Eigen::Matrix<T, 15, 15> A_upd = A + T(0.5) * (u * u.transpose());
    llt.rankUpdate(u, T(0.5));
    CHECK(llt.info() == Eigen::Success);
    CHECK((llt.reconstructedMatrix() - A_upd).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(30));

    Eigen::LDLT<Eigen::Matrix<T, 15, 15>> ldlt(A);
    CHECK(ldlt.info() == Eigen::Success);
    Eigen::Matrix<T, 15, 1> xl = ldlt.solve(b);
    CHECK((A * xl - b).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(20));
    Eigen::Matrix<T, 15, 5> Xl = ldlt.solve(B);
    CHECK((A * Xl - B).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(20));
    const T rcl = ldlt.rcond();
    CHECK(rcl > T(0) && rcl <= T(1));
  }

  // 6. HouseholderQR & SelfAdjointEigenSolver (tests makeHouseholder + BlockHouseholder pruning, zero malloc).
  {
    auto A = make_mat<T, 12, 9, Eigen::ColMajor>(60);
    Eigen::HouseholderQR<Eigen::Matrix<T, 12, 9>> qr(A);
    Eigen::Matrix<T, 12, 12> Q = qr.householderQ();
    Eigen::Matrix<T, 12, 9> R = qr.matrixQR().template triangularView<Eigen::Upper>();
    CHECK((Q * R - A).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(20));

    auto S_raw = make_mat<T, 9, 9, Eigen::ColMajor>(61);
    Eigen::Matrix<T, 9, 9> S = S_raw + S_raw.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 9, 9>> eig(S);
    CHECK(eig.info() == Eigen::Success);
    Eigen::Matrix<T, 9, 9> recon = eig.eigenvectors() * eig.eigenvalues().asDiagonal() * eig.eigenvectors().transpose();
    CHECK((recon - S).cwiseAbs().maxCoeff() <= tol_scale<T>() * T(30));
  }

  // 7. Dot, normalize, normalized, quaternion rotate/product.
  {
    auto v1 = make_mat<T, 15, 1, Eigen::ColMajor>(70);
    auto v2 = make_mat<T, 15, 1, Eigen::ColMajor>(71);
    T ref_dot = T(0);
    for (int i = 0; i < 15; ++i) ref_dot += v1(i) * v2(i);
    CHECK(std::abs(v1.dot(v2) - ref_dot) <= tol_scale<T>() * T(10));

    auto v3 = make_mat<T, 3, 1, Eigen::ColMajor>(72);
    auto vn = v3.normalized();
    CHECK(std::abs(vn.norm() - T(1)) <= tol_scale<T>());
    auto v3_copy = v3;
    v3_copy.normalize();
    CHECK((v3_copy - vn).cwiseAbs().maxCoeff() <= tol_scale<T>());

    // Zero vector normalize leaves unchanged.
    Eigen::Matrix<T, 3, 1> vz = Eigen::Matrix<T, 3, 1>::Zero();
    vz.normalize();
    CHECK(vz.squaredNorm() == T(0));

    Eigen::Quaternion<T> q(T(0.9), T(0.1), T(-0.2), T(0.3));
    q.normalize();
    Eigen::Matrix<T, 3, 1> vr = q * v3;
    Eigen::Matrix<T, 3, 1> vr_mat = q.toRotationMatrix() * v3;
    CHECK((vr - vr_mat).cwiseAbs().maxCoeff() <= tol_scale<T>());
  }

  // 8. Bitwise allFinite() and hasNaN() on contiguous + non-contiguous blocks + edge values.
  {
    auto M = make_mat<T, 15, 15, Eigen::ColMajor>(80);
    CHECK(M.allFinite());
    CHECK(!M.hasNaN());

    // Subnormal and negative zero must be finite and not NaN.
    M(0, 0) = -T(0);
    M(1, 0) = std::numeric_limits<T>::denorm_min();
    M(2, 0) = std::numeric_limits<T>::max();
    M(3, 0) = -std::numeric_limits<T>::max();
    CHECK(M.allFinite());
    CHECK(!M.hasNaN());

    // Inf: not finite, not NaN.
    M(14, 14) = std::numeric_limits<T>::infinity();
    CHECK(!M.allFinite());
    CHECK(!M.hasNaN());

    M(14, 14) = -std::numeric_limits<T>::infinity();
    CHECK(!M.allFinite());
    CHECK(!M.hasNaN());

    // Quiet NaN at various offsets (0, 3, 4, 14*15+14).
    M(14, 14) = T(1);
    const std::ptrdiff_t positions[] = {0, 1, 2, 3, 4, 7, 17, 224};
    for (std::ptrdiff_t idx : positions) {
      M.data()[idx] = std::numeric_limits<T>::quiet_NaN();
      CHECK(!M.allFinite());
      CHECK(M.hasNaN());
      M.data()[idx] = T(1);
    }

    // Non-contiguous block (falls back safely).
    M(10, 10) = std::numeric_limits<T>::quiet_NaN();
    CHECK(M.block(0, 0, 5, 5).allFinite());
    CHECK(!M.block(0, 0, 5, 5).hasNaN());
    CHECK(!M.block(8, 8, 5, 5).allFinite());
    CHECK(M.block(8, 8, 5, 5).hasNaN());
  }

  // 9. cwiseMin / cwiseMax (finite, PropagateNumbers, PropagateNaN).
  {
    Eigen::Matrix<T, 4, 1> a(T(-2), T(3), T(0), T(5));
    Eigen::Matrix<T, 4, 1> b(T(1), T(-4), T(2), T(5));
    Eigen::Matrix<T, 4, 1> cmin = a.cwiseMin(b);
    Eigen::Matrix<T, 4, 1> cmax = a.cwiseMax(b);
    CHECK(cmin(0) == T(-2) && cmin(1) == T(-4) && cmin(2) == T(0) && cmin(3) == T(5));
    CHECK(cmax(0) == T(1) && cmax(1) == T(3) && cmax(2) == T(2) && cmax(3) == T(5));

    Eigen::Matrix<T, 2, 1> an(std::numeric_limits<T>::quiet_NaN(), T(2));
    Eigen::Matrix<T, 2, 1> bn(T(1), std::numeric_limits<T>::quiet_NaN());
    auto pn = an.template cwiseMax<Eigen::PropagateNumbers>(bn);
    CHECK(pn(0) == T(1) && pn(1) == T(2));
    auto pnan = an.template cwiseMax<Eigen::PropagateNaN>(bn);
    CHECK(std::isnan(pnan(0)) && std::isnan(pnan(1)));
  }
}

int main() {
  static_assert(EIGEN_MCU == 1, "EIGEN_MCU must be 1");
#if EIGEN_MCU_ARCH_CORTEX_M
  static_assert(EIGEN_MCU_SMALL_GEMM == 1, "EIGEN_MCU_SMALL_GEMM must be 1 on Cortex-M");
#endif

  test_all_for_scalar<float>();
#ifndef TEST_FLOAT_ONLY
  test_all_for_scalar<double>();
#endif

  if (g_failures == 0) {
#ifdef TEST_MCU_BAREMETAL
    print("ALL PASSED\n");
#else
    std::printf("ALL PASSED\n");
#endif
    return 0;
  }
#ifdef TEST_MCU_BAREMETAL
  print("FAILED: ");
  print_u(static_cast<std::uint64_t>(g_failures));
  print(" checks\n");
#else
  std::printf("FAILED: %d checks\n", g_failures);
#endif
  return 1;
}
