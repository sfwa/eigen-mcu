#!/usr/bin/env bash
# This file is part of eigen-mcu, an MCU-targeted fork of Eigen.
#
# Copyright (C) 2026 SFWA contributors
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Package a clean eigen-mcu source release archive compatible with CMake
# ExternalProject_Add / FetchContent (such as sfwa/ukf and sfwa/nmpc) and
# standalone CMake installs.
#
# Usage: scripts/package_release.sh [--out DIR] [--tag TAG] [--verify]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/dist"
EXPECTED_TAG=""
VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --tag)
      EXPECTED_TAG="$2"
      shift 2
      ;;
    --verify)
      VERIFY=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

cd "${ROOT}"

read_macro() {
  local file="$1" macro="$2"
  sed -n "s/^#define ${macro}[[:space:]]\\+\"\\?\\([^\"]*\\)\"\\?$/\\1/p" "${file}" | head -n 1
}

EIG_MAJOR="$(read_macro Eigen/Version EIGEN_MAJOR_VERSION)"
EIG_MINOR="$(read_macro Eigen/Version EIGEN_MINOR_VERSION)"
EIG_PATCH="$(read_macro Eigen/Version EIGEN_PATCH_VERSION)"
EIG_STR="$(read_macro Eigen/Version EIGEN_VERSION_STRING)"
MCU_MAJOR="$(read_macro Eigen/src/Core/util/McuConfig.h EIGEN_MCU_VERSION_MAJOR)"
MCU_MINOR="$(read_macro Eigen/src/Core/util/McuConfig.h EIGEN_MCU_VERSION_MINOR)"
MCU_PATCH="$(read_macro Eigen/src/Core/util/McuConfig.h EIGEN_MCU_VERSION_PATCH)"

VERSION="${EIG_MAJOR}.${EIG_MINOR}.${EIG_PATCH}"
if [[ "${VERSION}" != "${EIG_STR}" ]]; then
  echo "ERROR: Eigen/Version mismatch: ${VERSION} vs EIGEN_VERSION_STRING=${EIG_STR}" >&2
  exit 1
fi
if [[ "${VERSION}" != "${MCU_MAJOR}.${MCU_MINOR}.${MCU_PATCH}" ]]; then
  echo "ERROR: McuConfig.h version (${MCU_MAJOR}.${MCU_MINOR}.${MCU_PATCH}) != Eigen/Version (${VERSION})" >&2
  exit 1
fi

if [[ -n "${EXPECTED_TAG}" ]]; then
  if [[ "${EXPECTED_TAG}" != "v${VERSION}" && "${EXPECTED_TAG}" != "${VERSION}" ]]; then
    echo "ERROR: Tag '${EXPECTED_TAG}' does not match version '${VERSION}' (expected v${VERSION} or ${VERSION})" >&2
    exit 1
  fi
fi

# Verify all upstream license, notice, and signature files are present.
for req in LICENSE COPYING.MPL2 COPYING.BSD COPYING.APACHE COPYING.MINPACK COPYING.README CHANGELOG.md signature_of_eigen3_matrix_library CMakeLists.txt Eigen/Core Eigen/src/Core/util/McuConfig.h; do
  if [[ ! -f "${req}" ]]; then
    echo "ERROR: Required file missing: ${req}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"
ARCHIVE_NAME="eigen-mcu-${VERSION}.tar.gz"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"
SUM_PATH="${ARCHIVE_PATH}.sha256"

git archive --format=tar.gz --prefix="eigen-mcu-${VERSION}/" -o "${ARCHIVE_PATH}" HEAD
(
  cd "${OUT_DIR}"
  sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
)

echo "Packaged ${ARCHIVE_PATH}"
cat "${SUM_PATH}"

if [[ "${VERIFY}" -eq 1 ]]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT
  (
    cd "${OUT_DIR}"
    sha256sum -c "${ARCHIVE_NAME}.sha256"
  )
  tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
  SRC="${TMP_DIR}/eigen-mcu-${VERSION}"
  for req in LICENSE COPYING.MPL2 COPYING.BSD COPYING.APACHE COPYING.MINPACK COPYING.README CHANGELOG.md Eigen/Core Eigen/src/Core/util/McuConfig.h; do
    test -f "${SRC}/${req}"
  done
  cmake -S "${SRC}" -B "${TMP_DIR}/build" \
    -DBUILD_TESTING=OFF -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF \
    -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_DEMOS=OFF \
    -DCMAKE_INSTALL_PREFIX="${TMP_DIR}/install"
  cmake --install "${TMP_DIR}/build"
  test -f "${TMP_DIR}/install/include/eigen3/Eigen/Core"
  test -f "${TMP_DIR}/install/include/eigen3/Eigen/src/Core/util/McuConfig.h"
  test -f "${TMP_DIR}/install/share/eigen3/cmake/Eigen3Config.cmake"
  echo "Verified release archive extraction and CMake install."
fi
