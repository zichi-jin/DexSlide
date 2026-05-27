#!/usr/bin/env bash
# TASK-004 Sophus v1.22.10 source build helper for online tracking mode.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Clones Sophus 1.22.10 into umi_mono/external/Sophus and builds it idempotently.
# Default behavior keeps artifacts inside the external/Sophus tree only.
# Optional install step writes to /usr/local and requires explicit confirmation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMI_MONO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${UMI_MONO_DIR}/external"
SOURCE_DIR="${EXTERNAL_DIR}/Sophus"
BUILD_DIR="${SOURCE_DIR}/build"
LOG_FILE="${BUILD_DIR}/build_sophus.log"
REPO_URL="https://github.com/strasdat/Sophus"
REQUIRED_TAG="1.22.10"

INSTALL=false
CLEAN=false
JOBS="$(nproc)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--install] [--clean] [--jobs N] [--help]

Options:
  --install  Run 'sudo cmake --install build' after a successful build
  --clean    Remove ${BUILD_DIR} before configuring
  --jobs N   Override parallel build jobs (default: nproc = ${JOBS})
  --help     Show this help text and exit
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

tag_matches() {
  local described_tag="$1"
  [[ "${described_tag}" == *"${REQUIRED_TAG}"* ]]
}

clone_repo() {
  local attempt current_tag cloned_tag

  mkdir -p "${EXTERNAL_DIR}"
  echo "Using Sophus tag: ${REQUIRED_TAG}"

  if [[ -d "${SOURCE_DIR}/.git" ]]; then
    current_tag="$(git -C "${SOURCE_DIR}" describe --tags 2>/dev/null || true)"
    if tag_matches "${current_tag}"; then
      return 0
    fi
  fi

  rm -rf "${SOURCE_DIR}"

  for attempt in 1 2; do
    if git clone --depth 1 --branch "${REQUIRED_TAG}" "${REPO_URL}" "${SOURCE_DIR}"; then
      break
    fi

    if [[ "${attempt}" -eq 1 ]]; then
      echo "Clone attempt 1 failed; retrying once..." >&2
      rm -rf "${SOURCE_DIR}"
      continue
    fi

    die "git clone failed after 2 attempts"
  done

  cloned_tag="$(git -C "${SOURCE_DIR}" describe --tags 2>/dev/null || true)"
  tag_matches "${cloned_tag}" || die "expected git describe --tags to contain '${REQUIRED_TAG}', got '${cloned_tag:-unknown}'"
}

verify_build_output() {
  local sophus_config=""

  sophus_config="$(find "${BUILD_DIR}" -name 'SophusConfig.cmake' -print -quit 2>/dev/null || true)"
  [[ -n "${sophus_config}" ]] || die "build output missing: expected SophusConfig.cmake under ${BUILD_DIR}"

  sed -n '1p' "${SOURCE_DIR}/sophus/se3.hpp" >/dev/null || die "cannot read ${SOURCE_DIR}/sophus/se3.hpp"
}

verify_install_output() {
  [[ -f "/usr/local/include/sophus/se3.hpp" ]] || die "missing /usr/local/include/sophus/se3.hpp after install"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      INSTALL=true
      shift
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires a value"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--jobs value must be a positive integer"
      JOBS="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
done

clone_repo

if [[ "${CLEAN}" == true ]]; then
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
: > "${LOG_FILE}"

(
  cd "${SOURCE_DIR}"
  cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SOPHUS_TESTS=OFF \
    -DBUILD_SOPHUS_EXAMPLES=OFF
  cmake --build build -j"${JOBS}"
) 2>&1 | tee -a "${LOG_FILE}"

verify_build_output

if [[ "${INSTALL}" == false ]]; then
  echo "Build complete; install step not requested"
  exit 0
fi

echo "About to: sudo cmake --install build (writes to /usr/local). Type yes to proceed:"
read -r confirmation
if [[ "${confirmation}" != "yes" ]]; then
  echo "Aborted"
  exit 2
fi

sudo cmake --install "${BUILD_DIR}"
verify_install_output

echo "Install complete and verified"
exit 0
