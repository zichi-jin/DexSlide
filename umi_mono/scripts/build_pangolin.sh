#!/usr/bin/env bash
# TASK-003 Pangolin v0.8 source build helper for online tracking mode.
# Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Clones Pangolin v0.8 into umi_mono/external/Pangolin and builds it idempotently.
# Optional install step writes only to /usr/local and requires explicit confirmation.
# Does not modify apt, apt sources, or files outside the Pangolin external tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMI_MONO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${UMI_MONO_DIR}/external"
SOURCE_DIR="${EXTERNAL_DIR}/Pangolin"
BUILD_DIR="${SOURCE_DIR}/build"
LOG_FILE="${SOURCE_DIR}/build_pangolin.log"
REPO_URL="https://github.com/stevenlovegrove/Pangolin"
REQUIRED_TAG="v0.8"

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

clone_repo() {
  local attempt

  mkdir -p "${EXTERNAL_DIR}"

  if [[ -d "${SOURCE_DIR}/.git" ]]; then
    local current_tag=""
    current_tag="$(git -C "${SOURCE_DIR}" describe --tags 2>/dev/null || true)"
    if [[ "${current_tag}" == "${REQUIRED_TAG}" ]]; then
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

  local cloned_tag=""
  cloned_tag="$(git -C "${SOURCE_DIR}" describe --tags 2>/dev/null || true)"
  [[ "${cloned_tag}" == "${REQUIRED_TAG}" ]] || die "expected tag '${REQUIRED_TAG}', got '${cloned_tag:-unknown}'"
}

verify_build_output() {
  if [[ -f "${BUILD_DIR}/src/libpango_core.so" || -f "${BUILD_DIR}/src/libpango_core.a" ]]; then
    return 0
  fi

  die "build output missing: expected ${BUILD_DIR}/src/libpango_core.so or .a"
}

verify_install_output() {
  [[ -f "/usr/local/include/pangolin/pangolin.h" ]] || die "missing /usr/local/include/pangolin/pangolin.h after install"
  [[ -f "/usr/local/lib/cmake/Pangolin/PangolinConfig.cmake" ]] || die "missing /usr/local/lib/cmake/Pangolin/PangolinConfig.cmake after install"
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
    -DBUILD_PANGOLIN_PYTHON=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF
  cmake --build build -j"${JOBS}"
) 2>&1 | tee -a "${LOG_FILE}"

verify_build_output

if [[ "${INSTALL}" == false ]]; then
  echo "Build complete; install step not requested"
  exit 0
fi

echo "About to: sudo make install (writes to /usr/local). Type yes to proceed:"
read -r confirmation
if [[ "${confirmation}" != "yes" ]]; then
  echo "Aborted"
  exit 2
fi

sudo cmake --install "${BUILD_DIR}"
verify_install_output

echo "Install complete and verified"
exit 0
