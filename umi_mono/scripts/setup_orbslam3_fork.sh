#!/usr/bin/env bash
# TASK-006 ORB_SLAM3 fork setup helper for online tracking mode.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Clones cheng-chi/ORB_SLAM3 as a regular git clone and pins it to a commit SHA.
# Unpacks ORB vocabulary when needed and records the pinned SHA atomically.
# Does not use submodules or modify top-level git metadata.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMI_MONO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${UMI_MONO_DIR}/external"
TARGET_DIR="${EXTERNAL_DIR}/ORB_SLAM3_fork"
REPO_URL="https://github.com/cheng-chi/ORB_SLAM3"
BRANCH_NAME=""
DEFAULT_BRANCH_CANDIDATES=("main" "master")
RESOLVED_SHA=""
CLEAN=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--sha <SHA>] [--clean] [--help]

Options:
  --sha SHA  Pin to a specific commit SHA (default: latest main HEAD, falling back to master)
  --clean    Remove ${TARGET_DIR} before cloning
  --help     Show this help text and exit
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

resolve_sha() {
  local candidate sha

  if [[ -n "${RESOLVED_SHA}" ]]; then
    BRANCH_NAME="${BRANCH_NAME:-main}"
    return 0
  fi

  for candidate in "${DEFAULT_BRANCH_CANDIDATES[@]}"; do
    sha="$(git ls-remote "${REPO_URL}" "${candidate}" | awk '{print $1}')"
    if [[ -n "${sha}" ]]; then
      BRANCH_NAME="${candidate}"
      RESOLVED_SHA="${sha}"
      return 0
    fi
  done

  die "failed to resolve latest SHA for main or master"
}

current_head() {
  if [[ -d "${TARGET_DIR}/.git" ]]; then
    git -C "${TARGET_DIR}" rev-parse HEAD 2>/dev/null || true
  else
    true
  fi
}

ensure_checkout() {
  local head_sha=""

  mkdir -p "${EXTERNAL_DIR}"

  if [[ "${CLEAN}" == true ]]; then
    rm -rf "${TARGET_DIR}"
  fi

  if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    rm -rf "${TARGET_DIR}"
    git clone "${REPO_URL}" "${TARGET_DIR}"
  fi

  head_sha="$(current_head)"
  if [[ "${head_sha}" != "${RESOLVED_SHA}" ]]; then
    git -C "${TARGET_DIR}" fetch origin
    git -C "${TARGET_DIR}" checkout "${RESOLVED_SHA}"
  fi
}

write_pinned_sha() {
  local tmp_file="${TARGET_DIR}/.pinned_sha.tmp"
  printf '%s\n' "${RESOLVED_SHA}" > "${tmp_file}"
  mv "${tmp_file}" "${TARGET_DIR}/.pinned_sha"
}

maybe_unpack_vocabulary() {
  local vocab_dir="${TARGET_DIR}/Vocabulary"
  local vocab_tar="${vocab_dir}/ORBvoc.txt.tar.gz"
  local vocab_txt="${vocab_dir}/ORBvoc.txt"

  if [[ -f "${vocab_tar}" && ! -f "${vocab_txt}" ]]; then
    tar -xzf "${vocab_tar}" -C "${vocab_dir}"
  fi
}

verify_setup() {
  local head_sha=""

  head_sha="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
  [[ "${head_sha}" == "${RESOLVED_SHA}" ]] || die "HEAD ${head_sha} does not match pinned SHA ${RESOLVED_SHA}"
  [[ -d "${TARGET_DIR}/Examples/Monocular-Inertial" ]] || die "missing ${TARGET_DIR}/Examples/Monocular-Inertial"
  [[ -f "${TARGET_DIR}/include/System.h" ]] || die "missing ${TARGET_DIR}/include/System.h"
}

print_summary() {
  local examples_count=0
  local orbvoc_size="missing"
  local orbvoc_path="${TARGET_DIR}/Vocabulary/ORBvoc.txt"
  local item_w=5
  local value_w=5

  examples_count="$(find "${TARGET_DIR}/Examples/Monocular-Inertial" -maxdepth 1 -type f | wc -l | tr -d ' ')"
  if [[ -f "${orbvoc_path}" ]]; then
    orbvoc_size="$(stat -c%s "${orbvoc_path}")"
  fi

  local -a items=("SHA" "branch" "Examples/Monocular-Inertial/ files count" "Vocabulary/ORBvoc.txt size")
  local -a values=("${RESOLVED_SHA}" "${BRANCH_NAME}" "${examples_count}" "${orbvoc_size}")
  local i

  for i in "${!items[@]}"; do
    (( ${#items[i]} > item_w )) && item_w=${#items[i]}
    (( ${#values[i]} > value_w )) && value_w=${#values[i]}
  done

  printf "%-${item_w}s | %-${value_w}s\n" "Item" "Value"
  printf "%-${item_w}s-+-%-${value_w}s\n" \
    "$(printf '%*s' "${item_w}" '' | tr ' ' '-')" \
    "$(printf '%*s' "${value_w}" '' | tr ' ' '-')"

  for i in "${!items[@]}"; do
    printf "%-${item_w}s | %-${value_w}s\n" "${items[i]}" "${values[i]}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sha)
      [[ $# -ge 2 ]] || die "--sha requires a value"
      RESOLVED_SHA="$2"
      shift 2
      ;;
    --clean)
      CLEAN=true
      shift
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

resolve_sha
ensure_checkout
write_pinned_sha
maybe_unpack_vocabulary
verify_setup
print_summary
