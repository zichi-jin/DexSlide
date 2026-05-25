#!/usr/bin/env bash
# TASK-007 native ORB_SLAM3 fork build wrapper for online tracking mode.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Builds vendored third-party components and the fork root idempotently with CMake.
# Writes logs and build artifacts only inside the ORB_SLAM3_fork tree.
# Does not modify source files, install system packages, or use sudo.

set -euo pipefail

FORK="/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork"
LOG_FILE="${FORK}/build_orbslam3.log"
EXTERNAL_PANGOLIN="/data/codes/DexSlide/umi_mono/external/Pangolin/build"
JOBS="$(nproc)"
CLEAN=false

declare -a SUMMARY_STEPS=()
declare -a SUMMARY_STATUS=()
declare -a SUMMARY_SIZES=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [--clean] [--jobs N] [--help]

Options:
  --clean    Remove all build directories under ${FORK} before building
  --jobs N   Override parallel build jobs (default: nproc = ${JOBS})
  --help     Show this help text and exit
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

add_summary() {
  SUMMARY_STEPS+=("$1")
  SUMMARY_STATUS+=("$2")
  SUMMARY_SIZES+=("$3")
}

artifact_size_or_missing() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    stat -c%s "${path}"
  else
    echo "missing"
  fi
}

ensure_source_dir() {
  local dir="$1"
  local label="$2"
  [[ -d "${dir}" ]] || die "${label} source directory missing: ${dir}"
  [[ -f "${dir}/CMakeLists.txt" ]] || die "${label} is not buildable: missing ${dir}/CMakeLists.txt"
}

run_logged() {
  local label="$1"
  shift
  echo "==> ${label}" | tee -a "${LOG_FILE}"
  "$@" 2>&1 | tee -a "${LOG_FILE}"
}

verify_file_exists() {
  local label="$1"
  local path="$2"
  [[ -f "${path}" ]] || die "${label} build finished but expected artifact is missing: ${path}"
}

verify_executable_exists() {
  local label="$1"
  local path="$2"
  [[ -x "${path}" ]] || die "${label} build finished but expected executable is missing or not executable: ${path}"
}

build_component() {
  local label="$1"
  local source_dir="$2"
  local build_type="$3"
  shift 3
  local extra_args=("$@")

  ensure_source_dir "${source_dir}" "${label}"
  run_logged "${label}: configure" cmake -B "${source_dir}/build" -S "${source_dir}" -DCMAKE_BUILD_TYPE="${build_type}" "${extra_args[@]}"
  run_logged "${label}: build" cmake --build "${source_dir}/build" -j"${JOBS}"
}

print_summary() {
  local step_w=4
  local status_w=6
  local size_w=20
  local i

  for i in "${!SUMMARY_STEPS[@]}"; do
    (( ${#SUMMARY_STEPS[i]} > step_w )) && step_w=${#SUMMARY_STEPS[i]}
    (( ${#SUMMARY_STATUS[i]} > status_w )) && status_w=${#SUMMARY_STATUS[i]}
    (( ${#SUMMARY_SIZES[i]} > size_w )) && size_w=${#SUMMARY_SIZES[i]}
  done

  printf "%-${step_w}s | %-${status_w}s | %-${size_w}s\n" "Step" "Status" "Output artifact size"
  printf "%-${step_w}s-+-%-${status_w}s-+-%-${size_w}s\n" \
    "$(printf '%*s' "${step_w}" '' | tr ' ' '-')" \
    "$(printf '%*s' "${status_w}" '' | tr ' ' '-')" \
    "$(printf '%*s' "${size_w}" '' | tr ' ' '-')"

  for i in "${!SUMMARY_STEPS[@]}"; do
    printf "%-${step_w}s | %-${status_w}s | %-${size_w}s\n" \
      "${SUMMARY_STEPS[i]}" "${SUMMARY_STATUS[i]}" "${SUMMARY_SIZES[i]}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

[[ -d "${FORK}" ]] || die "ORB_SLAM3 fork directory not found: ${FORK}"
[[ -f "${EXTERNAL_PANGOLIN}/PangolinConfig.cmake" ]] || {
  echo "External Pangolin v0.8 missing. Run scripts/build_pangolin.sh first." >&2
  exit 1
}

if [[ "${CLEAN}" == true ]]; then
  rm -rf \
    "${FORK}/Thirdparty/DBoW2/build" \
    "${FORK}/Thirdparty/g2o/build" \
    "${FORK}/Thirdparty/Sophus/build" \
    "${FORK}/Thirdparty/Pangolin/build" \
    "${FORK}/build"
fi

: > "${LOG_FILE}"

build_component "DBoW2" "${FORK}/Thirdparty/DBoW2" "Release"
verify_file_exists "DBoW2" "${FORK}/Thirdparty/DBoW2/lib/libDBoW2.so"
add_summary "DBoW2" "OK" "$(artifact_size_or_missing "${FORK}/Thirdparty/DBoW2/lib/libDBoW2.so")"

build_component "g2o" "${FORK}/Thirdparty/g2o" "Release"
verify_file_exists "g2o" "${FORK}/Thirdparty/g2o/lib/libg2o.so"
add_summary "g2o" "OK" "$(artifact_size_or_missing "${FORK}/Thirdparty/g2o/lib/libg2o.so")"

build_component "Sophus" "${FORK}/Thirdparty/Sophus" "Release"
verify_file_exists "Sophus" "${FORK}/Thirdparty/Sophus/build/SophusConfig.cmake"
add_summary "Sophus" "OK" "$(artifact_size_or_missing "${FORK}/Thirdparty/Sophus/build/SophusConfig.cmake")"

if [[ ! -f "${FORK}/Thirdparty/Pangolin/CMakeLists.txt" ]]; then
  echo "Skipping Thirdparty/Pangolin (empty in this fork — using external/Pangolin via CMAKE_PREFIX_PATH)" | tee -a "${LOG_FILE}"
  add_summary "Pangolin" "SKIPPED" "external build"
else
  build_component "Pangolin" "${FORK}/Thirdparty/Pangolin" "Release"
  if [[ -f "${FORK}/Thirdparty/Pangolin/build/src/libpango_core.so" ]]; then
    add_summary "Pangolin" "OK" "$(artifact_size_or_missing "${FORK}/Thirdparty/Pangolin/build/src/libpango_core.so")"
  elif [[ -f "${FORK}/Thirdparty/Pangolin/build/src/libpango_core.a" ]]; then
    add_summary "Pangolin" "OK" "$(artifact_size_or_missing "${FORK}/Thirdparty/Pangolin/build/src/libpango_core.a")"
  else
    die "Pangolin build finished but expected artifact is missing under ${FORK}/Thirdparty/Pangolin/build/src"
  fi
fi

build_component "ORB_SLAM3" "${FORK}" "RelWithDebInfo" "-DCMAKE_PREFIX_PATH=${EXTERNAL_PANGOLIN}"
verify_file_exists "ORB_SLAM3" "${FORK}/lib/libORB_SLAM3.so"
verify_executable_exists "ORB_SLAM3" "${FORK}/Examples/Monocular-Inertial/gopro_slam"
add_summary "ORB_SLAM3" "OK" "$(artifact_size_or_missing "${FORK}/lib/libORB_SLAM3.so")"

if ldd "${FORK}/Examples/Monocular-Inertial/gopro_slam" 2>&1 | grep -q "not found"; then
  die "ldd reported unresolved shared libraries for ${FORK}/Examples/Monocular-Inertial/gopro_slam"
fi

print_summary
exit 0
