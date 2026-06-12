#!/usr/bin/env bash
# TASK-001 host environment check for online tracking mode.
# Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Verifies required host tools and writable workspace preconditions.
# Read-only checks only; no installation or system modification.
# Prints Name | Detected | Required | Status and exits non-zero on hard misses.

set -euo pipefail

ROOT_DIR="/home/jzq/MyJob/DexSlide"

declare -a ROW_NAMES=()
declare -a ROW_DETECTED=()
declare -a ROW_REQUIRED=()
declare -a ROW_STATUS=()
declare -a MISSING_KEYS=()

declare -A HINTS=(
  ["gcc"]="sudo apt install gcc"
  ["gxx"]="sudo apt install g++"
  ["cmake"]="sudo apt install cmake"
  ["python3"]="sudo apt install python3"
  ["pkg-config"]="sudo apt install pkg-config"
  ["git"]="sudo apt install git"
  ["apt-get"]="sudo apt install apt"
  ["lsb_release"]="sudo apt install lsb-release"
)

version_ge() {
  local detected="$1"
  local required="$2"
  local -a detected_parts=()
  local -a required_parts=()
  local i max_len detected_num required_num

  IFS='.' read -r -a detected_parts <<<"$detected"
  IFS='.' read -r -a required_parts <<<"$required"

  max_len=${#detected_parts[@]}
  if (( ${#required_parts[@]} > max_len )); then
    max_len=${#required_parts[@]}
  fi

  for (( i=0; i<max_len; i++ )); do
    detected_num=${detected_parts[i]:-0}
    required_num=${required_parts[i]:-0}
    (( detected_num > required_num )) && return 0
    (( detected_num < required_num )) && return 1
  done

  return 0
}

add_row() {
  local name="$1"
  local detected="$2"
  local required="$3"
  local status="$4"
  local missing_key="${5:-}"

  ROW_NAMES+=("$name")
  ROW_DETECTED+=("$detected")
  ROW_REQUIRED+=("$required")
  ROW_STATUS+=("$status")

  if [[ "$status" == "[MISSING]" && -n "$missing_key" ]]; then
    MISSING_KEYS+=("$missing_key")
  fi
}

check_versioned_tool() {
  local key="$1"
  local name="$2"
  local required="$3"
  shift 3
  local cmd=("$@")
  local detected=""
  local status="[MISSING]"

  if command -v "${cmd[0]}" >/dev/null 2>&1; then
    detected="$("${cmd[@]}" 2>/dev/null | head -n 1 | grep -Eo '[0-9]+([.][0-9]+)*' || true)"
    if [[ -n "$detected" ]] && version_ge "$detected" "$required"; then
      status="[OK]"
    else
      detected="${detected:-unparseable}"
    fi
  else
    detected="not found"
  fi

  add_row "$name" "$detected" ">= $required" "$status" "$key"
}

check_presence_tool() {
  local key="$1"
  local name="$2"
  local cmd_name="$3"
  local detected="not found"
  local status="[MISSING]"

  if command -v "$cmd_name" >/dev/null 2>&1; then
    detected="$(command -v "$cmd_name")"
    status="[OK]"
  fi

  add_row "$name" "$detected" "present" "$status" "$key"
}

check_ubuntu_version() {
  local detected="not found"
  local status="[MISSING]"

  if command -v lsb_release >/dev/null 2>&1; then
    detected="$(lsb_release -rs 2>/dev/null || true)"
    if [[ -n "$detected" ]] && version_ge "$detected" "22.04"; then
      status="[OK]"
    else
      detected="${detected:-unparseable}"
    fi
  fi

  add_row "Ubuntu version" "$detected" ">= 22.04" "$status" "ubuntu_version"
}

check_python3() {
  local detected="not found"
  local status="[MISSING]"

  if command -v python3 >/dev/null 2>&1; then
    detected="$(python3 --version 2>/dev/null | grep -Eo '[0-9]+([.][0-9]+)*' || true)"
    if [[ -n "$detected" ]] && version_ge "$detected" "3.10"; then
      if version_ge "$detected" "3.13"; then
        status="[WARN]"
        echo "system python >= 3.13; ROS2 jazzy uses its own python3.10" >&2
      else
        status="[OK]"
      fi
    else
      detected="${detected:-unparseable}"
    fi
  fi

  add_row "python3" "$detected" ">= 3.10" "$status" "python3"
}

check_writable_root() {
  local detected="no"
  local status="[MISSING]"

  if [[ -w "$ROOT_DIR" ]]; then
    detected="yes"
    status="[OK]"
  fi

  add_row "$ROOT_DIR writable" "$detected" "writable" "$status" "root_writable"
}

print_table() {
  local name_w=4
  local detected_w=8
  local required_w=8
  local status_w=6
  local i

  for i in "${!ROW_NAMES[@]}"; do
    (( ${#ROW_NAMES[i]} > name_w )) && name_w=${#ROW_NAMES[i]}
    (( ${#ROW_DETECTED[i]} > detected_w )) && detected_w=${#ROW_DETECTED[i]}
    (( ${#ROW_REQUIRED[i]} > required_w )) && required_w=${#ROW_REQUIRED[i]}
    (( ${#ROW_STATUS[i]} > status_w )) && status_w=${#ROW_STATUS[i]}
  done

  printf "%-${name_w}s | %-${detected_w}s | %-${required_w}s | %-${status_w}s\n" \
    "Name" "Detected" "Required" "Status"
  printf "%-${name_w}s-+-%-${detected_w}s-+-%-${required_w}s-+-%-${status_w}s\n" \
    "$(printf '%*s' "$name_w" '' | tr ' ' '-')" \
    "$(printf '%*s' "$detected_w" '' | tr ' ' '-')" \
    "$(printf '%*s' "$required_w" '' | tr ' ' '-')" \
    "$(printf '%*s' "$status_w" '' | tr ' ' '-')"

  for i in "${!ROW_NAMES[@]}"; do
    printf "%-${name_w}s | %-${detected_w}s | %-${required_w}s | %-${status_w}s\n" \
      "${ROW_NAMES[i]}" "${ROW_DETECTED[i]}" "${ROW_REQUIRED[i]}" "${ROW_STATUS[i]}"
  done
}

print_failure_hints() {
  local seen=""
  local key

  echo "FAIL — install missing tools first"
  for key in "${MISSING_KEYS[@]}"; do
    case "$key" in
      ubuntu_version)
        echo "Upgrade host to Ubuntu 22.04 or newer"
        ;;
      root_writable)
        echo "Ensure current user can write to $ROOT_DIR"
        ;;
      *)
        if [[ " $seen " != *" $key "* ]]; then
          echo "${HINTS[$key]}"
          seen+=" $key"
        fi
        ;;
    esac
  done
}

check_ubuntu_version
check_versioned_tool "gcc" "GCC" "11.0" gcc -dumpversion
check_versioned_tool "gxx" "g++" "11.0" g++ -dumpversion
check_versioned_tool "cmake" "CMake" "3.22" cmake --version
check_python3
check_presence_tool "pkg-config" "pkg-config" "pkg-config"
check_presence_tool "git" "git" "git"
check_presence_tool "apt-get" "apt-get" "apt-get"
check_presence_tool "lsb_release" "lsb_release" "lsb_release"
check_writable_root

print_table

if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
  print_failure_hints
  exit 1
fi

echo "PASS — host environment OK"
exit 0
