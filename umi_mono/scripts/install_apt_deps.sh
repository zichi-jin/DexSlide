#!/usr/bin/env bash
# TASK-002 apt dependency installer for online tracking mode.
# Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Installs only apt-managed Phase 0 dependencies for host preparation.
# Defaults to dry-run and never uses sudo unless '--apply' is passed.
# Does not remove, purge, reinstall, or modify apt sources configuration.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_apt_deps.sh [--apply|--help]

Options:
  --apply  Run 'sudo apt update' and install missing apt packages after confirmation
  --help   Show this help text and exit

Default mode is dry-run: print package status without running sudo.
EOF
}

MODE="dry-run"
if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --apply)
      MODE="apply"
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
fi

PACKAGES=(
  build-essential
  pkg-config
  git
  cmake
  libopencv-dev
  libeigen3-dev
  libboost-serialization-dev
  libboost-system-dev
  libboost-filesystem-dev
  libboost-thread-dev
  libssl-dev
  libusb-1.0-0-dev
  libsqlite3-dev
  libglew-dev
  libgl1-mesa-dev
  libglu1-mesa-dev
  libegl1-mesa-dev
  libwayland-dev
  libxkbcommon-dev
  wayland-protocols
  libavcodec-dev
  libavformat-dev
  libavutil-dev
  libswscale-dev
  libavdevice-dev
  libzmq3-dev
  curl
  wget
  unzip
)

declare -a PACKAGE_NAMES=()
declare -a PACKAGE_STATUS=()
declare -a TO_INSTALL=()

is_installed() {
  local pkg="$1"
  dpkg -s "$pkg" 2>/dev/null | grep -q "Status: install ok installed"
}

is_in_cache() {
  local pkg="$1"
  apt-cache show "$pkg" >/dev/null 2>&1
}

collect_statuses() {
  PACKAGE_NAMES=()
  PACKAGE_STATUS=()
  TO_INSTALL=()

  local pkg status
  for pkg in "${PACKAGES[@]}"; do
    if is_installed "$pkg"; then
      status="installed"
    elif is_in_cache "$pkg"; then
      status="to-install"
      TO_INSTALL+=("$pkg")
    else
      status="not-in-cache"
    fi

    PACKAGE_NAMES+=("$pkg")
    PACKAGE_STATUS+=("$status")
  done
}

print_table() {
  local package_w=7
  local status_w=6
  local i

  for i in "${!PACKAGE_NAMES[@]}"; do
    (( ${#PACKAGE_NAMES[i]} > package_w )) && package_w=${#PACKAGE_NAMES[i]}
    (( ${#PACKAGE_STATUS[i]} > status_w )) && status_w=${#PACKAGE_STATUS[i]}
  done

  printf "%-${package_w}s | %-${status_w}s\n" "Package" "Status"
  printf "%-${package_w}s-+-%-${status_w}s\n" \
    "$(printf '%*s' "$package_w" '' | tr ' ' '-')" \
    "$(printf '%*s' "$status_w" '' | tr ' ' '-')"

  for i in "${!PACKAGE_NAMES[@]}"; do
    printf "%-${package_w}s | %-${status_w}s\n" "${PACKAGE_NAMES[i]}" "${PACKAGE_STATUS[i]}"
  done
}

print_summary() {
  local installed_count=0
  local to_install_count=0
  local unavailable_count=0
  local status

  for status in "${PACKAGE_STATUS[@]}"; do
    case "$status" in
      installed)
        ((installed_count += 1))
        ;;
      to-install)
        ((to_install_count += 1))
        ;;
      not-in-cache)
        ((unavailable_count += 1))
        ;;
    esac
  done

  echo "Summary: ${installed_count} already installed, ${to_install_count} to install, ${unavailable_count} unavailable"
}

maybe_print_cppzmq_note() {
  if ! apt-cache search cppzmq | grep -q '.'; then
    echo "cppzmq is header-only; will be cloned from github at build time"
  fi
}

collect_statuses
print_table
print_summary
maybe_print_cppzmq_note

if [[ "$MODE" == "dry-run" ]]; then
  echo "DRY-RUN — re-run with --apply to install"
  exit 0
fi

if [[ ${#TO_INSTALL[@]} -eq 0 ]]; then
  echo "Nothing to install, all targets already present"
  exit 0
fi

sudo apt update
echo "About to: sudo apt install -y ${TO_INSTALL[*]}"
read -r confirmation
if [[ "$confirmation" != "yes" ]]; then
  echo "Aborted"
  exit 2
fi

sudo apt install -y "${TO_INSTALL[@]}"

collect_statuses
print_table
print_summary
maybe_print_cppzmq_note

if [[ ${#TO_INSTALL[@]} -gt 0 ]]; then
  echo "Verification failed: some target packages are still not installed"
  exit 1
fi

echo "Install complete and verified"
exit 0
