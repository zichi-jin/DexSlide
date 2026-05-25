#!/usr/bin/env bash
# TASK-005 librealsense2 dev package installer for online tracking mode.
# Tracker: /data/codes/DexSlide/umi_mono/docs/online_tracking_implementation.md
# Installs librealsense2-dev from a pre-configured Intel PPA when explicitly asked.
# Default mode is dry-run and never uses sudo without --apply and confirmation.
# Does not modify apt source files or remove any package.

set -euo pipefail

PPA_FILE="/etc/apt/sources.list.d/librealsense.list"
PACKAGES=(
  librealsense2-dev
  librealsense2-dkms
  librealsense2-utils
)

usage() {
  cat <<'EOF'
Usage: install_librealsense.sh [--apply|--help]

Options:
  --apply  Run 'sudo apt update && sudo apt install -y librealsense2-dev' after confirmation
  --help   Show this help text and exit

Default mode is dry-run: print package status without using sudo.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

is_installed() {
  local pkg="$1"
  dpkg -s "$pkg" 2>/dev/null | grep -q "Status: install ok installed"
}

is_in_cache() {
  local pkg="$1"
  apt-cache show "$pkg" >/dev/null 2>&1
}

package_status() {
  local pkg="$1"
  if is_installed "$pkg"; then
    echo "installed"
  elif is_in_cache "$pkg"; then
    echo "to-install"
  else
    echo "not-in-cache"
  fi
}

print_table() {
  local package_w=7
  local status_w=6
  local pkg status

  for pkg in "${PACKAGES[@]}"; do
    (( ${#pkg} > package_w )) && package_w=${#pkg}
    status="$(package_status "$pkg")"
    (( ${#status} > status_w )) && status_w=${#status}
  done

  printf "%-${package_w}s | %-${status_w}s\n" "Package" "Status"
  printf "%-${package_w}s-+-%-${status_w}s\n" \
    "$(printf '%*s' "$package_w" '' | tr ' ' '-')" \
    "$(printf '%*s' "$status_w" '' | tr ' ' '-')"

  for pkg in "${PACKAGES[@]}"; do
    status="$(package_status "$pkg")"
    printf "%-${package_w}s | %-${status_w}s\n" "$pkg" "$status"
  done
}

MODE="dry-run"

if [[ $# -gt 1 ]]; then
  usage >&2
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
      usage >&2
      exit 1
      ;;
  esac
fi

print_table
echo "Note: librealsense2-dkms is optional."

dev_status="$(package_status librealsense2-dev)"

if [[ "$MODE" == "dry-run" ]]; then
  if [[ "$dev_status" == "to-install" ]]; then
    echo "Run with --apply to install via sudo apt install -y"
  fi
  exit 0
fi

[[ -f "$PPA_FILE" ]] || die "Intel PPA not configured. See https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md"

echo "About to: sudo apt install -y librealsense2-dev. Type yes to proceed:"
read -r confirmation
if [[ "$confirmation" != "yes" ]]; then
  echo "Aborted"
  exit 2
fi

sudo apt update
sudo apt install -y librealsense2-dev

pkg-config --modversion realsense2 >/dev/null 2>&1 || die "pkg-config --modversion realsense2 failed after install"

echo "librealsense2-dev install verified"
exit 0
