#!/usr/bin/env bash
# Populate native/libs/ with the headers and shared libraries needed to
# cross-compile bench_native for the reCamera Pro (RV1126B, aarch64).
#
# Sources, in priority order:
#   librknnrt.so  $RECAMERA_RKNNRT -> recamera-rknn-dev skill asset -> the device
#   rknn_api.h    $RECAMERA_SDK_ROOT/project/app/recamera_ipc/common/rc_infer/include
#   rga headers   $RECAMERA_SDK_ROOT/media/rga/rga/{include,im2d_api}
#   librga.so     $RECAMERA_RGA -> the device (/oem/usr/lib/librga.so)
#
# librknnrt.so is verified against the SHA-256 of the Seeed-qualified RKNN 2.3.2
# runtime that ships on the board; a mismatch aborts the build.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS="$HERE/libs"
SDK_ROOT="${RECAMERA_SDK_ROOT:-/home/seeed/recamera_pro/c_sdk/recamera-pro-sdk-v1.0.10}"
SKILL_RUNTIME="${RECAMERA_SKILL_ASSETS:-$HOME/.agents/skills/recamera-rknn-dev/assets/recamera-pro-runtime}"
RKNNRT_SHA256="d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
DEVICE_IP="${DEVICE_IP:-192.168.42.1}"
DEVICE_USER="${DEVICE_USER:-root}"
DEVICE_PASS="${DEVICE_PASS:-recamera.1}"

mkdir -p "$LIBS/include"
sha() { sha256sum "$1" | cut -d' ' -f1; }
pull() {  # pull <remote path> <local path>
  sshpass -p "$DEVICE_PASS" scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR "$DEVICE_USER@$DEVICE_IP:$1" "$2"
}

# ---- librknnrt.so -----------------------------------------------------------
if [ ! -f "$LIBS/librknnrt.so" ]; then
  if [ -n "${RECAMERA_RKNNRT:-}" ] && [ -f "$RECAMERA_RKNNRT" ]; then
    cp "$RECAMERA_RKNNRT" "$LIBS/librknnrt.so"
  elif [ -f "$SKILL_RUNTIME/librknnrt.so" ]; then
    cp "$SKILL_RUNTIME/librknnrt.so" "$LIBS/librknnrt.so"
  else
    echo "[libs] pulling librknnrt.so from the device"
    pull /oem/usr/lib/librknnrt.so "$LIBS/librknnrt.so"
  fi
fi
if [ "$(sha "$LIBS/librknnrt.so")" != "$RKNNRT_SHA256" ]; then
  echo "[libs] FATAL: librknnrt.so SHA-256 mismatch" >&2
  echo "       expected $RKNNRT_SHA256" >&2
  echo "       got      $(sha "$LIBS/librknnrt.so")" >&2
  exit 1
fi
[ -f "$SKILL_RUNTIME/SHA256SUMS" ] && cp "$SKILL_RUNTIME/SHA256SUMS" "$LIBS/SHA256SUMS"

# ---- headers ----------------------------------------------------------------
RKNN_INC="$SDK_ROOT/project/app/recamera_ipc/common/rc_infer/include"
RGA_DIR="$SDK_ROOT/media/rga/rga"
if [ ! -f "$LIBS/include/rknn_api.h" ]; then
  if [ -f "$RKNN_INC/rknn_api.h" ]; then
    cp "$RKNN_INC/rknn_api.h" "$LIBS/include/"
  else
    echo "[libs] FATAL: rknn_api.h not found (set RECAMERA_SDK_ROOT)" >&2
    exit 1
  fi
fi
if [ ! -f "$LIBS/include/rga.h" ]; then
  if [ -d "$RGA_DIR/include" ]; then
    cp "$RGA_DIR"/include/*.h "$LIBS/include/"
    mkdir -p "$LIBS/include/im2d_api"
    cp "$RGA_DIR"/im2d_api/*.h "$RGA_DIR"/im2d_api/*.hpp "$LIBS/include/im2d_api/" 2>/dev/null || true
  else
    echo "[libs] FATAL: RGA headers not found (set RECAMERA_SDK_ROOT)" >&2
    exit 1
  fi
fi

# ---- librga.so --------------------------------------------------------------
if [ ! -f "$LIBS/librga.so" ]; then
  if [ -n "${RECAMERA_RGA:-}" ] && [ -f "$RECAMERA_RGA" ]; then
    cp "$RECAMERA_RGA" "$LIBS/librga.so"
  else
    echo "[libs] pulling librga.so from the device"
    pull /oem/usr/lib/librga.so "$LIBS/librga.so"
  fi
fi

echo "[libs] ready: $(ls "$LIBS" | tr '\n' ' ')"
