#!/usr/bin/env bash
# Cross-compile bench_native for the reCamera Pro (RV1126B, aarch64).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="${RECAMERA_SDK_ROOT:-/home/seeed/recamera_pro/c_sdk/recamera-pro-sdk-v1.0.10}"
TC="$SDK_ROOT/tools/linux/toolchain/aarch64-rockchip1240-linux-gnu"
export RECAMERA_SYSROOT="$TC/aarch64-rockchip1240-linux-gnu/sysroot"
export CC="$TC/bin/aarch64-rockchip1240-linux-gnu-gcc"
export CXX="$TC/bin/aarch64-rockchip1240-linux-gnu-g++"
# headers + link libraries (RKNN Runtime / RGA); fetched once into native/libs
[ -f "$HERE/libs/librknnrt.so" ] && [ -f "$HERE/libs/librga.so" ] || "$HERE/fetch_libs.sh"
BUILD="$HERE/build"
mkdir -p "$BUILD"
cmake -S "$HERE" -B "$BUILD" \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" -j"$(nproc)"
# The board resolves vendor libs from /oem/usr/lib; never leak host paths.
"$TC/bin/aarch64-rockchip1240-linux-gnu-readelf" -d "$BUILD/bench_native" | grep -E "RPATH|RUNPATH" || true
echo "built: $BUILD/bench_native"
