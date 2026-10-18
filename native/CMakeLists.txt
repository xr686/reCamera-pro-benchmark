cmake_minimum_required(VERSION 3.16)
project(recamera_bench_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(RECAMERA_SYSROOT STREQUAL "")
  set(RECAMERA_SYSROOT "$ENV{RECAMERA_SYSROOT}")
endif()
if(RECAMERA_SYSROOT STREQUAL "")
  message(FATAL_ERROR "RECAMERA_SYSROOT must be set (build.sh does this)")
endif()

# Vendor headers/libs shipped by the reCamera Pro SDK but absent from the sysroot.
set(VENDOR_LIBDIR "${CMAKE_CURRENT_SOURCE_DIR}/libs")

# The board resolves vendor libs from /oem/usr/lib; never leak build-tree paths.
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "/oem/usr/lib")

add_compile_options(-O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti
                    -Wall -Wextra -Wno-unused-parameter)
add_link_options(-Wl,--gc-sections)

include_directories(SYSTEM ${VENDOR_LIBDIR}/include ${VENDOR_LIBDIR}/include/im2d_api)

add_executable(bench_native bench_native.cpp)
# Link the vendor libs by path so no host directory ends up in DT_RPATH; at
# runtime the board resolves both from /oem/usr/lib (see CMAKE_INSTALL_RPATH).
target_link_libraries(bench_native PRIVATE
  ${VENDOR_LIBDIR}/librknnrt.so
  ${VENDOR_LIBDIR}/librga.so
  pthread dl)
