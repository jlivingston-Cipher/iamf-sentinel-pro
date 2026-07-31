# sentinel-dsp — static C++ kernel of the Sentinel Pro numeric core (ADR-6b).
# Build:   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# Static:  add -DSENTINEL_DSP_STATIC=ON (mostly-static glibc; use a musl
#          toolchain for fully-static releases per ADR-6).
cmake_minimum_required(VERSION 3.16)
project(sentinel_dsp CXX)

option(SENTINEL_DSP_STATIC "Link libstdc++/libgcc statically" ON)

add_executable(sentinel-dsp src/main.cpp)
set_target_properties(sentinel-dsp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(sentinel-dsp PRIVATE -O2 -Wall -Wextra)
if(SENTINEL_DSP_STATIC)
  target_link_options(sentinel-dsp PRIVATE -static-libstdc++ -static-libgcc)
endif()
target_link_libraries(sentinel-dsp m)
