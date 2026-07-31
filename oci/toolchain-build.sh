#!/bin/bash
# oci/toolchain-build.sh — path (a): build the pinned IAMF toolchain from source
# and assemble the Toolchain.discover layout, self-contained (doc 39).
#
# A faithful adaptation of wp3-scripts/build_toolchain.sh S0–S7 (docs 10/15) +
# toolchain-addendum.sh (the iamfdec two-step incl. the /usr/local prefix quirk),
# plus the prune/assembly step that the doc-38 export-import realization performed
# host-side. S8 (essence generation) is deliberately omitted: it serves WP1
# experiments, not the oracle binaries.
#
# Pins (docs 10/15, unchanged): iamf-tools 93597f1 · libiamf v1.1.0 ·
# FFmpeg n8.1.2 (GitHub mirror) · abseil 20260107.1 · protobuf v33.5 ·
# fdk-aac ee76460 as source pin iff apt libfdk-aac-dev is unavailable.
#
# Usage: toolchain-build.sh [OUT] [WORK]
#   OUT   assembled toolchain root (default /opt/iamf-toolchain)
#   WORK  scratch build root       (default /tmp/tc-build)
# Env:
#   PORT_CMAKE    path to the iamf-tools CMake port file (default: port/CMakeLists.txt
#                 next to this script)
#   TCB_SKIP_APT  =1 to skip the S0 apt step (deps preinstalled, e.g. a cached
#                 docker layer)
#
# Output layout (== Toolchain.discover, docs 36/38):
#   $OUT/src/build-iamf/decoder_main
#   $OUT/src/build-iamfdec/iamfdec
#   $OUT/bin/ffmpeg-install/bin/ffmpeg
#   $OUT/lib/                       <- shared-lib bundle (rule below)
#
# Lib-bundle rule (doc 39 D-A4, self-contained): transitive ldd closure of the
# three binaries minus the glibc-family allowlist. In this stage every lib
# resolves, so ldd reports the full closure (the doc-38 in-container
# under-reporting applied to a runtime image with libs missing).
set -uo pipefail
OUT=${1:-/opt/iamf-toolchain}
WORK=${2:-/tmp/tc-build}
HERE=$(cd "$(dirname "$0")" && pwd)
PORT_CMAKE=${PORT_CMAKE:-$HERE/port/CMakeLists.txt}
SRC=$WORK/src
DEPS=$WORK/deps
LOG=$WORK/build-logs
mkdir -p "$SRC" "$DEPS" "$LOG" "$OUT"
export DEBIAN_FRONTEND=noninteractive
stage() { echo "=== TCB STAGE $1 [$(date +%H:%M:%S)] ==="; }
fail() { # fail NAME [LOGFILE] — dump the log tail so the docker build log carries it
  echo "TCB-FAIL: $1"
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---"; tail -40 "$2"; }
  exit 1
}

# ---------- S0: build dependencies ----------
FDK_FROM_APT=1
if [ "${TCB_SKIP_APT:-0}" != 1 ]; then
  stage s0-apt
  apt-get update > "$LOG/apt.log" 2>&1 || fail apt-update "$LOG/apt.log"
  apt-get install -y --no-install-recommends git ca-certificates cmake make g++ \
    nasm pkg-config zlib1g-dev libopus-dev libflac-dev libexpat1-dev libeigen3-dev \
    >> "$LOG/apt.log" 2>&1 || fail apt-install "$LOG/apt.log"
  apt-get install -y --no-install-recommends libfdk-aac-dev >> "$LOG/apt.log" 2>&1 \
    || { echo "note: libfdk-aac-dev unavailable (using source pin)"; FDK_FROM_APT=0; }
else
  dpkg -s libfdk-aac-dev > /dev/null 2>&1 || FDK_FROM_APT=0
fi

# ---------- S1: clones (pins == build_toolchain.sh) ----------
stage s1-clones
git config --global advice.detachedHead false
clone() { # name url ref shallow(y/n)
  local name=$1 url=$2 ref=$3 shallow=$4
  [ -d "$SRC/$name/.git" ] && { echo "clone $name: exists"; return 0; }
  if [ "$shallow" = y ]; then
    git clone --depth 1 ${ref:+--branch $ref} "$url" "$SRC/$name" > "$LOG/clone-$name.log" 2>&1
  else
    git clone "$url" "$SRC/$name" > "$LOG/clone-$name.log" 2>&1 && \
      ( cd "$SRC/$name" && git checkout "$ref" >> "$LOG/clone-$name.log" 2>&1 )
  fi
  [ -d "$SRC/$name" ] && echo "clone $name: OK" || { echo "clone $name: FAILED"; return 1; }
}
clone abseil-cpp https://github.com/abseil/abseil-cpp 20260107.1 y || \
  clone abseil-cpp https://github.com/abseil/abseil-cpp "" n || fail clone-absl
clone protobuf https://github.com/protocolbuffers/protobuf v33.5 y || \
  clone protobuf https://github.com/protocolbuffers/protobuf "" n || fail clone-pb
clone iamf-tools https://github.com/AOMediaCodec/iamf-tools 93597f1 n || fail clone-iamftools
# fdk_aac is ALWAYS cloned: the iamf-tools port compiles it from the source pin
# (fdk_aac_static target); the apt package only feeds libiamf's dep_codecs (S6).
clone fdk_aac https://github.com/mstorsjo/fdk-aac ee76460 n || fail clone-fdk
clone loudness_ebur128 https://github.com/google/loudness_ebur128 "" y || fail clone-lebur
clone obr https://github.com/google/obr "" y || fail clone-obr
clone audio_to_tactile https://github.com/google/audio-to-tactile "" y || fail clone-a2t
clone pffft https://github.com/marton78/pffft "" y || fail clone-pffft
clone libiamf https://github.com/AOMediaCodec/libiamf v1.1.0 y || fail clone-libiamf
clone FFmpeg https://github.com/FFmpeg/FFmpeg n8.1.2 y || fail clone-ffmpeg
( cd "$SRC/protobuf" && git submodule update --init --depth 1 third_party/jsoncpp 2>/dev/null; true )

# ---------- S2: header shims (F11) + port file ----------
stage s2-shims
mkdir -p "$SRC/shims/flac/include" "$SRC/shims/opus/include" "$SRC/shims/expat/expat/lib"
ln -sf /usr/include/FLAC "$SRC/shims/flac/include/FLAC"
for h in /usr/include/opus/*.h; do ln -sf "$h" "$SRC/shims/opus/include/"; done
ln -sf /usr/include/expat.h "$SRC/shims/expat/expat/lib/expat.h"
ln -sf /usr/include/expat_external.h "$SRC/shims/expat/expat/lib/expat_external.h"
ln -sf /usr/include/expat_config.h "$SRC/shims/expat/expat/lib/expat_config.h" 2>/dev/null || true
mkdir -p "$SRC/port" && cp "$PORT_CMAKE" "$SRC/port/CMakeLists.txt" || fail port-file

# ---------- S3: abseil ----------
stage s3-abseil
cmake -S "$SRC/abseil-cpp" -B "$SRC/build-absl" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_CXX_STANDARD=20 -DABSL_PROPAGATE_CXX_STD=ON \
  -DBUILD_SHARED_LIBS=OFF -DABSL_BUILD_TESTING=OFF > "$LOG/absl-cfg.log" 2>&1 || fail absl-cfg "$LOG/absl-cfg.log"
cmake --build "$SRC/build-absl" -j2 --target install > "$LOG/absl-build.log" 2>&1 || fail absl-build "$LOG/absl-build.log"

# ---------- S4: protobuf ----------
stage s4-protobuf
cmake -S "$SRC/protobuf" -B "$SRC/build-protobuf" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_PREFIX_PATH="$DEPS" -DCMAKE_CXX_STANDARD=20 \
  -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
  -Dprotobuf_BUILD_SHARED_LIBS=OFF > "$LOG/pb-cfg.log" 2>&1 || fail pb-cfg "$LOG/pb-cfg.log"
cmake --build "$SRC/build-protobuf" -j2 --target install > "$LOG/pb-build.log" 2>&1 || fail pb-build "$LOG/pb-build.log"

# ---------- S5: iamf-tools via the CMake port ----------
stage s5-iamf-tools-port
cmake -S "$SRC/port" -B "$SRC/build-iamf" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$DEPS" > "$LOG/port-cfg.log" 2>&1 || fail port-cfg "$LOG/port-cfg.log"
cmake --build "$SRC/build-iamf" -j2 > "$LOG/port-build.log" 2>&1 || fail port-build "$LOG/port-build.log"
for t in encoder_main decoder_main probe_main; do
  [ -x "$SRC/build-iamf/$t" ] || fail "missing-$t"
done

# ---------- S6: libiamf ----------
stage s6-libiamf
LI=$SRC/libiamf/code
[ -d "$LI" ] || LI=$SRC/libiamf
DC=$LI/dep_codecs
mkdir -p "$DC/lib" "$DC/include"
for so in libopus libFLAC libfdk-aac; do
  for f in /usr/lib/x86_64-linux-gnu/$so.so* /usr/lib/x86_64-linux-gnu/$so.a; do
    [ -e "$f" ] && ln -sf "$f" "$DC/lib/"; done
done
ln -sfn /usr/include/opus "$DC/include/opus"
ln -sfn /usr/include/FLAC "$DC/include/FLAC"
[ -d /usr/include/fdk-aac ] && ln -sfn /usr/include/fdk-aac "$DC/include/fdk-aac"
cmake -S "$LI" -B "$SRC/build-libiamf" -DCMAKE_BUILD_TYPE=Release \
  -DIAMF_ENABLE_BINAURALIZER=OFF -DENABLE_BUILD_CODECS=OFF > "$LOG/libiamf-cfg.log" 2>&1 || fail libiamf-cfg "$LOG/libiamf-cfg.log"
cmake --build "$SRC/build-libiamf" -j2 > "$LOG/libiamf-build.log" 2>&1 || fail libiamf-build "$LOG/libiamf-build.log"

# ---------- S7: FFmpeg n8.1.2 (lean LGPL — flags == build_toolchain.sh) ----------
stage s7-ffmpeg
( cd "$SRC/FFmpeg" && \
  ./configure --prefix="$OUT/bin/ffmpeg-install" --disable-doc --disable-ffplay \
    --disable-debug --enable-libopus > "$LOG/ff-cfg.log" 2>&1 && \
  make -j2 > "$LOG/ff-build.log" 2>&1 && \
  make install > "$LOG/ff-install.log" 2>&1 ) || fail ffmpeg "$LOG/ff-cfg.log"

# ---------- Addendum: iamfdec two-step (F26; /usr/local prefix quirk) ----------
stage addendum-iamfdec
cmake --install "$SRC/build-libiamf" > "$LOG/libiamf-install.log" 2>&1 || fail libiamf-install "$LOG/libiamf-install.log"
ldconfig 2>/dev/null || true
cmake -S "$LI/test/tools/iamfdec" -B "$SRC/build-iamfdec" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local > "$LOG/iamfdec-cfg.log" 2>&1 || fail iamfdec-cfg "$LOG/iamfdec-cfg.log"
cmake --build "$SRC/build-iamfdec" -j2 > "$LOG/iamfdec-build.log" 2>&1 || fail iamfdec-build "$LOG/iamfdec-build.log"
[ -x "$SRC/build-iamfdec/iamfdec" ] || fail iamfdec-missing

# ---------- Prune/assemble: the Toolchain.discover layout + lib bundle ----------
stage assemble
mkdir -p "$OUT/src/build-iamf" "$OUT/src/build-iamfdec" "$OUT/lib"
cp "$SRC/build-iamf/decoder_main" "$OUT/src/build-iamf/decoder_main"
cp "$SRC/build-iamfdec/iamfdec" "$OUT/src/build-iamfdec/iamfdec"
[ -x "$OUT/bin/ffmpeg-install/bin/ffmpeg" ] || fail ffmpeg-not-installed
# trim ffmpeg-install to the single discover binary (== the doc-38 export shape)
find "$OUT/bin/ffmpeg-install" -mindepth 1 -maxdepth 1 ! -name bin -exec rm -rf {} +
find "$OUT/bin/ffmpeg-install/bin" -mindepth 1 ! -name ffmpeg -exec rm -rf {} +
ALLOW="ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libgcc_s.so.1 libstdc++.so.6 libz.so.1"
allowed() { local s; for s in $ALLOW; do [ "$1" = "$s" ] && return 0; done; return 1; }
BUNDLED=""
for bin in "$OUT/src/build-iamf/decoder_main" "$OUT/src/build-iamfdec/iamfdec" \
           "$OUT/bin/ffmpeg-install/bin/ffmpeg"; do
  while read -r name arrow path rest; do
    [ "$arrow" = "=>" ] && [ -e "$path" ] || continue
    allowed "$name" && continue
    case " $BUNDLED " in *" $name "*) continue;; esac
    cp -L "$path" "$OUT/lib/$name"
    BUNDLED="$BUNDLED $name"
  done < <(ldd "$bin")
done
echo "lib bundle:$BUNDLED"
echo "TCB manifest:"
for f in src/build-iamf/decoder_main src/build-iamfdec/iamfdec \
         bin/ffmpeg-install/bin/ffmpeg; do
  [ -x "$OUT/$f" ] && echo "  OK $f" || fail "manifest-$f"
done
ls "$OUT/lib" | sed 's/^/  lib: /'
echo "TCB COMPLETE $(date +%H:%M:%S)"
