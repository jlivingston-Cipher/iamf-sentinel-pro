#!/bin/bash
# oci/toolchain-build.sh — path (a): build the pinned IAMF toolchain from source
# and assemble the Toolchain.discover layout, self-contained (doc 39).
#
# Pins (IA-0072, 2026-11-11: one toolchain on every machine, the same as the
# Mac's):
#   iamf-tools v4.0.0 (28e5b07), built by its OWN build_cmake.sh, which pins
#     abseil 20260526.0 · protobuf v36.0 · fdk-aac ee76460 · loudness_ebur128
#     e9e7314 · obr 478dc7c · audio-to-tactile d3f449f · pffft a4b0359
#   libiamf v2.0.1 (63cafdd; OAR submodule a7a5cc8), static, iamfdec built
#     in-tree (IAMF_TEST_TOOL=ON) with libiamf's own codec builds
#   FFmpeg n8.1.3 (1041abd, GitHub mirror)
# These replace (docs 10/15/39): iamf-tools 93597f1 through our CMake port
# (upstreamed as iamf-tools' build_cmake.sh, #78/#80; the port repository is
# archived) with four UNPINNED clones (loudness_ebur128, obr, audio-to-tactile,
# pffft) · libiamf v1.1.0 and its iamfdec two-step (F26) · FFmpeg n8.1.2.
# Measured on the Mac before the move (s324): v4.0.0 gives the same bytes as
# v3.0.0 on every Loom golden encode and decode and on the ADM corpus.
#
# Usage: toolchain-build.sh [OUT] [WORK]
#   OUT   assembled toolchain root (default /opt/iamf-toolchain)
#   WORK  scratch build root       (default /tmp/tc-build)
# Env:
#   TCB_SKIP_APT  =1 to skip the S0 apt step (deps preinstalled, e.g. a cached
#                 docker layer)
#   JOBS          parallel build jobs (default: nproc)
#
# Output layout (== Toolchain.discover, docs 36/38):
#   $OUT/src/build-iamf/decoder_main
#   $OUT/src/build-iamfdec/iamfdec
#   $OUT/bin/ffmpeg-install/bin/ffmpeg
#   $OUT/lib/                       <- shared-lib bundle (rule below)
#
# Lib-bundle rule (doc 39 D-A4, self-contained): transitive ldd closure of the
# three binaries minus the glibc-family allowlist.
set -uo pipefail
OUT=${1:-/opt/iamf-toolchain}
WORK=${2:-/tmp/tc-build}
SRC=$WORK/src
LOG=$WORK/build-logs
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 2)}
mkdir -p "$SRC" "$LOG" "$OUT"
export DEBIAN_FRONTEND=noninteractive
# libiamf's codec stage unpacks flac-1.4.2.tar.xz with CMake, which refuses
# under the C locale ("Pathname can't be converted from UTF-8").
export LANG=C.UTF-8 LC_ALL=C.UTF-8
IAMF_TOOLS_TAG=v4.0.0;  IAMF_TOOLS_SHA=28e5b072c669cbd9959c84284cedde5e15bca982
LIBIAMF_TAG=v2.0.1;     LIBIAMF_SHA=63cafdd9b2d7bdba08cfeec2f008718198d7362e
FFMPEG_TAG=n8.1.3;      FFMPEG_SHA=1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7
# libiamf v2 needs CMake >= 3.28; Debian bookworm ships 3.25. Kitware's
# release binary is used when the system one is older (sha256 from the
# release's cmake-3.31.6-SHA-256.txt, read 2026-11-11).
CMAKE_FALLBACK_VER=3.31.6
CMAKE_FALLBACK_SHA_x86_64=5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf
CMAKE_FALLBACK_SHA_aarch64=b4cc788d63112b2749b40627e719eb5d3b8ed8f00c36d77189f4019cfe64bc9e
stage() { echo "=== TCB STAGE $1 [$(date +%H:%M:%S)] ==="; }
fail() { # fail NAME [LOGFILE] — dump the log tail so the docker build log carries it
  echo "TCB-FAIL: $1"
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---"; tail -40 "$2"; }
  exit 1
}

# ---------- S0: build dependencies ----------
if [ "${TCB_SKIP_APT:-0}" != 1 ]; then
  stage s0-apt
  apt-get update > "$LOG/apt.log" 2>&1 || fail apt-update "$LOG/apt.log"
  apt-get install -y --no-install-recommends git ca-certificates curl cmake make g++ \
    nasm pkg-config zlib1g-dev libopus-dev libflac-dev libexpat1-dev libeigen3-dev \
    >> "$LOG/apt.log" 2>&1 || fail apt-install "$LOG/apt.log"
fi
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
CMV=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
if ! ver_ge "${CMV:-0}" 3.28; then
  stage s0-cmake-fallback
  arch=$(uname -m); eval "sha=\${CMAKE_FALLBACK_SHA_$arch:-}"
  [ -n "$sha" ] || fail "cmake ${CMV:-none} < 3.28 and no fallback for $arch"
  t=cmake-$CMAKE_FALLBACK_VER-linux-$arch
  curl -fsSL -o "$WORK/$t.tar.gz" \
    "https://github.com/Kitware/CMake/releases/download/v$CMAKE_FALLBACK_VER/$t.tar.gz" || fail cmake-download
  echo "$sha  $WORK/$t.tar.gz" | sha256sum -c - > "$LOG/cmake-sha.log" 2>&1 || fail cmake-sha "$LOG/cmake-sha.log"
  tar -xzf "$WORK/$t.tar.gz" -C "$WORK" && rm -f "$WORK/$t.tar.gz"
  export PATH="$WORK/$t/bin:$PATH"
fi
echo "cmake: $(cmake --version | head -1)"

# ---------- S1: clones, each checked against its pinned commit ----------
stage s1-clones
git config --global advice.detachedHead false
pinned() { # name expected_sha
  local got; got=$(git -C "$SRC/$1" rev-parse HEAD 2>/dev/null)
  [ "$got" = "$2" ] || fail "pin $1: HEAD $got, expected $2"
  echo "clone $1: $got"
}
[ -d "$SRC/iamf-tools/.git" ] || git clone --depth 1 --branch "$IAMF_TOOLS_TAG" \
  https://github.com/AOMediaCodec/iamf-tools "$SRC/iamf-tools" > "$LOG/clone-iamf-tools.log" 2>&1 \
  || fail clone-iamftools "$LOG/clone-iamf-tools.log"
pinned iamf-tools "$IAMF_TOOLS_SHA"
# libiamf: code/ only (a full clone with its conformance vectors is ~8 GB);
# the OAR submodule comes with it.
if [ ! -d "$SRC/libiamf/.git" ]; then
  { git clone --depth 1 --branch "$LIBIAMF_TAG" --filter=blob:none --sparse \
      https://github.com/AOMediaCodec/libiamf "$SRC/libiamf" \
    && git -C "$SRC/libiamf" sparse-checkout set code \
    && git -C "$SRC/libiamf" submodule update --init --recursive --depth 1; } \
    > "$LOG/clone-libiamf.log" 2>&1 || fail clone-libiamf "$LOG/clone-libiamf.log"
fi
pinned libiamf "$LIBIAMF_SHA"
[ -d "$SRC/FFmpeg/.git" ] || git clone --depth 1 --branch "$FFMPEG_TAG" \
  https://github.com/FFmpeg/FFmpeg "$SRC/FFmpeg" > "$LOG/clone-ffmpeg.log" 2>&1 \
  || fail clone-ffmpeg "$LOG/clone-ffmpeg.log"
pinned FFmpeg "$FFMPEG_SHA"

# ---------- S2–S5: iamf-tools by its own build_cmake.sh ----------
# It clones and builds abseil and protobuf, makes the header shims, and builds
# encoder_main / decoder_main / probe_main into $WORK/it/src/build-iamf.
stage s2-iamf-tools
JOBS=$JOBS bash "$SRC/iamf-tools/build_cmake.sh" "$WORK/it" > "$LOG/iamf-tools.log" 2>&1 \
  || fail iamf-tools-build_cmake "$LOG/iamf-tools.log"
IT=$WORK/it/src/build-iamf
for t in encoder_main decoder_main probe_main; do
  [ -x "$IT/$t" ] || fail "missing-$t" "$LOG/iamf-tools.log"
done
grep -E '^clone ' "$LOG/iamf-tools.log" | sed 's/^/  build_cmake.sh /'

# ---------- S6: libiamf + iamfdec (in-tree test tool) ----------
stage s6-libiamf
cflags=""
case "$(uname -m)" in
  # OAR at a7a5cc8 compiles its NEON matrix render to a bare `ret` unless this
  # macro is defined on the command line (measured on arm64 at v2.0.0, s317).
  arm64|aarch64) cflags=-Ddef_oar_arch_arm ;;
esac
cmake -S "$SRC/libiamf/code" -B "$WORK/build-libiamf" -DCMAKE_BUILD_TYPE=Release \
  -DIAMF_TEST_TOOL=ON -DIAMF_BUILD_SHARED_LIB=OFF ${cflags:+-DCMAKE_C_FLAGS=$cflags} \
  > "$LOG/libiamf-cfg.log" 2>&1 || fail libiamf-cfg "$LOG/libiamf-cfg.log"
cmake --build "$WORK/build-libiamf" -j"$JOBS" > "$LOG/libiamf-build.log" 2>&1 \
  || fail libiamf-build "$LOG/libiamf-build.log"
IAMFDEC=$WORK/build-libiamf/test/tools/iamfdec/iamfdec
[ -x "$IAMFDEC" ] || fail iamfdec-missing "$LOG/libiamf-build.log"

# ---------- S7: FFmpeg (lean LGPL) ----------
stage s7-ffmpeg
( cd "$SRC/FFmpeg" && \
  ./configure --prefix="$OUT/bin/ffmpeg-install" --disable-doc --disable-ffplay \
    --disable-debug --enable-libopus > "$LOG/ff-cfg.log" 2>&1 && \
  make -j"$JOBS" > "$LOG/ff-build.log" 2>&1 && \
  make install > "$LOG/ff-install.log" 2>&1 ) || fail ffmpeg "$LOG/ff-cfg.log"

# ---------- S8: pin verification — encode, then decode with both oracles ----------
# iamf-tools' own LPCM and Opus test vectors through encoder_main, each decoded
# to stereo by decoder_main and by iamfdec; a decode that is silent fails the
# build (the OAR NEON defect above wrote all-zero samples and exited 0).
stage s8-smoke
TD=$SRC/iamf-tools/iamf/cli/testdata; SM=$WORK/smoke; mkdir -p "$SM"
FF=$OUT/bin/ffmpeg-install/bin/ffmpeg
peak() { "$FF" -hide_banner -nostats -i "$1" -af volumedetect -f null - 2>&1 \
         | awk '/max_volume/{print $5}'; }
for t in test_000002 test_000021; do
  "$IT/encoder_main" --user_metadata_filename="$TD/$t.textproto" \
    --input_wav_directory="$TD" --output_iamf_directory="$SM" > "$LOG/smoke-$t-enc.log" 2>&1
  [ -s "$SM/$t.iamf" ] || fail "smoke $t: encoder_main wrote nothing" "$LOG/smoke-$t-enc.log"
  "$IT/decoder_main" --input_filename="$SM/$t.iamf" --output_filename="$SM/$t.dm.wav" \
    --output_layout=2.0 > "$LOG/smoke-$t-dm.log" 2>&1 || fail "smoke $t: decoder_main" "$LOG/smoke-$t-dm.log"
  "$IAMFDEC" -i0 -s0 -disable_limiter -o3 "$SM/$t.id.wav" "$SM/$t.iamf" > "$LOG/smoke-$t-id.log" 2>&1 \
    || fail "smoke $t: iamfdec" "$LOG/smoke-$t-id.log"
  for w in dm id; do
    p=$(peak "$SM/$t.$w.wav")
    echo "smoke $t $w: iamf $(sha256sum "$SM/$t.iamf" | cut -c1-16) wav $(sha256sum "$SM/$t.$w.wav" | cut -c1-16) peak ${p:-?} dB"
    awk -v p="${p:--inf}" 'BEGIN{exit !(p+0 > -60)}' || fail "smoke $t $w: silent or unreadable (peak ${p:-?})"
  done
done

# ---------- Prune/assemble: the Toolchain.discover layout + lib bundle ----------
stage assemble
mkdir -p "$OUT/src/build-iamf" "$OUT/src/build-iamfdec" "$OUT/lib"
cp "$IT/decoder_main" "$OUT/src/build-iamf/decoder_main"
cp "$IAMFDEC" "$OUT/src/build-iamfdec/iamfdec"
[ -x "$OUT/bin/ffmpeg-install/bin/ffmpeg" ] || fail ffmpeg-not-installed
# trim ffmpeg-install to the single discover binary (== the doc-38 export shape)
find "$OUT/bin/ffmpeg-install" -mindepth 1 -maxdepth 1 ! -name bin -exec rm -rf {} +
find "$OUT/bin/ffmpeg-install/bin" -mindepth 1 ! -name ffmpeg -exec rm -rf {} +
ALLOW="ld-linux-x86-64.so.2 ld-linux-aarch64.so.1 libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libgcc_s.so.1 libstdc++.so.6 libz.so.1"
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
  [ -x "$OUT/$f" ] || fail "manifest-$f"
  echo "  OK $f $(sha256sum "$OUT/$f" | cut -c1-16)"
done
echo "  pins: iamf-tools $IAMF_TOOLS_TAG $IAMF_TOOLS_SHA · libiamf $LIBIAMF_TAG $LIBIAMF_SHA · FFmpeg $FFMPEG_TAG $FFMPEG_SHA"
ls "$OUT/lib" | sed 's/^/  lib: /'
echo "TCB COMPLETE $(date +%H:%M:%S)"
