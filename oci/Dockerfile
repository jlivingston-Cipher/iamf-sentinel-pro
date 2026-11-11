# ============================================================================
# BUILT + SMOKE-TESTED, STAGE 2 LANDED (docs 36 + 38, July 21, 2026) —
# supersedes the doc-35 D-S9 "UNBUILT DRAFT" status. Doc 36 built stage 1 and
# smoke-tested end-to-end with a bind-mounted toolchain (one edit: the
# `COPY --from=oss` named context, doc 36 D-O1). Doc 38 landed stage 2: the
# pinned toolchain + its 11-shared-lib bundle ride IN the image (export-import
# realization, `tc` named context below), and the smoke battery re-ran with
# NO bind mounts: containerized `validate --l3` byte-identical to host on the
# F4 pair, kernel↔reference differential 10/10 across the container boundary.
# Doc 39 realized path (a): stage 2 can now build the toolchain from source
# in-image (TOOLCHAIN_REALIZATION=source — no `tc` context needed), proven
# equivalent to the export-import image on the full battery (3-way byte-
# identical `validate --l3`, differential 10/10, contracts held).
# Doc 41 landed runtime hygiene: the repo trees now enter only the discarded
# `pybuild` stage, which builds the two wheels; the runtime stage installs
# them via a BuildKit mount (offline pip). No source tree, pip build/
# artifact, kernel C++ source, or oci/ machinery reaches any runtime layer
# (the doc-40 D-R6 release note, executed; IP posture per docs 26/32).
# Caveat for release CI (docs 36/38): sandbox builds used locally-constructed
# stand-in base images (registries blocked there) — rebuild against the real
# pinned bases.
# ============================================================================
#
# The Pro deployable (ADR-6 → ADR-6b): pinned toolchain + Sentinel, with the
# Python scientific stack REMOVED from the runtime layer. Measurement runs on
# the compiled sentinel-dsp kernel; dsp.py stays present as the reference but
# never imports numpy in this image (backend switch, doc 35). The adm-compare
# variant (numpy + EAR venv) is a separate, fatter image — see the note at the
# bottom.

# Stage-2 realization selector (doc 39 D-A1): must be declared before the
# first FROM so the `FROM toolchain-${TOOLCHAIN_REALIZATION}` line can see it.
#   import (default) = doc-38 export-import (needs --build-context tc=…)
#   source           = doc-39 in-image source build (self-contained, no tc)
ARG TOOLCHAIN_REALIZATION=import

# ---- Stage 1: kernel ------------------------------------------------------
FROM debian:bookworm-slim AS kernel
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make \
    && rm -rf /var/lib/apt/lists/*
COPY sentinel-dsp /src/sentinel-dsp
RUN cmake -S /src/sentinel-dsp -B /build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j

# ---- Stage 2: pinned toolchain --------------------------------------------
# The reference oracles (decoder_main @ iamf-tools v4.0.0, iamfdec @ libiamf
# v2.0.1, ffmpeg n8.1.3 -- the Mac's toolchain, IA-0072) built by
# oci/toolchain-build.sh; iamf-tools by its own build_cmake.sh (until IA-0072:
# 93597f1 / v1.1.0 / n8.1.2 through our CMake port, docs 10/15). Two sanctioned
# realizations, selected by TOOLCHAIN_REALIZATION (docs 36→38→39):
#   (a) source (doc 39, executed — release CI, self-contained): stage
#       `toolchain-source` runs oci/toolchain-build.sh (S0–S8 + prune,
#       ≈25–45 min) against the pins above; the lib/ bundle is the ldd
#       closure of the three binaries minus the glibc-family allowlist (rule
#       in the script) — no `tc` context needed:
#         docker build -f oci/Dockerfile \
#           --build-context oss=../sentinel-oss \
#           --build-arg TOOLCHAIN_REALIZATION=source .
#       Pin verification (in-image decode round-trip) required before tagging;
#       the script's S8 now does one (encode, decode with both oracles, refuse
#       a silent result), and .github/workflows/toolchain.yml runs the script
#       in debian:trixie on every change to it.
#   (b) import (doc 38, executed — the default): COPY a pruned, validated
#       host build via a named context — build with
#         docker build -f oci/Dockerfile \
#           --build-context oss=../sentinel-oss \
#           --build-context tc=<toolchain-export> .
#       where <toolchain-export> holds exactly:
#         src/build-iamf/decoder_main
#         src/build-iamfdec/iamfdec
#         bin/ffmpeg-install/bin/ffmpeg     (Toolchain.discover layout)
#         lib/                              ← the bundled shared libs
#       The lib/ bundle is doc 36's 11: libopus.so.0 libFLAC.so.12
#       libogg.so.0 libfdk-aac.so.2 libiamf.so libxcb.so.1 libxcb-shm.so.0
#       libXau.so.6 libXdmcp.so.6 libbsd.so.0 libmd.so.0. libiamf.so comes
#       from the toolchain build itself; the rest map to distro packages:
#       libopus0 libflac12* libogg0 libfdk-aac2 libxcb1 libxcb-shm0 libxau6
#       libxdmcp6 libbsd0 libmd0. (Path (a) bundles fewer when ffmpeg's
#       configure detects no x11/xcb in the clean stage — doc 39 D-A4.)
# TOOLCHAIN_REALIZATION is declared globally above stage 1 (FROM-line scope).

FROM debian:bookworm-slim AS toolchain-import
COPY --from=tc . /opt/iamf-toolchain

# trixie, not bookworm (IA-0072): the runtime stage's python:3.11-slim IS Debian trixie
# (same digest as 3.11-slim-trixie, read 2026-11-11), so the oracles are now built against the
# glibc they run on; and bookworm's GCC 12.2 cannot compile iamf-tools v4.0.0 (an ambiguous
# absl::NoDestructor brace-init in iamf/cli/channel_label.cc; toolchain.yml run 43120145811).
FROM debian:trixie-slim AS toolchain-source
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl cmake make g++ nasm pkg-config zlib1g-dev \
      libopus-dev libflac-dev libexpat1-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY oci/toolchain-build.sh /usr/local/bin/toolchain-build.sh
RUN TCB_SKIP_APT=1 \
    bash /usr/local/bin/toolchain-build.sh /opt/iamf-toolchain /tmp/tc-build \
    && rm -rf /tmp/tc-build

FROM toolchain-${TOOLCHAIN_REALIZATION} AS toolchain

# ---- Stage 3: pybuild (doc 41 D-H1) ---------------------------------------
# The two repo trees enter THIS stage only; it builds the two wheels and is
# then discarded. Under layered images, install-then-prune in the runtime
# stage could never achieve the IP goal (a later `rm` does not unpublish an
# earlier COPY layer), so hygiene is realized as build-stage isolation.
# The OSS core is a *named context* (doc 36 D-O1): the two repos are siblings
# after the D1 split, so no single context contains both. Build with
#   docker build -f oci/Dockerfile --build-context oss=../sentinel-oss .
# (from the Pro repo root; in the monorepo: --build-context oss=sentinel-oss,
# context sentinel-pro). Mirrors the dual-checkout in dsp-differential.yml.
# This stage is the only pip step that needs the network (build backend).
FROM python:3.11-slim AS pybuild
COPY --from=oss . /src/sentinel-oss
COPY . /src/sentinel-pro
RUN pip wheel --no-deps --no-cache-dir --wheel-dir /wheels \
      /src/sentinel-oss /src/sentinel-pro

# ---- Stage 4: runtime -----------------------------------------------------
FROM python:3.11-slim
# No numpy, no scipy: installing the two wheels brings only stdlib-dependent
# code (iamf-sentinel has zero deps; iamf-sentinel-pro's numpy is an extra
# since doc 35 — deliberately NOT installed here). The wheels arrive via a
# BuildKit mount from the pybuild stage — nothing but site-packages content
# and the `sentinel` entry point lands in this layer, and pip runs offline
# (--no-index): the wheel set is closed (pro depends only on the oss wheel).
RUN --mount=type=bind,from=pybuild,source=/wheels,target=/wheels \
    pip install --no-index --no-cache-dir /wheels/*.whl
COPY --from=kernel /build/sentinel-dsp /usr/local/bin/sentinel-dsp
COPY --from=toolchain /opt/iamf-toolchain /opt/iamf-toolchain
ENV SENTINEL_TOOLCHAIN=/opt/iamf-toolchain \
    LD_LIBRARY_PATH=/opt/iamf-toolchain/lib
# sentinel-dsp is on PATH → dsp.analyze() selects the kernel backend; a
# missing/broken kernel in this image is an execution error, never a silent
# numpy fallback (there is no numpy to fall back to — by design).
ENTRYPOINT ["sentinel"]
CMD ["--help"]

# adm-compare variant: realized as oci/Dockerfile.adm (doc 40 — landed, not
# published in v1). FROM this image: `pip install numpy>=1.24 scipy>=1.10`
# (scipy is required — adm_compare's matched filter imports scipy.signal;
# the earlier numpy-only note here was wrong, caught by executing the stage)
# plus the EAR venv (ear==2.1.0 with setuptools<81 — the F30 recipe, doc 31)
# at /opt/earvenv. Kept out of the default image: it is exactly the
# install-rot surface ADR-6b shed from the measurement path.
