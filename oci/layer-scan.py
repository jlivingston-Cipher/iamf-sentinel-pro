#!/usr/bin/env python3
"""oci/layer-scan.py — image-hygiene audit (doc 41 E-H3; release-CI tool).

Scans every layer of a `docker save` tarball and reports (a) violations of the
runtime-hygiene contract — no /opt/sentinel-{pro,oss} source trees, no kernel
C++ source, no pip build//*.egg-info artifacts, no oci/ build machinery in any
runtime layer (the doc-40 D-R6 note, executed by the pybuild wheel stage) —
and (b) presence of the required runtime set (site-packages trees, dist-infos,
profiles, `sentinel` CLI, kernel binary, toolchain discover layout + lib/).

Usage: docker save <image> -o img.tar && python3 oci/layer-scan.py img.tar
Note (stand-in bases only): Ubuntu-noble stand-ins carry the distro's own
apt python3-setuptools egg-info in the BASE layer — attributable, absent on
the real python:3.11-slim base (verified in doc 41).
"""
import tarfile
import json
import sys
import re
outer = tarfile.open(sys.argv[1])
manifest = json.load(outer.extractfile('manifest.json'))
paths = []
for layer in manifest[0]['Layers']:
    lt = tarfile.open(fileobj=outer.extractfile(layer))
    paths += [m.name for m in lt.getmembers()]
P = set(paths)
def match(rx): return sorted(p for p in P if re.search(rx, p))
viol = {}
viol['opt_source_trees'] = match(r'^opt/sentinel-(pro|oss)(/|$)')
viol['kernel_cpp_source'] = match(r'\.(cpp|cc|hpp)$|sentinel-dsp/(src|tests|CMakeLists)')
viol['egg_info_or_pip_build'] = match(r'\.egg-info|^opt/.*/build/')
viol['oci_machinery'] = match(r'(^|/)oci/')
req = {}
req['site_pkg_sentinel'] = bool(match(r'(dist|site)-packages/sentinel/__init__\.py$'))
req['site_pkg_sentinel_pro'] = bool(match(r'(dist|site)-packages/sentinel_pro/__init__\.py$'))
req['dist_infos'] = len(match(r'(dist|site)-packages/iamf_sentinel.*\.dist-info/METADATA$'))
req['profiles'] = len(match(r'(dist|site)-packages/sentinel(_pro)?/profiles/.*\.toml$'))
req['cli'] = bool(match(r'^usr/local/bin/sentinel$'))
req['kernel_bin'] = bool(match(r'^usr/local/bin/sentinel-dsp$'))
req['toolchain'] = [bool(match(r'^opt/iamf-toolchain/src/build-iamf/decoder_main$')),
                    bool(match(r'^opt/iamf-toolchain/src/build-iamfdec/iamfdec$')),
                    bool(match(r'^opt/iamf-toolchain/bin/ffmpeg-install/bin/ffmpeg$')),
                    len(match(r'^opt/iamf-toolchain/lib/'))]
print(json.dumps({'total_paths': len(P), 'violations': viol, 'required': req}, indent=1))
