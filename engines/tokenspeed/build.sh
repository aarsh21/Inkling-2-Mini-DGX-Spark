#!/usr/bin/env bash
# Build tokenspeed-inkling2mini:pr3682-52fdeee2: TokenSpeed at the head of lightseekorg/tokenspeed#3682
# ("day 0 support for Inkling 2 Mini", 52fdeee2) on the newest published image (nightly-c47e7a, 2026-11-16,
# no Inkling 2 Mini), whose CUDA toolchain and torch it reuses.
# The PR sits 156 commits after the image and its runtime imports a newer engine
# (tokenspeed_engine.scheduler), so a Python-only overlay fails at startup. This build therefore:
#   1. builds and installs the engine wheel (Rust ts-host + C++ scheduler) from the PR checkout,
#   2. rebuilds tokenspeed-kernel from the PR for sm_121a: the image's compiled kernels target only
#      sm_90a/100a/103a ("no kernel image is available for execution on the device" on GB10),
#   3. installs the PR's tokenspeed package with its pinned gateway (smg) dependencies.
# sm12x-skinny-mm.patch keeps GB10 off one Triton GEMM whose tiles need more shared memory than sm_12x has
# (147 KB of 99 KB); those calls take the generic GEMM path the PR already uses on other platforms.
set -euo pipefail
pr_head=52fdeee2a879cb04417183649855fbadb4fda8c4
here=$(cd "$(dirname "$0")" && pwd); work=$(mktemp -d)
git -C "$work" init -q src
git -C "$work/src" fetch -q --depth 1 https://github.com/lightseekorg/tokenspeed "$pr_head"
git -C "$work/src" checkout -q FETCH_HEAD
git -C "$work/src" apply "$here/sm12x-skinny-mm.patch"
cat > "$work/Dockerfile" <<'DOCKERFILE'
FROM lightseekorg/tokenspeed:nightly-c47e7a
ENV PIP_BREAK_SYSTEM_PACKAGES=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
COPY src /workspace
WORKDIR /workspace
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
RUN export PATH=/root/.cargo/bin:$PATH CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-12} \
 && WORKSPACE=/workspace bash test/ci_system/install_tokenspeed_engine.sh
RUN MAX_JOBS=${MAX_JOBS:-8} FLASHINFER_CUDA_ARCH_LIST="12.1a" TOKENSPEED_KERNEL_BACKEND=cuda \
    python3 -m pip install tokenspeed-kernel/python/ --no-build-isolation
RUN python3 -m pip install ./python --no-build-isolation \
 && python3 -c "import tokenspeed_engine.scheduler"   # the kernel package needs a GPU to import
WORKDIR /
LABEL org.opencontainers.image.source=https://github.com/lightseekorg/tokenspeed/pull/3682
DOCKERFILE
docker build --network host -t tokenspeed-inkling2mini:pr3682-52fdeee2 "$work"
rm -rf "$work"
