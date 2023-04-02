# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.
# The NVidia drivers need to *exactly* match between the host machine and the
# docker image.

FROM gcr.io/iree-oss/frontends@sha256:41e82bb62a3850c01d0252aa67aebd7816e47da0132af677cd70fc76731c4b5d

# We use .deb files that we host because we have to pin the version exactly to
# match the host machine and packages routinely dissapear from the Ubuntu
# apt repositories.
ARG NVIDIA_GL_DEB="libnvidia-gl-460_460.39-0ubuntu0.18.04.1_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-460_460.39-0ubuntu0.18.04.1_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-460_460.39-0ubuntu0.18.04.1_all.deb"

WORKDIR /install-nvidia
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMMON_DEB}" \
  && wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_GL_DEB}" \
  && wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMPUTE_DEB}" \
  && apt-get install "./${NVIDIA_COMMON_DEB}" \
  "./${NVIDIA_GL_DEB}" \
  "./${NVIDIA_COMPUTE_DEB}" \
  && rm -rf /install-nvidia

WORKDIR /
