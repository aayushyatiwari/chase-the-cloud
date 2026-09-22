# Image for chase-the-cloud (INSAT cloud-motion forecasting).
#
#   docker build -t chase-the-cloud:cu118 .
#   docker run --gpus all -it -v /path/to/persistent:/workspace chase-the-cloud:cu118
#
# ---------------------------------------------------------------------------
# Driver constraint -- please read before changing any version here:
#
# The target node runs NVIDIA driver 470.256.02 (caps at CUDA 11.4) on 2x
# Quadro RTX 6000. PyTorch is therefore pinned to a CUDA 11.8 build, which
# runs on driver 470 via CUDA minor-version compatibility (needs >= 450.80.02).
#
# A CUDA 12.x build REQUIRES driver >= 525 and will fail at runtime with
# "CUDA driver version is insufficient for CUDA runtime version". If the
# driver is upgraded to 525+, torch can move to a cu12x build -- otherwise
# please leave the pins in environment.yml as they are.
#
# No CUDA toolkit is needed in the image: the pip wheels vendor the CUDA 11.8
# runtime, and the driver is injected at runtime by nvidia-container-toolkit.
# That is why this uses a plain miniconda base rather than an nvidia/cuda one.
# ---------------------------------------------------------------------------

FROM continuumio/miniconda3:24.9.2-0

# git: wandb reads the repo commit for run provenance.
# libgl1 + libglib2.0-0: matplotlib/Pillow image backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Build the env as its own layer so it is cached across code changes.
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Make the env the default interpreter for both RUN and an interactive shell,
# so no `conda activate` is needed on exec.
ENV PATH=/opt/conda/envs/chase-the-cloud/bin:$PATH
ENV CONDA_DEFAULT_ENV=chase-the-cloud

# Fail the build here rather than at the user's first training run.
# Only checks that the CUDA 11.8 build is present -- no GPU is visible at
# build time, so is_available() cannot be tested until the container runs.
RUN python -c "import torch, h5py, netCDF4, wandb, yaml, matplotlib; \
assert torch.version.cuda == '11.8', f'wrong CUDA build: {torch.version.cuda}'; \
print('build ok:', torch.__version__)"

CMD ["/bin/bash"]
