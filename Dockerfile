FROM nvcr.io/nvidia/pytorch:25.04-py3

#####################################
# RCP Storage creds
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}
#####################################

RUN mkdir -p /workspace/lingua_modified
WORKDIR /workspace
ENV PYTHONPATH="/workspace:/workspace/lingua_modified"

COPY ./lingua_modified/requirements.txt /workspace/lingua_modified/requirements.txt
RUN pip install --no-cache-dir ninja

RUN pip install --no-cache-dir --no-deps \
    https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp312-cp312-manylinux_2_28_x86_64.whl
RUN pip install --no-cache-dir -r /workspace/lingua_modified/requirements.txt
# Use prebuilt wheel to avoid 10-30min source compile (nvcr pytorch:25.04 has PyTorch 2.7, Python 3.12, CUDA 12)
RUN pip install --no-cache-dir --no-deps \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

RUN python --version && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import xformers; print(f'xformers installed')" && \
    python -c "import omegaconf; print(f'omegaconf version: {omegaconf.__version__}')" && \
    echo "Environment setup complete!"

USER ${LDAP_USERNAME}

COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} ./ /workspace/

CMD ["/bin/bash"]