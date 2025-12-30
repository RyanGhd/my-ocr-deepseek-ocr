FROM us-docker.pkg.dev/gcp-wow-wiq-014-test/pa-docker-registry/workbench-container:latest

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      torch==2.6.0 \
      transformers==4.46.3 \
      tokenizers==0.20.3 \
      einops \
      addict \
      easydict \
      huggingface_hub && \
    pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Set the model directory
ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Download the DeepSeek-OCR model during build (when internet is available)
RUN mkdir -p /opt/huggingface && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='deepseek-ai/DeepSeek-OCR', local_dir='/opt/models/deepseek-ocr', local_dir_use_symlinks=False)"

# Set permissions
RUN chmod -R 755 /opt/models