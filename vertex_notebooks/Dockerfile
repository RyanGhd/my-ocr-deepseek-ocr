FROM us-docker.pkg.dev/gcp-wow-wiq-014-test/pa-docker-registry/workbench-container:latest

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      torch==2.6.0 \
      torchvision \
      transformers==4.46.3 \
      tokenizers==0.20.3 \
      einops \
      addict \
      easydict \
      huggingface_hub && \
    pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Download the DeepSeek-OCR model during build (with online mode enabled)
RUN mkdir -p /opt/models && \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='deepseek-ai/DeepSeek-OCR', local_dir='/opt/models/deepseek-ocr', local_dir_use_symlinks=False)"

# Set permissions
RUN chmod -R 755 /opt/models

# Now set offline mode for runtime (after download is complete)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1