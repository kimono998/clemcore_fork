FROM ghcr.io/huggingface/openenv-base:latest

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# WORK_DIR is /app
RUN git clone https://github.com/clp-research/clembench \
    && cd clembench \
    && pip install -r requirements.txt \
    && clem list games
ENV CLEMBENCH_HOME /app/clembench

# Run the server with reading defaults from the env vars
CMD ["clem", "serve", "--host", "0.0.0.0", "--port", "8000"]
