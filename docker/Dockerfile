FROM python:3.11-slim

# Open3D needs these at runtime even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libglib2.0-0t64 \
    libgl1 \
    libgomp1 \
    libidn2-0 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY slam ./slam
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["python"]
