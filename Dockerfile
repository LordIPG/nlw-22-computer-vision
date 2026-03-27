# Use an older, more stable Python image for better MediaPipe support
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    libdbus-1-3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/

# Set working directory
WORKDIR /app

# Copy the specific application directory
# We assume the Dockerfile is in the root and we want to deploy computer_vision_app
COPY computer_vision_app/ /app/

# Sync dependencies using uv
# --frozen ensures we use the exact versions from uv.lock
RUN /uv/bin/uv sync --frozen --no-dev

# Expose the port
EXPOSE 10000

# Start the application
# We use /app/.venv/bin/python to skip 'uv run' overhead in production
CMD ["/app/.venv/bin/python", "app.py"]
