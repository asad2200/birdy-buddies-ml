FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies including audio libraries
RUN yum -y install \
    libsndfile \
    mesa-libGL \
    gcc \
    gcc-c++ \
    tar \
    xz \
    && yum clean all

# Install FFmpeg (crucial for librosa audio reading)
RUN cd /tmp && \
    curl -O https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-* && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python packages directly
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    --no-cache-dir
    # --only-binary llvmlite
    # --prefer-binary

# Copy the Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY model.pt ${LAMBDA_TASK_ROOT}

# Set environment variables for audio processing
ENV TORCH_HOME="/tmp"
ENV ULTRALYTICS_CONFIG_DIR="/tmp"
ENV MPLCONFIGDIR="/tmp"
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}"

# DISABLE ALL CACHING - This is the key part!
ENV NUMBA_CACHE_DIR="/tmp/numba_cache"
ENV LIBROSA_CACHE_DIR="/tmp/librosa_cache"
ENV LIBROSA_CACHE_LEVEL="0"


ENV NUMBA_THREADING_LAYER="workqueue"
ENV NUMBA_NUM_THREADS="1"
ENV OMP_NUM_THREADS="1"
ENV MKL_NUM_THREADS="1"
# CRITICAL: Force joblib to use threading instead of multiprocessing
ENV JOBLIB_MULTIPROCESSING="0"
ENV JOBLIB_TEMP_FOLDER="/tmp"
# Disable parallel processing that causes segfaults in Lambda
ENV OPENBLAS_NUM_THREADS="1"
ENV VECLIB_MAXIMUM_THREADS="1"
ENV NUMEXPR_NUM_THREADS="1"
ENV PYTHONFAULTHANDLER="1"

# FFmpeg path
ENV PATH="/usr/local/bin:${PATH}"

# Create temporary directories with proper permissions
RUN mkdir -p /tmp/models /tmp/audio /tmp/video /tmp/images /tmp/numba_cache /tmp/librosa_cache && \
    chmod 755 /tmp/models /tmp/audio /tmp/video /tmp/images /tmp/numba_cache /tmp/librosa_cache

# Set the handler
CMD ["lambda_function.lambda_handler"]