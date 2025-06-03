FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum -y install \
    libsndfile \
    mesa-libGL \
    gcc \
    gcc-c++ \
    && yum clean all

# Copy requirements
COPY requirements.txt .

# Install Python packages directly
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    --no-cache-dir \
    --only-binary=llvmlite,numba \
    --prefer-binary

# Set environment variables
ENV TORCH_HOME="/tmp"
ENV ULTRALYTICS_CONFIG_DIR="/tmp"
ENV MPLCONFIGDIR="/tmp"

# Create temporary directories
RUN mkdir -p /tmp/models /tmp/audio /tmp/video /tmp/images

# Set the handler
CMD ["lambda_function.lambda_handler"]