FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum -y install \
    libsndfile \
    mesa-libGL \
    gcc \
    gcc-c++ \
    && yum clean all

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python packages directly
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    --no-cache-dir \
    --only-binary=llvmlite,numba \
    --prefer-binary

# Copy the Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY model.pt ${LAMBDA_TASK_ROOT}

# Set environment variables
ENV TORCH_HOME="/tmp"
ENV ULTRALYTICS_CONFIG_DIR="/tmp"
ENV MPLCONFIGDIR="/tmp"
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}"

# Create temporary directories with proper permissions
RUN mkdir -p /tmp/models /tmp/audio /tmp/video /tmp/images && \
    chmod 755 /tmp/models /tmp/audio /tmp/video /tmp/images

# Set the handler
CMD ["lambda_function.lambda_handler"]