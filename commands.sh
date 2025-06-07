AWS_ACCOUNT_ID="278152733922"  # Replace with your account ID
AWS_REGION="us-east-1"   


docker build -t birdtag-ml-dependencies:v2 .

docker buildx build --platform linux/amd64 -t birdtag-ml-dependencies:v2 . --load

docker buildx build --platform linux/arm64/v8 -t birdtag-ml-dependencies:v2 .

docker build --platform linux/amd64 -t birdtag-ml-dependencies:v2 .


# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag the image
docker tag birdtag-ml-dependencies:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birdtag-ml-dependencies:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birdtag-ml-dependencies:latest

--------------------------------

docker build -t birtag-inference:latest .

docker buildx build --platform linux/arm64/v8 -t birtag-inference:v1 .

# Run a throw-away container from the image
docker run --rm --entrypoint '' birtag-inference:v1 /usr/local/bin/ffmpeg -version
# Expected: version information, *not* "Exec format error"

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag birtag-inference:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birtag-inference:latest

docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birtag-inference:latest


docker buildx build --platform linux/arm64/v8 -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birtag-inference:latest . --push


docker buildx build --platform linux/amd64 -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birtag-inference:amd3 . --push


# docker run --rm -v "$PWD/test_audios/crow-64028.mp3":/tmp/sample.mp3:ro -it --entrypoint /bin/bash birtag-inference:v2

# from birdnetlib.analyzer import Analyzer
# from birdnetlib import Recording
# from datetime import datetime

# analyzer = Analyzer()
# print('✓ BirdNET Analyzer initialized successfully')

# recording = Recording(
#     analyzer,
#     "/tmp/sample.mp3",
#     date=datetime.now(),
#     min_conf=0.1
# )
# print('✓ BirdNET Recording object created successfully')

# # Try analysis (this might take a moment)
# recording.analyze()
# print(f'✓ BirdNET analysis completed with {(recording.detections)} detections')


# import os, librosa
# assert os.path.exists("/tmp/sample.mp3"), f"{"/tmp/sample.mp3"} not found"
# print("File size:", os.path.getsize("/tmp/sample.mp3"))
# y, sr = librosa.load("/tmp/sample.mp3", sr=None)

# y, sr = librosa.load("/tmp/sample.wav", sr=None)