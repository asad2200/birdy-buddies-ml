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


docker build -t birtag-inference:latest .

docker tag birtag-inference:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birtag-inference:latest

docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/birtag-inference:latest
