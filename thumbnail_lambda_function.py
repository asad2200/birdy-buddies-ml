import json
import boto3
import urllib.parse
from PIL import Image, ImageOps
import io
import os

# Initialize clients
s3_client = boto3.client('s3')
lambda_client = boto3.client("lambda")

BIRD_LAMBDA_ARN = os.environ.get("BIRD_LAMBDA_ARN",
                                 "arn:aws:lambda:us-east-1:278152733922:function:BirdTag-Inference")

def lambda_handler(event, context):
    """
    Lambda function to create thumbnails for uploaded images
    Triggered by S3 PUT events
    """
    
    try:
        # Get bucket and object key from the S3 event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        print(f"Processing file: {key} from bucket: {bucket}")
        
        # Check if the file is an image
        if not is_image_file(key):
            print(f"File {key} is not an image. Skipping thumbnail generation.")

            # Pass original event and get response
            invoke_next_lambda(event)

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'File {key} is not an image. No thumbnail created.',
                })
            }
        
        # Download the image from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        # Create thumbnail
        thumbnail_data = create_thumbnail(image_data)
        
        # Generate thumbnail key (add -thumb suffix before extension)
        thumbnail_key = generate_thumbnail_key(key)
        
        # Upload thumbnail back to S3
        s3_client.put_object(
            Bucket=bucket,  # You can use a different bucket for thumbnails
            Key=thumbnail_key,
            Body=thumbnail_data,
            ContentType='image/jpeg',
            Metadata={
                'original-file': key,
                'file-type': 'thumbnail'
            }
        )
        
        print(f"Thumbnail created successfully: {thumbnail_key}")

        # Pass original event and get response
        invoke_next_lambda(event)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Thumbnail created successfully',
                'original_file': key,
                'thumbnail_file': thumbnail_key,
            })
        }
        
    except Exception as e:
        print(f"Error processing file {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def is_image_file(key):
    """
    Check if the file is an image based on its extension
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return any(key.lower().endswith(ext) for ext in image_extensions)

def create_thumbnail(image_data, max_size=(200, 200)):
    """
    Create a thumbnail from image data
    """
    try:
        # Open image with PIL
        with Image.open(io.BytesIO(image_data)) as image:
            # Convert to RGB if necessary (for PNG with transparency)
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save thumbnail to bytes
            thumbnail_buffer = io.BytesIO()
            image.save(thumbnail_buffer, format='JPEG', quality=85, optimize=True)
            thumbnail_buffer.seek(0)
            
            return thumbnail_buffer.getvalue()
            
    except Exception as e:
        raise Exception(f"Error creating thumbnail: {str(e)}")

def generate_thumbnail_key(original_key):
    """
    Generate thumbnail key by adding -thumb suffix and saving in thumbnails folder
    Example: images/bird1.jpg -> thumbnails/bird1-thumb.jpg
    """
    # Extract just the filename (remove any existing path)
    if '/' in original_key:
        filename = original_key.rsplit('/', 1)[1]
    else:
        filename = original_key
    
    # Split filename and extension
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        thumbnail_filename = f"{name}-thumb.{ext}"
    else:
        thumbnail_filename = f"{filename}-thumb"
    
    # Always save thumbnails in the "thumbnails" folder
    return f"thumbnails/{thumbnail_filename}"

def invoke_next_lambda(original_event: dict):
    """
    Pass the original S3 event to bird detection Lambda and return its response
    """
    try:
        lambda_client.invoke(
            FunctionName=BIRD_LAMBDA_ARN,
            InvocationType="Event",  # Asynchronous invocation
            Payload=json.dumps(original_event).encode(),
        )
        
        print("Successfully invoked bird-detection Lambda asynchronously")
        
    except Exception as exc:
        print(f"Could not invoke bird-detection Lambda: {exc}")
