import json
import boto3
import os
import tempfile
import logging
from typing import Dict, List, Any
from datetime import datetime
import urllib.parse
from ultralytics import YOLO
import supervision as sv
import cv2 as cv
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'your-model-bucket')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/model.pt')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'image-metadata')
THUMBNAIL_BUCKET = os.environ.get('THUMBNAIL_BUCKET', 'your-thumbnail-bucket')

# Global variables for model caching
cached_model = None
model_last_modified = None

def lambda_handler(event, context):
    """Main Lambda handler function for processing images, videos, and audio files"""
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Handle S3 event trigger
        if 'Records' in event:
            for record in event['Records']:
                if record['eventSource'] == 'aws:s3':
                    bucket = record['s3']['bucket']['name']
                    key = urllib.parse.unquote_plus(record['s3']['object']['key'])
                    
                    # Process the media file
                    process_media_file(bucket, key)
        
        # Handle direct invocation
        elif 'bucket' in event and 'key' in event:
            bucket = event['bucket']
            key = event['key']
            process_media_file(bucket, key)
        
        else:
            raise ValueError("Invalid event format. Expected S3 event or direct invocation with bucket/key.")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Media processing completed successfully'
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def should_reload_model() -> bool:
    """Check if model should be reloaded from S3"""
    global model_last_modified
    
    try:
        # Get current model metadata from S3
        response = s3_client.head_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        current_last_modified = response['LastModified']
        
        # Check if model needs to be reloaded
        if model_last_modified is None or model_last_modified != current_last_modified:
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking model metadata: {str(e)}")
        return True  # Force reload on error

def load_model_from_s3():
    return YOLO('./model.pt') #TODO: remove this

    """Download and load YOLO model from S3"""
    global cached_model, model_last_modified
    
    try:
        if should_reload_model():
            logger.info(f"Loading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
            
            # Download model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
                s3_client.download_fileobj(MODEL_BUCKET, MODEL_KEY, temp_file)
                temp_file_path = temp_file.name
            
            try:
                # Load YOLO model
                cached_model = YOLO(temp_file_path)
                
                # Update last modified time
                response = s3_client.head_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
                model_last_modified = response['LastModified']
                
                logger.info("YOLO model loaded successfully")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            logger.info("Using cached model")
            
        return cached_model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def download_media_file(bucket: str, key: str) -> str:
    """Download media file from S3 to temporary location"""
    try:
        # Determine file extension
        file_ext = os.path.splitext(key)[1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        
        s3_client.download_fileobj(bucket, key, temp_file)
        temp_file.close()
        
        logger.info(f"Downloaded {bucket}/{key} to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error downloading media file: {str(e)}")
        raise

def get_file_type(file_path: str) -> str:
    """Determine if file is image, video, or audio based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in image_extensions:
        return 'image'
    elif file_ext in video_extensions:
        return 'video'
    elif file_ext in audio_extensions:
        return 'audio'
    else:
        return 'unknown'

def image_prediction(image_path: str, confidence: float = 0.5) -> List[Dict[str, int]]:
    """
    Run YOLO prediction on image and return detected class names with counts
    
    Parameters:
        image_path (str): Path to the image file
        confidence (float): Confidence threshold for detections
        
    Returns:
        List[Dict[str, int]]: List of dictionaries with class names as keys and counts as values
    """
    try:
        # Load model
        model = load_model_from_s3()
        class_dict = model.names
        
        # Load image
        img = cv.imread(image_path)
        if img is None:
            logger.error("Couldn't load the image")
            return []
        
        # Run prediction
        result = model(img)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter by confidence and count class occurrences
        tags = []
        if detections.class_id is not None:
            filtered_detections = detections[(detections.confidence > confidence)]
            
            # Count occurrences of each class
            class_counts = {}
            for cls_id in filtered_detections.class_id:
                class_name = class_dict[cls_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Convert to list of dictionaries
            tags = [{class_name: count} for class_name, count in class_counts.items()]
        
        logger.info(f"Image prediction tags: {tags}")
        return tags
        
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        return []

def audio_prediction(audio_path: str, confidence: float = 0.5, min_conf: float = 0.25) -> List[Dict[str, int]]:
    """
    Run BirdNET prediction on audio file and return detected species with counts
    
    Parameters:
        audio_path (str): Path to the audio file
        confidence (float): Confidence threshold for detections
        min_conf (float): Minimum confidence threshold for BirdNET
        
    Returns:
        List[Dict[str, int]]: List of dictionaries with species names as keys and counts as values
    """
    try:
        # Initialize BirdNET analyzer
        analyzer = Analyzer()
        
        # Create recording object with current date
        recording = Recording(
            analyzer,
            audio_path,
            lat=None,  # Optional: Add latitude if available
            lon=None,  # Optional: Add longitude if available
            date=datetime.now(),
            min_conf=min_conf
        )
        
        # Run prediction
        recording.analyze()
        
        # Count species occurrences above confidence threshold
        species_counts = {}
        for detection in recording.detections:
            if detection['confidence'] >= confidence:
                species_name = detection['common_name']
                species_counts[species_name] = species_counts.get(species_name, 0) + 1
        
        # Convert to list of dictionaries
        tags = [{species_name: count} for species_name, count in species_counts.items()]
        
        logger.info(f"Audio prediction species: {tags}")
        return tags
        
    except Exception as e:
        logger.error(f"Error in audio prediction: {str(e)}")
        return []

def video_prediction(video_path: str, confidence: float = 0.5, sample_frames: int = 10) -> List[Dict[str, int]]:
    """
    Run YOLO prediction on video frames and return detected class names with counts
    
    Parameters:
        video_path (str): Path to the video file
        confidence (float): Confidence threshold for detections
        sample_frames (int): Number of frames to sample for prediction
        
    Returns:
        List[Dict[str, int]]: List of dictionaries with class names as keys and counts as values
    """
    try:
        # Load model
        model = load_model_from_s3()
        class_dict = model.names
        
        # Open video
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Couldn't open the video")
            return []
        
        # Get video info
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames <= sample_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // sample_frames
            frame_indices = [i * step for i in range(sample_frames)]
        
        all_class_counts = {}
        
        # Process sampled frames
        for frame_idx in frame_indices:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Run prediction on frame
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter by confidence and count class occurrences
            if detections.class_id is not None:
                filtered_detections = detections[(detections.confidence > confidence)]
                
                # Count detections in this frame
                for cls_id in filtered_detections.class_id:
                    class_name = class_dict[cls_id]
                    all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
        
        # Convert to list of dictionaries
        tags = [{class_name: count} for class_name, count in all_class_counts.items()]
        
        cap.release()
        logger.info(f"Video prediction tags: {tags}")
        return tags
        
    except Exception as e:
        logger.error(f"Error in video prediction: {str(e)}")
        return []


def save_to_database(image_url: str, thumbnail_url: str, tags: List[str], metadata: Dict[str, Any]):
    """Save prediction results to DynamoDB"""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Create item for DynamoDB
        item = {
            'image_id': metadata.get('image_id', ''),
            'media_url': image_url,
            'thumbnail_url': thumbnail_url,
            'tags': tags,
            'media_type': metadata.get('media_type', 'image'),
            'created_at': datetime.utcnow().isoformat(),
            'file_size': metadata.get('file_size', 0),
            'content_type': metadata.get('content_type', ''),
            'processed_at': datetime.utcnow().isoformat(),
            'bucket': metadata.get('bucket', ''),
            'key': metadata.get('key', '')
        }
        
        # Add additional metadata if available
        if 'width' in metadata:
            item['width'] = metadata['width']
        if 'height' in metadata:
            item['height'] = metadata['height']
        if 'duration' in metadata:
            item['duration'] = metadata['duration']
            
        response = table.put_item(Item=item)
        logger.info(f"Successfully saved to DynamoDB: {item['image_id']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error saving to DynamoDB: {str(e)}")
        raise

def process_media_file(bucket: str, key: str):
    """Process a single media file (image, video, or audio)"""
    try:
        # Generate unique ID
        media_id = f"{bucket}_{key.replace('/', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Download media file
        media_path = download_media_file(bucket, key)
        
        try:
            # Determine file type
            file_type = get_file_type(media_path)
            
            # Run appropriate prediction
            if file_type == 'image':
                tags = image_prediction(media_path)
            elif file_type == 'video':
                tags = video_prediction(media_path)
            elif file_type == 'audio':
                tags = audio_prediction(media_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                tags = []
            
            # Generate URLs
            media_url = f"s3://{bucket}/{key}"
            thumbnail_url = generate_thumbnail_url(bucket, key)
            
            # Get file metadata
            response = s3_client.head_object(Bucket=bucket, Key=key)
            
            # Prepare metadata
            metadata = {
                'image_id': media_id,
                'media_type': file_type,
                'file_size': response.get('ContentLength', 0),
                'content_type': response.get('ContentType', ''),
                'bucket': bucket,
                'key': key
            }
            
            # Save to database
            save_to_database(media_url, thumbnail_url, tags, metadata)
            
            logger.info(f"Successfully processed {file_type}: {media_id} with {len(tags)} tags")
            
        finally:
            # Clean up temporary file
            if os.path.exists(media_path):
                os.unlink(media_path)
                
    except Exception as e:
        logger.error(f"Error processing media file {bucket}/{key}: {str(e)}")
        raise

def generate_thumbnail_url(bucket: str, key: str) -> str:
    """Generate thumbnail URL based on original media path"""
    file_name = os.path.splitext(os.path.basename(key))[0]
    file_ext = os.path.splitext(key)[1]
    thumbnail_key = f"thumbnails/thumb_{file_name}{file_ext}"
    
    return f"s3://{THUMBNAIL_BUCKET}/{thumbnail_key}"