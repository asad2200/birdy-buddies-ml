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
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'birdtag-media-storage-aj')
THUMBNAIL_BUCKET = os.environ.get('THUMBNAIL_BUCKET', 'birdtag-media-storage-aj')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/model.pt')

METADATA_TABLE = os.environ.get('METADATA_TABLE', 'BirdTagMetadata-DDB')
TAG_INDEX_TABLE = os.environ.get('TAG_INDEX_TABLE', 'BirdTagTagIndex')

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
                    
                    # Extract user_sub from event or set default
                    # You may need to modify this based on how you pass user_sub
                    user_sub = event.get('user_sub', 'default_user')
                    
                    # Process the media file
                    process_media_file(bucket, key, user_sub)
        
        # Handle direct invocation
        elif 'bucket' in event and 'key' in event:
            bucket = event['bucket']
            key = event['key']
            user_sub = event.get('user_sub', 'default_user')
            process_media_file(bucket, key, user_sub)
        
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

def save_to_database(file_url: str, thumbnail_url: str, tags: List[Dict[str, int]], 
                    user_sub: str, main_file_sk: str, file_type: str, iso_timestamp: str):
    """Save prediction results to DynamoDB with metadata and tag index tables"""
    try:
        # Get table references
        metadata_table = dynamodb.Table(METADATA_TABLE)
        tag_index_table = dynamodb.Table(TAG_INDEX_TABLE)
        
        # Convert tags from List[Dict[str, int]] to Dict[str, int]
        detected_tags = {}
        for tag_dict in tags:
            detected_tags.update(tag_dict)
        
        # --- Write the main metadata record into BirdTagMetadata-DDB ---
        metadata_item = {
            "PK": f"USER#{user_sub}",
            "SK": main_file_sk,
            "fileURL": file_url,
            "thumbURL": thumbnail_url,
            "fileType": file_type,
            "uploadedAt": iso_timestamp,
            "tags": detected_tags  # e.g. { "crow": 2, "pigeon": 1 }
        }
        
        metadata_table.put_item(Item=metadata_item)
        logger.info(f"Successfully saved metadata to DynamoDB: {main_file_sk}")
        
        # --- Write one inverted-index item per detected species into BirdTagTagIndex ---
        if detected_tags:
            with tag_index_table.batch_writer() as batch:
                for species, count in detected_tags.items():
                    tag_index_item = {
                        "PK": f"TAG#{species}",  # e.g. "TAG#crow"
                        "SK": main_file_sk,      # Matches metadata SK
                        "fileURL": file_url,     # Store the original S3 URL
                        "thumbURL": thumbnail_url,  # Store the thumbnail URL
                        "tagCount": count,       # Numeric count of that species
                        "uploadedAt": iso_timestamp,  # Record when inference ran
                        "userPK": f"USER#{user_sub}"  # Same PK as metadata
                    }
                    batch.put_item(Item=tag_index_item)
            
            logger.info(f"Successfully saved {len(detected_tags)} tag index items to DynamoDB")
        
    except Exception as e:
        logger.error(f"Error saving to DynamoDB: {str(e)}")
        raise

def process_media_file(bucket: str, key: str, user_sub: str):
    """Process a single media file (image, video, or audio)"""
    try:
        # Generate unique SK for this file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        main_file_sk = f"FILE#{timestamp}#{key.replace('/', '_')}"
        iso_timestamp = datetime.utcnow().isoformat()
        
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
            file_url = build_http_url(bucket, key)
            
            if file_type == 'image':
                thumbnail_key = generate_thumbnail_key(key)
                thumbnail_url = build_http_url(THUMBNAIL_BUCKET, thumbnail_key)
            else:
                thumbnail_url = None
            
            # Save to database
            save_to_database(file_url, thumbnail_url, tags, user_sub, main_file_sk, file_type, iso_timestamp)
            
            logger.info(f"Successfully processed {file_type}: {main_file_sk} with {len(tags)} tag types")
            
        finally:
            # Clean up temporary file
            if os.path.exists(media_path):
                os.unlink(media_path)
                
    except Exception as e:
        logger.error(f"Error processing media file {bucket}/{key}: {str(e)}")
        raise

def build_http_url(bucket: str, key: str) -> str:
    """
    Return an HTTPS, virtual-hosted–style URL:
    https://<bucket>.s3.<region>.amazonaws.com/<key>
    """
    quoted_key = urllib.parse.quote(key, safe="/")  # preserve the '/'s
    return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{quoted_key}"

def generate_thumbnail_key(key: str) -> str:
    """Return the S3 *key* where the thumbnail will live."""
    file_name, file_ext = os.path.splitext(os.path.basename(key))
    return f"thumbnails/{file_name}-thumb{file_ext}"