import os
import boto3
from datetime import datetime
import uuid

# Initialize DynamoDB tables (resource mode)
dynamodb = boto3.resource('dynamodb')
metadata_table = dynamodb.Table('BirdTagMetadata-DDB')
tag_index_table = dynamodb.Table('BirdTagTagIndex')

def lambda_handler(event, context):
    """
    Triggered by an S3 ObjectCreated event. Responsibilities:
      1. Read the S3 event to get bucket and object key.
      2. Generate and upload a thumbnail under "thumbnails/{uuid}.jpg".
      3. Build file and thumbnail URLs.
      4. Run the pretrained inference model on the downloaded file.
      5. Construct a unique DynamoDB Sort Key (SK) using timestamp + UUID.
      6. Persist metadata in BirdTagMetadata-DDB.
      7. For each detected species, write inverted-index entries in BirdTagTagIndex.
    """



    # --- 1. Extract bucket and key from the S3 event ---
    record     = event['Records'][0]
    bucket     = record['s3']['bucket']['name']
    upload_key = record['s3']['object']['key']
    # e.g. "uploads/550e8400-e29b-41d4-a716-446655440000.jpg"
    # The client or presign step ensured the filename is a UUID plus extension.
    # image stored in the S3 for original file will be like uploads/UUDI.jpg, and for thumbnail it will be thumbnails/UUID.jpg

    # --- 2. Generate a thumbnail and put it under "thumbnails/{uuid}.jpg" ---
    # Extract the UUID portion from the upload_key's filename:
    uuid_suffix = upload_key.split("/")[-1]               # "550e8400-e29b-41d4-a716-446655440000.jpg"
    thumb_key   = upload_key.replace("uploads/", "thumbnails/")
    # Now thumb_key is, e.g., "thumbnails/550e8400-e29b-41d4-a716-446655440000.jpg"

    # Download the original file locally for processing
    s3 = boto3.client('s3')
    local_upload_path = f"/tmp/{uuid_suffix}"
    s3.download_file(bucket, upload_key, local_upload_path)

    # (Your teammate should insert thumbnail-generation code here.)
    # For example:
    #   from PIL import Image
    #   img = Image.open(local_upload_path)
    #   img.thumbnail((200, 200))
    #   img.save(f"/tmp/{uuid_suffix}")
    #   s3.upload_file(f"/tmp/{uuid_suffix}", bucket, thumb_key)

    # After running image processing, upload the thumbnail to S3:
    # (Ensure the local thumbnail file exists at /tmp/{uuid_suffix})
    s3.upload_file(f"/tmp/{uuid_suffix}", bucket, thumb_key)

    # --- 3. Build S3 URLs using the bucket and keys ---
    file_url      = f"https://{bucket}.s3.amazonaws.com/{upload_key}"
    thumbnail_url = f"https://{bucket}.s3.amazonaws.com/{thumb_key}"
    file_type     = "image"  # or "video"/"audio" based on the file extension or metadata

    # --- 4. Run your inference model on the downloaded file ---
    # Replace the following pseudocode with your actual model invocation:
    #   from my_model import load_model, infer_species
    #   model = load_model('/opt/models/bird_detector')
    #   detected_tags = infer_species(model, local_upload_path)
    #
    # For demonstration, we’ll hardcode:
    detected_tags = {"crow": 2, "pigeon": 1}
    # 'detected_tags' maps each species to its count in the image.

    # Retrieve the Cognito sub (uploader's user ID) from the event (if available)
    # If using a Lambda trigger without identity, you might fetch user_sub elsewhere.
    user_sub = record.get('userIdentity', {}).get('principalId', 'unknown-user')

    # --- 5. Build SK using timestamp + the UUID suffix (without ".jpg") ---
    iso_timestamp = datetime.utcnow().isoformat() + "Z"
    file_uuid     = uuid_suffix.replace(".jpg", "")  # remove extension to get pure UUID
    main_file_sk  = f"FILE#{iso_timestamp}#{file_uuid}"
    # Example SK: "FILE#2025-06-05T14:22:30.123Z#550e8400-e29b-41d4-a716-446655440000"

    # --- 6. Write the main metadata record into BirdTagMetadata-DDB ---
    metadata_item = {
        "PK":         f"USER#{user_sub}",
        "SK":         main_file_sk,
        "fileURL":    file_url,
        "thumbURL":   thumbnail_url,
        "fileType":   file_type,
        "uploadedAt": iso_timestamp,
        "tags":       detected_tags           # e.g. { "crow": 2, "pigeon": 1 }
    }
    metadata_table.put_item(Item=metadata_item)

    # --- 7. Write one inverted-index item per detected species into BirdTagTagIndex ---
    with tag_index_table.batch_writer() as batch:
        for species, count in detected_tags.items():
            tag_index_item = {
                "PK":         f"TAG#{species}",       # e.g. "TAG#crow"
                "SK":         main_file_sk,           # Matches metadata SK
                "fileURL":    file_url,               # Store the original S3 URL
                "thumbURL":   thumbnail_url,          # Store the thumbnail URL
                "tagCount":   count,                  # Numeric count of that species
                "uploadedAt": iso_timestamp,          # Record when inference ran
                "userPK":     f"USER#{user_sub}"      # Same PK as metadata
            }
            batch.put_item(Item=tag_index_item)

    # --- 8. (Optional) Publish SNS notifications for confirmed subscribers ---
    # e.g. for each species in detected_tags:
    #   subs = subscription_table.query(KeyConditionExpression=Key("PK").eq(f"TAG#{species}"))
    #   for sub in subs["Items"]:
    #       if sub["status"] == "CONFIRMED":
    #           sns.publish(TopicArn=sub["subscriptionArn"], Message=f"New {species} spotted!")

    return {
        "statusCode": 200,
        "body": {
            "message":   "Tagging and indexing complete",
            "fileSK":    main_file_sk,
            "detected":  detected_tags
        }
    }