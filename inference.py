from lambda_function import image_prediction, video_prediction, audio_prediction

# print(image_prediction('./test_images/crows_3.jpg'))
# print(video_prediction('./test_videos/kingfisher.mp4'))
# print(audio_prediction('./test_audios/crow-64028.mp3'))



# call inference lambda function ---------------------------------

# import boto3, base64, json, os

# lambda_client = boto3.client("lambda")
# BIRD_LAMBDA_ARN = os.environ.get("BIRD_LAMBDA_ARN",
#                                  "arn:aws:lambda:us-east-1:278152733922:function:BirdTag-Inference")

# def lambda_handler(event, context):
#     # 1. get the raw bytes (e.g. API Gateway binary body)
#     body_b64 = event["body"]               # still base-64
#     file_bytes = base64.b64decode(body_b64)

#     # 2. build payload for lambda_inference
#     invoke_payload = {
#         "fileB64": base64.b64encode(file_bytes).decode(),   # JSON-friendly
#         "filename": event.get("queryStringParameters", {}).get("filename", "upload.jpg"),
#         "contentType": event["headers"].get("Content-Type", "application/octet-stream")
#     }

#     # 3. synchronous invoke
#     response = lambda_client.invoke(
#         FunctionName=BIRD_LAMBDA_ARN,
#         InvocationType="RequestResponse",
#         Payload=json.dumps(invoke_payload).encode(),
#     )

#     result = json.load(response["Payload"])   # {"statusCode":200,"body":"..."}
#     tags = json.loads(result["body"])["results"][0]["tags"]