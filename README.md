# Birdy Buddies

## Setup Instructions

### 1. Create and Activate a Virtual Environment

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

Make sure you have `pip` updated:
```bash
pip install --upgrade pip
```

Then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run Inference

To run the inference script, activate your virtual environment (if not already active) and run:
```bash
python inference.py
```

You can uncomment the relevant lines in `inference.py` to test image, video, or audio prediction, e.g.:
```python
print(image_prediction('./test_images/crows_3.jpg'))
print(video_prediction('./test_videos/kingfisher.mp4'))
print(audio_prediction('./test_audios/crow-64028.mp3'))
```

---

## Notes
- Ensure you have Python 3.8 or newer installed.
- If you encounter issues with dependencies, check the `requirements.txt` for version compatibility.
- For AWS Lambda or cloud deployment, additional environment variables and AWS credentials may be required.
