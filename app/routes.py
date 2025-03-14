import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from app.models.hand_model import TranslationModel
from app.utils.video_processing import process_video

main = Blueprint('main', __name__)

# Load the model
model = TranslationModel(model_path="app/models/your_model.pth")

@main.route('/')
def home():
    return jsonify({"message": "Welcome to the Sign Language Translation API!"})

@main.route('/translate', methods=['POST'])
def translate():
    """
    Process video directly from the request without saving to disk.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video = request.files['file']
    #return jsonify({"filename": video.filename})

    # Convert the video file into a format readable by OpenCV
    video_bytes = np.frombuffer(video.read(), np.uint8)
    video_capture = cv2.VideoCapture(cv2.imdecode(video_bytes, cv2.IMREAD_UNCHANGED))

    if not video_capture.isOpened():
        return jsonify({"error": "Invalid video file"}), 400

    # Process the video and extract keypoints
    keypoints = process_video(video_capture)

    # Predict translation using the model
    translation = model.predict(keypoints)

    return jsonify({"translation": translation})
