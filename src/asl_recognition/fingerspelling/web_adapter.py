import threading
import time
from flask import Flask, Response, render_template
import io
from PIL import Image
import numpy as np

# Global variable to store the latest frame.
latest_frame = None

# We still import cv2 to patch the functions that your original program calls.
import cv2 as cv
_original_imshow = cv.imshow
_original_waitKey = cv.waitKey

def patched_imshow(window_name, frame, *args, **kwargs):
    global latest_frame
    # Instead of displaying the frame via OpenCV, save a copy
    # Note: frame is expected to be a numpy array (e.g., in RGB format).
    latest_frame = frame.copy()

def patched_waitKey(delay=1):
    # Immediately return a dummy key code to allow the loop in webcam.py to continue.
    return -1

# Apply monkey-patching. This will intercept calls from your original code.
cv.imshow = patched_imshow
cv.waitKey = patched_waitKey

# Import your unchanged original application.
import webcam  # Assumes your original file is named webcam.py and contains a main() function

def run_original_program():
    # Call the main() from webcam.py, which starts the video processing pipeline.
    webcam.main()

# Set up a minimal Flask app.
app = Flask(__name__)

def generate_frames():
    global latest_frame
    while True:
        # Wait for a frame to become available.
        if latest_frame is None:
            time.sleep(0.01)
            continue
        try:
            # Convert the numpy array frame to a PIL Image.
            # (If needed, adjust the conversion if your frame is in BGR or RGB.)
            img = Image.fromarray(latest_frame)
            # Use an in-memory bytes buffer to save the image as JPEG.
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG')
            frame_bytes = img_io.getvalue()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print("Error generating frame:", e)
        time.sleep(0.03)  # Adjust frame delay as necessary

@app.route('/')
def index():
    # Renders an HTML page that displays the video stream.
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import torch
    # Use the spawn method (especially needed on Windows)
    torch.multiprocessing.set_start_method('spawn')
    
    # Run the original application in a background daemon thread.
    original_thread = threading.Thread(target=run_original_program, daemon=True)
    original_thread.start()

    # Run the Flask app without the auto-reloader.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
