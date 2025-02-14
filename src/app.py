from flask import Flask, render_template, jsonify, Response
import cv2
import time
from asl_recognition.test import recognize_sign  # Import the updated function

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Global variables for gesture recognition
frame_buffer = []
gesture_sentence = ""
word_count = 0
is_recording = False
is_detecting = False
start_time = time.time()

def gen_frames():
    global frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Call the recognize_sign function
        result = recognize_sign(frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time)

        # Get the processed frame
        frame = result['frame']

        # Encode the frame to send to the frontend (as JPEG)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_sentence')
def get_gesture_sentence():
    return jsonify({'gesture_sentence': gesture_sentence.strip()})

@app.route('/state_info')
def get_state_info():
    result = recognize_sign(frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time)
    return jsonify({
        'state_text': result['state_text'],
        'remaining_time': result['remaining_time'],
        'word_count_text': result['word_count_text']
    })

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global is_recording, is_detecting
    # Toggle start/stop detection
    is_detecting = not is_detecting
    is_recording = not is_recording
    return jsonify({
        'is_detecting': is_detecting,
        'is_recording': is_recording
    })

@app.route('/reset_gesture', methods=['POST'])
def reset_gesture():
    global gesture_sentence
    # Reset the gesture sentence
    gesture_sentence = ""
    return jsonify({
        'gesture_sentence': gesture_sentence
    })

@app.route('/get_frame_data')
def get_frame_data():
    # Assume you capture a frame here (camera logic)
    frame = "dummy_frame_data"  # Replace with actual frame capture logic

    return jsonify({
        'frame': frame,
        'gesture_sentence': gesture_sentence,
        'state_text': "Capturing" if is_detecting else "Idle",
        'word_count_text': f"Detected Words: 0"
    })

if __name__ == '__main__':
    app.run(debug=True)
