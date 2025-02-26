from flask import Flask, render_template, jsonify, Response
import cv2
import time
import threading
import asl_recognition.test   # Import the updated function

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)
lock = threading.Lock()  # Ensures thread safety

# Global variables for gesture recognition
frame_buffer = []
gesture_sentence = ""
word_count = 0
is_recording = False
is_detecting = False
start_time = time.time()
frame = None

def gen_frames():
    global frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            result = asl_recognition.test.recognize_sign(frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time)
            gesture_sentence = result['gesture_sentence']
            word_count = result['word_count']
            frame = result['frame']

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_sentence')
def get_gesture_sentence():
    with lock:
        return jsonify({'gesture_sentence': gesture_sentence.strip()})

@app.route('/state_info')
def get_state_info():
    with lock:
        if frame is None:
            return jsonify({'error': 'No frame available'})
        result = asl_recognition.test.recognize_sign(frame, frame_buffer, gesture_sentence, word_count, is_recording, is_detecting, start_time)
        return jsonify({
            'state_text': result['state_text'],
            'remaining_time': result['remaining_time'],
            'word_count_text': result['word_count_text']
        })

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global is_recording, is_detecting
    with lock:
        is_detecting = not is_detecting
        is_recording = not is_recording
        print("Toggle Detection Called: is_detecting =", is_detecting, "is_recording =", is_recording)
    return jsonify({'is_detecting': is_detecting, 'is_recording': is_recording})


@app.route('/reset_gesture', methods=['POST'])
def reset_gesture():
    global gesture_sentence
    with lock:
        gesture_sentence = ""
    return jsonify({'gesture_sentence': gesture_sentence})

@app.route('/get_frame_data')
def get_frame_data():
    with lock:
        return jsonify({
            'frame': "Available" if frame is not None else "No frame",
            'gesture_sentence': gesture_sentence,
            'state_text': "Capturing" if is_detecting else "Idle",
            'word_count_text': f"Detected Words: {word_count}"
        })

def release_camera():
    cap.release()

import atexit
atexit.register(release_camera)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

