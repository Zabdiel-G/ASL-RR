from flask import Flask, render_template, jsonify, Response
import cv2
import time
import threading
import numpy as np
import atexit
import asl_recognition.test  # Import the updated function
import chatbot.chatbot
from flask_cors import CORS
from asl_response.processor import match_sentence_to_words, load_default_pose
from asl_response.animator import animate_sentence
from asl_response.response_utils import draw_keypoints
from asl_response.config import FRAME_WIDTH, FRAME_HEIGHT

app = Flask(__name__)
CORS(app)

# Initialize the webcam

cap = cv2.VideoCapture(0)
lock = threading.Lock()  # Ensures thread safety

# Global variables for gesture recognition and animation
frame_buffer = []
gesture_sentence = ""
word_count = 0
is_recording = False
is_detecting = False
start_time = time.time()
frame = None

# Animation variables
current_sentence = "Hello my name M A R C"
default_pose, default_hand_left, default_hand_right = load_default_pose()

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

def gen_animation():
    while True:
        # Here, you can get the animated frame based on the sentence
        if current_sentence:
            words = match_sentence_to_words(current_sentence)  # Get words based on sentence
            for frame in animate_sentence(words):
                ok, buf = cv2.imencode('.jpg', frame)
                if not ok:
                    continue
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' 
                    + buf.tobytes() + b'\r\n')
        else:
            time.sleep(0.03)  # Delay for smooth animation (30 FPS)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/animation_feed')
def animation_feed():
    # resource intensive
    return Response(gen_animation(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat_from_gesture', methods=['GET'])
def chat_from_gesture():
    global gesture_sentence, current_sentence
    with lock:
        current_gesture = gesture_sentence.strip()

    if not current_gesture:
        return jsonify({"error": "No gesture sentence available."}), 400

    # Process the gesture sentence using functions from your chatbot module
    english_interpretation = chatbot.chatbot.interpret_asl_input(current_gesture)
    english_response = chatbot.chatbot.generate_english_response(english_interpretation)
    asl_response = chatbot.chatbot.generate_asl_response(english_response)



    # Update the sentence for animation
    with lock:
        current_sentence = asl_response

    return jsonify({
        "gesture_sentence": current_gesture,
        "english_interpretation": english_interpretation,
        "english_response": english_response,
        "asl_response": asl_response
    })

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

atexit.register(release_camera)

if __name__ == '__main__':
    threading.Thread(target=gen_animation, daemon=True).start()  # Start animation thread
    app.run(host='0.0.0.0', port=5000, debug=True)
