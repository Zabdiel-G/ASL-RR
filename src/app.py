# app.py

from flask import Flask, render_template, request, Response, jsonify# type: ignore
from multiprocessing import freeze_support
from io import BytesIO
from PIL import Image # type: ignore
import numpy as np # type: ignore
import configparser
import torch # type: ignore
from torchvision import transforms # type: ignore
import atexit
import cv2
import threading


from asl_recognition.fingerspelling.chicago_fs_wild import PriorToMap, ToTensor, Normalize, Batchify
from asl_recognition.fingerspelling.mictranet import init_lstm_hidden, MiCTRANet
from asl_recognition.fingerspelling.utils import get_ctc_vocab
from pipeline import VideoProcessingPipeline


from asl_response.processor import match_sentence_to_words, load_default_pose
from asl_response.animator import animate_sentence
from asl_response.response_utils import draw_keypoints
from asl_response.config import FRAME_WIDTH, FRAME_HEIGHT

import chatbot.chatbot
import time

fps_buffer = []

# Model-related globals
encoder = None
h0 = None
h = None
tsfm = None
inv_vocab_map = None
device = None
cfg = None
sentence = ''
last_letter = '_'
current_sentence = ''
lock = threading.Lock() 

# Detection state
is_recording = False
is_detecting = False
# Buffers for raw frames and decoded words
frame_buffer    = []
sentence_buffer = []

# Last decoded finger‑spelling sentence
gesture_sentence = ''

def gen_animation():
    while True:
        # If there's no active response, use a default idle gesture
        if current_sentence:
            words = match_sentence_to_words(current_sentence)
        else:
            # Use an idle gloss (this assumes you have defined an idle gesture in your system)
            words = ["hello"]
        for aniframe in animate_sentence(words):
            ok, buf = cv2.imencode('.jpg', aniframe)
            if not ok:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' 
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.03)




def predict_proba(dequeue=True):
    global h

    imgs, prior = pipeline.get_model_input(dequeue=dequeue)
    sample = tsfm({'imgs': imgs, 'priors': prior})

    # Move all inputs to the correct device
    sample['imgs'] = sample['imgs'].to(device)
    sample['maps'] = sample['maps'].to(device)
    h = tuple(t.to(device) for t in h)  # make sure h0 and h1 (LSTM) are also on device
    print(f"sample['imgs'] on {sample['imgs'].device}")
    print(f"Model on {next(encoder.parameters()).device}")

    try:
        with torch.no_grad():
            print("sample['imgs'].device:", sample['imgs'].device)
            print("sample['maps'].device:", sample['maps'].device)
            print("hidden[0] device:", h[0].device)
            print("hidden[1] device:", h[1].device)
            print("Model param device:", next(encoder.parameters()).device)
            probs_tensor, h = encoder(sample['imgs'], h, sample['maps'])
    except Exception as e:
        print(f"[Server] Error during inference: {e}")
        raise


    return probs_tensor.cpu().numpy().squeeze()


def greedy_decode(probs, sentence, last_letter):
    letter = inv_vocab_map[np.argmax(probs)]
    if letter != '_' and letter != last_letter:
        sentence += letter.upper()
        return letter, sentence, True
    return letter, sentence, False


if not torch.cuda.is_available():
    raise RuntimeError("[Server] CUDA is not available. GPU is required for this setup.")

device = torch.device('cuda')
print(f"[Server] Using device: {device}")

app = Flask(__name__, template_folder='templates', static_folder='static')

# Will hold our pipeline once we create it
pipeline = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/animation_feed')
def animation_feed():
    # resource intensive
    return Response(gen_animation(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat_from_gesture', methods=['GET'])
def chat_from_gesture():
    global gesture_sentence, current_sentence
    current_gesture = gesture_sentence.strip()

    if not current_gesture:
        return jsonify({"error": "No gesture sentence available."}), 400

    # Process the gesture sentence using functions from your chatbot module
    english_interpretation = chatbot.chatbot.interpret_asl_input(current_gesture)
    english_response = chatbot.chatbot.generate_english_response(english_interpretation)
    asl_response = chatbot.chatbot.generate_asl_response(english_response)

    # Update the sentence for animation
    current_sentence = asl_response

    return jsonify({
        "gesture_sentence": current_gesture,
        "english_interpretation": english_interpretation,
        "english_response": english_response,
        "asl_response": asl_response
    })

@app.route('/gesture_sentence')
def get_gesture_sentence():
    global gesture_sentence
    return jsonify({'gesture_sentence': gesture_sentence.strip()})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global is_recording, is_detecting, frame_buffer, sentence_buffer
    with lock:
        # Toggle detection state
        is_detecting = not is_detecting
        is_recording = not is_recording
        print("[DEBUG] New state: is_detecting =", is_detecting, "is_recording =", is_recording)
        # When turning off detection, clear stale data
        if not is_detecting:
            frame_buffer.clear()
            sentence_buffer.clear()
    return jsonify({'is_detecting': is_detecting, 'is_recording': is_recording})

@app.route('/reset_gesture', methods=['POST'])
def reset_gesture():
    global gesture_sentence, current_sentence, sentence
    with lock:
        gesture_sentence = ""
        current_sentence = ""
        sentence = ""
        # Also clear the buffer that holds recognized words
        from asl_recognition.test import sentence_buffer
        sentence_buffer.clear()
    return jsonify({'gesture_sentence': gesture_sentence})

@app.route('/translate_english', methods=['POST'])
def translate_english():
    data = request.get_json()
    english_sentence = data.get("english_sentence", "").strip()
    if not english_sentence:
        return jsonify({"error": "No English sentence provided."}), 400

    try:
        # Directly use the English sentence as input (skip gesture interpretation)
        english_response = chatbot.chatbot.generate_english_response(english_sentence)
        asl_response = chatbot.chatbot.generate_asl_response(english_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Update global current_sentence for the animation feed, if applicable.
    global current_sentence
    current_sentence = asl_response

    return jsonify({
        "english_sentence": english_sentence,
        "english_response": english_response,
        "asl_response": asl_response
    })


@app.route('/frame', methods=['POST'])
def handle_frame():
    global pipeline, encoder, h0, h, tsfm, inv_vocab_map, device, cfg, is_detecting, sentence

    # Load config once
    if cfg is None:
        print("[Server] Loading config from asl_response/fingerspelling/conf.ini")
        cfg = configparser.ConfigParser()
        cfg.read('asl_recognition/fingerspelling/conf.ini')

        print("[Server] Config sections and values:")
        for section in cfg.sections():
            print(f"  [{section}]")
            for key, value in cfg[section].items():
                print(f"    {key} = {value}")

    # Initialize pipeline once
    if pipeline is None:
        print("[Server] Initializing VideoProcessingPipeline…")
        model_cfg = cfg['MODEL']
        img_cfg   = cfg['IMAGE']

        pipeline = VideoProcessingPipeline(
            img_size     = model_cfg.getint('img_size'),
            img_cfg      = img_cfg,
            frames_window= model_cfg.getint('frames_window', 13),
            flows_window = model_cfg.getint('flows_window', 5),
            skip_frames  = model_cfg.getint('skip_frames', 2),
            denoising    = bool(model_cfg.getint('denoising', 1))
        )
        print("[Server] Pipeline initialized.")

        # Register cleanup
        atexit.register(lambda: (
            print("[Server] Terminating pipeline..."),
            pipeline.terminate()
        ))

    # Initialize model once
    if encoder is None:
        print("[Server] Initializing model...")
        model_cfg = cfg['MODEL']
        img_cfg   = cfg['IMAGE']
        lang_cfg  = cfg['LANG']
        char_list = lang_cfg['chars']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, inv_vocab_map, char_list = get_ctc_vocab(lang_cfg['chars'])
        print(f"[Server] char_list = {char_list}")
        print(f"[Server] len(char_list) = {len(char_list)}")

        tsfm = transforms.Compose([
            PriorToMap(model_cfg.getint('map_size')),
            ToTensor(),
            Normalize(
                [float(x) for x in img_cfg['img_mean'].split(',')],
                [float(x) for x in img_cfg['img_std'].split(',')]
            ),
            Batchify()
        ])

        h0 = init_lstm_hidden(1, model_cfg.getint('hidden_size'), device=device)
        h = h0

        encoder = MiCTRANet(
            backbone    = model_cfg.get('backbone'),
            hidden_size = model_cfg.getint('hidden_size'),
            attn_size   = model_cfg.getint('attn_size'),
            output_size = len(get_ctc_vocab(lang_cfg['chars'])[2]), 
            mode        = 'online'
        ).to(device)  # <- move to CUDA first

        encoder.load_state_dict(torch.load(model_cfg['model_pth'], map_location=device))
        encoder.eval()
        #print(f"[Server] Model weights on: {next(encoder.parameters()).device}")
        for param in encoder.parameters():
            assert param.device == device, f"Param still on {param.device}, expected {device}"
        encoder.eval()
        inv_vocab_map = inv_vocab_map
        #print(f"[Server] Model loaded on device: {next(encoder.parameters()).device}")

    # Decode incoming frame
    img = Image.open(BytesIO(request.data)).convert('RGB')
    arr = np.array(img)
    #print(f"[Server] Received frame, shape={arr.shape}")
    if(is_detecting):
        pipeline.enqueue_frame(arr)
    now = time.time()
    fps_buffer.append(now)

    # Keep only recent 30 timestamps
    if len(fps_buffer) > 30:
        fps_buffer.pop(0)

    # Calculate instantaneous FPS
    if len(fps_buffer) > 1:
        elapsed = fps_buffer[-1] - fps_buffer[0]
        fps = (len(fps_buffer) - 1) / elapsed
        print(f"[Server] Approx FPS: {fps:.2f}")
    #print(f"[Debug] Frames: {len(pipeline.img_frames)} / {pipeline.frames_window}")
    #print(f"[Debug] Priors: {len(pipeline.priors)} / 1")
    # print("[Debug] Should run inference:",
    #   pipeline.total_frames >= pipeline.frames_window * pipeline.skip_frames,
    #   "and", len(pipeline.priors) >= 1)
    #print(f"[Server] Enqueued frame #{pipeline.total_frames}")

    # Run inference if enough frames
    global sentence, last_letter, gesture_sentence

# Try to fetch new priors (if available)
    try:
        prior, _ = pipeline.q_parent.get(block=False)
        pipeline.priors.append(prior)
        #print(f"[Server] Dequeued prior → Total stored: {len(pipeline.priors)}")
    except Exception:
        pass

    # Then run inference if ready
    if (
        pipeline.total_frames >= pipeline.frames_window * pipeline.skip_frames
        and len(pipeline.priors) >= 1
    ):
        probs = predict_proba(dequeue=False)  # don't dequeue again inside
        last_letter, sentence, new_letter = greedy_decode(probs, sentence, last_letter)
        print(f"[Server] Predicted: {last_letter}, Full Sentence: {sentence}")
        gesture_sentence = sentence
        return {"prediction": last_letter, "sentence": sentence}


    return Response(status=204)

if __name__ == '__main__':
    # Required on Windows for multiprocessing.spawn
    freeze_support()
    app.run(
        host='0.0.0.0',
        port=5000,
        ssl_context=('cert.pem', 'private.key'),  # Path to your SSL certificate and key
    )
