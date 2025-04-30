import time
from datetime import datetime
import os
import argparse
import configparser
import collections  
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch.multiprocessing import Process, Queue
from torchvision import transforms

from asl_recognition.fingerspelling.chicago_fs_wild import (PriorToMap, ToTensor, Normalize, Batchify)
from asl_recognition.fingerspelling.mictranet import *
from asl_recognition.fingerspelling.utils import get_ctc_vocab

ESCAPE_KEY = 27
SPACE_KEY = 32
ENTER_KEY = 13
BACKSPACE_KEY = 8
DELETE_KEY = 255
CAM_RES = (640, 480)

class VideoProcessingPipeline(object):
    def __init__(self, img_size, img_cfg, frames_window=13, flows_window=5,
                 skip_frames=2, cam_res=(640, 480), denoising=True):
        if frames_window not in [9, 13, 17, 21]:
            raise ValueError('Invalid window size for webcam frames: `%s`' % str(frames_window))
        if flows_window not in [3, 5, 7, 9]:
            raise ValueError('Invalid window size for optical flows: `%s`' % str(flows_window))
        if flows_window > frames_window:
            raise ValueError('Optical flow window cannot be wider than camera frames window')
        self.img_size = img_size
        self.opt_size = img_size // 2
        self.frames_window = frames_window
        self.flows_window = flows_window
        self.skip_frames = skip_frames
        self.total_frames = 0
        self.cam_res = cam_res
        self.denoising = denoising
        self.img_frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)] * (self.frames_window // 2)
        self.gray_frames = [np.zeros((self.opt_size, self.opt_size), dtype=np.uint8)] * (self.frames_window // 2)
        self.priors = []
        self.q_parent, self.q_prior = Queue(), Queue()
        if self.denoising:
            self.q_denoise = Queue()
            self.p_denoise = Process(target=denoise_frame,
                                     args=(self.q_denoise, self.q_prior,
                                           img_cfg.getint('h'),
                                           img_cfg.getint('template_window_size'),
                                           img_cfg.getint('search_window_size')))
            self.p_denoise.start()
            print('Denoising enabled')
        else:
            print('Denoising disabled')
        self.p_prior = Process(target=calc_attention_prior,
                               args=(self.opt_size, self.flows_window,
                                     self.q_prior, self.q_parent))
        self.p_prior.start()
        self.cap = cv.VideoCapture(0)
        if self.cap.isOpened():
            self.cap_fps = int(round(self.cap.get(cv.CAP_PROP_FPS)))
            self.cap.set(3, self.cam_res[0])
            self.cap.set(4, self.cam_res[1])
            print('Device @%d FPS' % self.cap_fps)
        else:
            raise IOError('Failed to open webcam capture')
        self.last_frame = collections.deque(maxlen=self.cap_fps)
        self.last_cropped_frame = collections.deque(maxlen=self.cap_fps)
        for i in range((frames_window // 2) + 1):
            self.acquire_next_frame(enable_skip=False)
        while len(self.priors) == 0:
            if not self.q_parent.empty():
                prior, flow = self.q_parent.get(block=False)
                self.priors.append(prior)
            time.sleep(0.01)

    def _center_crop(self, img, target_shape):
        h, w = target_shape
        y, x = img.shape[:2]
        start_y = max(0, y // 2 - (h // 2))
        start_x = max(0, x // 2 - (w // 2))
        return img[start_y:start_y + h, start_x:start_x + w]

    def acquire_next_frame(self, enable_skip=True):
        ret, frame = self.cap.read()
        if not ret:
            self.terminate()
            raise IOError('Failed to read the next frame from webcam')
        self.total_frames += 1
        if not enable_skip:
            return self._preprocess_frame(frame)
        elif (self.total_frames % self.skip_frames) == 0:
            return self._preprocess_frame(frame)
        return None

    def _preprocess_frame(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb = self._center_crop(rgb, (self.cam_res[1], self.cam_res[1]))
        self.last_frame.append(frame)
        self.last_cropped_frame.append(rgb)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (self.opt_size, self.opt_size))
        rgb = cv.resize(rgb, (self.img_size, self.img_size))
        if self.denoising:
            self.q_denoise.put(gray)
        else:
            self.q_prior.put(gray)
        self.img_frames.append(rgb)
        self.gray_frames.append(gray)
        return frame

    def get_model_input(self, dequeue=True):
        if dequeue:
            prior, flow = self.q_parent.get(block=False)
            self.priors.append(prior)
        n_frames = self.frames_window
        assert len(self.img_frames) >= n_frames
        assert len(self.gray_frames) >= n_frames
        assert len(self.priors) == 1
        imgs = np.stack(self.img_frames[:self.frames_window], axis=0)
        self.img_frames.pop(0)
        self.gray_frames.pop(0)
        return imgs, [self.priors.pop(0)]

    def terminate(self):
        if self.denoising:
            self.q_denoise.put(None)
            time.sleep(0.2)
            self.p_denoise.terminate()
        else:
            self.q_prior.put(None)
            time.sleep(0.2)
        self.p_prior.terminate()
        time.sleep(0.1)
        if self.denoising:
            self.p_denoise.join(timeout=0.5)
        self.p_prior.join(timeout=0.5)
        if self.denoising:
            self.q_denoise.close()
        self.q_parent.close()
        self.cap.release()

def denoise_frame(q_denoise, q_prior, h=3, template_window_size=7, search_window_size=21):
    while True:
        while not q_denoise.empty():
            gray = q_denoise.get(block=False)
            if gray is None:
                q_prior.put(None)
                print('Exiting denoising process')
                return 0
            gray = cv.fastNlMeansDenoising(np.uint8(np.clip(gray, 0, 255)),
                                           None, h=h,
                                           templateWindowSize=template_window_size,
                                           searchWindowSize=search_window_size)
            q_prior.put(gray)
        time.sleep(0.01)

def calc_attention_prior(opt_size, flows_window, q_prior, q_parent):
    prev_gray = None
    opt_flows = [np.zeros((opt_size, opt_size), dtype=np.uint8)] * (1 + flows_window // 2)
    while True:
        while not q_prior.empty():
            next_gray = q_prior.get(block=False)
            if next_gray is None:
                print('Exiting optical flow process')
                return 0
            if prev_gray is not None:
                flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                                   pyr_scale=0.5, levels=3,
                                                   winsize=15, iterations=3,
                                                   poly_n=5, poly_sigma=1.2, flags=0)
                mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
                if (mag.max() - mag.min()) == 0:
                    flow = np.zeros_like(mag)
                elif mag.max() == np.inf:
                    mag = np.nan_to_num(mag, copy=True, posinf=mag.min())
                    flow = (mag - mag.min()) / float(mag.max() - mag.min())
                else:
                    flow = (mag - mag.min()) / float(mag.max() - mag.min())
                prev_gray = next_gray
                opt_flows.append(flow)
                if len(opt_flows) < flows_window:
                    continue
                flows = np.stack(opt_flows, axis=0)
                prior = 255 * np.mean(flows, axis=0)
                prior = prior.astype('uint8')
                q_parent.put((prior, opt_flows[flows_window // 2]))
                opt_flows.pop(0)
            else:
                prev_gray = next_gray
        time.sleep(0.01)

class PlayerWindow(object):
    def __init__(self, vpp, inv_vocab_map, char_list):
        self.vpp = vpp
        self.inv_vocab_map = inv_vocab_map
        # Filter out unwanted characters
        self.char_list = [c for c in char_list if c not in ("'", '&', '.', '@')]
        self.window_name = "Fingerspelling Practice"
        cv.namedWindow(self.window_name)
        cv.moveWindow(self.window_name, 450, 280)
        self.last_frame = None

    def draw_banner(self, frame, fps=None, banner_color=(0,235,235), is_recording=False, rec_color=(255,0,0)):
        # FPS bar drawing:
        # if fps is not None:
        #     cv.putText(frame, '{:>2.1f}FPS'.format(fps), (10,25),
        #                cv.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 1, cv.LINE_AA)
        if is_recording:
            cv.circle(frame, (460,20), 10, rec_color, -1, cv.LINE_AA)
        return frame

    def draw_canvas(self, frame, probs, pred, n_lines, is_recording, fps):
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = self.draw_banner(frame, fps, (255,105,57), is_recording, (255,105,57))
        if pred is not None:
            # cv.putText(frame, pred, (10, frame.shape[0]-20),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv.LINE_AA)
            print("Prediction:", pred)
        # The following block draws letters and bars is commented out.
        """
        if probs is not None:
            max_width = 100
            bar_height = 15
            start_x = 10
            start_y = 40
            for i, letter in enumerate(self.char_list):
                prob = probs[i] if i < len(probs) else 0
                bar_width = int(prob * max_width)
                cv.rectangle(frame, (start_x, start_y + i*(bar_height+5)),
                             (start_x+bar_width, start_y+i*(bar_height+5)+bar_height),
                             (119,199,105), -1)
                cv.putText(frame, letter, (start_x+max_width+10,
                             start_y + i*(bar_height+5)+bar_height-2),
                             cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv.LINE_AA)
        """
        self.last_frame = frame.copy()
        cv.imshow(self.window_name, frame)

    def save_frame(self, outdir, n_frames):
        if self.last_frame is not None:
            cv.imwrite(os.path.join(outdir, 'plot', '{:04d}.png'.format(n_frames)), self.last_frame)
            cv.imwrite(os.path.join(outdir, 'raw', '{:04d}.png'.format(n_frames)), self.last_frame)

def main():
    parser = argparse.ArgumentParser(description='Fingerspelling Practice')
    parser.add_argument('--conf', type=str, default='conf.ini', help='configuration file')
    parser.add_argument('--gpu_id', type=str, default='0', help='CUDA enabled GPU device (default: 0)')
    parser.add_argument('--frames_window', type=int, default=13, help='images window size used for each prediction step')
    parser.add_argument('--flows_window', type=int, default=5, help='optical flow window size used to calculate attention prior')
    parser.add_argument('--skip_frames', type=int, default=2, help='video frames downsampling ratio')
    parser.add_argument('--denoising', type=int, default=1, help='denoise frames from low quality webcams: 1 for True, 0 for False')
    args = parser.parse_args()
    frames_window = args.frames_window
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg = config['MODEL'], config['LANG']
    img_cfg, data_cfg = config['IMAGE'], config['DATA']
    char_list = lang_cfg['chars']
    hidden_size = model_cfg.getint('hidden_size')
    attn_size = model_cfg.getint('attn_size')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Compute device: ' + device)
    device = torch.device(device)
    vocab_map, inv_vocab_map, char_list = get_ctc_vocab(char_list)
    img_mean = [float(x) for x in img_cfg['img_mean'].split(',')]
    img_std = [float(x) for x in img_cfg['img_std'].split(',')]
    tsfm = transforms.Compose([PriorToMap(model_cfg.getint('map_size')),
                               ToTensor(),
                               Normalize(img_mean, img_std),
                               Batchify()])
    print('Loading model from: %s' % model_cfg['model_pth'])
    h0 = init_lstm_hidden(1, hidden_size, device=device)
    h = h0
    encoder = MiCTRANet(backbone=model_cfg.get('backbone'),
                        hidden_size=hidden_size,
                        attn_size=attn_size,
                        output_size=len(char_list),
                        mode='online')
    encoder.load_state_dict(torch.load(model_cfg['model_pth']))
    encoder.to(device)
    encoder.eval()
    vpp = VideoProcessingPipeline(model_cfg.getint('img_size'), img_cfg,
                                  frames_window=args.frames_window,
                                  flows_window=args.flows_window,
                                  skip_frames=args.skip_frames,
                                  denoising=bool(args.denoising))
    pw = PlayerWindow(vpp, inv_vocab_map, char_list)
    def predict_proba(h, dequeue):
        imgs, prior = vpp.get_model_input(dequeue=dequeue)
        sample = tsfm({'imgs': imgs, 'priors': prior})
        with torch.no_grad():
            probs, h = encoder(sample['imgs'].to(device), h,
                               sample['maps'].to(device))
        p = probs.cpu().numpy().squeeze()
        return p, h
    def greedy_decode(probs, sentence, last_letter):
        letter = inv_vocab_map[np.argmax(probs)]
        if (letter is not '_') & (last_letter != letter):
            sentence += letter.upper()
            return letter, sentence, True
        else:
            return letter, sentence, False
    prob, h = predict_proba(h, dequeue=False)
    torch.cuda.synchronize()
    _ = vpp.last_frame.popleft()
    last_cropped_frame = vpp.last_cropped_frame.popleft()
    pw.draw_canvas(last_cropped_frame, np.zeros(28), pred=None,
                   n_lines=1, is_recording=False, fps=None)
    run_times = collections.deque([(vpp.cap_fps * args.skip_frames) / 1000] * 3,
                                  maxlen=2 * vpp.cap_fps)
    frame_start = time.perf_counter()
    last_letter = '_'
    sentence = ''
    n_lines = 0
    is_recording = False
    outdir = None
    n_frames = 0
    while True:
        frame = vpp.acquire_next_frame()
        while (not vpp.q_parent.empty()) & (len(vpp.img_frames) >= frames_window):
            probs, h = predict_proba(h, dequeue=True)
            last_letter, sentence, new_letter_found = greedy_decode(probs, sentence, last_letter)
            _ = vpp.last_frame.popleft()
            last_cropped_frame = vpp.last_cropped_frame.popleft()
            pw.draw_canvas(last_cropped_frame, probs, sentence, n_lines,
                           is_recording, 1 / np.mean(run_times))
            if is_recording:
                n_frames += 1
                pw.save_frame(outdir, n_frames)
        key = cv.waitKey(1)
        if key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                outdir = os.path.join('data', 'recordings',
                                      str(datetime.now()).split('.')[0].replace(' ', '@'))
                os.makedirs(os.path.join(outdir, 'plot'))
                os.makedirs(os.path.join(outdir, 'raw'))
                h = h0
                last_letter = '_'
                sentence = ''
                n_lines = 0
            if not is_recording:
                n_frames = 0
        elif key == SPACE_KEY:
            h = h0
            last_letter = '_'
            sentence += ' '
        elif key == ENTER_KEY:
            h = h0
            last_letter = '_'
            sentence += '\n'
            if n_lines == 2:
                sentence = '\n'.join(sentence.split('\n')[1:])
            else:
                n_lines += 1
        elif key == BACKSPACE_KEY:
            if len(sentence) > 0:
                h = h0
                last_letter = '_'
                end_letter = sentence[-1]
                sentence = sentence[:-1]
                if end_letter == '\n':
                    n_lines -= 1
        elif key == DELETE_KEY:
            h = h0
            last_letter = '_'
            sentence = ''
            n_lines = 0
        elif (key == ESCAPE_KEY) | (key == ord('q')) | (key == ord('x')):
            break
        if frame is not None:
            frame_end = time.perf_counter()
            run_times.append(frame_end - frame_start)
            frame_start = frame_end
    vpp.terminate()
    cv.destroyAllWindows()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()


