import time
import collections
import numpy as np
import cv2 as cv
from torch.multiprocessing import Process, Queue


def denoise_frame(q_denoise, q_prior, h=3, template_window_size=7, search_window_size=21):
    """
    Worker process to denoise incoming gray frames.
    """
    while True:
        gray = q_denoise.get()
        if gray is None:
            # signal end of stream
            q_prior.put(None)
            return
        # apply fastNlMeansDenoising
        clean = cv.fastNlMeansDenoising(
            np.uint8(np.clip(gray, 0, 255)),
            None,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
        q_prior.put(clean)


def calc_attention_prior(opt_size, flows_window, q_prior, q_parent):
    """
    Worker process to compute optical-flow attention priors.
    """
    prev_gray = None
    buffer    = []
    while True:
        next_gray = q_prior.get()
        #print("[Flow] Received grayscale frame for flow computation")
        if next_gray is None:
            return
        if prev_gray is not None:
            flow = cv.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5, levels=3,
                winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # normalize magnitude
            norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
            norm = np.nan_to_num(norm)
            buffer.append(norm)
        if len(buffer) >= flows_window:
            prior = (255 * np.mean(buffer[-flows_window:], axis=0)).astype('uint8')
            #print("[Flow] Pushing prior to q_parent")  # ADD THIS
            q_parent.put((prior, None))
            #print("[Flow] Prior pushed")               # AND THIS
            buffer.pop(0)
        prev_gray = next_gray


class VideoProcessingPipeline:
    """
    Pipeline for processing incoming RGB frames from a web client.

    Methods:
      - enqueue_frame(frame: np.ndarray): queue a new frame for processing
      - get_model_input(dequeue: bool = True): return (imgs, [prior]) ready for the model
      - terminate(): gracefully shut down processes
    """
    def __init__(self,
                 img_size: int,
                 img_cfg,
                 frames_window: int = 13,
                 flows_window: int = 5,
                 skip_frames: int = 2,
                 denoising: bool = True):
        # configuration
        self.img_size      = img_size
        self.opt_size      = img_size // 2
        self.frames_window = frames_window
        self.flows_window  = flows_window
        self.skip_frames   = skip_frames
        self.denoising     = denoising

        # buffers
        self.total_frames = 0
        self.img_frames   = collections.deque(maxlen=frames_window)
        self.gray_frames  = collections.deque(maxlen=frames_window)
        self.priors       = []

        # queues & background workers
        self.q_parent = Queue()
        self.q_prior  = Queue()
        if denoising:
            self.q_denoise = Queue()
            self.p_denoise = Process(
                target=denoise_frame,
                args=(self.q_denoise,
                      self.q_prior,
                      img_cfg.getint('h'),
                      img_cfg.getint('template_window_size'),
                      img_cfg.getint('search_window_size'))
            )
            self.p_denoise.start()
        self.p_prior = Process(
            target=calc_attention_prior,
            args=(self.opt_size,
                  flows_window,
                  self.q_prior,
                  self.q_parent)
        )
        self.p_prior.start()

    def enqueue_frame(self, frame: np.ndarray):
        """
        Queue a new RGB frame (H×W×3 uint8) for processing.
        Automatically applies skip_frames downsampling.
        """
        self.total_frames += 1
        if (self.total_frames % self.skip_frames) != 0:
            return
        # preprocess
        self._preprocess_frame(frame)

    def _center_crop(self, img: np.ndarray, target_shape: tuple):
        h, w = target_shape
        y, x = img.shape[:2]
        start_y = max(0, y//2 - h//2)
        start_x = max(0, x//2 - w//2)
        return img[start_y:start_y+h, start_x:start_x+w]

    def _preprocess_frame(self, frame: np.ndarray):
        # convert BGR→RGB
        rgb  = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # center-crop square
        side = min(rgb.shape[0], rgb.shape[1])
        rgb  = self._center_crop(rgb, (side, side))
        # grayscale & resize for prior
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (self.opt_size, self.opt_size))
        # resize RGB for model
        rgb  = cv.resize(rgb, (self.img_size, self.img_size))

        # send to denoiser or directly to prior
        if self.denoising:
            self.q_denoise.put(gray)
        else:
            self.q_prior.put(gray)

        # store for model
        self.img_frames.append(rgb)
        self.gray_frames.append(gray)

    def get_model_input(self, dequeue: bool = True):
        if dequeue and not self.q_parent.empty():
            prior, _ = self.q_parent.get(block=False)
            self.priors.append(prior)
            print(f"[Pipeline] Dequeued prior. Total now: {len(self.priors)}")

        assert len(self.img_frames)  >= self.frames_window
        assert len(self.gray_frames) >= self.frames_window
        assert len(self.priors)      >= 1

        imgs  = np.stack(list(self.img_frames)[-self.frames_window:], axis=0)
        prior = self.priors.pop(0)
        return imgs, [prior]


    def terminate(self):
        """
        Gracefully stop background processes.
        """
        if self.denoising:
            self.q_denoise.put(None)
            time.sleep(0.1)
            self.p_denoise.terminate()
        self.q_prior.put(None)
        time.sleep(0.1)
        self.p_prior.terminate()
        time.sleep(0.1)
        self.q_parent.close()
        if self.denoising:
            self.q_denoise.close()
