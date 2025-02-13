import torch as th
import numpy as np
import random
from collections import deque
from models.TGCN.tgcn_model import GCN_muti_att  # Import your model class

def load_model(model_path, input_feature=100):
    model = GCN_muti_att(input_feature=input_feature, hidden_feature=256, num_class=2000, p_dropout=0.3, num_stage=24)
    model.load_state_dict(th.load(model_path, map_location=th.device('cpu')))
    model.eval()
    return model

def load_text_model(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return markovify.NewlineText(text, state_size=1)

def make_prediction(
    frame_buffer: deque,
    model: th.nn.Module,
    class_mapping: dict,
    frames_to_extract: int = 50,
    input_shape: tuple[int, int] = (55, 100)
) -> list[list[str, float]]:
    # Processes the frame buffer to make a prediction using the trained model.

    # Args:
    #     frame_buffer (deque): A deque containing preprocessed frames.
    #     model (torch.nn.Module): The trained model for making predictions.
    #     class_mapping (dict): A dictionary mapping class indices to class names.
    #     frames_to_extract (int, optional): Number of frames to sample from the buffer. Defaults to 50.
    #     input_shape (Tuple[int, int], optional): The shape to which input data is reshaped. Defaults to (55, 100).

    # Returns:
    #     List[List[str, float]]: A 2D list where each sublist contains a word and its corresponding probability.
    
    if len(frame_buffer) < frames_to_extract:
        raise ValueError("Not enough frames in the buffer to perform prediction.")

    # Sample frames from the buffer
    sampled_frames = random.sample(list(frame_buffer), frames_to_extract)

    # Stack and reshape the sampled frames
    input_data = np.stack(sampled_frames).T
    input_data = input_data.reshape(*input_shape)

    # Convert to tensor and add batch dimension
    input_tensor = th.tensor(input_data, dtype=th.float32).unsqueeze(0)

    # Perform inference
    model.eval()
    with th.no_grad():
        outputs = model(input_tensor)
        probabilities = th.softmax(outputs, dim=1)
        top_probs, top_classes = th.topk(probabilities, k=5, dim=1)
        top_probs = top_probs[0].cpu().numpy()
        top_classes = top_classes[0].cpu().numpy()

    # Map class indices to labels and pair with probabilities
    top_predictions = [
        [class_mapping.get(cls, "Unknown"), float(prob)]
        for cls, prob in zip(top_classes, top_probs)
    ]

    return top_predictions