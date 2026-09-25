import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
from src.tasks.config.utils import CONFIG, DEVICE
from memfof import MEMFOF

def load_flow_mappings():
    mapping_path = Path("src/tasks/config/flow_mappings.json")
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            return json.load(f)
    return {}

FLOW_MAPPINGS = load_flow_mappings()

def load_model(model_name=CONFIG["flow_estimation"]["model_name"], device=DEVICE):
    model = MEMFOF.from_pretrained(model_name).eval().to(device)
    return model

# Lazy model to avoid consuming GPU at import time
_flow_model = None

def unload_model():
    global _flow_model
    _flow_model = None
    import torch
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _get_flow_model():

    global _flow_model
    if _flow_model is None:
        _flow_model = load_model()
    return _flow_model

def predict_flows(frames, model=None):
    """
    Process a sequence of frames and predict optical flow using the full temporal context.

    Args:
        frames (list[np.ndarray] or torch.Tensor): Sequence of frames.
                                                   Expected shape [N, C, H, W] or list of [H, W, C].
        model: The MEMFOF model. If None, uses the lazy loaded model.

    Returns:
        forward_flows (list[np.ndarray]): List of forward flow maps [C=2, H, W].
    """
    if model is None:
        model = _get_flow_model()

    # Convert frames to torch tensor [N, C, H, W] if they are a list of [H, W, C]
    if isinstance(frames, list):
        # Assume images are [H, W, C] and uint8
        tensor_frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    elif isinstance(frames, torch.Tensor):
        tensor_frames = frames.float()
    else:
        raise TypeError("frames must be a list of np.ndarray or a torch.Tensor")

    num_frames = tensor_frames.shape[0]
    if num_frames < 3:
        return [np.zeros((2, tensor_frames.shape[2], tensor_frames.shape[3]), dtype=np.float32)] * num_frames

    with torch.inference_mode():
        # Pass the entire sequence as a single batch: [Batch=1, Time=N, Channels=3, H, W]
        input_tensor = tensor_frames.unsqueeze(0).to(DEVICE)

        # MEMFOF can crash if input resolution is too small for its pyramid levels
        # Resize to a larger standard size to ensure stability (e.g., 384x384)
        target_h, target_w = 384, 384
        B, T, C, H, W = input_tensor.shape

        if H != target_h or W != target_w:
            # Reshape to [B*T, C, H, W] to use 2D interpolation
            input_tensor = input_tensor.view(B * T, C, H, W)
            input_tensor = torch.nn.functional.interpolate(
                input_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            # Reshape back to [B, T, C, target_h, target_w]
            input_tensor = input_tensor.view(B, T, C, target_h, target_w)

        print(f"DEBUG: MEMFOF input shape (resized): {input_tensor.shape}")

        output = model(input_tensor)

        # Extract the final refined flow prediction
        # output["flow"][-1] is the final flow prediction tensor
        flow_tensor_raw = output["flow"][-1]

        # Remove batch dimension and move to CPU
        # Expected raw shape: [1, C=2, T=N-2, H, W] or [1, T=N-2, C=2, H, W]
        flow_tensor = flow_tensor_raw.squeeze(0).cpu().numpy()

        # Check dimensions and normalize to [T, C, H, W]
        # If [C=2, T, H, W], move C to second dim
        if flow_tensor.ndim == 4 and flow_tensor.shape[0] == 2:
            # [2, T, H, W] -> [T, 2, H, W]
            flow_tensor = np.moveaxis(flow_tensor, 0, 1)

        # Now it should be [T, 2, H, W]
        if flow_tensor.ndim != 4 or flow_tensor.shape[1] != 2:
            raise ValueError(f"Unexpected flow tensor shape after processing: {flow_tensor.shape}. Expected [T, 2, H, W]")

        # Resize back to original H, W if resizing was applied
        if H != target_h or W != target_w:
            # flow_tensor shape: [T, 2, H_resized, W_resized]
            # torch.nn.functional.interpolate expects [B, C, H, W]
            # we can process each frame in the sequence
            resized_flows = []
            for i in range(flow_tensor.shape[0]):
                frame_flow = torch.from_numpy(flow_tensor[i]).unsqueeze(0)
                frame_flow = torch.nn.functional.interpolate(
                    frame_flow, size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0).numpy()
                resized_flows.append(frame_flow)
            flow_tensor = np.stack(resized_flows)

        forward_flows = [flow_tensor[i] for i in range(flow_tensor.shape[0])]

    # Pad the end to match the original number of frames (since flow is between frames)
    while len(forward_flows) < num_frames:
        forward_flows.append(np.zeros_like(forward_flows[-1] if forward_flows else np.zeros((2, 0, 0))))

    return forward_flows

def mode_flow(flow_map, segmentation_mask):
    """
    Calculate the average flow vector for a segmented object.

    Args:
        flow_map (np.ndarray): Flow map of shape [2, H, W] (u, v components).
        segmentation_mask (np.ndarray): Binary mask (True for object pixels).

    Returns:
        tuple | None: (avg_u, avg_v) representing the flow orientation and magnitude.
    """
    binary_mask = segmentation_mask > 0

    if not np.any(binary_mask):
        return None

    # Extract flow values for the mask
    # flow_map[0] is u, flow_map[1] is v
    u_vals = flow_map[0][binary_mask]
    v_vals = flow_map[1][binary_mask]

    avg_u = np.mean(u_vals)
    avg_v = np.mean(v_vals)

    return (avg_u, avg_v)

def describe_flow(flow_vector, side="front"):
    """
    Convert a flow vector (avg_u, avg_v) into a textual description relative to the environment.
    """
    if flow_vector is None:
        return "no flow data"

    u, v = flow_vector
    magnitude = np.sqrt(u**2 + v**2)

    if magnitude < 0.5:
        return "stationary"

    # Get mapping for the specific camera side
    mapping = FLOW_MAPPINGS.get(side, FLOW_MAPPINGS.get("front"))

    # Horizontal (u)
    if u > 0.5:
        h_dir = mapping.get("u_pos", "right")
    elif u < -0.5:
        h_dir = mapping.get("u_neg", "left")
    else:
        h_dir = "stable horizontally"

    # Vertical (v)
    if v > 0.5:
        v_dir = mapping.get("v_pos", "down")
    elif v < -0.5:
        v_dir = mapping.get("v_neg", "up")
    else:
        v_dir = "stable vertically"

    return f"moving {h_dir} and {v_dir} (relative to environment)"
