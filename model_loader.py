import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# EXACT same architecture you trained
class Model(nn.Module):
    def __init__(self, num_classes: int = 2, latent_dim: int = 2048,
                 lstm_layers: int = 1, hidden_dim: int = 2048,
                 bidirectional: bool = False):
        super(Model, self).__init__()
        # Backbone: ResNeXt50; use weights=None to avoid deprecated 'pretrained'
        base = models.resnext50_32x4d(weights=None)
        # Remove last 2 layers (same as training script)
        self.model = nn.Sequential(*list(base.children())[:-2])

        # Keep LSTM arguments identical to your training (bias issue included)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        x: [B, T, 3, 112, 112]
        returns: fmap (unused here), logits [B, num_classes]
        """
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x)                       # [B*T, 2048, H', W']
        x = self.avgpool(fmap)                     # [B*T, 2048, 1, 1]
        x = x.view(batch_size, seq_length, 2048)   # [B, T, 2048]

        # LSTM
        x_lstm, _ = self.lstm(x, None)
        out = self.dp(self.linear1(torch.mean(x_lstm, dim=1)))  # [B, num_classes]
        return fmap, out


def load_model(model_path: str, device: str = None):
    """
    Load the trained model checkpoint safely (CPU/GPU).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(num_classes=2)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # Handle state_dict vs full model saving
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")

    model.to(device)
    model.eval()
    return model, device


def predict_video(model: nn.Module, device: str, frames_tensor: torch.Tensor):
    """
    Predict real/fake label and confidence for given video frames.
    frames_tensor: [1, T, 3, 112, 112]
    """
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        _, logits = model(frames_tensor)  # [1, 2]
        probs = F.softmax(logits, dim=1).squeeze(0)  # [2]
        pred_idx = int(torch.argmax(probs).item())
        conf = float(probs[pred_idx].item())
        label = "REAL" if pred_idx == 1 else "FAKE"
    return label, conf
