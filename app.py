import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# 1. Basic setup
# ----------------------------------------------------
app = FastAPI(title="Solar Flare Prediction API")

# Allow calls from browser etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your saved model (checkpoint)
CHECKPOINT_PATH = os.path.join("models", "flare_cnn_multitask_subset_e5.pth")


# ----------------------------------------------------
# 2. Model definition (same as in notebook)
# ----------------------------------------------------
class FlareCNNMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 64)
        self.reg_head = nn.Linear(64, 1)  # log(flux)
        self.cls_head = nn.Linear(64, 1)  # M/X yes-no

    def forward(self, x):
        # x: (B, 4, 10, 256, 256)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = F.adaptive_avg_pool2d(x, 1)  # (B*T, 128, 1, 1)
        x = x.view(B, T, 128)            # (B, 4, 128)
        x = x.mean(dim=1)                # (B, 128)

        x = F.relu(self.fc1(x))          # (B, 64)

        reg_out = self.reg_head(x).squeeze(-1)   # (B,)
        cls_logit = self.cls_head(x).squeeze(-1) # (B,)
        return reg_out, cls_logit


# ----------------------------------------------------
# 3. Load the trained model
# ----------------------------------------------------
model = FlareCNNMultiTask().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ----------------------------------------------------
# 4. Helper functions
# ----------------------------------------------------
def flux_to_class(flux):
    """Convert flux (W/m^2) to flare class string."""
    if flux >= 1e-4:
        return "X"
    elif flux >= 1e-5:
        return "M"
    elif flux >= 1e-6:
        return "C"
    elif flux >= 1e-7:
        return "B"
    else:
        return "A"


class PredictionResponse(BaseModel):
    peak_flux_w_m2: float
    log_flux: float
    flare_class: str
    is_mx_flare: bool
    mx_probability: float


# ----------------------------------------------------
# 5. Simple endpoints
# ----------------------------------------------------
@app.get("/health")
def health():
    """Just to check if API is running."""
    return {
        "status": "ok",
        "device": str(device),
        "checkpoint": CHECKPOINT_PATH,
    }


@app.post("/predict_npy", response_model=PredictionResponse)
async def predict_npy(file: UploadFile = File(...)):
    """
    Predict from a .npy file.

    Input:
      - file: .npy file with shape (40, 256, 256), dtype uint8
        (this is 4 timesteps x 10 channels, same as training)

    Output:
      - JSON with flux, class, M/X probability
    """
    try:
        # Read uploaded file into memory
        raw = await file.read()
        arr = np.load(io.BytesIO(raw))

        if arr.shape != (40, 256, 256):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (40, 256, 256), got {arr.shape}"
            )

        # Prepare tensor: (1, 4, 10, 256, 256)
        arr = arr.reshape(1, 4, 10, 256, 256).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).to(device)

        # Run model
        with torch.no_grad():
            reg_out, cls_logit = model(x)
            log_flux = reg_out.item()
            peak_flux = float(10 ** log_flux)
            mx_prob = float(torch.sigmoid(cls_logit).item())

        return PredictionResponse(
            peak_flux_w_m2=peak_flux,
            log_flux=float(log_flux),
            flare_class=flux_to_class(peak_flux),
            is_mx_flare=mx_prob >= 0.5,
            mx_probability=mx_prob,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
