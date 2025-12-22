# Solar Flare Prediction with CNN 
This project predicts solar flare intensity and class using SDO (Solar Dynamics Observatory) imagery and exposes the trained model through a simple REST API. It is built around the SDOBenchmark dataset and a custom multitask convolutional neural network that performs both regression (peak X‑ray flux) and classification (M/X vs non‑M/X flare) from multi‑channel image sequences.
​

## Features
- Multitask CNN model:

  - Predicts log10(peak X‑ray flux) as a regression target.

  - Predicts whether the flare is M/X class as a binary classification target.
​

- Uses multi‑channel SDO image sequences (10 wavelengths × 4 time steps = 40 images per sample).
​

- Jupyter notebooks for:

  - Data loading and exploration.

  - Baseline models.

  - CNN training and evaluation.

  - Inference and test metrics.
​

- FastAPI server that exposes the model via HTTP endpoints for easy integration.
​

- Tech Stack
  - Language: Python
​

  - Data & ML: NumPy, Pandas, scikit‑learn, PyTorch
​

  - Serving: FastAPI, Uvicorn
​

  - Environment & Tools: Jupyter Notebooks, Git, GitHub
​
Project Structure



```
Solar_Flare_Prediction/
├─ 01_load_and_visualize.ipynb      # Data loading, exploration, baseline
├─ 02_cnn_model.ipynb               # CNN dataset, model, and training
├─ 04_inference_test.ipynb          # Inference and test metrics for the CNN
├─ app.py                           # FastAPI app exposing the trained model
├─ sample_flare.npy                 # Example input sample (40, 256, 256)
├─ models/
│   └─ flare_cnn_multitask_subset_e5.pth   # Trained CNN checkpoint
├─ data/
│   └─ SDOBenchmark-data-full/      # SDOBenchmark dataset (not tracked in git)
└─ README.md 
```
Not all files are tracked in Git (e.g., large data files and models are typically ignored using .gitignore).
​

## Dataset
The project uses the SDOBenchmark dataset, which provides SDO image patches and associated GOES X‑ray flux labels. Each training sample consists of:

- 4 time steps.

- 10 channels/wavelengths per time step (94, 131, 171, 193, 211, 304, 335, 1700, continuum, magnetogram).

- Images resized to 256 × 256 pixels, giving 40 images per sample.
​

Labels include:

- Peak X‑ray flux in W/m².

- Derived log10(flux) for regression.

- Binary label is_MX indicating whether the flare is M or X class (flux ≥ 10<sup>-5</sup> W/m²).
​

## Model
The core model is a multitask CNN implemented in PyTorch:
​

- Input shape: (B,4,10,256,256) (batch, time, channels, height, width).

- Three convolutional blocks with max pooling.

- Global average pooling across spatial dimensions.

- Shared fully‑connected layer.

- Two heads:

  - Regression head: predicts log<sub>10</sub>(peak flux).

  - Classification head: predicts M/X probability (BCEWithLogitsLoss).
​

Training details (in the current version):

- Subset of ~300 samples for quick experimentation.

- 80/20 train–validation split.

- Adam optimizer with a small learning rate.

- Joint loss = regression MAE + classification BCE.
​

## Notebooks

1.``01_load_and_visualize.ipynb``
  - Loads the SDOBenchmark CSVs.

  - Explores label distributions (flux, flare classes).

  - May include a simple baseline model (e.g., RandomForest) for comparison.
​

2.``02_cnn_model.ipynb``
  - Implements:

    - Dataset class that loads and stacks 40 images per sample.

    - The multitask CNN model.
  
    - Training loop and validation metrics.
​

3.``04_inference_test.ipynb``
- Loads the saved model checkpoint.

- Builds a test set and evaluates:

  - Log‑flux MAE and flux MAE.

  - M/X classification metrics (precision, recall, F1, confusion matrix).

- Generates diagnostic plots (e.g., predicted vs true flux).
​

## API
The ``app.py`` file provides a simple FastAPI service:​

Endpoints
- ``GET /health``
Returns a small JSON with status, device, and loaded checkpoint path.

- ``POST /predict_npy``
Accepts:

- A .npy file with shape (40, 256, 256) representing 4 time steps × 10 channels of 256×256 images.

Returns JSON with:

- peak_flux_w_m2

- log_flux

- flare_class (A/B/C/M/X)

- is_mx_flare (boolean)

- mx_probability ∈[0,1].
​

Running the API
From the project root:

```
conda activate solarflare   # or your env
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Then open:

- Swagger UI: http://127.0.0.1:8000/docs

- Health check: http://127.0.0.1:8000/health
​

Example request (curl):
```
curl -X POST "http://127.0.0.1:8000/predict_npy" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_flare.npy"
```
Example response:

```
{
  "peak_flux_w_m2": 5.7e-06,
  "log_flux": -5.24,
  "flare_class": "C",
  "is_mx_flare": false,
  "mx_probability": 0.10
}
```
This corresponds to a predicted C‑class flare with low probability of being M/X.
​

##How to Run Locally
1. Clone the repository:
```
git clone https://github.com/<your-username>/solar-flare-prediction.git
cd solar-flare-prediction
```
2. Create and activate a Python environment (conda or venv).

3. Install dependencies (if you provide a requirements.txt):

```
pip install -r requirements.txt
```
4. Place the SDOBenchmark data under data/SDOBenchmark-data-full/ as expected by the notebooks and scripts.
​

5. Run the notebooks in order to reproduce training and evaluation or directly start the API with the existing checkpoint.
​

## Future Improvements

- Train on the full SDOBenchmark dataset rather than a small subset.

- Use more advanced architectures (e.g., ResNet backbones, attention, 3D CNNs).

- Implement better handling of class imbalance (focal loss, reweighting, data sampling).

- Add additional API endpoints and a small web UI (e.g., Streamlit) for interactive use.
