import os
import shutil
import tempfile
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import your ViViT model and load_video function
from vivit import ViViT, load_video

# Read class names from truncated_classes.txt (for mapping predicted index to label)
data_subset = []
with open("truncated_classes.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("//"):
            data_subset.append(line)

# Create FastAPI app
app = FastAPI(title="Fun ViViT Inference Dashboard")

# Optionally mount a static directory if you wish to add separate CSS/JS files.
# app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML for the upload form page with fun CSS animations.
UPLOAD_FORM = """
<!DOCTYPE html>
<html>
<head>
  <title>ViViT Inference Dashboard</title>
  <style>
    body { 
      background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
      font-family: 'Comic Sans MS', cursive, sans-serif; 
      display: flex; 
      justify-content: center; 
      align-items: center; 
      height: 100vh; 
      margin: 0;
    }
    .container {
      background: #fff; 
      border-radius: 10px; 
      padding: 30px 50px; 
      text-align: center;
      box-shadow: 0 0 20px rgba(0,0,0,0.2);
      animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
      0% { transform: translateY(0px); }
      50% { transform: translateY(-20px); }
      100% { transform: translateY(0px); }
    }
    h1 { 
      color: #e52e71; 
      margin-bottom: 20px; 
      animation: hue 10s infinite linear;
    }
    @keyframes hue {
      from { filter: hue-rotate(0deg); }
      to { filter: hue-rotate(360deg); }
    }
    input[type="file"] {
      margin: 10px 0;
    }
    input[type="submit"] {
      padding: 10px 20px;
      font-size: 1.1em;
      background: linear-gradient(90deg, #ff8a00, #e52e71);
      border: none;
      color: #fff;
      border-radius: 5px;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    input[type="submit"]:hover {
      background: linear-gradient(90deg, #e52e71, #ff8a00);
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>ViViT Inference Dashboard</h1>
    <form action="/inference" enctype="multipart/form-data" method="post">
      <p>Select a model checkpoint (*.pth):</p>
      <input name="checkpoint" type="file" accept=".pth" required><br>
      <p>Select a video for inference:</p>
      <input name="video" type="file" accept="video/*" required><br>
      <input type="submit" value="Run Inference">
    </form>
  </div>
</body>
</html>
"""

# HTML for displaying inference result.
RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Inference Result</title>
  <style>
    body {{ 
      background: linear-gradient(135deg, #B5FFFC, #FFDEE9);
      font-family: 'Comic Sans MS', cursive, sans-serif;
      display: flex; 
      justify-content: center; 
      align-items: center; 
      height: 100vh; 
      margin: 0;
    }}
    .result-container {{
      background: #fff; 
      border-radius: 10px; 
      padding: 40px 60px; 
      text-align: center;
      box-shadow: 0 0 20px rgba(0,0,0,0.2);
      animation: pop 0.5s ease-out;
    }}
    @keyframes pop {{
      0% {{ transform: scale(0.8); opacity: 0; }}
      100% {{ transform: scale(1); opacity: 1; }}
    }}
    h1 {{ color: #ff8a00; }}
    p {{ font-size: 1.2em; }}
  </style>
</head>
<body>
  <div class="result-container">
    <h1>Inference Result</h1>
    <p>Predicted class: <strong>{pred_class}</strong></p>
    <p>Confidence (if available): {confidence}</p>
    <br>
    <a href="/">Run Another Inference</a>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=UPLOAD_FORM)

@app.post("/inference", response_class=HTMLResponse)
async def inference(checkpoint: UploadFile = File(...), video: UploadFile = File(...)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Save uploaded checkpoint and video files to temporary locations
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, checkpoint.filename)
        video_path = os.path.join(tmpdir, video.filename)
        with open(checkpoint_path, "wb") as f:
            f.write(await checkpoint.read())
        with open(video_path, "wb") as f:
            f.write(await video.read())

        # Initialize the model architecture; assume same parameters as used during training.
        num_classes = len(data_subset)
        model = ViViT(image_size=224, patch_size=16, num_classes=num_classes, num_frames=16)
        # Load the checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"] if "model_state_dict" in checkpoint_data else checkpoint_data)
        model.to(device)
        model.eval()

        # Run inference on the provided video.
        # load_video returns a tensor of shape [1, T, C, H, W]; we remove the extra batch dimension.
        video_tensor = load_video(video_path, num_frames=16, image_size=224)
        if video_tensor.ndim == 5 and video_tensor.shape[0] == 1:
            video_tensor = video_tensor.squeeze(0)
        video_tensor = video_tensor.to(device)

        with torch.no_grad():
            output = model(video_tensor.unsqueeze(0))  # Add batch dimension for inference: [1, T, C, H, W]
            probabilities = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_idx].item()

        pred_class = data_subset[pred_idx] if pred_idx < len(data_subset) else f"Class {pred_idx}"

    result_page = RESULT_HTML.format(pred_class=pred_class, confidence=f"{confidence*100:.2f}%")
    return HTMLResponse(content=result_page)
