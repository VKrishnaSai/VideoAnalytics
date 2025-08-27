import os
import shutil
import tempfile
import torch
import torch.nn as nn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
from io import BytesIO

from vivit import ViViT, load_video_new

# Load class names consistently with training
class_index_file = "./ucfTrainTestlist/classInd.txt"
class_indices = {}
with open(class_index_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            idx, class_name = parts
            class_indices[int(idx)] = class_name
data_subset = [class_indices[i] for i in sorted(class_indices.keys())]  # Same as training

app = FastAPI(title="Fun ViViT Inference Dashboard")

UPLOAD_FORM = """
<!DOCTYPE html>
<html>
<head>
  <title>ViViT Inference Dashboard</title>
  <style>
    body { 
      background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
      font-family: 'Courier New', cursive, sans-serif; 
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

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Inference Result</title>
  <style>
    body {{ 
      background: linear-gradient(135deg, #B5FFFC, #FFDEE9);
      font-family: 'Courier New', cursive, sans-serif;
      display: flex; 
      justify-content: center; 
      align-items: center; 
      min-height: 100vh; 
      margin: 0;
    }}
    .result-container {{
      background: #fff; 
      border-radius: 10px; 
      padding: 40px 60px; 
      text-align: center;
      box-shadow: 0 0 20px rgba(0,0,0,0.2);
      animation: pop 0.5s ease-out;
      max-width: 800px;
    }}
    @keyframes pop {{
      0% {{ transform: scale(0.8); opacity: 0; }}
      100% {{ transform: scale(1); opacity: 1; }}
    }}
    h1 {{ color: #ff8a00; }}
    p {{ font-size: 1.2em; }}
    .predictions {{
      text-align: left;
      max-height: 300px;
      overflow-y: auto;
      margin: 20px 0;
    }}
    .prediction-item {{
      margin: 5px 0;
    }}
    img {{
      max-width: 100%;
      height: auto;
      margin-top: 20px;
      border-radius: 5px;
    }}
  </style>
</head>
<body>
  <div class="result-container">
    <h1>Inference Result</h1>
    <p>Top Predicted Class: <strong>{top_pred_class}</strong> (Confidence: {top_confidence}%)</p>
    <div class="predictions">
      <h3>All Predictions:</h3>
      {all_predictions}
    </div>
    <p>Attention Heatmap (Middle Frame):</p>
    <img src="data:image/png;base64,{heatmap_image}" alt="Attention Heatmap">
    <br><br>
    <a href="/">Run Another Inference</a>
  </div>
</body>
</html>
"""

def generate_attention_heatmap(video_tensor, space_attn_weights, temporal_attn_weights, image_size=224):
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)
    elif video_tensor.dim() != 4:
        raise ValueError(f"Unexpected video_tensor shape: {video_tensor.shape}, expected [T, C, H, W] or [1, T, C, H, W]")

    mid_frame_idx = video_tensor.shape[0] // 2
    frame = video_tensor[mid_frame_idx].detach().cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    frame = (frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    num_patches = (image_size // 16) ** 2
    space_attn = space_attn_weights[-1].mean(dim=1)  # [b*t, n, n] -> [b*t, n]
    space_attn = space_attn[mid_frame_idx, 1:, 1:].mean(dim=0)  # [n-1,]

    patch_grid_size = int(np.sqrt(num_patches))
    heatmap = space_attn.reshape(patch_grid_size, patch_grid_size).detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0.0)

    _, buffer = cv2.imencode('.png', superimposed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=UPLOAD_FORM)

@app.post("/inference", response_class=HTMLResponse)
async def inference(checkpoint: UploadFile = File(...), video: UploadFile = File(...)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, checkpoint.filename)
        video_path = os.path.join(tmpdir, video.filename)
        with open(checkpoint_path, "wb") as f:
            f.write(await checkpoint.read())
        with open(video_path, "wb") as f:
            f.write(await video.read())

        num_classes = len(data_subset)
        model = ViViT(image_size=224, patch_size=16, num_classes=num_classes, num_frames=16)
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"] if "model_state_dict" in checkpoint_data else checkpoint_data)
        model.to(device)
        model.eval()

        video_tensor = load_video_new(video_path, num_frames=16, image_size=224)  # [1, T, C, H, W]
        video_tensor = video_tensor.to(device)

        with torch.no_grad():
            output, (space_attn_weights, temporal_attn_weights) = model(video_tensor, return_attention=True)
            probabilities = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probabilities).item()
            top_confidence = probabilities[pred_idx].item() * 100
            top_pred_class = data_subset[pred_idx] if pred_idx < len(data_subset) else f"Class {pred_idx}"

        # Ensure class names match 0-indexed model output
        probs_with_labels = [(data_subset[i], prob.item() * 100) for i, prob in enumerate(probabilities)]
        probs_with_labels.sort(key=lambda x: x[1], reverse=True)
        all_predictions_html = "".join(
            f'<p class="prediction-item">{label}: {confidence:.2f}%</p>'
            for label, confidence in probs_with_labels
        )

        heatmap_image = generate_attention_heatmap(video_tensor, space_attn_weights, temporal_attn_weights)

    result_page = RESULT_HTML.format(
        top_pred_class=top_pred_class,
        top_confidence=f"{top_confidence:.2f}",
        all_predictions=all_predictions_html,
        heatmap_image=heatmap_image
    )
    return HTMLResponse(content=result_page)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)