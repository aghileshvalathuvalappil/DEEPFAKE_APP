import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from model_loader import load_model, predict_video
from utils import extract_face_frames

# Flask app initialization
app = Flask(__name__)

# Paths and settings
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB upload limit

# Allowed extensions
ALLOWED_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}

def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

# Load model safely
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model, device = load_model(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, filename, error = None, None, None, None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part in the request."
            return render_template("index.html", error=error)

        file = request.files["file"]
        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(save_path)
            filename = fname

            # Extract frames
            frames_tensor = extract_face_frames(save_path, target_frames=20)
            if frames_tensor is None:
                error = "Could not extract faces/frames from the video. Try another clip."
                return render_template("index.html", error=error, filename=filename)

            # Predict
            label, conf = predict_video(model, device, frames_tensor)
            prediction = label
            confidence = round(conf * 100.0, 2)

        else:
            error = "Unsupported file format. Allowed: mp4, avi, mov, mkv, webm."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        error=error
    )


if __name__ == "__main__":
    # Use 0.0.0.0 for Ubuntu server (accessible externally if needed)
    app.run(host="0.0.0.0", port=5000, debug=True)
