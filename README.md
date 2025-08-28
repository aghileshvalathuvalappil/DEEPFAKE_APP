Deepfake Detection System

This project implements a deepfake detection framework using state-of-the-art deep learning techniques. The system can analyse both images and videos to distinguish between authentic and manipulated media. It has been deployed with a user-friendly web interface on AWS, ensuring accessibility for non-technical stakeholders such as journalists, educators, and content moderators.

🔑 Key Features

Deep learning–based classifier (EfficientNet-B0 baseline, extendable to hybrid/transformer models).

Supports both image and video deepfake detection.

AWS-hosted web application for scalability and real-world usability.

Ethical design considerations, including dataset bias awareness and transparency in predictions.

📦 Model Storage

Due to GitHub’s file size limitations (100 MB per file), the trained model could not be uploaded here.
Instead, it is securely stored on OneDrive. The download link is provided in the project documentation.

⚙️ Usage Instructions

Download or clone the repository

Get a copy of this project to your local machine.

Set up the environment

Create a virtual environment (recommended) and install the dependencies listed in the requirements.txt file using your preferred package manager.

Obtain the trained model

The model file is too large for GitHub and is therefore provided through OneDrive.

Download the model from the link in the documentation.

Place the model file into the models folder inside the project directory.

Run the application

Start the application by running the main file (e.g., app.py).

Once running, open your web browser and go to http://127.0.0.1:5000 to access the system for local host testing 

Access via AWS (Recommended)

The system is already deployed and accessible through the AWS link provided in the project documentation.

Simply open the link in your web browser to upload media files (images or videos) and view the detection results.

📊 System Workflow

Below is the high-level architecture of the project:

         ┌─────────────┐
         │  User Upload │
         └──────┬──────┘
                │
        ┌───────▼────────┐
        │ Preprocessing   │ (frame extraction, resizing, normalisation)
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Deep Learning   │ (EfficientNet-B0 / hybrid models)
        │  Classification │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Prediction API  │ (Flask + Gunicorn)
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Web Interface   │ (AWS-hosted front-end)
        └────────────────┘


## 🖥️ Project Structure

```plaintext
deepfake_app/
│── models/                # Pretrained model checkpoints (deepfake_detector.pt)
│── static/                # Uploaded media, processed outputs, CSS
│── templates/             # HTML templates (index.html, results.html)
│── app.py                 # Flask application entry point
│── model_loader.py        # Model loading and inference logic
│── utils.py               # Preprocessing utilities (frame extraction, face detection)
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation

## 👨‍💻 Author

**Aghilesh Valathu Valappil** – CSCT Masters Project  
