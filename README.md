# X-Ray Image Analysis Using Autoencoder with Deep Feature Extraction

An academic, full-stack Deep Learning web platform for unsupervised chest radiograph feature extraction, image reconstruction, and difference-residual anomaly analysis using Convolutional Autoencoders.

> **Academic Disclaimer:** This application is an educational and research prototype developed for college project demonstrations and Deep Learning viva-voce examinations. It is **explicitly not a certified medical diagnostic system** and must never be utilized for clinical decision-making.

---

## 1. Project Overview

Medical chest X-rays (CXRs) exhibit high visual complexity characterized by overlapping rib structures, soft parenchymal tissue, and variable exposures. This project implements an end-to-end Deep Learning pipeline:
1. **Multi-Stage Preprocessing Pipeline:** Conversion to single-channel grayscale, bilinear interpolation to $224 \times 224$, $[0.0, 1.0]$ tensor normalization, and batch expansion.
2. **Convolutional Autoencoder (ConvAE):** 4-layer CNN encoder compressing high-dimensional radiographs into a compact $128$-dimensional bottleneck latent vector.
3. **Deconvolutional Decoder:** 4-layer transposed convolution network learning spatial upsampling to reconstruct the original radiograph.
4. **Reconstruction Residual Estimation:** Pixel-wise absolute difference maps $|I_{\text{orig}} - I_{\text{recon}}|$ transformed into colorized Jet heatmaps highlighting structural abnormalities.
5. **Dimensionality Reduction:** Interactive 2D manifold projections via Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).
6. **Latent Space Similarity Retrieval:** Sub-millisecond top-$k$ nearest neighbor search using cosine similarity across the learned latent feature database.

---

## 2. Problem Statement

Traditional computer-vision methods for chest radiograph analysis rely heavily on handcrafted spatial filters and rigid intensity thresholds, failing to adapt to subtle pathological variations such as pulmonary infiltrates, consolidations, or nodular opacities. Furthermore, supervised deep neural networks require thousands of expert radiologist annotations, which are scarce and expensive. 

**Proposed Solution:** An unsupervised Convolutional Autoencoder trained primarily on normal chest radiographs. The model learns the intrinsic manifold of healthy thoracic anatomy; when an abnormal radiograph is presented, the model fails to reconstruct unfamiliar pathological lesions accurately, manifesting in elevated reconstruction error (MSE) and visible localized residual hot-spots in the difference map.

---

## 3. Technology Stack

### Frontend
- **Framework:** React.js 18 + Vite
- **Styling:** Tailwind CSS + Vanilla CSS Tokens
- **Routing:** React Router DOM (v7)
- **Animations:** Framer Motion
- **Icons:** Lucide React
- **Charts & Visualizations:** Recharts (Interactive Scatter & Convergence Line Charts)
- **HTTP Client:** Axios

### Backend
- **Framework:** FastAPI
- **Server:** Uvicorn (ASGI)
- **Validation:** Pydantic v2
- **Image Processing:** OpenCV (`opencv-python-headless`), Pillow (`PIL`), NumPy
- **Data Handling:** Pandas, SQLite3 (`database.sqlite3`)

### Deep Learning
- **Framework:** PyTorch (`torch`) + `torchvision`
- **Manifold Reductions:** Scikit-Learn (`PCA`, `TSNE`, Cosine Metric)
- **Execution Target:** CPU / CUDA (auto-detected)

---

## 4. Deep Learning Architecture Pipeline

```text
       Chest X-Ray Input (Raw PNG/JPG)
                      │
                      ▼
        Image Preprocessing Pipeline
  [Grayscale ➔ Resize (224×224) ➔ Normalization [0,1]]
                      │
                      ▼
           Convolutional Encoder
     Conv2D (1➔32) ➔ ReLU ➔ MaxPool (112×112)
     Conv2D (32➔64) ➔ ReLU ➔ MaxPool (56×56)
     Conv2D (64➔128) ➔ ReLU ➔ MaxPool (28×28)
     Conv2D (128➔256) ➔ ReLU ➔ MaxPool (14×14)
                      │
                      ▼
     Flatten & Linear Bottleneck Layer
          [128-D Deep Feature Vector]
             /                 \
            /                   \
           ▼                     ▼
Latent Space Analysis      Linear Expansion & Unflatten
• Cosine Similarity            to (256, 14, 14)
• 2D PCA Manifold                │
• 2D t-SNE Clustering            ▼
                       Convolutional Decoder
                 ConvTranspose2D (256➔128) ➔ ReLU (28×28)
                 ConvTranspose2D (128➔64) ➔ ReLU (56×56)
                 ConvTranspose2D (64➔32) ➔ ReLU (112×112)
                 ConvTranspose2D (32➔1) ➔ Sigmoid (224×224)
                                 │
                                 ▼
                     Reconstructed X-Ray Image
                                 │
                                 ▼
                    Difference Residual Heatmap
                   |Original - Reconstructed|
                                 │
                                 ▼
                   Mathematical Metric Evaluation
                   [MSE, MAE, RMSE, PSNR, SSIM]
```

---

## 5. Directory Structure

```text
Deep Learning/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application lifecycle & static mounts
│   │   ├── config.py                # Hyperparameters, directories & device resolution
│   │   ├── api/
│   │   │   ├── upload.py            # File validation & secure disk storage
│   │   │   ├── analysis.py          # Preprocessing, reconstruction & metrics API
│   │   │   ├── training.py          # Background training worker & checkpoint API
│   │   │   └── history.py           # Paginated analysis database log & deletion
│   │   ├── models/
│   │   │   ├── autoencoder.py       # PyTorch ConvAutoencoder architecture
│   │   │   └── database.py          # SQLite schema & query methods
│   │   ├── services/
│   │   │   ├── preprocessing.py     # 6-stage image transformation
│   │   │   ├── feature_extraction.py# PCA & t-SNE dimensionality reductions
│   │   │   ├── reconstruction.py    # Model evaluation & difference map generation
│   │   │   ├── similarity.py        # Latent vector cosine similarity matcher
│   │   │   └── synthetic_data.py    # Realistic synthetic CXR procedural generator
│   │   └── utils/
│   │       └── metrics.py           # MSE, MAE, RMSE, PSNR, SSIM calculations
│   ├── datasets/                    # Train, validation, and test splits
│   ├── uploads/                     # Uploaded user images
│   ├── results/                     # Preprocessed stages, reconstructions & heatmaps
│   ├── saved_models/                # PyTorch .pth weight checkpoints
│   ├── requirements.txt             # Python dependencies
│   ├── train.py                     # Standalone CLI & programmatic training script
│   └── database.sqlite3             # SQLite historical database
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           # Global status badges, branding & theme toggle
│   │   │   ├── Sidebar.jsx          # Module navigation & academic disclaimer
│   │   │   ├── MetricCard.jsx       # Reusable KPI card with hover elevation
│   │   │   ├── ImageUploader.jsx    # Drag-and-drop uploader with sample selectors
│   │   │   ├── XRayViewer.jsx       # Radiologic split-slider, zoom & diff viewer
│   │   │   ├── FeatureChart.jsx     # Recharts scatter plot for PCA & t-SNE
│   │   │   └── LoadingPipeline.jsx  # Animated 6-step deep learning progress
│   │   ├── pages/
│   │   │   ├── Landing.jsx          # Hero presentation & research statistics
│   │   │   ├── Dashboard.jsx        # Telemetry cards & recent evaluations
│   │   │   ├── Analyze.jsx          # Image upload & intermediate artifact cards
│   │   │   ├── Results.jsx          # Detailed report, vector inspector & similar CXRs
│   │   │   ├── Features.jsx         # PCA/t-SNE latent manifold explorer
│   │   │   ├── Reconstruction.jsx   # Dedicated comparative viewer & formula cards
│   │   │   ├── History.jsx          # Audit table with search, sorting & pagination
│   │   │   ├── Training.jsx         # Live Recharts loss curves & checkpoint manager
│   │   │   ├── Model.jsx            # Flowchart, theory cards & PyTorch layer breakdown
│   │   │   └── About.jsx            # Viva defense Q&A guide & research roadmap
│   │   ├── services/api.js          # Axios API endpoints wrapper
│   │   ├── context/AppContext.jsx   # Theme persistence, notifications & global stats
│   │   ├── App.jsx                  # React Router configuration
│   │   ├── main.jsx                 # Application entry point
│   │   └── index.css                # Tailwind directives & glassmorphism tokens
│   ├── index.html                   # HTML template & fonts
│   ├── vite.config.js               # Dev server port 5173 with backend proxy
│   └── tailwind.config.js           # Medical AI color palette configuration
└── README.md
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10+ (64-bit)
- Node.js v18+ & npm v9+

### 1. Setup Backend Environment

```bash
# Navigate to project root
cd "d:/GitHub/Deep Learning"

# Create a virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Setup Frontend Environment

```bash
# Navigate to frontend directory
cd frontend

# Install node dependencies
npm install
```

---

## 7. Running the Application

### Start the FastAPI Backend
From the project root with the virtual environment activated:
```bash
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
```
* The backend API documentation is available at `http://127.0.0.1:8000/docs`

### Start the React Frontend
In a separate terminal:
```bash
cd frontend
npm run dev
```
* Open your browser and navigate to: **`http://127.0.0.1:5173/`**

---

## 8. Model Training

### Option A: Via the Web Dashboard
1. Open the web application and navigate to **Model Training** (`/model-training`).
2. Configure **Epochs** (default: 20), **Batch Size** (default: 16), **Learning Rate** (default: 0.001), and **Latent Dimension** (default: 128).
3. Click **Start Training**. The backend executes the PyTorch loop asynchronously in a background thread while Recharts streams real-time loss curves.
4. Save the model checkpoint using the **Model Checkpoint Registry**.

### Option B: Via Command Line (CLI)
```bash
cd backend
..\.venv\Scripts\python.exe train.py --epochs 20 --batch-size 16 --lr 0.001 --latent-dim 128 --optimizer adam
```

---

## 9. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Validates file size (max 10MB) & format (.png, .jpg, .jpeg) |
| `POST` | `/api/analyze` | Executes end-to-end preprocessing, inference, and stores metrics |
| `GET` | `/api/results/{id}` | Retrieves metrics, feature vectors, and similar images |
| `GET` | `/api/results/{id}/image/{key}` | Serves original, reconstructed, diff heatmap, or stage images |
| `GET` | `/api/features-distribution` | Computes 2D PCA & t-SNE coordinates for all evaluated images |
| `GET` | `/api/history` | Paginated search, sorting, and listing of past analyses |
| `DELETE` | `/api/history/{id}` | Deletes an analysis record from SQLite database |
| `POST` | `/api/train` | Launches asynchronous PyTorch training worker |
| `GET` | `/api/train/status` | Returns active epoch, losses, and epoch history |
| `POST` | `/api/train/stop` | Signals training worker to halt execution gracefully |
| `POST` | `/api/train/save` | Saves active weights to a named `.pth` checkpoint |
| `POST` | `/api/train/load` | Restores a selected checkpoint into the active model |
| `GET` | `/api/model` | Returns PyTorch layer graph and parameter hierarchy |
| `GET` | `/api/stats` | High-level KPI telemetry for dashboard counters |

---

## 10. Key Features for Academic Defense & Viva

- **No Fabricated Medical Claims:** Uses strictly scientific terminology (**Reconstruction Error (MSE)**, **Structural Residual**, **Feature Similarity**) instead of making unfounded medical diagnostic assertions.
- **Explainable Heatmap Visualization:** Computes absolute difference residual $|I_{\text{orig}} - I_{\text{recon}}|$, overlaying a calibrated Jet colormap to visually highlight structural anomalies.
- **Real Dynamic Metrics:** Computes Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
- **Interactive Radiologic Viewer:** Features split-screen slider, side-by-side mode, difference heatmap toggle, inverted contrast (negative radiography mode), and zoom controls.
- **Dimensionality Reduction:** 2D PCA and t-SNE scatter plots illustrate how deep representations cluster in the latent space.

---

## 11. Limitations & Future Work

### Limitations
1. **Unsupervised Residual Limitations:** Elevated reconstruction residuals can sometimes arise from variations in patient positioning, body habitus, or hardware calibration rather than true pathology.
2. **Resolution:** Default resolution is set to $224 \times 224$ for standard convolutional efficiency. Micro-nodules may benefit from higher-resolution inputs ($512 \times 512$ or native DICOM resolutions).

### Future Roadmap
- Implementation of **Vision Transformers (ViT)** and Masked Autoencoders (MAE).
- **Grad-CAM** visual attention maps for interpretability.
- Native **DICOM** format ingestion with window/level HU adjustments.
- Multi-head supervised classification fine-tuning on public datasets (NIH ChestX-ray14, CheXpert).
- Distributed GPU acceleration and federated clinical deployment.

---

## 12. License & Citation

Distributed under the Apache 2.0 License. Developed as an academic deep learning capstone and viva-voce demonstration project.
# Deep-learning-Mini-project
