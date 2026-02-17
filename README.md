# Lidar Perception & Cinematic Rendering Pipeline

A comprehensive Python-based pipeline for processing, analyzing, and visualizing Lidar point cloud data. This project includes modules for preprocessing, object detection, classification, and tracking, along with a specialized cinematic rendering engine for creating high-quality visualizations.

## 🚀 Features

-   **Data Loading**: Efficient loading of Lidar frames from CSV/PCD formats.
-   **Preprocessing**:
    -   Voxel grid downsampling for performance.
    -   Ground plane removal using RANSAC.
    -   Statistical outlier removal.
-   **Object Detection**:
    -   DBSCAN clustering to identify object candidates.
    -   Oriented Bounding Box (OBB) and Axis-Aligned Bounding Box (AABB) generation.
-   **Classification**:
    -   Heuristic-based classification (Car, Pedestrian, Cyclist) based on geometric properties.
-   **Tracking**:
    -   Multi-object tracking with ID persistence.
    -   Motion analysis (e.g., "approaching" status).
-   **Cinematic Visualization**:
    -   High-quality 3D rendering using Open3D.
    -   Custom camera paths (Side View, etc.).
    -   Distance-based color mapping.
    -   Overlay of object labels and trajectory information.
    -   Branding and HUD integration.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Lidar-Perception.git
    cd Lidar-Perception
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

The project is configured via `configs/config.yaml`. You can adjust parameters for preprocessing, detection, tracking, and rendering there.

**Example `configs/config.yaml`:**
```yaml
paths:
  dataset: "data/raw/dataset"
  results: "results/videos"

preprocess:
  voxel_size: 0.10
  ransac_distance_threshold: 0.15

video:
  fps: 10
  frame_repeat: 2
  width: 1280
  height: 720
```

## 💻 Usage

### 1. Run Cinematic Video Rendering
Generate a high-quality video of the Lidar scene with object tracking and branding.
```bash
python -m src.run_cinematic_video
```
*Output will be saved to `results/videos/`.*

### 2. Run Single Frame Detection
Visualize detection results for a specific frame.
```bash
python -m src.run_detect_one
```

### 3. Run Tracking Sequence
Process a sequence of frames and visualize object tracking.
```bash
python -m src.run_track_sequence
```

## 📂 Project Structure

```
Lidar/
├── assets/                 # Logos and branding assets
├── configs/                # Configuration files
│   └── config.yaml
├── data/                   # Input data (Lidar frames)
├── results/                # Output videos and reports
├── src/                    # Source code
│   ├── detect.py           # Object detection logic
│   ├── track.py            # Object tracking logic
│   ├── viz.py              # Visualization utilities
│   ├── run_cinematic_video.py # Main rendering script
│   └── ...
├── requirements.txt        # python dependencies
└── README.md               # Project documentation
```

## 🎓 Branding

This project includes support for institutional branding (e.g., IU International University of Applied Sciences) in the rendered videos. Place your logo in `assets/iu_logo.png` to enable it.
