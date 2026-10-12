# Dog Breed Classification Using MobileNetV2-Based Transfer Learning

## 1. Project Overview
This project develops an end-to-end Deep Learning system capable of taking an arbitrary dog image as input and accurately predicting its specific breed among 120 recognized dog breeds. The system leverages Transfer Learning with a MobileNetV2 convolutional neural network pretrained on the ImageNet dataset.

## 2. Objective
The primary objective of this deep learning application is to classify dog breeds efficiently and accurately. By utilizing a lightweight and highly efficient architecture (MobileNetV2) and adapting pretrained feature extractors via fine-tuning and transfer learning, the system achieves strong classification performance while remaining optimized for CPU and edge inference.

## 3. Dataset Architecture & Handling
The dataset utilized is the benchmark **Kaggle Dog Breed Identification** dataset:
- **Source**: [Kaggle Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification)
- **Structure**:
  - `dog_breed_classification/dataset/train/`: Directory containing 10,222 training images in `.jpg` format.
  - `dog_breed_classification/dataset/labels.csv`: Ground-truth mapping table (`id,breed`).

### Label Mapping Pipeline
The unique image ID in `labels.csv` is mapped dynamically to the corresponding image file in the training folder:
```
Training Images + labels.csv
          ↓
 Image ID → Image Filename (.jpg) → Breed Label
  (e.g., 000bec180eb18c7604dcecc8fe0dba07 → 000bec180eb18c7604dcecc8fe0dba07.jpg → boston_bull)
```
- **Image Names**: Preserved intact (no renaming).
- **Directory Layout**: Flat image directory coupled with metadata table (no artificial subfolder reorganizations).
- **Format Verification**: Determined dynamically by inspecting disk files rather than assuming extensions.

## 4. Planned Model Architecture
- **Base Architecture**: MobileNetV2 (pretrained on ImageNet, frozen feature extractor).
- **Classification Head**:
  - `GlobalAveragePooling2D`
  - `Dense(256, activation="relu")`
  - `Dropout(0.3)`
  - `Dense(NUM_CLASSES, activation="softmax")`
- **Optimizer**: Adam (Initial Learning Rate = 0.001)
- **Loss Function**: `SparseCategoricalCrossentropy`
- **Metrics**: Accuracy, Top-5 Accuracy
- **Training Status**: **Not started** (Model architecture built, compiled, and validated in Phase 3; actual training will occur in Phase 4).

## 5. Current Phase
**Phase 7 — Build the Dog Breed Classification Web Application**
- Status: **COMPLETED**
- **Application Name**: PawBreed AI — Dog Breed Classifier
- **Web Framework**: Streamlit
- **Main Entrypoint**: `dog_breed_classification/app/streamlit_app.py`
- **Cached Final Model**: `dog_breed_classification/models/dog_breed_classifier_final.keras`
- **Class Mapping**: `dog_breed_classification/models/class_names.json` (120 dog breeds)
- **Features**:
  - Image upload support (JPG, JPEG, PNG)
  - Curated quick-test samples selector
  - Real-time preview with dimension and mode inspection
  - Official MobileNetV2 preprocessing (`preprocess_input`, [-1, 1] range)
  - Top-1 winner card with confidence percentage
  - Top-5 ranked breed candidate distribution with visual probability bars
  - Detailed collapsible metric table
  - Domain-specific disclaimer for non-dog inputs

## 6. Project Structure
```
dog_breed_classification/
│
├── app/
│   └── streamlit_app.py       # PawBreed AI Streamlit web application
│
├── dataset/
│   ├── train/                 # 10,222 raw training images (.jpg)
│   ├── labels.csv             # Image ID to Breed mappings
│   └── README.md              # Dataset notes & path instructions
│
├── notebooks/                 # Exploratory and prototyping notebooks
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── dataset_analysis.py    # Phase 1 verification & EDA script
│   ├── data_preprocessing.py  # Phase 2 pipeline & single image preprocessing
│   ├── model.py               # Phase 3 MobileNetV2 architecture
│   ├── train_initial.py       # Phase 4 frozen base training
│   ├── train_finetune.py      # Phase 5 fine-tuning script
│   └── evaluate_final.py      # Phase 6 test evaluation & model selector
│
├── models/                    # Trained model checkpoints
│   ├── dog_breed_classifier_final.keras   # Final selected Phase 5 model
│   ├── mobilenetv2_finetuned_best.keras   # Phase 5 checkpoint
│   ├── mobilenetv2_initial_best.keras     # Phase 4 checkpoint
│   └── class_names.json                   # 120 breed names mapping
│
├── outputs/                   # Generated charts, statistics, and reports
│   ├── phase6_final_report.txt      # Final test evaluation metrics
│   ├── final_metrics.json           # JSON test metrics
│   ├── confusion_matrix.png         # 120x120 test confusion matrix
│   ├── test_prediction_gallery.png  # Test sample visualization
│   └── top5_predictions.csv         # Top-5 predictions sample
│
├── requirements.txt           # Project dependencies (TensorFlow, Streamlit, etc.)
└── README.md                  # Project documentation
```

## 7. How to Launch the Web Application

### Step 1: (Optional) Windows Virtual Environment Setup
Open PowerShell or Command Prompt:
```powershell
# Navigate to project root
cd c:\Users\Admin\Desktop\Deep_Learning_mini

# Create a virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1
# (or in Command Prompt: .\venv\Scripts\activate.bat)

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Launch Streamlit Web App
From the project workspace root:
```bash
python -m streamlit run dog_breed_classification/app/streamlit_app.py
```
Or from `dog_breed_classification` directory:
```bash
cd dog_breed_classification
python -m streamlit run app/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 8. Deployment Preparation & Guidelines
- **Required Dependencies**: Defined in `requirements.txt` (`tensorflow>=2.16.0`, `streamlit>=1.30.0`, `Pillow>=10.2.0`, `numpy>=1.26.0`, `pandas>=2.2.0`).
- **Model File**: Ensure `models/dog_breed_classifier_final.keras` (~30 MB) and `models/class_names.json` are included in the deployment repository. Git LFS is recommended if hosted on GitHub.
- **Resource Limits**:
  - Memory: TensorFlow runtime + model requires approximately 500 MB – 1 GB RAM during inference.
  - Free tiers on Streamlit Community Cloud or Render are sufficient with CPU inference.
- **Status**: Verified locally on `http://localhost:8501` (Not yet deployed to a public cloud URL).

## 9. How to Run Phase 1 Dataset Analysis
Execute the dataset verification and analysis script:
```bash
python dog_breed_classification/src/dataset_analysis.py
```
