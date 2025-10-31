# Joint Fairness-Aware Decision Support System for Face Recognition-Based Financial Transaction Authentication with Integrated Presentation Attack Detection (FR+PAD)
[cite_start][cite: 100]

## Introduction

[cite_start]In the context of rapid digital transformation, identity authentication plays a critical role in ensuring transaction security, mitigating fraud, and enhancing user experience in online financial services[cite: 107]. [cite_start]However, these systems face two fundamental challenges: (1) **Security**, against sophisticated Presentation Attacks (PAs) such as deepfakes, print attacks, and 3D masks [cite: 111][cite_start]; and (2) **Fairness**, as numerous studies (e.g., Gender Shades, NIST FRVT) have demonstrated that AI models exhibit significant bias, leading to higher error rates for specific demographic groups[cite: 115, 116].

[cite_start]To address these limitations, this project presents a **Fairness-Aware Decision Support System (DSS)**[cite: 129]. [cite_start]Our architecture moves beyond optimizing FR or PAD in isolation [cite: 120] [cite_start]and instead designs a closed-loop pipeline to proactively measure and mitigate bias, fulfilling both technical and emerging legal requirements[cite: 122, 126].

Our framework integrates four key components:
1.  [cite_start]**YOLOv8/YOLOv10:** For high-performance face localization (pre-processing)[cite: 153, 160].
2.  [cite_start]**ArcFace (ResNet-50):** As a robust feature extractor for Face Recognition (FR)[cite: 142, 155, 156].
3.  [cite_start]**Ensemble PAD (Presentation Attack Detection):** Integrating both **DeepPixBiS** and **Vision Transformer (ViT)** models for robust spoof detection[cite: 142, 158].
4.  [cite_start]**DSS Engine & Feedback Loop:** A decision logic core designed to incorporate **adaptive thresholds** [cite: 134, 351] [cite_start]and a **feedback loop** (with `needs_review` and `retraining_set` directories) for the system to continuously learn from difficult cases and improve fairness over time [cite: 366-369].

[cite_start]This research presents a comprehensive DSS solution that achieves high accuracy on standard benchmarks (such as CelebA-Spoofing [cite: 150][cite_start]) and establishes an infrastructure to measure and significantly reduce the Fairness Gap between demographic groups[cite: 144].

## Key Features

* [cite_start]**Joint FR+PAD System:** Concurrently processes both Face Recognition (FR) and Presentation Attack Detection (PAD) in a unified pipeline, simulating real-world eKYC and transaction authentication scenarios[cite: 133].
* [cite_start]**Advanced Spoof Detection (Ensemble PAD):** Combines pixel-based (DeepPixBiS) and global-feature-based (ViT) approaches to enhance spoof detection robustness[cite: 142, 158, 299].
* [cite_start]**Fairness-Centric Design:** Moves beyond static thresholds, architected with a three-tier bias reduction strategy (data, model, and decision-level) [cite: 134] [cite_start]and an **adaptive threshold** mechanism[cite: 351].
* [cite_start]**Continuous Learning:** Integrated `needs_review`  and `retraining_set` [cite: 631] [cite_start]directories allow administrators to review uncertain cases, re-label them, and feed the data back for retraining, enabling the system to adapt to new threats and reduce bias [cite: 366-369].
* [cite_start]**End-to-End Demonstration:** Includes scripts for database creation (`create_DB.py`)  and a full-featured web application (`app.py` using Streamlit) for registration, authentication, and detailed analysis results.

## Project Structure

[cite_start](This structure is based on Appendix 2 of the research paper [cite: 618-648])


. ├── config/ │ └── config.yaml # Defines all system configurations (paths, thresholds, model handles)  │ ├── data/ │ ├── needs_review/ # Stores images flagged as 'uncertain' for admin review │ └── retraining_set/ # Curated dataset for model fine-tuning (feedback loop)  │ ├── src/ │ ├── analysis/ │ │ ├── camera_quality.py # Module for assessing input image quality │ │ ├── predictor.py # Core FaceAnalysisDSS class, manages the full pipeline │ │ └── review_handler.py # Logic for managing the 'Admin review' process  │ │ │ ├── evaluation/ │ │ └── eval.ipynb # Jupyter Notebook for system evaluation and fairness metrics  │ │ │ ├── model/ │ │ ├── arcface.py # ArcFace model architecture (FR) │ │ ├── deeppixbis.py # DeepPixBiS model architecture (PAD) │ │ └── vit.py # Vision Transformer model architecture (PAD)  │ │ │ ├── test_img/ │ │ └── person 1/ # Sample images used by create_DB.py  │ │ │ └── ultil.py # Utility functions (config loading, model asset loading)  │ ├── .gitignore ├── app.py # Streamlit web demo application (UI/UX) ├── create_DB.py # Script to create the face recognition DB ├── known_faces_db.pt # The persistent PyTorch DB for known embeddings ├── main.py # Main entry point for non-interactive demo (e.g., webcam) └── requirements.txt # Python dependencies

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/KhanhGiauTen/Fairness_Aware_Face_Recognition_Financial_Authentication_with_PAD.git](https://github.com/KhanhGiauTen/Fairness_Aware_Face_Recognition_Financial_Authentication_with_PAD.git)
    cd Fairness_Aware_Face_Recognition_Financial_Authentication_with_PAD
    ```
    [cite: 617]

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    

## API Token Setup (Kaggle / Hugging Face)

This system utilizes `ultil.py`  to load pre-trained model checkpoints from Kaggle Hub or Hugging Face Hub, as defined in `config.yaml`.

**For Kaggle:**
1.  Navigate to your Kaggle account settings.
2.  In the "API" section, click "Create New API Token" to download `kaggle.json`.
3.  Place the `kaggle.json` file in the root of this project, OR in the system's default path:
    * *Linux/macOS:* `~/.kaggle/kaggle.json`
    * *Windows:* `C:\Users\<username>\.kaggle\kaggle.json`
4.  (Recommended) Set file permissions:
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```

**For Hugging Face:**
* No token is typically required for public models.
* For private models, you may need to log in via `huggingface-cli login`.

## Usage & Replication of Results

Follow these steps to run the system. A GPU-accelerated environment is highly recommended.

### Step 0: Data and Model Preparation
1.  **Download Data:** Download necessary datasets (e.g., CelebA-Spoofing [cite: 150], TripletsCelebA [cite: 150], Face Detection Dataset [cite: 150]) and place them in accessible paths.
2.  **Configure Models:** Open `config.yaml`. Ensure the `handle` (Kaggle/HF path) or `local` path for all models (YOLO, ArcFace, ViT, DeepPixBiS) is correct.

### Step 1: Create the Identity Database (FR Database)
1.  Place labeled portrait images (structured by folder, e.g., `src/test_img/person 1/face.jpg` ) in the input directory.
2.  Run the `create_DB.py` script to generate the `known_faces_db.pt` file. This file contains the feature embeddings for known users.
    ```bash
    python create_DB.py
    ```

### Step 2: Run the Web Demo Application (Streamlit)
1.  This is the recommended method for interacting with the system.
2.  Run the following command from the root directory:
    ```bash
    streamlit run app.py
    ```
    
3.  Open your browser and navigate to the provided address (usually `http://localhost:8501`).
4.  Use the interface to:
    * **Register Identity:** Add new users to the database.
    * **Live Webcam:** View real-time analysis.
    * **Upload Image:** Analyze a static image and view detailed results.

### Step 3: (Optional) Run Basic Webcam Demo
1.  If you prefer not to use the Streamlit UI, you can run `main.py`.
    ```bash
    python main.py
    ```

### Step 4: Evaluate Performance and Fairness
1.  Open and run the Jupyter Notebook `src/evaluation/eval.ipynb`.
2.  This notebook will load the models and run them on a test dataset.
3.  Analyze the outputs (Accuracy, EER, APCER/BPCER, and Fairness Gaps) [cite: 164] to replicate the research findings.

## Pre-trained Models

(Links provided from user's previous draft)

| Model | Weight File (Example) | Download Link |
| :--- | :--- | :--- |
| **YOLOv8 (Face Detection)** | `yolov8n-face.pt` | `https://huggingface.co/KhanhGiauTen/YOLO_FaceDetection` |
| **ArcFace (FR)** | `arcface_resnet50.pth` | `https://huggingface.co/KhanhGiauTen/ArcFace` |
| **ViT (PAD)** | `vit_pad_model.pth` | `https://huggingface.co/KhanhGiauTen/Vision_Transformer` |
| **DeepPixBiS (PAD)** | `deeppixbis_model.pth` | `https://huggingface.co/KhanhGiauTen/DeepPixBis` |

## Results

This section presents the actual experimental results from the research paper (Chapter 4)[cite: 371].

### 1. Security Performance (PAD Module)
The system was evaluated on the CelebA-Spoofing dataset[cite: 374]. The hybrid model combining YOLOv8 and Vision Transformer (ViT) demonstrated strong performance[cite: 417].

| Metric | Result |
| :--- | :--- |
| Accuracy | 92.96% [cite: 405] |
| F1-Score | 95.08% [cite: 407] |
| ROC AUC | 97.76% [cite: 409] |
| **APCER** (Attack Error) | **2.88%** [cite: 411] |
| **BPCER** (Bona Fide Error) | **16.73%** [cite: 412] |
| ACER (Average Error) | 9.8% [cite: 414] |

**Discussion:** The model shows excellent discriminative power (ROC AUC 0.978) [cite: 421, 462] and a very low APCER (2.88%)[cite: 419, 487], indicating it is highly effective at blocking spoofing attempts. However, this comes with a trade-off: a high BPCER (16.73%) [cite: 412, 422, 490] suggests the model is "overcautious," which may impact legitimate users[cite: 423].

### 2. Recognition Performance (FR Module)
The model combining ResNet-50 and ArcFace was evaluated on the CelebA Face Recognition Triplets dataset[cite: 384, 436].

| Metric | Result |
| :--- | :--- |
| ROC-AUC | 94.89% [cite: 426] |
| EER (Equal Error Rate) | 10.57% [cite: 426] |
| TPR@FPR<=1e-3 | 25.63% [cite: 426] |
| Balanced Accuracy | 89.43% [cite: 432] |
| Recall | 89.43% [cite: 431] |
| F1-score | 18.8% [cite: 433] |
| Precision | 9.5% [cite: 434] |

**Discussion:** The model demonstrates a good ability to distinguish between individuals (ROC-AUC 0.9489) [cite: 437, 510] and a reasonable EER (10.57%)[cite: 438, 533]. The high Recall (89.43%) [cite: 439] is promising. However, the low Precision and F1-score are noted as being due to a large class imbalance in the test set[cite: 440].

## Contributions

* Developed a comprehensive and extensible **integrated FR and PAD DSS**[cite: 575].
* Demonstrated innovation by **jointly integrating FR (ArcFace/ResNet-50) and PAD (ViT/DeepPixBiS)** in a unified framework[cite: 576].
* Designed a **fairness-centric architecture** with a **Feedback Loop** (`needs_review` and `retraining_set` [cite: 630, 631]) enabling continuous learning and bias mitigation[cite: 366].
* Applied **fairness-aware strategies** at the data, model, and decision levels to enhance generalization across demographic groups[cite: 577].
* Provided a **full-featured Streamlit demo (`app.py`)**  for visualizing and interacting with the system.
* Structured the project with clean, modularized code (`ultil.py`, `predictor.py`) for ease of maintenance and reproducibility[cite: 637, 648].

## Future Work

(Based on Section 5.3: Limitations and further research directions [cite: 581])

* **Enhance FR Accuracy:** Improve the ArcFace module's accuracy through fine-tuning on domain-specific datasets [cite: 601] and exploring alternative architectures (e.g., CosFace, AdaFace, ViT backbones)[cite: 602, 603].
* **Improve Generalization:** Incorporate advanced domain adaptation and generalization techniques [cite: 605] and image enhancement networks for low-light or degraded inputs[cite: 604].
* **Integrate a Scalable DBMS:** Develop a dedicated hybrid database infrastructure (e.g., PostgreSQL for metadata + Milvus/FAISS for vector retrieval) to improve scalability, latency, and data integrity[cite: 597, 606].
* **Investigate Fairness Mechanisms:** Fully implement and investigate fairness-aware mechanisms, such as adversarial debiasing or fairness-regularized loss functions, to mitigate demographic biases[cite: 608, 609].

## Acknowledgments

Acknowledgments to the research community for providing the datasets (CelebA-Spoofing, CelebA, Face Detection Dataset [cite: 150]) and open-source model architectures.
