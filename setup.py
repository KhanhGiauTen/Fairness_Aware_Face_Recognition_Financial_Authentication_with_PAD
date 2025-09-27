import os

# Cấu trúc thư mục và file
structure = {
    "data/raw/face": [],
    "data/raw/transactions": [],
    "data/raw/demographics": [],
    "data/processed/face_aligned": [],
    "data/processed/face_embeddings": [],
    "data/processed/transactions_cleaned": [],
    "capture": ["mobile_capture.py", "web_capture.js", "desktop_capture.py"],
    "preprocessing": [
        "face_detect.py", "face_align.py", "face_augment.py",
        "pad_detection.py", "transaction_clean.py"
    ],
    "feature_extraction": [
        "backbone.py", "debias_training.py", "fairness_nas.py", "embeddings.py"
    ],
    "face_matching": ["similarity.py", "threshold_adaptive.py", "calibration.py"],
    "fraud_model/model_checkpoint": [],
    "fraud_model": ["train_fraud_model.py", "inference_fraud.py"],
    "dss_engine": ["decision_rules.py", "decision_layer.py", "workflow.py", "config.yaml"],
    "fairness_audit/reports": [],
    "fairness_audit": ["audit_metrics.py", "bias_logging.py", "bias_mitigation.py"],
    "ui_dashboard/api": [],
    "ui_dashboard/templates": [],
    "ui_dashboard/static": [],
    "ui_dashboard": ["app.py"],
    "evaluation/plots": [],
    "evaluation": ["model_eval.py", "dss_eval.py"],
    "monitoring": ["drift_detection.py", "retrain_scheduler.py", "metrics_dashboard.py"],
    "docs": [
        "architecture_diagram.png", "pipeline_overview.md",
        "dataset_overview.md", "methodology.md", "references.md"
    ]
}

root_files = ["requirements.txt", "README.md", "data/README.md"]

def create_structure(base_dir="banking-dss"):
    for path, files in structure.items():
        dir_path = os.path.join(base_dir, path)
        os.makedirs(dir_path, exist_ok=True)
        for f in files:
            file_path = os.path.join(dir_path, f)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as fp:
                    fp.write("")

    # Tạo file gốc
    for f in root_files:
        file_path = os.path.join(base_dir, f)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write("")

if __name__ == "__main__":
    create_structure()
    print("✅ Repo structure for 'banking-dss' created successfully!")
