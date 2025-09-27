# Bias-Mitigation-in-Face-Recognition-Based-Financial-Transaction-Authentication
Research and Development of a Decision Support System for Bias Mitigation in Face Recognition-Based Financial Transaction Authentication

# Banking-DSS

## Giới thiệu  
Banking-DSS là một hệ thống **Decision Support System (DSS)** hỗ trợ xác thực giao dịch tài chính thông qua **nhận dạng khuôn mặt** và **phát hiện gian lận**, đồng thời tích hợp các kỹ thuật **fairness audit** để giảm thiểu thiên lệch (bias).  

Hệ thống gồm các module chính:  
- **Xử lý dữ liệu**: chuẩn bị dữ liệu khuôn mặt, giao dịch và nhân khẩu học.  
- **Nhận dạng khuôn mặt**: huấn luyện và suy luận mô hình xác thực người dùng.  
- **Phát hiện gian lận**: phát hiện giao dịch bất thường.  
- **Fairness Audit**: đánh giá và giảm bias giữa các nhóm nhân khẩu học.  
- **DSS Engine**: kết hợp kết quả nhiều mô hình, đưa ra quyết định cuối cùng.  
- **UI Dashboard**: giao diện quản trị và demo DSS.  
- **Evaluation**: đánh giá hệ thống tổng thể.  

---

## Cấu trúc thư mục
banking-dss/
│
├── data/
│ ├── raw/
│ │ ├── face/ # Dữ liệu khuôn mặt gốc (VGGFace2, LFW, FairFace,...)
│ │ ├── transactions/ # Dữ liệu giao dịch (Credit Card Fraud, synthetic data)
│ │ └── demographics/ # Dữ liệu nhân khẩu học (UTKFace)
│ ├── processed/
│ │ ├── face_embeddings/ # Vector đặc trưng khuôn mặt đã trích xuất
│ │ └── transactions_cleaned/
│ └── README.md
│
├── preprocessing/
│ ├── face_preprocess.py # Resize, align, augment ảnh khuôn mặt
│ ├── transaction_clean.py # Làm sạch dữ liệu giao dịch
│ └── demographics_merge.py # Gắn nhãn nhân khẩu học cho dữ liệu
│
├── face_model/
│ ├── train_face_model.py # Train mô hình nhận dạng khuôn mặt
│ ├── inference_face.py # Inference xác thực khuôn mặt
│ ├── model_checkpoint/ # Lưu model
│ └── utils.py
│
├── fraud_model/
│ ├── train_fraud_model.py # Train mô hình phát hiện gian lận
│ ├── inference_fraud.py # Inference risk score giao dịch
│ └── model_checkpoint/
│
├── fairness_audit/
│ ├── audit_metrics.py # Tính fairness metrics (FMR, FNMR, disparate impact)
│ ├── bias_mitigation.py # Kỹ thuật giảm bias (threshold per-group, calibration)
│ └── reports/ # Lưu báo cáo audit
│
├── dss_engine/
│ ├── decision_rules.py # Luật kết hợp kết quả face match + fraud score
│ ├── decision_workflow.py # Pipeline ra quyết định
│ └── config.yaml # Tham số threshold, rule weights
│
├── ui_dashboard/
│ ├── app.py # Web UI (Streamlit/FastAPI + React)
│ ├── templates/ # Giao diện
│ ├── static/
│ └── api/ # API gọi DSS engine
│
├── evaluation/
│ ├── model_eval.py # Đánh giá hiệu năng từng mô hình
│ ├── dss_eval.py # Đánh giá hệ DSS tổng thể
│ └── plots/ # Biểu đồ kết quả
│
├── docs/
│ ├── architecture_diagram.png
│ ├── dataset_overview.md
│ ├── methodology.md
│ └── references.md
│
├── requirements.txt
└── README.md


---

## Cài đặt




