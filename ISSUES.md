# 📋 Project Issues - Banking DSS

Danh sách các issue được phân theo module.  
Mỗi issue có mô tả kèm Input/Output/Format để tiện triển khai và test.

---

## 1. Data

- [ ] **Thu thập dataset khuôn mặt**
  - Input: VGGFace2, LFW, FairFace, UTKFace (ảnh JPG/PNG).
  - Output: Lưu vào `data/raw/face/` (cấu trúc theo userID).
  - Format: 256x256 PNG/JPG, kèm metadata JSON.

- [ ] **Thu thập dữ liệu giao dịch**
  - Input: Credit Card Fraud dataset (CSV), synthetic transactions.
  - Output: Lưu vào `data/raw/transactions/`.
  - Format: CSV chuẩn UTF-8, schema chuẩn hóa.

- [ ] **Thu thập dữ liệu nhân khẩu học**
  - Input: UTKFace (gender, age, skin tone).
  - Output: Lưu vào `data/raw/demographics/`.
  - Format: CSV (user_id, gender, age, race).

- [ ] **Sinh dữ liệu synthetic cho nhóm ít**
  - Input: dữ liệu imbalance.
  - Output: bộ dữ liệu augmented.
  - Format: giữ schema gốc.

---

## 2. Capture

- [ ] **Desktop/Kiosk capture**
  - Input: OpenCV/GStreamer.
  - Output: Frame JPG.
  - Format: 256x256, auto-exposure.

---

## 3. Preprocessing

- [ ] **Face detection**
  - Input: Ảnh gốc.
  - Output: Bounding box + crop khuôn mặt.
  - Format: JSON (x, y, w, h) + ảnh PNG.

- [ ] **Face alignment**
  - Input: Ảnh crop.
  - Output: Ảnh aligned 256x256.
  - Format: PNG, normalized histogram.

- [ ] **Data augmentation (fairness)**
  - Input: Ảnh aligned.
  - Output: Ảnh augmented.
  - Format: đa điều kiện ánh sáng/góc chụp.

- [ ] **PAD (Presentation Attack Detection)**
  - Input: Ảnh/video.
  - Output: label {genuine, spoof}.
  - Format: JSON {score: float, label: str}.

- [ ] **Transaction cleaning**
  - Input: CSV giao dịch raw.
  - Output: cleaned CSV.
  - Format: chuẩn hóa missing value, encoding UTF-8.

---

## 4. Feature Extraction

- [ ] **Backbone training**
  - Input: face dataset (aligned).
  - Output: trained model checkpoint.
  - Format: `.pt` / `.h5`.

- [ ] **Embedding generation**
  - Input: Ảnh aligned.
  - Output: vector embedding 512-dim.
  - Format: `.npy` hoặc `.pkl`.

- [ ] **Adversarial debiasing**
  - Input: ảnh + demographic labels.
  - Output: embeddings ít chứa bias.
  - Format: checkpoint + log fairness.

- [ ] **Reweighting training**
  - Input: imbalance dataset.
  - Output: balanced training process.
  - Format: log file + metrics.

---

## 5. Face Matching

- [ ] **Similarity computation**
  - Input: 2 embeddings.
  - Output: score ∈ [0,1].
  - Format: float.

- [ ] **Adaptive threshold**
  - Input: score + demographic group.
  - Output: decision {accept/reject}.
  - Format: JSON {score: float, decision: str}.

- [ ] **Confidence calibration**
  - Input: raw score.
  - Output: calibrated probability.
  - Format: float, reliability diagram.

---

## 6. Fraud Model

- [ ] **Train fraud detection**
  - Input: transaction CSV.
  - Output: model checkpoint.
  - Format: `.pkl` / `.pt`.

- [ ] **Inference fraud score**
  - Input: transaction record.
  - Output: fraud score ∈ [0,1].
  - Format: JSON {id, score}.

---

## 7. DSS Engine

- [ ] **Decision rules**
  - Input: face score + fraud score.
  - Output: combined decision.
  - Format: JSON {face: float, fraud: float, final: str}.

- [ ] **Decision layer (grey zone)**
  - Input: combined score.
  - Output: {approve, reject, manual_review}.
  - Format: JSON.

- [ ] **Workflow orchestration**
  - Input: user session.
  - Output: full decision trace.
  - Format: log file + API response.

---

## 8. Fairness Audit

- [ ] **Compute metrics**
  - Input: prediction logs.
  - Output: FMR, FNMR, disparate impact.
  - Format: CSV/JSON metrics.

- [ ] **Bias logging**
  - Input: decision logs.
  - Output: aggregated bias per group.
  - Format: dashboard JSON.

- [ ] **Bias mitigation**
  - Input: scores.
  - Output: calibrated thresholds.
  - Format: config file update.

---

## 9. UI Dashboard

- [ ] **User upload + camera**
  - Input: selfie + ID doc.
  - Output: preview image.
  - Format: base64 or PNG.

- [ ] **Display decision**
  - Input: DSS decision JSON.
  - Output: UI status (approve/reject/review).
  - Format: Web UI.

- [ ] **Fairness monitoring view**
  - Input: metrics JSON.
  - Output: charts/plots.
  - Format: interactive dashboard.

---

## 10. Evaluation

- [ ] **Evaluate models**
  - Input: test dataset.
  - Output: accuracy, ROC, FPR/FNR.
  - Format: CSV + plots.

- [ ] **Evaluate DSS pipeline**
  - Input: simulated sessions.
  - Output: overall performance.
  - Format: report + confusion matrix.

---

## 11. Monitoring

- [ ] **Drift detection**
  - Input: live data.
  - Output: drift score.
  - Format: alert log.

- [ ] **Retrain scheduler**
  - Input: time schedule + data.
  - Output: new model checkpoint.
  - Format: `.pt` / `.pkl`.

- [ ] **Fairness metrics dashboard**
  - Input: logs.
  - Output: trends over time.
  - Format: dashboard charts.

---
