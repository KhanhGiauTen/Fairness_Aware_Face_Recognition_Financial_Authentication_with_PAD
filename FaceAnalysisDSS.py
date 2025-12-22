## 1. Nhập thư viện và Thiết lập
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import kagglehub
import math
import yaml
from ultralytics import YOLO
from torchvision.models import vit_b_16

# ==========================================================
## 2. Kiến trúc Model (Giữ nguyên định nghĩa Inline)
# ==========================================================

# --- Kiến trúc DeepPixBis ---
class DeepPixBiS(nn.Module):
    def __init__(self):
        super(DeepPixBiS, self).__init__()
        densenet = models.densenet121(weights=None) 
        self.features = densenet.features
        self.binary_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        feat = self.features(x)
        return self.binary_branch(feat)

# --- Kiến trúc ArcFace ---
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet34', in_features=512, output_dim=512, pretrained=False):
        super().__init__()
        resnet = models.resnet34(weights=None)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.fc = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=512):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(output_dim=feature_dim)
    def forward(self, x):
        features = self.feature_extractor(x)
        return F.normalize(features)

# --- Kiến trúc ViT (PAD Model) ---
class ViTModel(nn.Module):
    def __init__(self, num_classes=2, feature_dim=768):
        super().__init__()
        vit_backbone = vit_b_16(weights=None) 
        self.conv_proj = vit_backbone.conv_proj 
        self.class_token = vit_backbone.class_token
        self.pos_embedding = vit_backbone.encoder.pos_embedding 
        self.encoder_blocks = vit_backbone.encoder.layers 
        self.encoder_norm = vit_backbone.encoder.ln 
        self.classifier = nn.Linear(feature_dim, num_classes) 

    def forward(self, x):
        n = x.shape[0]
        x = self.conv_proj(x) 
        x = x.reshape(n, self.pos_embedding.shape[-1], -1).permute(0, 2, 1) 
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) 
        x = x + self.pos_embedding
        x = self.encoder_blocks(x)
        cls_token_output = self.encoder_norm(x[:, 0])
        return self.classifier(cls_token_output)

# ==========================================================
## 3. Class Pipeline Chính: FaceAnalysisDSS (NÂNG CẤP)
# ==========================================================

class FaceAnalysisDSS:
    def __init__(self, config): 
        print(">> Đang khởi tạo hệ thống DSS (Research Mode)...")
        self.config = config
        self.device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
        print(f">> Sử dụng thiết bị: {self.device}")

        # --- 1. Load Params ---
        self.ens_cfg = config['ensemble']
        self.adapt_cfg = config['adaptive_threshold']
        self.fair_cfg = config['fairness']

        # --- 2. Load Models ---
        self._load_models()
        
        # --- 3. Load Database ---
        db_path = config.get('database', {}).get('path') or config.get('known_db_path')
        if db_path:
            self.known_db = self._load_database(db_path)
        else:
            print("Warning: Không tìm thấy đường dẫn database trong config. Chạy ở chế độ không matching.")
            self.known_db = {}

        # --- 4. Transforms ---
        # Chuẩn ImageNet cho ViT/DeepPixBiS
        self.pad_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Chuẩn [-1, 1] cho ArcFace
        self.fr_transform = transforms.Compose([
            transforms.Resize((112, 112)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

    def _load_from_kaggle_hub(self, handle, filename):
        """Helper để tải và trả về đường dẫn file"""
        try:
            path = kagglehub.model_download(handle)
            full_path = os.path.join(path, filename)
            if not os.path.exists(full_path):
                # Fallback: tìm trong thư mục con nếu cấu trúc khác
                for root, dirs, files in os.walk(path):
                    if filename in files:
                        return os.path.join(root, filename)
            return full_path
        except Exception as e:
            print(f"Lỗi tải Kaggle Hub ({handle}): {e}")
            return None

    def _load_models(self):
        models_cfg = self.config['models']

        # --- Helper: Hàm load trọng số an toàn ---
        def safe_load_weights(model, path, model_name):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                # Kiểm tra nếu là checkpoint đầy đủ (có key 'model_state_dict')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    print(f"  -> Loading {model_name} from checkpoint['model_state_dict']...")
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Nếu file chỉ chứa trọng số
                    model.load_state_dict(checkpoint)
                model.eval()
            except Exception as e:
                print(f"ERROR loading {model_name}: {e}")
                # (Tùy chọn) Khởi tạo ngẫu nhiên nếu load lỗi để không crash app
                # pass 

        # 1. YOLO (Ultralytics tự xử lý việc load)
        print("Loading YOLO...")
        yolo_path = self._load_from_kaggle_hub(models_cfg['yolo']['handle'], models_cfg['yolo']['filename'])
        if yolo_path:
            self.yolo_model = YOLO(yolo_path)
        else:
            print("Warning: Không tìm thấy YOLO, dùng yolov8n.pt mặc định.")
            self.yolo_model = YOLO("yolov8n.pt")

        # 2. DeepPixBiS (Sửa lỗi load checkpoint tại đây)
        print("Loading DeepPixBiS...")
        pix_path = self._load_from_kaggle_hub(models_cfg['deeppixbis']['handle'], models_cfg['deeppixbis']['filename'])
        self.deepix_model = DeepPixBiS().to(self.device)
        if pix_path:
            safe_load_weights(self.deepix_model, pix_path, "DeepPixBiS")

        # 3. ViT (Áp dụng tương tự cho ViT vì có thể cùng định dạng checkpoint)
        print("Loading ViT...")
        vit_path = self._load_from_kaggle_hub(models_cfg['vit']['handle'], models_cfg['vit']['filename'])
        self.vit_model = ViTModel(num_classes=models_cfg['vit']['num_classes']).to(self.device)
        if vit_path:
            safe_load_weights(self.vit_model, vit_path, "ViT")

        # 4. ArcFace
        print("Loading ArcFace...")
        arc_path = self._load_from_kaggle_hub(models_cfg['arcface']['handle'], models_cfg['arcface']['filename'])
        self.arcface_model = ArcFaceModel(num_classes=models_cfg['arcface']['num_classes']).to(self.device)
        if arc_path:
            safe_load_weights(self.arcface_model, arc_path, "ArcFace")

    def _load_database(self, path):
        if os.path.exists(path):
            print(f"Loading known faces from {path}")
            return torch.load(path, map_location=self.device)
        print("Warning: Database not found. Running in demo mode without matching.")
        return {}

    # ==========================================================
    # 4. CÁC HÀM LOGIC NGHIÊN CỨU (SCIENTIFIC CORE)
    # ==========================================================

    def _assess_image_quality(self, face_img_cv2):
        """Tính điểm chất lượng ảnh (Quality Score)"""
        gray = cv2.cvtColor(face_img_cv2, cv2.COLOR_BGR2GRAY)
        # Độ nét (Blur) dùng Laplacian Variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, max(0.0, (laplacian_var - 50) / 450)) # Chuẩn hóa [0,1]
        
        # Độ sáng (Brightness)
        mean_brightness = np.mean(gray)
        if 80 <= mean_brightness <= 200: bright_score = 1.0
        else: bright_score = max(0.0, 1.0 - (abs(mean_brightness - 140) / 140))
        
        # Tổng hợp: 70% độ nét, 30% độ sáng
        return 0.7 * blur_score + 0.3 * bright_score

    def _calculate_adaptive_threshold(self, quality_score):
        """Adaptive Thresholding: Tăng ngưỡng khi ảnh xấu"""
        base = self.adapt_cfg['base_fr_threshold']
        alpha = self.adapt_cfg['quality_penalty_weight']
        penalty = alpha * (1.0 - quality_score)
        return min(0.95, base + penalty)

    def _predict_demographics(self, face_img):
        """Giả lập phân loại nhóm để demo tính năng Fairness"""
        # Trong thực tế, dùng model FairFace ở đây.
        # Demo: trả về 'unknown' (dùng bias mặc định)
        return 'unknown'

    def _calibrate_fairness_score(self, raw_score, demographic_group):
        """Fairness Calibration: Cộng điểm bù cho nhóm yếu thế"""
        if not self.fair_cfg['enable']: return raw_score, 0.0
        
        offset = self.fair_cfg['bias_offset'].get(demographic_group, 0.0)
        return min(1.0, raw_score + offset), offset

    # ==========================================================
    # 5. Pipeline Xử lý (Thay thế process_frame cũ)
    # ==========================================================

    def analyze_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # B1: Detect Face
        yolo_cfg = self.config['models']['yolo']
        yolo_conf = yolo_cfg.get('conf_threshold', 0.5)
        yolo_results = self.yolo_model(frame_rgb, verbose=False, conf=yolo_conf)
        pipeline_results = []

        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            face_crop_rgb = frame_rgb[y1:y2, x1:x2]
            if face_crop_rgb.size == 0: continue
            
            face_pil = Image.fromarray(face_crop_rgb)
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)

            # --- A. Quality Assessment ---
            q_score = self._assess_image_quality(face_crop_bgr)
            if q_score < self.adapt_cfg['min_quality_score']:
                # Ảnh quá xấu -> Bỏ qua hoặc đánh dấu
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (128, 128, 128), 2)
                continue

            # --- B. PAD Ensemble (Joint Learning) ---
            pad_input = self.pad_transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # ViT Score
                vit_out = self.vit_model(pad_input)
                vit_prob = F.softmax(vit_out, dim=1)[0][1].item() # Class 1 = Spoof? Kiểm tra lại logic train
                # Lưu ý: Logic train của bạn: 0=Live, 1=Spoof hay ngược lại? 
                # Dựa vào code cũ: "spoof_prob_vit = vit_probs[0, 1]", giả sử 1 là Spoof.
                
                # DeepPixBiS Score (0-1, thường 1 là Spoof hoặc ngược lại, cần verify)
                # Giả sử DeepPixBiS output sigmoid là xác suất Spoof
                pix_prob = self.deepix_model(pad_input).item()

            # Weighted Fusion
            w_vit = self.ens_cfg['vit_weight']
            w_pix = self.ens_cfg['pixbis_weight']
            fused_spoof_prob = (w_vit * vit_prob) + (w_pix * pix_prob)
            
            is_spoof = fused_spoof_prob > self.ens_cfg['spoof_threshold']

            # --- C. Face Recognition & Fairness ---
            if not is_spoof:
                fr_input = self.fr_transform(face_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.arcface_model(fr_input)
                
                # Matching
                best_sim = 0.0
                best_id = "Unknown"
                if len(self.known_db) > 0:
                    # So sánh với database (Dictionary {name: tensor})
                    # Cần đảm bảo database lưu tensor đúng shape
                    for name, db_emb in self.known_db.items():
                        # db_emb shape [1, 512] or [512]
                        sim = F.cosine_similarity(emb, db_emb.to(self.device).view(1, -1)).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_id = name
                
                # Fairness Calibration
                demo_group = self._predict_demographics(face_crop_bgr)
                calibrated_sim, offset = self._calibrate_fairness_score(best_sim, demo_group)
                
                # Adaptive Threshold
                final_thresh = self._calculate_adaptive_threshold(q_score)
                
                is_match = calibrated_sim > final_thresh
                
                # --- Drawing ---
                color = (0, 255, 0) if is_match else (0, 165, 255) # Xanh hoặc Cam (Unknown)
                label = f"{best_id} ({calibrated_sim:.2f})" if is_match else f"Unknown ({calibrated_sim:.2f})"
                
                # Debug info
                # print(f"Raw: {best_sim:.2f} | Calib: {calibrated_sim:.2f} | Thresh: {final_thresh:.2f}")

            else:
                # Spoof detected
                color = (0, 0, 255)
                label = f"FAKE ({fused_spoof_prob:.2f})"

            # Vẽ bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            pipeline_results.append({
                "bbox": [x1, y1, x2, y2],
                "status": "FAKE" if is_spoof else ("PASS" if is_match else "UNKNOWN"),
                "scores": {
                    "pad": fused_spoof_prob,
                    "fr": calibrated_sim if not is_spoof else 0.0,
                    "quality": q_score
                }
            })

        return frame_bgr, pipeline_results

    def run_demo(self):
        print(">> Đang mở camera... (Nhấn 'q' để thoát)")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, results = self.analyze_frame(frame)
            cv2.imshow('Joint Fairness-Aware DSS', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# ==========================================================
## 4. Main Block
# ==========================================================

if __name__ == "__main__":
    # Load Config
    print(">> Loading configuration...")
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'config/config.yaml'.")
        exit()

    # Login Kaggle (Optional check)
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        try:
            kagglehub.login()
        except Exception as exc:
            print(f"Warning: Lỗi đăng nhập Kaggle: {exc}")
    else:
        print("Warning: Bỏ qua đăng nhập Kaggle vì chưa thiết lập biến môi trường KAGGLE_USERNAME/KAGGLE_KEY.")

    # Init DSS
    dss = FaceAnalysisDSS(config)
    
    # Run
    dss.run_demo()