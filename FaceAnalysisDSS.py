## 1. Nhập thư viện và Thiết lập
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset # Bỏ comment nếu cần type hinting
import cv2
import numpy as np
from PIL import Image
# from tqdm import tqdm # Bỏ comment nếu cần
import os
import shutil
import kagglehub # Đảm bảo đã import và login
import math
from ultralytics import YOLO
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ==========================================================
## 2. Kiến trúc Model 
# ==========================================================

# --- Kiến trúc DeepPixBis  ---
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
        binary_out = self.binary_branch(feat)
        return binary_out 

# --- Kiến trúc ArcFace ---
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
    def forward(self, input, label): 
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet34', in_features = 512, output_dim=512, pretrained=False):
        super().__init__()
        assert backbone_name == 'resnet34', "Phiên bản này chỉ hỗ trợ resnet34"
        resnet = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        actual_in_features = 512
        if output_dim != actual_in_features:
            self.fc = nn.Linear(actual_in_features, output_dim)
        else:
            self.fc = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=512, scale=30.0, margin=0.5):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(
            backbone_name='resnet34',
            output_dim=feature_dim,
            pretrained=False
        )
    def forward(self, x, labels=None):
        features = self.feature_extractor(x)
        return F.normalize(features)

# --- Kiến trúc ViT (PAD Model) ---
class ViTModel(nn.Module):
    def __init__(self, num_classes=2, feature_dim=768, pretrained=False):
        super().__init__()
        vit_backbone = vit_b_16(weights=None) 
        
        self.conv_proj = vit_backbone.conv_proj 
        self.class_token = vit_backbone.class_token
        self.pos_embedding = vit_backbone.encoder.pos_embedding 
        self.encoder_blocks = vit_backbone.encoder.layers 
        self.encoder_norm = vit_backbone.encoder.ln 
        
        self.classifier = nn.Linear(feature_dim, num_classes) 

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv_proj(x) 
        x = x.reshape(n, self.pos_embedding.shape[-1], -1) 
        x = x.permute(0, 2, 1) 
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) 
        x = x + self.pos_embedding
        x = self.encoder_blocks(x)
        cls_token_output = x[:, 0]
        cls_token_output = self.encoder_norm(cls_token_output)
        logits = self.classifier(cls_token_output) 
        return logits

# ==========================================================
## 3. Class Pipeline Chính FaceAnalysisDSS 
# ==========================================================

class FaceAnalysisDSS:
    def __init__(self, yolo_handle, deepix_handle, arcface_handle, vit_handle, known_db_path, num_arcface_classes=1000, num_vit_classes=2,vit_spoof_threshold=0.7): 
        print("Đang khởi tạo hệ thống DSS...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

        # --- Tải các Model ---
        self.yolo_model = self._load_yolo_from_kaggle_hub(yolo_handle) 
        
        # self.deepix_model = self._load_from_kaggle_hub(
        #     model_architecture_class=DeepPixBiS, 
        #     model_handle=deepix_handle, 
        #     model_filename="DeepPixBis.pt" 
        # )
        
        # Tải ArcFaceModel 
        self.arcface_model = self._load_from_kaggle_hub(
            model_architecture_class=ArcFaceModel,
            model_handle=arcface_handle,
            model_filename="ArcFace.pt", 
            num_classes=num_arcface_classes, 
            feature_dim=512 
        )

        # Tải ViTModel (dùng làm Spoof Detector chính)
        self.vit_model = self._load_from_kaggle_hub(
            model_architecture_class=ViTModel,
            model_handle=vit_handle,
            model_filename="ViT.pt", 
            num_classes=num_vit_classes, # Phải là 2 (Live/Spoof)
            pretrained=False 
        )
        # # DeepPixBis (kích thước input của DenseNet) - Comment lại
        # self.deepix_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        # ])
        
        # ArcFace 
        self.arcface_transform = transforms.Compose([
            transforms.Resize((112, 112)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])
        
        # ViT (Spoof Detector)
        # Dùng chuẩn ImageNet cho ViT
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def _get_identity(self, embedding, threshold=0.4):
        if self.known_embeddings.shape[0] == 0: 
            return "KhongCoCSDL", 0.0
        cos_sim = F.linear(embedding, self.known_embeddings) 
        best_score, best_idx = torch.max(cos_sim, dim=1)
        score = best_score.item()
        if score > threshold:
            return self.known_names[best_idx.item()], score
        else:
            return "KhongXacDinh", score

    # (ĐÃ CẬP NHẬT) Hàm xử lý chính của pipeline
    def process_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # --- Bước 1: Phát hiện khuôn mặt (YOLO) ---
        yolo_results = self.yolo_model(frame_rgb, verbose=False, conf=0.6) 
        detections = yolo_results[0].boxes
        pipeline_results = [] 

        # --- Bước 2: Xử lý từng khuôn mặt ---
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            face_crop_rgb = frame_rgb[y1:y2, x1:x2]
            if face_crop_rgb.shape[0] < 10 or face_crop_rgb.shape[1] < 10: continue 
            face_pil = Image.fromarray(face_crop_rgb)

            # --- Bước 3: Chống giả mạo (ViT là chính) ---
            vit_input = self.vit_transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vit_logits = self.vit_model(vit_input) 
                vit_probs = F.softmax(vit_logits, dim=1)
                spoof_prob_vit = vit_probs[0, 1].item() # Xác suất là Spoof (lớp 1) từ ViT
                vit_pred = vit_logits.argmax(-1).item() 
                status_label = self.vit_labels.get(vit_pred) # THAT hoặc GIA MAO
                
            if spoof_prob_vit > self.vit_spoof_threshold: 
                status_label = "GIA MAO" # Gán lại nhãn nếu vượt ngưỡng
                color = (0, 0, 255) # Màu đỏ
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                spoof_text = f"{status_label} (ViT: {float(spoof_prob_vit):.2f})" if isinstance(spoof_prob_vit, (float, int)) else f"{status_label} (ViT: {spoof_prob_vit})"
                cv2.putText(frame_bgr, spoof_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pipeline_results.append({
                    "bbox": [x1, y1, x2, y2], 
                    "status": status_label, 
                    "xac_suat_gia_mao_vit": spoof_prob_vit
                 })
                continue
            # --- (COMMENT LẠI) Bước 3 Phụ: Chống giả mạo (DeepPixBis) ---
            # spoof_prob_deepix = -1.0 # Giá trị mặc định nếu không chạy
            # if hasattr(self, 'deepix_model'): # Kiểm tra nếu model được load
            #     deepix_input = self.deepix_transform(face_pil).unsqueeze(0).to(self.device)
            #     with torch.no_grad():
            #         spoof_prob_deepix = self.deepix_model(deepix_input).item() 

            # --- Quyết định DSS 1: Kiểm tra giả mạo (Dựa trên ViT) ---
            # (Sau này có thể kết hợp cả spoof_prob_vit và spoof_prob_deepix)
            final_spoof_decision = (status_label == "GIA MAO") # Quyết định dựa trên ViT
            
            if final_spoof_decision:
                status = "GIA MAO"
                color = (0, 0, 255) # Màu đỏ
                # Hiển thị xác suất của ViT (hoặc cả hai nếu muốn)
                display_prob = spoof_prob_vit 
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                display_text = f"{status} (ViT: {float(display_prob):.2f})" if isinstance(display_prob, (float, int)) else f"{status} (ViT: {display_prob})"
                cv2.putText(frame_bgr, display_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pipeline_results.append({
                    "bbox": [x1, y1, x2, y2], 
                    "status": status, 
                    "xac_suat_gia_mao_vit": spoof_prob_vit,
                    # "xac_suat_gia_mao_deepix": spoof_prob_deepix # Thêm nếu cần
                 })
                continue 

            # --- Nếu là khuôn mặt thật (LIVE / THAT) ---
            status = "THAT"
            color = (0, 255, 0) # Màu xanh lá

            # --- Bước 4: Định danh (ArcFace) ---
            arcface_input = self.arcface_transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.arcface_model(arcface_input) 
                identity, id_score = self._get_identity(embedding)

            # --- Bước 5: Tổng hợp kết quả và vẽ ---
            result_text = f"{identity} ({float(id_score):.2f})" if isinstance(id_score, (float, int)) else f"{identity} ({id_score})" 
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, result_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            pipeline_results.append({
                "bbox": [x1, y1, x2, y2], 
                "status": status, 
                "dinh_danh": identity,
                "diem_dinh_danh": id_score,
                "xac_suat_gia_mao_vit": spoof_prob_vit, # Vẫn lưu lại để tham khảo
                # "xac_suat_gia_mao_deepix": spoof_prob_deepix
            })
            
        return frame_bgr, pipeline_results

    def run_demo(self):
        print("Đang mở camera... (Nhấn 'q' để thoát)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): 
            print("Lỗi: Không thể mở camera.")
            return
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Lỗi: Không thể đọc khung hình.")
                break
            processed_frame, results = self.process_frame(frame)
            if results: 
                live_faces = [r for r in results if r['status'] == 'THAT']
                spoof_faces = len(results) - len(live_faces)
                print(f"Kết quả khung hình: {len(live_faces)} THAT, {spoof_faces} GIA MAO | Chi tiết: {results}")
            cv2.imshow('He thong DSS Phan tich Khuon mat (Nhan \'q\' de thoat)', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Đã đóng camera.")


# ==========================================================
## 4. Khối Khởi tạo và Thực thi 
# ==========================================================

if __name__ == "__main__":
    # --- Đăng nhập Kaggle Hub ---
    try:
        print("Đang thử đăng nhập Kaggle Hub...")
        kagglehub.login()
        print("Đăng nhập Kaggle Hub thành công.")
    except Exception as e:
        print(f"Lỗi đăng nhập Kaggle Hub: {e}. Vui lòng đảm bảo đã cấu hình Kaggle API token.")
        exit() 

    # --- Định nghĩa Handle/Đường dẫn Model và Tham số ---
    YOLO_HANDLE = "khnhnguyn222/yolo-facedetection/pyTorch/default" 
    DEEPIX_HANDLE = "khnhnguyn222/deeppixbis/pyTorch/default" # Giữ lại handle      
    ARCFACE_HANDLE = "khnhnguyn222/arcface-resnet34/pyTorch/default" 
    VIT_HANDLE = "khnhnguyn222/vision-transformer/pyTorch/default" 
    
    KNOWN_DB_PATH = "/kaggle/input/arcface-database/known_faces_db.pt" # Ví dụ
    NUM_ARCFACE_CLASSES_TRAINED = 1000 
    NUM_VIT_CLASSES_TRAINED = 2         

    # --- Khởi tạo và Chạy Pipeline ---
    try:
        dss_pipeline = FaceAnalysisDSS(
            yolo_handle=YOLO_HANDLE,
            deepix_handle=DEEPIX_HANDLE, # Vẫn truyền handle vào
            arcface_handle=ARCFACE_HANDLE,
            vit_handle=VIT_HANDLE,
            known_db_path=KNOWN_DB_PATH,
            num_arcface_classes=NUM_ARCFACE_CLASSES_TRAINED,
            num_vit_classes=NUM_VIT_CLASSES_TRAINED, # Phải là 2
            vit_spoof_threshold=0.7
        )
        
        dss_pipeline.run_demo()
        
    except Exception as e:
        print(f"\n--- Lỗi Pipeline ---")
        import traceback
        traceback.print_exc() 
        print("\nĐã xảy ra lỗi trong quá trình khởi tạo hoặc chạy DSS.")
        print("Vui lòng kiểm tra lại handle/đường dẫn model, định nghĩa class, các tham số bắt buộc (như num_classes), và sự tồn tại của file.")

    print("\n--- Hoàn tất Demo DSS ---")