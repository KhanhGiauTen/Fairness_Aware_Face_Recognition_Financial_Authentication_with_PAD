from pathlib import Path
import os
import logging
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


from src.model.deeppixbis import DeepPixBiS
from src.model.arcface import ArcFaceModel
from src.model.vit import ViTModel


from src.ultil import get_model_entry, load_model_asset, load_checkpoint_to_model, get_source

logger = logging.getLogger(__name__)

class FaceAnalysisDSS:
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    @classmethod
    def from_config(cls, cfg: dict):
        inst = cls()
        logger.info(f"Khởi tạo FaceAnalysisDSS trên thiết bị: {inst.device}")

        model_names = ['yolo', 'arcface', 'vit', 'deeppixbis']

        for name in model_names:
            logger.info(f"Đang xử lý model: {name}")
            entry = get_model_entry(cfg, name)

            if not entry:
                if name == 'deeppixbis': 
                    logger.warning(f"Không tìm thấy cấu hình cho model '{name}'. Bỏ qua DeepPixBis.")
                    inst.deepix_model = None
                    continue
                else:
                    raise ValueError(f"Thiếu cấu hình bắt buộc cho model '{name}' trong file config.")

            handle = entry.get('handle')
            filename = entry.get('filename')
            source = get_source(cfg, name)

            if not handle:
                raise ValueError(f"Thiếu 'handle' (hoặc đường dẫn local) cho model '{name}' trong file config.")

            model_path = None
            try:
                logger.info(f"Đang tải '{name}' từ nguồn '{source}'...")
                model_path = load_model_asset(source, handle, filename)
                logger.info(f"Đã lấy tài nguyên cho '{name}' tại: {model_path}")

                if name == 'yolo':
                    from ultralytics import YOLO
                    inst.yolo_model = YOLO(str(model_path))
                    logger.info(f"Đã load YOLO.")

                elif name == 'arcface':
                    num_classes = entry.get('num_classes', 1000) 
                    feature_dim = entry.get('feature_dim', 512) 
                    inst.arcface_model = ArcFaceModel(num_classes=num_classes, feature_dim=feature_dim)
                    load_checkpoint_to_model(inst.arcface_model, model_path, device=inst.device)
                    logger.info(f"Đã load ArcFace.")

                elif name == 'vit':
                    num_vit_classes = entry.get('num_classes', 2) 
                    inst.vit_model = ViTModel(num_classes=num_vit_classes)
                    load_checkpoint_to_model(inst.vit_model, model_path, device=inst.device)
                    logger.info(f"Đã load ViT.")

                elif name == 'deeppixbis':
                    inst.deepix_model = DeepPixBiS()
                    load_checkpoint_to_model(inst.deepix_model, model_path, device=inst.device)
                    logger.info(f"Đã load DeepPixBis.")

            except Exception as e:

                if name != 'deeppixbis':
                    logger.error(f"Lỗi nghiêm trọng khi tải model '{name}' từ '{source}': {e}", exc_info=True) 
                    raise RuntimeError(f"Không thể tải model bắt buộc '{name}': {e}")
                else:
                    logger.warning(f"Không thể tải model tùy chọn 'deeppixbis' từ '{source}': {e}. Tiếp tục mà không có DeepPixBis.")
                    inst.deepix_model = None 
                    
        inst.vit_labels = {0: 'LIVE', 1: 'SPOOF'}
        inst.vit_spoof_threshold = float(cfg.get('vit_spoof_threshold', 0.7))
        logger.info(f"Ngưỡng ViT spoof: {inst.vit_spoof_threshold}")

        known_path = cfg.get('known_db_path')
        inst.known_embeddings, inst.known_names = inst._load_known_db(known_path)
        inst._init_transforms() 
        logger.info("Khởi tạo FaceAnalysisDSS hoàn tất.")
        return inst

    def _load_known_db(self, db_path):
        if not db_path or not os.path.exists(db_path):
            logger.warning("Database not found at %s. Using synthetic data.", db_path)
            known_embeddings = [torch.randn(512), torch.randn(512)]
            known_names = ["Person1", "Person2"]
        else:
            try:
                db_data = torch.load(db_path, map_location='cpu')
                if 'embeddings' not in db_data or 'names' not in db_data:
                    raise KeyError("Database file must contain keys 'embeddings' and 'names'.")
                known_embeddings = db_data['embeddings']
                known_names = db_data['names']
                logger.info(f"Đã tải {len(known_names)} định danh từ CSDL.")
            except Exception as e:
                logger.warning("Error loading DB file: %s. Using synthetic data.", e)
                known_embeddings = [torch.randn(512), torch.randn(512)]
                known_names = ["Person1", "Person2"]


        if isinstance(known_embeddings, torch.Tensor):
            embeddings_tensor = known_embeddings.to(self.device)
        elif isinstance(known_embeddings, list) and len(known_embeddings) > 0:
            try:
                embeddings_tensor = F.normalize(torch.stack(known_embeddings), p=2, dim=1).to(self.device)
            except Exception as stack_err:
                 logger.error(f"Lỗi khi stack/normalize CSDL embeddings: {stack_err}. Kiểm tra định dạng embeddings trong file DB.")
                 embeddings_tensor = torch.empty((0, 512)).to(self.device)
        else:
            logger.warning("CSDL rỗng hoặc không hợp lệ.")
            embeddings_tensor = torch.empty((0, 512)).to(self.device)

        return embeddings_tensor, known_names

    def _init_transforms(self):
        from torchvision import transforms
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Transform cho ArcFace
        self.arcface_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Transform cho ViT
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        if hasattr(self, 'deepix_model') and self.deepix_model is not None:
             self.deepix_transform = transforms.Compose([
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
             ])
             logger.info("Đã khởi tạo transform cho DeepPixBis.")
        else:
             self.deepix_transform = None
             logger.info("Bỏ qua transform cho DeepPixBis (model không được load).")
        logger.info("Đã khởi tạo các transforms còn lại.")

    def _get_identity(self, embedding, threshold=0.4):
        if self.known_embeddings.shape[0] == 0:
            return "NoDatabase", 0.0
        try:
            cos_sim = F.linear(embedding, self.known_embeddings)
            best_score, best_idx = torch.max(cos_sim, dim=1)
            score = best_score.item()
            if score > threshold:
                idx = best_idx.item()
                if 0 <= idx < len(self.known_names):
                    return self.known_names[idx], score
                else:
                    logger.error(f"Lỗi index định danh: index {idx} / số lượng tên {len(self.known_names)}")
                    return "IndexError", score 
            else:
                return "Unknown", score
        except Exception as e:
            logger.error(f"Lỗi trong _get_identity: {e}")
            return "ErrorGetID", 0.0 

    def process_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pipeline_results = []
        try:
            yolo_results = self.yolo_model(frame_rgb, verbose=False, conf=0.6)
            detections = yolo_results[0].boxes
        except Exception as e:
            logger.error(f"Lỗi khi chạy YOLO detection: {e}")
            detections = []

        # Đọc config uncertainty nếu có
        lower_spoof = getattr(self, 'uncertainty_lower', 0.65)
        upper_spoof = getattr(self, 'uncertainty_upper', 0.75)
        auto_save_enabled = getattr(self, 'auto_save_enabled', True)

        for box in detections:
            status_label = "ERROR"
            spoof_prob_vit = -1.0
            spoof_prob_dp = -1.0
            identity = "N/A"
            id_score = 0.0
            color = (0, 165, 255)

            try:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                face_crop_rgb = frame_rgb[y1:y2, x1:x2]
                if face_crop_rgb.shape[0] < 10 or face_crop_rgb.shape[1] < 10:
                    continue
                face_pil = Image.fromarray(face_crop_rgb)

                vit_input = self.vit_transform(face_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    vit_logits = self.vit_model(vit_input)
                    vit_probs = F.softmax(vit_logits, dim=1)
                    spoof_prob_vit = vit_probs[0, 1].item()

                if hasattr(self, 'deepix_model') and self.deepix_model is not None and self.deepix_transform is not None:
                    try:
                        dp_input = self.deepix_transform(face_pil).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            spoof_prob_dp = self.deepix_model(dp_input).item()
                    except Exception as e_dp:
                        logger.error(f"Lỗi khi chạy DeepPixBis: {e_dp}")

                reasons = []
                # Chỉ auto-save nếu spoof_prob_vit nằm trong khoảng uncertainty
                if lower_spoof <= spoof_prob_vit <= upper_spoof:
                    reasons.append('vit_uncertain')

                scores = {
                    'spoof_prob_vit': spoof_prob_vit
                }
                # Nếu vẫn dùng DeepPixBis, có thể lưu lại giá trị nhưng không dùng cho auto-save
                if spoof_prob_dp != -1.0:
                    scores['spoof_prob_dp'] = spoof_prob_dp

                if spoof_prob_vit > self.vit_spoof_threshold:
                    status_label = "SPOOF"
                    color = (0, 0, 255)
                    text = f"{status_label} (ViT:{float(spoof_prob_vit):.2f})" if isinstance(spoof_prob_vit, (float, int)) else f"{status_label} (ViT:{spoof_prob_vit})"
                    if spoof_prob_dp != -1.0:
                        text += f" (DP:{float(spoof_prob_dp):.2f})" if isinstance(spoof_prob_dp, (float, int)) else f" (DP:{spoof_prob_dp})"
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    pipeline_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "status": status_label,
                        "identity": "N/A",
                        "identity_score": 0.0,
                        "spoof_prob_vit": spoof_prob_vit,
                        "spoof_prob_dp": spoof_prob_dp,
                        "reasons": reasons,
                        "scores": scores
                    })
                    continue

                status_label = "LIVE"
                color = (0, 255, 0)
                arcface_input = self.arcface_transform(face_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.arcface_model(arcface_input)
                    identity, id_score = self._get_identity(embedding)

                scores['identity_score'] = id_score

                result_text = f"{identity} ({float(id_score):.2f})" if isinstance(id_score, (float, int)) else f"{identity} ({id_score})"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, result_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pipeline_results.append({
                    "bbox": [x1, y1, x2, y2],
                    "status": status_label,
                    "identity": identity,
                    "identity_score": id_score,
                    "spoof_prob_vit": spoof_prob_vit,
                    "spoof_prob_dp": spoof_prob_dp,
                    "reasons": reasons,
                    "scores": scores
                })

            except Exception as e_inner:
                logger.error(f"Lỗi khi xử lý bounding box {box.xyxy[0].int().tolist()}: {e_inner}", exc_info=True)
                pipeline_results.append({"bbox": box.xyxy[0].int().tolist() if box.xyxy.numel() > 0 else [], "status": "ERROR_PROCESSING"})

        return frame_bgr, pipeline_results

    def run_demo(self):
        logger.info("Opening camera... (press 'q' to exit)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): logger.error("Cannot open camera."); raise RuntimeError("Cannot open camera")
        while True:
            ret, frame = cap.read()
            if not ret: logger.error("Failed to read frame from camera."); break
            try:
                 processed_frame, results = self.process_frame(frame)
                 if results:
                     live_faces = [r for r in results if r['status'] == 'LIVE']
                     spoof_faces = len([r for r in results if r['status'] == 'SPOOF'])
                     error_faces = len(results) - len(live_faces) - spoof_faces
                     logger.info("Frame results: %d LIVE, %d SPOOF, %d ERROR | Details: %s", len(live_faces), spoof_faces, error_faces, results)
                 cv2.imshow('Face Analysis DSS (press q to exit)', processed_frame)
            except Exception as e:
                 logger.error(f"Lỗi nghiêm trọng trong vòng lặp xử lý frame: {e}", exc_info=True) 
                 cv2.putText(frame, "CRITICAL ERROR IN LOOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                 cv2.imshow('Face Analysis DSS (press q to exit)', frame) 

            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera closed.")