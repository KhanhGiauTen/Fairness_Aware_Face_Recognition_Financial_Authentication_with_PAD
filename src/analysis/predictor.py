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

from src.ultil import get_model_entry, load_from_kaggle, load_checkpoint_to_model

logger = logging.getLogger(__name__)



class FaceAnalysisDSS:
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    @classmethod
    def from_config(cls, cfg: dict):
        inst = cls()

        # YOLO
        yolo_entry = get_model_entry(cfg, 'yolo')
        yolo_handle = yolo_entry.get('handle')
        yolo_file = yolo_entry.get('filename')
        try:
            yolo_path = load_from_kaggle(yolo_handle, filename=yolo_file)
            from ultralytics import YOLO
            inst.yolo_model = YOLO(str(yolo_path)) if os.path.isfile(yolo_path) else YOLO(str(yolo_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO: {e}")

        arc_entry = get_model_entry(cfg, 'arcface')
        arc_handle = arc_entry.get('handle')
        arc_file = arc_entry.get('filename')
        num_classes = arc_entry.get('num_classes', 1000)
        feature_dim = arc_entry.get('feature_dim', 512)
        inst.arcface_model = ArcFaceModel(num_classes=num_classes, feature_dim=feature_dim).to(inst.device)
        try:
            arc_ckpt = load_from_kaggle(arc_handle, filename=arc_file)
            if os.path.isfile(arc_ckpt):
                load_checkpoint_to_model(inst.arcface_model, arc_ckpt, device=inst.device)
        except Exception as e:
            logger.warning("Could not load ArcFace checkpoint: %s", e)

        vit_entry = get_model_entry(cfg, 'vit')
        vit_handle = vit_entry.get('handle')
        vit_file = vit_entry.get('filename')
        num_vit_classes = vit_entry.get('num_classes', 2)
        inst.vit_model = ViTModel(num_classes=num_vit_classes).to(inst.device)
        try:
            vit_ckpt = load_from_kaggle(vit_handle, filename=vit_file)
            if os.path.isfile(vit_ckpt):
                load_checkpoint_to_model(inst.vit_model, vit_ckpt, device=inst.device)
        except Exception as e:
            logger.warning("Could not load ViT checkpoint: %s", e)

        deeppix_entry = get_model_entry(cfg, 'deeppixbis')
        if deeppix_entry:
            dp_handle = deeppix_entry.get('handle')
            dp_file = deeppix_entry.get('filename')
            try:
                inst.deepix_model = DeepPixBiS().to(inst.device)
                dp_ckpt = load_from_kaggle(dp_handle, filename=dp_file)
                if os.path.isfile(dp_ckpt):
                    load_checkpoint_to_model(inst.deepix_model, dp_ckpt, device=inst.device)
            except Exception as e:
                logger.warning("Could not load DeepPixBis checkpoint: %s", e)

        # labels and threshold
        inst.vit_labels = {0: 'LIVE', 1: 'SPOOF'}
        inst.vit_spoof_threshold = float(cfg.get('vit_spoof_threshold', 0.7))

        # load known database and transforms
        known_path = cfg.get('known_db_path')
        inst.known_embeddings, inst.known_names = inst._load_known_db(known_path)
        inst._init_transforms()
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
            except Exception as e:
                logger.warning("Error loading DB file: %s. Using synthetic data.", e)
                known_embeddings = [torch.randn(512), torch.randn(512)]
                known_names = ["Person1", "Person2"]
        if known_embeddings:
            embeddings_tensor = F.normalize(torch.stack(known_embeddings), p=2, dim=1).to(self.device)
        else:
            embeddings_tensor = torch.empty((0, 512)).to(self.device)
        return embeddings_tensor, known_names

    def _init_transforms(self):
        from torchvision import transforms
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.arcface_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def _get_identity(self, embedding, threshold=0.4):
        if self.known_embeddings.shape[0] == 0:
            return "NoDatabase", 0.0
        cos_sim = F.linear(embedding, self.known_embeddings)
        best_score, best_idx = torch.max(cos_sim, dim=1)
        score = best_score.item()
        if score > threshold:
            return self.known_names[best_idx.item()], score
        else:
            return "Unknown", score

    def process_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        yolo_results = self.yolo_model(frame_rgb, verbose=False, conf=0.6)
        detections = yolo_results[0].boxes
        pipeline_results = []
        for box in detections:
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
                vit_pred = vit_logits.argmax(-1).item()
                status_label = self.vit_labels.get(vit_pred)

                if spoof_prob_vit > self.vit_spoof_threshold:
                    status_label = "SPOOF"
                    color = (0, 0, 255)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, f"{status_label} (ViT: {spoof_prob_vit:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    pipeline_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "status": status_label,
                        "spoof_prob_vit": spoof_prob_vit
                    })
                    continue

                # If not spoof -> continue with identification
                status = "LIVE"
                color = (0, 255, 0)

            arcface_input = self.arcface_transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.arcface_model(arcface_input)
                identity, id_score = self._get_identity(embedding)

            result_text = f"{identity} ({id_score:.2f})"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, result_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            pipeline_results.append({
                "bbox": [x1, y1, x2, y2],
                "status": status,
                "identity": identity,
                "identity_score": id_score,
                "spoof_prob_vit": spoof_prob_vit,
            })

        return frame_bgr, pipeline_results

    def run_demo(self):
        logger.info("Opening camera... (press 'q' to exit)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera.")
            raise RuntimeError("Cannot open camera")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break
            processed_frame, results = self.process_frame(frame)
            if results:
                live_faces = [r for r in results if r['status'] == 'LIVE']
                spoof_faces = len(results) - len(live_faces)
                logger.info("Frame results: %d LIVE, %d SPOOF | Details: %s", len(live_faces), spoof_faces, results)
            cv2.imshow('Face Analysis DSS (press q to exit)', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera closed.")
