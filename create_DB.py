import os
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO
import logging
import kagglehub 
from pathlib import Path
import sys
try:
    from src.model.arcface import ArcFaceModel 

    from src.ultil import load_checkpoint_to_model, load_config, get_model_entry, load_model_asset 
except ImportError:
    # ... (xử lý lỗi import giữ nguyên) ...
    print("Lỗi: Không tìm thấy 'src.model.arcface' hoặc 'src.ultil'.")
    print("Hãy đảm bảo bạn chạy script này từ thư mục gốc của dự án.")
    exit()

# ... (CẤU HÌNH ĐƯỜNG DẪN giữ nguyên) ...
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_DIR = r"C:\Users\Acer\source\repos\DSS\Fairness_Aware_Face_Recognition_Financial_Authentication_with_PAD\src\test_img" 
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
OUTPUT_DB_PATH = os.path.join(PROJECT_ROOT, "known_faces_db.pt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Sử dụng thiết bị: {device}")

    try:
        cfg = load_config(CONFIG_PATH)
        logger.info(f"Đã tải config từ {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Không thể tải {CONFIG_PATH}: {e}")
        return
    try:
        logger.info("Đang tải YOLO model...")
        yolo_entry = get_model_entry(cfg, 'yolo')
        yolo_source = yolo_entry.get('source', 'kaggle').lower()
        yolo_handle = yolo_entry.get('handle')
        yolo_filename = yolo_entry.get('filename')
        
        yolo_model_path = load_model_asset(yolo_source, yolo_handle, yolo_filename)

        from ultralytics import YOLO 
        yolo_model = YOLO(str(yolo_model_path))
        logger.info(f"Tải YOLO model thành công từ {yolo_source}.")
    except Exception as e:
        logger.error(f"Lỗi khi tải YOLO: {e}")
        return

    # Tải ArcFace
    try:
        logger.info("Đang tải ArcFace model...")
        arc_entry = get_model_entry(cfg, 'arcface')
        arc_source = arc_entry.get('source', 'kaggle').lower()
        arc_handle = arc_entry.get('handle')
        arc_filename = arc_entry.get('filename')

        arcface_checkpoint_path = load_model_asset(arc_source, arc_handle, arc_filename)
              
        ARCFACE_NUM_CLASSES = arc_entry.get('num_classes', 1000) 
        ARCFACE_FEATURE_DIM = arc_entry.get('feature_dim', 512) 

        arcface_model = ArcFaceModel(
            num_classes=ARCFACE_NUM_CLASSES, 
            feature_dim=ARCFACE_FEATURE_DIM
        ) 
        
        load_checkpoint_to_model(arcface_model, arcface_checkpoint_path, device=device)
        arcface_model.eval() 
        logger.info(f"Tải ArcFace model thành công từ {arc_source}.")
    except Exception as e:
        logger.error(f"Lỗi khi tải ArcFace: {e}")
        return

    # --- 2. Định nghĩa Transform ---
    arcface_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    logger.info("Đã định nghĩa ArcFace transform.")

    # --- 3. Xử lý ảnh và tạo Embeddings ---
    all_embeddings = []
    all_names = []
    if not os.path.isdir(INPUT_IMAGE_DIR):
        logger.error(f"Thư mục ảnh đầu vào không tồn tại: {INPUT_IMAGE_DIR}"); return
    for person_name in tqdm(os.listdir(INPUT_IMAGE_DIR), desc="Processing People"):
        person_dir = os.path.join(INPUT_IMAGE_DIR, person_name)
        if not os.path.isdir(person_dir): continue
        person_embeddings = []
        image_list = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Đang xử lý {person_name} với {len(image_list)} ảnh...")
        for image_name in image_list:
            image_path = os.path.join(person_dir, image_name)
            try:
                frame_bgr = cv2.imread(image_path)
                if frame_bgr is None: logger.warning(f"Skip: {image_path}"); continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yolo_results = yolo_model(frame_rgb, verbose=False, conf=0.6)
                detections = yolo_results[0].boxes
                if len(detections) == 0: logger.warning(f"No face: {image_path}"); continue
                box = detections[0] 
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                face_crop_rgb = frame_rgb[y1:y2, x1:x2]
                if face_crop_rgb.shape[0] < 10 or face_crop_rgb.shape[1] < 10: continue
                face_pil = Image.fromarray(face_crop_rgb)
                arcface_input = arcface_transform(face_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = arcface_model(arcface_input)
                    person_embeddings.append(embedding.detach().cpu())
            except Exception as e: logger.error(f"Error processing {image_path}: {e}")
        if not person_embeddings: logger.warning(f"No embeddings for {person_name}"); continue
        avg_embedding = torch.mean(torch.stack(person_embeddings), dim=0)
        all_embeddings.append(avg_embedding.squeeze()) 
        all_names.append(person_name)
        logger.info(f"Finished {person_name} ({len(person_embeddings)} images).")

    # --- 4. Lưu Database ---
    if not all_embeddings: logger.error("No embeddings generated. DB not saved."); return
    final_embeddings_list = [emb.squeeze() for emb in all_embeddings]
    if not final_embeddings_list: logger.error("Final embedding list is empty."); return
    final_embeddings_tensor = F.normalize(torch.stack(final_embeddings_list), p=2, dim=1)
    database_to_save = {'embeddings': final_embeddings_tensor, 'names': all_names}
    torch.save(database_to_save, OUTPUT_DB_PATH)
    logger.info(f"--- SUCCESS! ---")
    logger.info(f"Database saved to: {OUTPUT_DB_PATH}")
    logger.info(f"Total identities registered: {len(all_names)}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent 
    src_dir = repo_root / "src" 
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        print(f"Added {src_dir} to sys.path")
    create_database()