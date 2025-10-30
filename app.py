import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path
import logging
import io
import pandas as pd # Thêm pandas để hiển thị bảng đẹp hơn

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Thêm thư mục src vào sys.path ---
try:
    APP_ROOT = Path(__file__).resolve().parent
    SRC_DIR = APP_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
        logger.info(f"Đã thêm {SRC_DIR} vào sys.path")
    from analysis.predictor import FaceAnalysisDSS
    from ultil import load_config
    from model.arcface import ArcFaceModel
except ImportError as e:
    st.error(f"Lỗi import: {e}. Đảm bảo bạn chạy Streamlit từ thư mục gốc và cấu trúc thư mục đúng.")
    st.stop()

# --- Hàm tải cấu hình và khởi tạo DSS (cache) ---
@st.cache_resource
def load_pipeline():
    """Tải cấu hình và khởi tạo pipeline DSS."""
    config_path = APP_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        st.error(f"Không tìm thấy file config tại: {config_path}")
        return None, None # Trả về None cho cả hai
    try:
        cfg = load_config(config_path)
        logger.info("Đang khởi tạo FaceAnalysisDSS từ config...")
        # (Bỏ comment phần login nếu cần thiết cho model private)
        # try:
        #      import kagglehub; kagglehub.login()
        # except Exception: logger.warning("Kaggle Hub login lỗi hoặc bỏ qua.")
        # try:
        #      import huggingface_hub; # huggingface_hub.login()
        # except Exception: logger.warning("HF Hub login lỗi hoặc bỏ qua.")

        # Truyền config vào khi khởi tạo để có review_handler
        pipeline = FaceAnalysisDSS.from_config(cfg)
        # Nếu pipeline không có review_handler, khởi tạo thủ công
        if not hasattr(pipeline, 'review_handler') or pipeline.review_handler is None:
            from src.analysis.review_handler import ReviewHandler
            pipeline.review_handler = ReviewHandler(cfg)
        logger.info("Khởi tạo FaceAnalysisDSS thành công.")
        pipeline.db_path = cfg.get('known_db_path', "known_faces_db.pt")
        # Lấy ngưỡng từ config để hiển thị
        pipeline.display_threshold = cfg.get('vit_spoof_threshold', 0.7) 
        return pipeline, cfg
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi khởi tạo DSS: {e}", exc_info=True)
        st.error(f"Lỗi khởi tạo Pipeline DSS: {e}")
        return None, None

# --- Khởi tạo Pipeline ---
pipeline_data = load_pipeline()
if pipeline_data and pipeline_data[0]:
    dss_pipeline, config_data = pipeline_data
    DB_PATH = dss_pipeline.db_path
    VIT_THRESHOLD_DISPLAY = dss_pipeline.display_threshold # Lấy ngưỡng để hiển thị
else:
    # Thông báo lỗi đã được hiển thị trong load_pipeline
    st.stop()

# --- Khởi tạo Session State ---
if 'registration_photos' not in st.session_state:
    st.session_state.registration_photos = []
if 'person_name' not in st.session_state:
    st.session_state.person_name = ""
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False # Trạng thái webcam

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide") 
st.title("🎭 Hệ thống DSS Phân tích Khuôn mặt")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Chế độ hoạt động")
    mode = st.radio(
        "Chọn chức năng:",
        ["Giới thiệu", "👤 Đăng ký Danh tính", "📷 Webcam Trực tiếp", "🖼️ Tải ảnh lên", "⚙️ Admin Review"],
        key="mode_selection"
    )
    st.divider()
    st.header("ℹ️ Thông tin")
    st.info(f"Ngưỡng phát hiện giả mạo (ViT): **{VIT_THRESHOLD_DISPLAY}**")
    with st.expander("Danh sách đã đăng ký"):
        if dss_pipeline and dss_pipeline.known_names:
            st.write(dss_pipeline.known_names)
        else:
            st.write("Chưa có ai trong cơ sở dữ liệu.")

# --- Nội dung chính ---

# --- Chế độ Giới thiệu ---
if mode == "Giới thiệu":
    st.header("Chào mừng!")
    st.markdown("""
    Đây là giao diện demo cho Hệ thống Hỗ trợ Quyết định (DSS) Phân tích Khuôn mặt.
    Hệ thống sử dụng kết hợp các mô hình AI để:
    
    1.  **Phát hiện khuôn mặt** (YOLO)
    2.  **Chống giả mạo (PAD)** bằng cách phân biệt ảnh thật/giả (ViT)
    3.  **Định danh khuôn mặt** nếu là ảnh thật (ArcFace)

    👈 Vui lòng chọn một chế độ hoạt động từ thanh bên trái.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200) 

# --- Chế độ Đăng ký ---
elif mode == "👤 Đăng ký Danh tính":
    st.header("👤 Đăng ký Danh tính Mới")
    st.info("Hướng dẫn: Nhập tên, chụp **ít nhất 3 ảnh** rõ mặt, đủ sáng, sau đó nhấn 'Bắt đầu Đăng ký'.")
    st.divider()

    col1_reg, col2_reg = st.columns([1, 2]) 
    with col1_reg:
        st.subheader("Bước 1: Nhập tên")
        st.session_state.person_name = st.text_input("Tên người đăng ký:", value=st.session_state.person_name, key="reg_name")

        st.subheader("Bước 2: Chụp ảnh")
        img_file_buffer = st.camera_input("Chụp ảnh (Nhìn thẳng)", key="reg_camera")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            st.session_state.registration_photos.append(pil_image)
            st.success(f"Đã thêm ảnh {len(st.session_state.registration_photos)}.")


    with col2_reg:
        st.subheader(f"Ảnh đã chụp ({len(st.session_state.registration_photos)}):")
        if st.session_state.registration_photos:
            cols_img = st.columns(5) 
            for idx, img in enumerate(st.session_state.registration_photos):
                cols_img[idx % 5].image(img, width=100, caption=f"Ảnh {idx+1}")
            
            if st.button("Xóa ảnh cuối", key="reg_delete"):
                if st.session_state.registration_photos:
                    st.session_state.registration_photos.pop()
                    st.rerun()

    st.divider()
    st.subheader("Bước 3: Hoàn tất")
    MIN_PHOTOS = 3
    can_register = bool(st.session_state.person_name.strip()) and len(st.session_state.registration_photos) >= MIN_PHOTOS

    if st.button("Bắt đầu Đăng ký", disabled=not can_register, type="primary", key="reg_submit"):
        if not st.session_state.person_name.strip(): st.warning("Vui lòng nhập tên.")
        elif len(st.session_state.registration_photos) < MIN_PHOTOS: st.warning(f"Cần ít nhất {MIN_PHOTOS} ảnh.")
        else:
            person_name = st.session_state.person_name.strip()
            photos_to_process = st.session_state.registration_photos.copy()
            st.session_state.registration_photos = [] # Reset state
            st.session_state.person_name = ""

            progress_bar = st.progress(0, text="Bắt đầu xử lý...")
            status_text = st.empty()
            person_embeddings = []
            valid_photos_count = 0

            for i, pil_img in enumerate(photos_to_process):
                status_text.text(f"Đang xử lý ảnh {i+1}/{len(photos_to_process)}...")
                try:
                    frame_rgb = np.array(pil_img)
                    yolo_results = dss_pipeline.yolo_model(frame_rgb, verbose=False, conf=0.6)
                    detections = yolo_results[0].boxes
                    if not detections: status_text.warning(f"Ảnh {i+1}: Không tìm thấy khuôn mặt."); continue
                    
                    # Tìm box lớn nhất
                    best_box = max(detections, key=lambda box: (box.xyxy[0][2]-box.xyxy[0][0])*(box.xyxy[0][3]-box.xyxy[0][1])).xyxy[0].int().tolist()
                    x1, y1, x2, y2 = best_box
                    face_crop_rgb = frame_rgb[y1:y2, x1:x2]
                    if face_crop_rgb.shape[0] < 50 or face_crop_rgb.shape[1] < 50: # Yêu cầu kích thước tối thiểu lớn hơn
                        status_text.warning(f"Ảnh {i+1}: Khuôn mặt quá nhỏ."); continue
                    
                    face_pil = Image.fromarray(face_crop_rgb)
                    arcface_input = dss_pipeline.arcface_transform(face_pil).unsqueeze(0).to(dss_pipeline.device)
                    with torch.no_grad():
                        embedding = dss_pipeline.arcface_model(arcface_input)
                        person_embeddings.append(embedding.detach().cpu())
                    valid_photos_count += 1
                except Exception as e_reg:
                    logger.error(f"Lỗi xử lý ảnh đăng ký {i+1}: {e_reg}")
                    status_text.error(f"Ảnh {i+1}: Lỗi xử lý.")
                
                # Cập nhật progress bar
                progress_bar.progress((i + 1) / len(photos_to_process), text=f"Đã xử lý ảnh {i+1}/{len(photos_to_process)}...")

            # Kết thúc xử lý ảnh
            status_text.text(f"Hoàn tất xử lý ảnh. Đã trích xuất {valid_photos_count} embeddings hợp lệ.")

            if not person_embeddings:
                st.error("Không trích xuất được đặc trưng nào. Đăng ký thất bại.")
            else:
                avg_embedding = torch.mean(torch.stack(person_embeddings), dim=0)
                
                # --- Cập nhật DB ---
                status_text.text("Đang cập nhật cơ sở dữ liệu...")
                known_embeddings_list = []
                known_names_list = []
                db_path_abs = DB_PATH
                if os.path.exists(db_path_abs):
                    try:
                        db_data = torch.load(db_path_abs, map_location='cpu')
                        if 'embeddings' in db_data and isinstance(db_data['embeddings'], torch.Tensor):
                            known_embeddings_list = list(db_data['embeddings'])
                        if 'names' in db_data: known_names_list = db_data['names']
                    except Exception as e_load_db:
                        logger.error(f"Lỗi tải DB hiện tại {db_path_abs}: {e_load_db}")
                        st.error(f"Lỗi tải DB hiện tại, không thể cập nhật."); st.stop()

                if person_name in known_names_list:
                    st.warning(f"Tên '{person_name}' đã tồn tại, sẽ ghi đè embedding.")
                    try:
                        idx = known_names_list.index(person_name)
                        known_embeddings_list[idx] = avg_embedding.squeeze()
                    except ValueError: st.error("Lỗi tìm index tên."); st.stop()
                else:
                    known_embeddings_list.append(avg_embedding.squeeze())
                    known_names_list.append(person_name)

                # --- Lưu và cập nhật pipeline ---
                if known_embeddings_list:
                    try:
                        final_embeddings_tensor = F.normalize(torch.stack(known_embeddings_list), p=2, dim=1)
                        torch.save({'embeddings': final_embeddings_tensor, 'names': known_names_list}, db_path_abs)
                        dss_pipeline.known_embeddings = final_embeddings_tensor.to(dss_pipeline.device)
                        dss_pipeline.known_names = known_names_list
                        logger.info(f"Đã cập nhật DB và pipeline cho '{person_name}'.")
                        status_text.empty() # Xóa text trạng thái
                        progress_bar.empty() # Xóa progress bar
                        st.success(f"✅ Đăng ký thành công cho '{person_name}'!")
                        st.balloons() # Hiệu ứng ăn mừng
                        # Chờ vài giây rồi rerun để người dùng đọc thông báo
                        import time; time.sleep(3); st.rerun()
                    except Exception as e_save_db:
                        logger.error(f"Lỗi lưu DB {db_path_abs}: {e_save_db}")
                        st.error("Lỗi khi lưu cơ sở dữ liệu.")
                else:
                    st.error("Danh sách embedding rỗng, không thể lưu.")
            
            # Nếu có lỗi mà chưa rerun, cho phép người dùng thử lại
            st.button("Thử lại Đăng ký")


# --- Chế độ Webcam ---
elif mode == "📷 Webcam Trực tiếp":
    st.header("📷 Phân tích Webcam Trực tiếp")
    st.info("Nhấn 'Bắt đầu' để mở webcam và 'Dừng lại' để đóng.")

    col1_cam, col2_cam = st.columns(2)
    with col1_cam:
        if st.button("Bắt đầu Webcam", key="start_cam"):
            st.session_state.webcam_running = True
    with col2_cam:
        if st.button("Dừng Webcam", key="stop_cam"):
            st.session_state.webcam_running = False
            # Có thể cần rerun để đóng camera ngay lập tức
            st.rerun() 

    # Placeholder cho video và kết quả
    FRAME_WINDOW = st.image([])
    status_placeholder = st.empty() # Để hiển thị trạng thái Live/Spoof count

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Lỗi: Không thể mở webcam.")
            st.session_state.webcam_running = False # Tắt trạng thái nếu không mở được
        else:
            status_placeholder.info("Đang chạy webcam...")
            while st.session_state.webcam_running: # Vòng lặp chạy khi state là True
                ret, frame = cap.read()
                if not ret:
                    st.error("Lỗi: Mất kết nối webcam.")
                    st.session_state.webcam_running = False # Dừng nếu mất kết nối
                    break
                
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame_bgr, results = dss_pipeline.process_frame(frame)
                    processed_frame_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(processed_frame_rgb)
                    
                    # Xử lý auto-save cho các case cần review
                    for result in results:
                        # Lấy khuôn mặt từ bounding box
                        if 'bbox' in result:
                            x1, y1, x2, y2 = result['bbox']
                            face_crop_rgb = frame_rgb[y1:y2, x1:x2]
                            
                            # Kiểm tra các điều kiện để auto-save
                            if 'vit_uncertain' in result.get('reasons', []) or \
                               'model_disagreement' in result.get('reasons', []):
                                try:
                                    # Lưu case để review
                                    dss_pipeline.review_handler.save_review_case(
                                        face_crop_rgb,
                                        scores=result.get('scores', {}),
                                        context={
                                            'camera_quality': result.get('camera_quality', 'unknown'),
                                            'decision': result.get('status', 'unknown'),
                                            'reasons': result.get('reasons', [])
                                        }
                                    )
                                except Exception as e:
                                    logger.error(f"Lỗi khi auto-save case: {e}")

                    # Cập nhật trạng thái
                    live_count = len([r for r in results if r['status'] == 'LIVE'])
                    spoof_count = len([r for r in results if r['status'] == 'SPOOF'])
                    error_count = len(results) - live_count - spoof_count
                    status_placeholder.markdown(f"**Kết quả:** `{live_count}` LIVE | `{spoof_count}` SPOOF | `{error_count}` ERROR")
                
                except Exception as e_cam_loop:
                     logger.error(f"Lỗi trong vòng lặp webcam: {e_cam_loop}")
                     status_placeholder.error("Có lỗi xảy ra khi xử lý frame.")
                     # Có thể thêm logic dừng nếu lỗi liên tục

            cap.release()
            status_placeholder.info("Đã dừng webcam.")
            # Clear ảnh cuối cùng (tùy chọn)
            # FRAME_WINDOW.empty() 

# --- Chế độ Tải ảnh lên ---
elif mode == "🖼️ Tải ảnh lên":
    st.header("🖼️ Phân tích Ảnh Tải lên")
    
    uploaded_file = st.file_uploader("Chọn một file ảnh (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"], key="upload_img")
    
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            frame_rgb = np.array(pil_image) 
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            col1_up, col2_up = st.columns(2)
            with col1_up:
                st.subheader("Ảnh gốc")
                st.image(pil_image, use_column_width=True)
            
            with col2_up:
                st.subheader("Kết quả phân tích")
                with st.spinner('⏳ Đang xử lý...'):
                    processed_frame_bgr, results = dss_pipeline.process_frame(frame_bgr.copy())
                processed_frame_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB)
                st.image(processed_frame_rgb, use_column_width=True)

            st.divider()
            st.subheader("📄 Chi tiết kết quả:")
            if results:
                # Chuyển đổi kết quả thành DataFrame để hiển thị đẹp hơn
                display_data = []
                for i, res in enumerate(results):
                     data_row = {
                         "Khuôn mặt": i + 1,
                         "Trạng thái": res.get("status", "N/A"),
                         "Định danh": res.get("identity", "N/A"),
                         "Điểm DĐ": f'{res.get("identity_score", 0.0):.3f}',
                         "Xác suất Giả mạo (ViT)": f'{res.get("spoof_prob_vit", -1.0):.3f}',
                         # Bỏ comment nếu dùng DeepPixBis
                         # "Xác suất Giả mạo (DP)": f'{res.get("spoof_prob_dp", -1.0):.3f}', 
                         "Bounding Box": str(res.get("bbox", [])),
                     }
                     display_data.append(data_row)
                
                df = pd.DataFrame(display_data)
                st.dataframe(df, use_container_width=True) # Hiển thị bảng
            else:
                st.info("ℹ️ Không phát hiện được khuôn mặt nào trong ảnh.")
        except Exception as e_upload:
            logger.error(f"Lỗi khi xử lý ảnh tải lên: {e_upload}", exc_info=True)
            st.error(f"Có lỗi xảy ra: {e_upload}")

# --- Chế độ Admin Review ---
elif mode == "⚙️ Admin Review":
    st.header("⚙️ Quản lý Review Cases")
    
    # Lấy danh sách các case cần review
    try:
        pending_cases = dss_pipeline.review_handler.get_pending_cases()
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách case: {str(e)}")
        st.stop()
    
    if not pending_cases:
        st.info("✨ Không có case nào cần review.")
        st.stop()
    
    # Hiển thị tổng quan
    st.info(f"📋 Tổng số case cần review: {len(pending_cases)}")
    
    # Chọn case để review
    selected_case = st.selectbox(
        "Chọn case để review:",
        list(pending_cases.keys()),
        format_func=lambda x: f"Case {x[:8]}... ({pending_cases[x]['timestamp']})"
    )
    
    if selected_case:
        case_data = pending_cases[selected_case]
        
        # Hiển thị thông tin case
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Thông tin Chi tiết")
            st.write(f"📅 Thời gian: {case_data['timestamp']}")
            st.write(f"📷 Camera Quality: {case_data.get('context', {}).get('camera_quality', 'N/A')}")
            
            # Hiển thị các điểm số
            if 'scores' in case_data:
                scores = case_data['scores']
                st.write("📊 Scores:")
                spoof_vit = scores.get('spoof_prob_vit', 'N/A')
                spoof_vit_str = f"{float(spoof_vit):.3f}" if isinstance(spoof_vit, (float, int)) else str(spoof_vit)
                st.write(f"- ViT Spoof: {spoof_vit_str}")
                if 'spoof_prob_dp' in scores:
                    spoof_dp = scores.get('spoof_prob_dp', 'N/A')
                    spoof_dp_str = f"{float(spoof_dp):.3f}" if isinstance(spoof_dp, (float, int)) else str(spoof_dp)
                    st.write(f"- DeepPixBis: {spoof_dp_str}")
                identity_score = scores.get('identity_score', 'N/A')
                identity_score_str = f"{float(identity_score):.3f}" if isinstance(identity_score, (float, int)) else str(identity_score)
                st.write(f"- Identity Score: {identity_score_str}")
            
            # Hiển thị lý do review
            if 'context' in case_data and 'reasons' in case_data['context']:
                st.write("❓ Lý do review:")
                for reason in case_data['context']['reasons']:
                    st.write(f"- {reason}")
        
        with col2:
            # Hiển thị ảnh
            try:
                image_path = Path(dss_pipeline.review_handler.review_paths['needs_review']) / f"{selected_case}.jpg"
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                caption="Ảnh cần review",
                                use_column_width=True)
                    else:
                        st.error("Không thể đọc ảnh")
                else:
                    st.error("Không tìm thấy file ảnh")
            except Exception as e:
                st.error(f"Lỗi khi hiển thị ảnh: {str(e)}")
        
        # Form gán nhãn
        with st.form("review_form"):
            st.subheader("Gán nhãn Ground Truth")
            
            # Chọn nhãn
            ground_truth = st.radio(
                "Phân loại:",
                ["LIVE", "SPOOF"],
                key=f"label_{selected_case}"
            )
            
            # Thêm tên nếu là LIVE
            person_name = None
            if ground_truth == "LIVE":
                add_to_db = st.checkbox("Thêm vào Database?")
                if add_to_db:
                    person_name = st.text_input("Tên người dùng:")
            
            # Nút submit
            submitted = st.form_submit_button("Xác nhận & Lưu")
            
            if submitted:
                try:
                    # Cập nhật nhãn
                    success = dss_pipeline.review_handler.update_case_label(
                        selected_case,
                        ground_truth,
                        identity=person_name
                    )
                    
                    if success:
                        # Nếu là LIVE và có tên, thêm vào DB
                        if ground_truth == "LIVE" and person_name:
                            try:
                                db_success = dss_pipeline.review_handler.append_identity_to_db(
                                    dss_pipeline,
                                    [image_path],
                                    person_name,
                                    min_images=1  # Cho phép 1 ảnh vì đã được review
                                )
                                if db_success:
                                    st.success(f"✅ Đã thêm {person_name} vào database!")
                                else:
                                    st.error("Lỗi khi thêm vào database")
                            except Exception as e:
                                st.error(f"Lỗi khi thêm vào DB: {str(e)}")
                        
                        st.success("✅ Đã cập nhật case thành công!")
                        import time; time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error("Lỗi khi cập nhật case")
                except Exception as e:
                    st.error(f"Lỗi trong quá trình review: {str(e)}")
                    logger.error(f"Review error: {e}", exc_info=True)

# --- Hiển thị khi chưa chọn chế độ ---
elif mode == "---":
    st.info("👈 Vui lòng chọn một chế độ từ thanh bên trái để bắt đầu.")

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption("© 2025 - Demo DSS")