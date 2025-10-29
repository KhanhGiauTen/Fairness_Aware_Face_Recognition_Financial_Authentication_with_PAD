from pathlib import Path
import sys
import argparse
import logging # Thêm logging
from src.ultil import load_config
from src.analysis.predictor import FaceAnalysisDSS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def build_parser():
    p = argparse.ArgumentParser(description="Run FaceAnalysis DSS demo")
    p.add_argument("--config", "-c", default="config/config.yaml", help="Path to config YAML")
    return p

def main():
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        logger.info(f"Added {src_dir} to sys.path")

    parser = build_parser()
    args = parser.parse_args()

    cfg_path = repo_root / args.config
    logger.info(f"Đang tải cấu hình từ: {cfg_path}")
    cfg = load_config(cfg_path)

    logger.info("Đang khởi tạo pipeline FaceAnalysisDSS...")
    pipeline = FaceAnalysisDSS.from_config(cfg)
    logger.info("Khởi tạo pipeline thành công. Bắt đầu demo...")
    pipeline.run_demo()
    logger.info("Demo hoàn tất.")


if __name__ == "__main__":
    main()