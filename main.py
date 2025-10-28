from pathlib import Path
import sys
import argparse
from src.ultil import load_config
from src.analysis.predictor import FaceAnalysisDSS

def build_parser():
    p = argparse.ArgumentParser(description="Run FaceAnalysis DSS demo")
    p.add_argument("--config", "-c", default="config/config.yaml", help="Path to config YAML")
    return p


def main():
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    parser = build_parser()
    args = parser.parse_args()

    cfg_path = repo_root / args.config
    cfg = load_config(cfg_path)

    try:
        import kagglehub
        try:
            kagglehub.login()
            print("KaggleHub: login successful")
        except Exception:
            print("KaggleHub: login skipped or failed (ensure credentials if needed)")
    except Exception:
        print("KaggleHub: not available")
        pass

    pipeline = FaceAnalysisDSS.from_config(cfg)
    pipeline.run_demo()


if __name__ == "__main__":
    main()
