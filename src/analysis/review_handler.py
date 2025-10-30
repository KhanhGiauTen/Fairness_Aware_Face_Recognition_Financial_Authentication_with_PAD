from pathlib import Path
import json
import shutil
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F
import tempfile
import shutil
from PIL import Image

class ReviewHandler:
    def __init__(self, config: Dict[str, Any]):
        self.review_paths = config.get('review_paths', {})
        self._init_directories()
        self.metadata_path = Path(self.review_paths.get('metadata', 'data/needs_review/metadata.json'))
        self.metadata = self._load_metadata()

    def _init_directories(self):
        """Initialize directory structure for review system"""
        for key, path in self.review_paths.items():
            # If path is a dict (e.g., retraining: {live:..., spoof:...}) create each subdir
            if isinstance(path, dict):
                for subkey, subpath in path.items():
                    Path(subpath).mkdir(parents=True, exist_ok=True)
            else:
                # If path looks like a file (has a suffix), create its parent dir
                try:
                    p = Path(path)
                    if p.suffix:  # likely a file (e.g., metadata.json)
                        p.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        p.mkdir(parents=True, exist_ok=True)
                except Exception:
                    # Fallback - try to create as directory
                    Path(path).mkdir(parents=True, exist_ok=True)
        logger.info("Review directories initialized")

    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new if not exists"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted metadata file, creating new")
                return {}
        return {}

    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_review_case(self, face_crop_rgb: np.ndarray, scores: Dict[str, float], 
                        context: Dict[str, Any], case_id: Optional[str] = None) -> str:
        """
        Save a case for review with face image and associated data
        Returns the case ID
        """
        if case_id is None:
            case_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        review_dir = Path(self.review_paths['needs_review'])
        image_path = review_dir / f"{case_id}.jpg"
        
        # Save image
        cv2.imwrite(str(image_path), cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        self.metadata[case_id] = {
            'timestamp': datetime.now().isoformat(),
            'scores': scores,
            'context': context,
            'status': 'pending'
        }
        self._save_metadata()
        
        logger.info(f"Saved review case {case_id}")
        return case_id

    def get_pending_cases(self) -> Dict[str, Dict]:
        """Get all cases pending review"""
        return {k: v for k, v in self.metadata.items() 
                if v.get('status') == 'pending'}

    def update_case_label(self, case_id: str, ground_truth: str, 
                         identity: Optional[str] = None) -> bool:
        """
        Update a case with admin-provided ground truth label
        Returns True if successful
        """
        if case_id not in self.metadata:
            logger.error(f"Case {case_id} not found")
            return False

        review_dir = Path(self.review_paths['needs_review'])
        source_image = review_dir / f"{case_id}.jpg"
        
        if not source_image.exists():
            logger.error(f"Image for case {case_id} not found")
            return False

        # Determine target directory based on ground truth
        if ground_truth.upper() == 'LIVE':
            target_dir = Path(self.review_paths['retraining']['live'])
        else:  # SPOOF
            target_dir = Path(self.review_paths['retraining']['spoof'])

        # Move image to appropriate retraining directory
        target_path = target_dir / f"{case_id}.jpg"
        shutil.move(str(source_image), str(target_path))

        # Update metadata
        self.metadata[case_id].update({
            'status': 'labeled',
            'ground_truth': ground_truth,
            'identity': identity,
            'labeled_at': datetime.now().isoformat()
        })
        self._save_metadata()

        logger.info(f"Updated case {case_id} with ground truth: {ground_truth}")
        return True

    def get_case_details(self, case_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Get the image and metadata for a specific case
        Returns (image, metadata)
        """
        if case_id not in self.metadata:
            raise KeyError(f"Case {case_id} not found")

        review_dir = Path(self.review_paths['needs_review'])
        image_path = review_dir / f"{case_id}.jpg"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image for case {case_id} not found")

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, self.metadata[case_id]

    def append_identity_to_db(self, pipeline, image_paths: list, person_name: str, db_path: Optional[str]=None, min_images: int = 3) -> bool:
        """
        Compute embedding(s) for images and append a single averaged embedding for person_name into DB.
        - pipeline: FaceAnalysisDSS instance (provides arcface_model, arcface_transform, device)
        - image_paths: list of file paths to images (RGB jpg/png)
        - person_name: name string
        - db_path: optional explicit path to db; if None, try pipeline.db_path or default
        Returns True on success.
        """
        if len(image_paths) < min_images:
            raise ValueError(f"Require at least {min_images} images to create reliable embedding.")

        # determine db path
        if db_path is None:
            if hasattr(pipeline, 'db_path') and pipeline.db_path:
                db_path = Path(pipeline.db_path)
            else:
                db_path = Path('database/known_faces_db.pt')
        else:
            db_path = Path(db_path)

        device = getattr(pipeline, 'device', torch.device('cpu'))

        emb_list = []
        for p in image_paths:
            try:
                img = Image.open(str(p)).convert('RGB')
                x = pipeline.arcface_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = pipeline.arcface_model(x)
                emb_list.append(e.squeeze().cpu())
            except Exception as e:
                logger.error(f"Error computing embedding for {p}: {e}")

        if len(emb_list) < min_images:
            raise RuntimeError(f"After processing images only {len(emb_list)} valid embeddings (min {min_images}).")

        # average and normalize
        emb_tensor = torch.stack(emb_list, dim=0)
        avg_emb = torch.mean(emb_tensor, dim=0, keepdim=True)
        avg_emb = F.normalize(avg_emb, p=2, dim=1)

        # load existing DB
        backup = db_path.with_suffix('.pt.bak')
        try:
            if db_path.exists():
                db = torch.load(str(db_path), map_location='cpu')
                embeddings = db.get('embeddings', torch.empty((0, avg_emb.size(1))))
                names = db.get('names', [])
                if not isinstance(embeddings, torch.Tensor):
                    # try to convert list
                    try:
                        embeddings = torch.stack(list(embeddings)) if isinstance(embeddings, list) and embeddings else torch.empty((0, avg_emb.size(1)))
                    except Exception:
                        embeddings = torch.empty((0, avg_emb.size(1)))
            else:
                embeddings = torch.empty((0, avg_emb.size(1)))
                names = []

            # append
            if embeddings.numel():
                new_embeddings = torch.cat([embeddings, avg_emb], dim=0)
            else:
                new_embeddings = avg_emb
            new_names = names + [person_name]

            # backup existing
            if db_path.exists():
                shutil.copy2(db_path, backup)

            # atomic write via temp file in same dir
            db_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pt', dir=str(db_path.parent))
            tmp_path = Path(tmp.name)
            tmp.close()
            torch.save({'embeddings': new_embeddings, 'names': new_names}, str(tmp_path))
            tmp_path.replace(db_path)

            # update pipeline in-memory if possible
            try:
                pipeline.known_embeddings = new_embeddings.to(device)
                pipeline.known_names = new_names
            except Exception:
                logger.warning("Could not update pipeline's in-memory DB (pipeline may not be running in this process)")

            return True
        except Exception as e:
            logger.error(f"Failed to append identity to DB: {e}")
            # try restore backup
            try:
                if backup.exists():
                    shutil.copy2(backup, db_path)
            except Exception:
                logger.exception("Failed to restore DB backup")
            raise