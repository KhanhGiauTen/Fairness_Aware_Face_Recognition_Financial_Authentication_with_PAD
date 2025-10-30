import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def assess_camera_quality(frame):
    """
    Assess camera quality based on various metrics:
    - Resolution
    - Brightness
    - Sharpness/Blur
    
    Returns:
    - str: 'low', 'mid', or 'high' based on camera quality
    """
    try:
        # Check resolution
        height, width = frame.shape[:2]
        resolution_score = 0
        if width >= 1920 or height >= 1080:  # Full HD or higher
            resolution_score = 2
        elif width >= 1280 or height >= 720:  # HD
            resolution_score = 1
        
        # Check brightness
        brightness_score = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 100:  # Good lighting
            brightness_score = 2
        elif mean_brightness > 50:  # Moderate lighting
            brightness_score = 1
        
        # Check sharpness (using Laplacian variance)
        sharpness_score = 0
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 1000:  # Very sharp image
            sharpness_score = 2
        elif laplacian_var > 500:  # Moderately sharp
            sharpness_score = 1
        
        # Calculate overall quality score
        total_score = resolution_score + brightness_score + sharpness_score
        
        # Determine quality level based on total score and minimum criteria
        if total_score >= 5 and min(resolution_score, brightness_score) >= 1:
            quality = 'high'
        elif total_score >= 2 and resolution_score >= 1:
            quality = 'mid'
        else:
            quality = 'low'
            
        logger.debug(f"Camera quality assessment - Resolution: {resolution_score}, "
                   f"Brightness: {brightness_score}, Sharpness: {sharpness_score}, "
                   f"Quality: {quality}")
        
        return quality
        
    except Exception as e:
        logger.error(f"Error assessing camera quality: {e}")
        return 'default'  # Return default on error