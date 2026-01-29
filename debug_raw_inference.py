import cv2
import coremltools as ct
from PIL import Image
import numpy as np

def debug_raw_output(video_path, model_path):
    print("Loading model...")
    model = ct.models.MLModel(model_path)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    cap.release()
    
    h, w = frame.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Preprocess exactly as in the detector
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (640, 640))
    pil_image = Image.fromarray(resized)
    
    print("Running inference...")
    predictions = model.predict({'image': pil_image})
    
    if 'var_1441' in predictions:
        output = predictions['var_1441']
        print(f"Output shape: {output.shape}")
        
        # Look at the first 10 detections
        detections = output[0]
        
        valid_detections = 0
        for i in range(len(detections)):
            det = detections[i]
            x, y, width, height, conf, cls = det
            if conf > 0.01: # Small threshold to see something
                print(f"Det {i}: [{x:.4f}, {y:.4f}, {width:.4f}, {height:.4f}], Conf: {conf:.4f}, Cls: {cls}")
                valid_detections += 1
                if valid_detections > 20:
                    break
    else:
        print("Output var_1441 not found")

if __name__ == "__main__":
    debug_raw_output("data/16.10.mp4", "yolo26n.mlpackage")
