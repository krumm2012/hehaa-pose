tennis_analyzer/
├── main.py                     # Main script to run the analysis
├── pose_estimator.py           # Handles yolo-pose detection and swing classification
├── ball_tracker.py             # Handles TrackNetV2 ball tracking
├── utils.py                    # Helper functions (drawing, coordinate transformations)
├── configs/
│   └── default_config.yaml     # Configuration parameters
├── models/                     # To store downloaded model weights (e.g., .pt files)
│   ├── yolov8n-pose.pt
│   └── tracknetv2_weights.pth  # (Or however TrackNetV2 weights are stored)
└── data/
    └── input_video.mp4         # Your input video
    └── output_video.mp4        # Annotated output# hehaa-pose
