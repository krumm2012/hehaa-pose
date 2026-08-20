"""OpenCV-only pose rendering without model runtime dependencies."""

import cv2


HEAD_KEYPOINTS = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
SKELETON = (
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "left_shoulder"),
    ("right_hip", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
)
COLORS = {
    "right_arm": (255, 140, 0),
    "left_arm": (135, 206, 235),
    "torso": (75, 0, 130),
    "legs": (50, 205, 50),
}


def _connection_color(name_a, name_b):
    names = name_a + name_b
    if "wrist" in names or "elbow" in names:
        return COLORS["right_arm"] if "right" in names else COLORS["left_arm"]
    if "hip" in names or "shoulder" in names:
        return COLORS["torso"]
    return COLORS["legs"]


def _keypoint_color(name):
    if "wrist" in name or "elbow" in name:
        return COLORS["right_arm"] if "right" in name else COLORS["left_arm"]
    if "shoulder" in name or "hip" in name:
        return COLORS["torso"]
    if "knee" in name or "ankle" in name:
        return COLORS["legs"]
    return (255, 0, 255)


def draw_pose_keypoints(frame, person_keypoints_list):
    if not person_keypoints_list:
        return frame

    keypoints = person_keypoints_list[0]
    if not any(
        point is not None and name not in HEAD_KEYPOINTS
        for name, point in keypoints.items()
    ):
        return frame

    for name_a, name_b in SKELETON:
        point_a = keypoints.get(name_a)
        point_b = keypoints.get(name_b)
        if point_a and point_b:
            cv2.line(
                frame,
                point_a,
                point_b,
                _connection_color(name_a, name_b),
                2,
            )

    for name, point in keypoints.items():
        if name in HEAD_KEYPOINTS or point is None:
            continue
        cv2.circle(frame, point, 5, _keypoint_color(name), -1)

    return frame
