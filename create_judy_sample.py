#!/usr/bin/env python3
# create_judy_sample.py
"""
创建示例Judy头像图片的脚本
如果没有真实的Judy头像，这个脚本会创建一个卡通风格的示例图像
"""

import cv2
import numpy as np
import os

def create_judy_avatar():
    """创建一个卡通风格的Judy头像示例"""
    
    # 创建一个300x300的画布，带透明通道
    size = 300
    img = np.zeros((size, size, 4), dtype=np.uint8)
    
    center = (size // 2, size // 2)
    
    # 脸部轮廓（淡粉色）
    face_radius = size // 3
    cv2.circle(img, center, face_radius, (220, 180, 255, 255), -1)
    
    # 头发（棕色）
    hair_points = np.array([
        [center[0] - face_radius + 20, center[1] - face_radius + 40],
        [center[0] - face_radius - 10, center[1] - face_radius + 10],
        [center[0] - face_radius - 20, center[1] - 20],
        [center[0] - face_radius, center[1] + 20],
        [center[0] - face_radius + 30, center[1] + face_radius - 20],
        [center[0] + face_radius - 30, center[1] + face_radius - 20],
        [center[0] + face_radius, center[1] + 20],
        [center[0] + face_radius + 20, center[1] - 20],
        [center[0] + face_radius + 10, center[1] - face_radius + 10],
        [center[0] + face_radius - 20, center[1] - face_radius + 40],
    ], np.int32)
    
    cv2.fillPoly(img, [hair_points], (101, 67, 33, 255))  # 棕色头发
    
    # 眼睛
    eye_offset_x = face_radius // 3
    eye_offset_y = face_radius // 4
    eye_radius = 8
    
    # 左眼
    left_eye = (center[0] - eye_offset_x, center[1] - eye_offset_y)
    cv2.circle(img, left_eye, eye_radius, (255, 255, 255, 255), -1)  # 白色眼球
    cv2.circle(img, left_eye, eye_radius - 3, (0, 100, 200, 255), -1)  # 蓝色瞳孔
    cv2.circle(img, (left_eye[0] - 2, left_eye[1] - 2), 2, (255, 255, 255, 255), -1)  # 高光
    
    # 右眼
    right_eye = (center[0] + eye_offset_x, center[1] - eye_offset_y)
    cv2.circle(img, right_eye, eye_radius, (255, 255, 255, 255), -1)  # 白色眼球
    cv2.circle(img, right_eye, eye_radius - 3, (0, 100, 200, 255), -1)  # 蓝色瞳孔
    cv2.circle(img, (right_eye[0] - 2, right_eye[1] - 2), 2, (255, 255, 255, 255), -1)  # 高光
    
    # 眉毛
    eyebrow_offset = 15
    cv2.ellipse(img, (left_eye[0], left_eye[1] - eyebrow_offset), (12, 4), 0, 0, 180, (101, 67, 33, 255), 3)
    cv2.ellipse(img, (right_eye[0], right_eye[1] - eyebrow_offset), (12, 4), 0, 0, 180, (101, 67, 33, 255), 3)
    
    # 鼻子（简单的点）
    nose_pos = (center[0], center[1] + 10)
    cv2.circle(img, nose_pos, 2, (200, 150, 180, 255), -1)
    
    # 嘴巴（笑容）
    mouth_center = (center[0], center[1] + face_radius // 3)
    mouth_width = face_radius // 3
    mouth_height = face_radius // 6
    cv2.ellipse(img, mouth_center, (mouth_width, mouth_height), 0, 0, 180, (200, 50, 50, 255), 3)
    
    # 腮红
    blush_offset_x = face_radius // 2
    blush_offset_y = face_radius // 6
    left_blush = (center[0] - blush_offset_x, center[1] + blush_offset_y)
    right_blush = (center[0] + blush_offset_x, center[1] + blush_offset_y)
    
    cv2.circle(img, left_blush, 12, (255, 150, 150, 100), -1)
    cv2.circle(img, right_blush, 12, (255, 150, 150, 100), -1)
    
    # 耳环（可选装饰）
    earring_left = (center[0] - face_radius + 10, center[1])
    earring_right = (center[0] + face_radius - 10, center[1])
    cv2.circle(img, earring_left, 4, (255, 215, 0, 255), -1)  # 金色耳环
    cv2.circle(img, earring_right, 4, (255, 215, 0, 255), -1)
    
    return img

def main():
    """主函数"""
    print("正在创建示例Judy头像...")
    
    # 确保assets目录存在
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"创建了 {assets_dir} 目录")
    
    # 创建Judy头像
    judy_img = create_judy_avatar()
    
    # 保存图片
    output_path = os.path.join(assets_dir, "judy_head.png")
    cv2.imwrite(output_path, judy_img)
    
    print(f"示例Judy头像已保存到: {output_path}")
    print("图像尺寸:", judy_img.shape)
    print("注意: 这是一个示例头像。您可以替换为真实的Judy头像图片以获得更好的效果。")
    
    # 创建一个备选的更简单的头像
    simple_judy = create_simple_judy()
    simple_output_path = os.path.join(assets_dir, "judy_head_simple.png")
    cv2.imwrite(simple_output_path, simple_judy)
    print(f"简单版Judy头像已保存到: {simple_output_path}")

def create_simple_judy():
    """创建一个更简单的Judy头像"""
    size = 200
    img = np.ones((size, size, 4), dtype=np.uint8) * 255
    
    center = (size // 2, size // 2)
    radius = size // 3
    
    # 脸部（浅肤色）
    cv2.circle(img, center, radius, (230, 200, 180, 255), -1)
    
    # 眼睛
    eye_offset = radius // 3
    cv2.circle(img, (center[0] - eye_offset, center[1] - eye_offset//2), 5, (0, 0, 0, 255), -1)
    cv2.circle(img, (center[0] + eye_offset, center[1] - eye_offset//2), 5, (0, 0, 0, 255), -1)
    
    # 嘴巴
    cv2.ellipse(img, (center[0], center[1] + eye_offset//2), (eye_offset, eye_offset//3), 0, 0, 180, (200, 0, 0, 255), 2)
    
    # 头发
    cv2.ellipse(img, (center[0], center[1] - radius//2), (radius + 10, radius//2), 0, 180, 360, (139, 69, 19, 255), -1)
    
    return img

if __name__ == "__main__":
    main() 