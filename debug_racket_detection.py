#!/usr/bin/env python3
"""
球拍检测调试脚本
显示所有原始检测结果，不应用过滤
"""

import cv2
import yaml
import numpy as np
import coremltools as ct
from PIL import Image
import time


def debug_racket_detection(config_path, video_path, output_path, max_frames=50):
    """
    调试球拍检测 - 显示所有原始检测
    """
    print("=" * 80)
    print("🔍 球拍检测调试 - 显示所有原始检测")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    unified_config = config.get('unified_detection', {})
    model_path = unified_config.get('model_path', 'yolo26n.mlpackage')
    
    # 加载模型
    print(f"\n🚀 加载模型: {model_path}")
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 视频: {width}x{height} @ {fps} FPS")
    
    # 创建输出
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 统计
    all_detections = []
    racket_class_id = 38  # tennis racket in COCO
    
    print(f"\n⏱️  处理 {max_frames} 帧...")
    print("-" * 80)
    
    for frame_num in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (640, 640))
        pil_image = Image.fromarray(resized)
        
        # 推理
        predictions = model.predict({'image': pil_image})
        
        # 解析
        display_frame = frame.copy()
        frame_rackets = []
        
        if 'var_1441' in predictions:
            output = predictions['var_1441']
            if len(output.shape) == 3:
                detections = output[0]
            else:
                detections = output
            
            for detection in detections:
                x_center, y_center, w, h, conf, cls = detection
                cls_id = int(cls)
                
                # 只看球拍
                if cls_id == racket_class_id:
                    # 缩放到原始尺寸
                    scale_x = width / 640
                    scale_y = height / 640
                    
                    x1 = int((x_center - w / 2) * scale_x)
                    y1 = int((y_center - h / 2) * scale_y)
                    x2 = int((x_center + w / 2) * scale_x)
                    y2 = int((y_center + h / 2) * scale_y)
                    
                    box_width = x2 - x1
                    box_height = y2 - y1
                    area = box_width * box_height
                    aspect_ratio = box_height / max(box_width, 1)
                    
                    racket_info = {
                        'frame': frame_num,
                        'conf': float(conf),
                        'box': [x1, y1, x2, y2],
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'width': box_width,
                        'height': box_height
                    }
                    
                    frame_rackets.append(racket_info)
                    all_detections.append(racket_info)
                    
                    # 绘制（所有检测，无论置信度）
                    # 颜色根据置信度
                    if conf >= 0.5:
                        color = (0, 255, 0)  # 绿色
                    elif conf >= 0.3:
                        color = (0, 255, 255)  # 黄色
                    else:
                        color = (0, 0, 255)  # 红色
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 信息
                    info = f"C:{conf:.2f} A:{area} R:{aspect_ratio:.1f}"
                    cv2.putText(display_frame, info, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 帧信息
        cv2.putText(display_frame, f"Frame {frame_num}: {len(frame_rackets)} rackets",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(display_frame)
        
        if frame_rackets:
            print(f"   帧 {frame_num}: 检测到 {len(frame_rackets)} 个球拍")
            for r in frame_rackets:
                print(f"      - 置信度:{r['conf']:.3f}, 面积:{r['area']}, 宽高比:{r['aspect_ratio']:.2f}, 尺寸:{r['width']}x{r['height']}")
    
    cap.release()
    out.release()
    
    # 统计
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    
    print(f"\n总检测数: {len(all_detections)}")
    
    if all_detections:
        confs = [d['conf'] for d in all_detections]
        areas = [d['area'] for d in all_detections]
        ratios = [d['aspect_ratio'] for d in all_detections]
        
        print(f"\n置信度:")
        print(f"   范围: {min(confs):.3f} - {max(confs):.3f}")
        print(f"   平均: {np.mean(confs):.3f}")
        print(f"   中位数: {np.median(confs):.3f}")
        
        print(f"\n面积 (像素²):")
        print(f"   范围: {min(areas):.0f} - {max(areas):.0f}")
        print(f"   平均: {np.mean(areas):.0f}")
        print(f"   中位数: {np.median(areas):.0f}")
        
        print(f"\n宽高比:")
        print(f"   范围: {min(ratios):.2f} - {max(ratios):.2f}")
        print(f"   平均: {np.mean(ratios):.2f}")
        print(f"   中位数: {np.median(ratios):.2f}")
        
        # 分析过滤影响
        print("\n" + "=" * 80)
        print("🔍 过滤分析")
        print("=" * 80)
        
        # 当前过滤条件
        conf_threshold = unified_config.get('racket_confidence_threshold', 0.3)
        min_area = 1000
        max_area = (width * height) / 4
        min_ratio = 0.5
        max_ratio = 5.0
        
        print(f"\n当前过滤条件:")
        print(f"   置信度 >= {conf_threshold}")
        print(f"   面积: {min_area} - {max_area:.0f}")
        print(f"   宽高比: {min_ratio} - {max_ratio}")
        
        # 应用过滤
        filtered = [d for d in all_detections if
                   d['conf'] >= conf_threshold and
                   min_area <= d['area'] <= max_area and
                   min_ratio <= d['aspect_ratio'] <= max_ratio]
        
        print(f"\n过滤结果:")
        print(f"   原始检测: {len(all_detections)}")
        print(f"   过滤后: {len(filtered)}")
        print(f"   被过滤: {len(all_detections) - len(filtered)} ({(1 - len(filtered)/len(all_detections))*100:.1f}%)")
        
        # 分析被过滤的原因
        if len(all_detections) > len(filtered):
            print(f"\n被过滤原因:")
            for d in all_detections:
                if d not in filtered:
                    reasons = []
                    if d['conf'] < conf_threshold:
                        reasons.append(f"置信度低({d['conf']:.3f})")
                    if d['area'] < min_area:
                        reasons.append(f"面积太小({d['area']})")
                    if d['area'] > max_area:
                        reasons.append(f"面积太大({d['area']})")
                    if d['aspect_ratio'] < min_ratio or d['aspect_ratio'] > max_ratio:
                        reasons.append(f"宽高比异常({d['aspect_ratio']:.2f})")
                    
                    print(f"   帧{d['frame']}: {', '.join(reasons)}")
    else:
        print("\n⚠️  没有检测到任何球拍！")
        print("\n可能原因:")
        print("   1. 模型未能识别球拍")
        print("   2. 视频中没有球拍")
        print("   3. 模型输出格式不正确")
    
    print(f"\n✅ 输出: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    config_path = "configs/yolo26_tennis_config.yaml"
    video_path = "data/16.10.mp4"
    output_path = "data/racket_debug.mp4"
    
    debug_racket_detection(config_path, video_path, output_path, max_frames=100)
