# main.py
import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector # <-- IMPORT
from full_swing_analyzer import FullSwingAnalyzer # <-- IMPORT NEW ANALYZER
import os
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import logging
import requests
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tennis_analyzer')

# 字体管理函数
def manage_fonts():
    """
    管理字体 - 确保必要的中文字体在系统中可用
    如果系统中没有可用的中文字体，会尝试下载
    返回一个最优的中文字体路径
    """
    # 字体目录
    font_dir = "fonts"
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
        logger.info(f"创建字体目录: {font_dir}")
    
    # 定义常用字体及其网络资源
    font_resources = {
        "SourceHanSansSC-Regular.otf": "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf",
        "NotoSansSC-Regular.otf": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
        "wqy-microhei.ttc": "https://github.com/anthonyfok/fonts-wqy-microhei/raw/master/wqy-microhei.ttc"
    }
    
    # 优先检查系统中已有的字体
    system_font_paths = [
        "/System/Library/Fonts/PingFang.ttc",           # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",      # macOS备选
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", # Linux
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Linux备选
        "C:/Windows/Fonts/simhei.ttf",                  # Windows
        "C:/Windows/Fonts/msyh.ttc",                    # Windows备选 
        "C:/Windows/Fonts/simsun.ttc",                  # Windows备选2
    ]
    
    # 检查系统字体
    for path in system_font_paths:
        if os.path.exists(path):
            logger.info(f"找到系统中文字体: {path}")
            # 验证字体是否可用
            if verify_font(path):
                return path
            else:
                logger.warning(f"系统字体 {path} 存在但验证失败，继续查找其他字体")
    
    # 检查已下载的字体
    for font_name in font_resources.keys():
        local_path = os.path.join(font_dir, font_name)
        if os.path.exists(local_path):
            logger.info(f"找到本地下载的字体: {local_path}")
            if verify_font(local_path):
                return local_path
            else:
                logger.warning(f"本地字体 {local_path} 存在但验证失败，尝试重新下载")
                # 尝试删除并重新下载
                try:
                    os.remove(local_path)
                except:
                    pass
    
    # 如果没有找到可用字体，尝试下载
    for font_name, font_url in font_resources.items():
        local_path = os.path.join(font_dir, font_name)
        logger.info(f"尝试下载字体 {font_name}...")
        
        try:
            # 使用requests下载（更好的进度和错误处理）
            response = requests.get(font_url, stream=True)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                logger.info(f"字体 {font_name} 下载成功")
                
                # 验证下载的字体
                if verify_font(local_path):
                    return local_path
            else:
                logger.error(f"下载字体 {font_name} 失败，HTTP状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"下载字体 {font_name} 时出错: {str(e)}")
            
            # 备用下载方法
            try:
                logger.info(f"尝试使用urllib.request下载字体 {font_name}...")
                urllib.request.urlretrieve(font_url, local_path)
                logger.info(f"字体 {font_name} 使用备用方法下载成功")
                
                # 验证下载的字体
                if verify_font(local_path):
                    return local_path
            except Exception as e2:
                logger.error(f"备用下载方法也失败: {str(e2)}")
    
    # 如果所有尝试都失败，返回None
    logger.error("无法找到或下载可用的中文字体，中文文本可能无法正确显示")
    return None

def verify_font(font_path):
    """验证字体文件是否可用于渲染中文"""
    try:
        # 尝试创建字体对象
        font = ImageFont.truetype(font_path, 24)
        # 尝试用此字体渲染中文
        img = Image.new('RGB', (100, 50), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "测试中文", font=font, fill=(255, 255, 255))
        logger.info(f"字体 {font_path} 验证成功，可以渲染中文")
        return True
    except Exception as e:
        logger.error(f"字体 {font_path} 验证失败: {str(e)}")
        return False

# 下载并保存字体
def download_font(font_url, save_path):
    """下载字体文件并保存到指定路径"""
    try:
        logger.info(f"尝试下载字体: {font_url}")
        urllib.request.urlretrieve(font_url, save_path)
        logger.info(f"字体下载成功并保存到: {save_path}")
        return True
    except Exception as e:
        logger.error(f"字体下载失败: {str(e)}")
        return False

# 全局字体路径
FONT_PATH = None

# 添加绘制中文文本的函数
def draw_chinese_text(img, text, position, font_size=20, text_color=(255, 255, 255), thickness=1):
    """
    在OpenCV图像上绘制中文文本
    :param img: OpenCV图像
    :param text: 要绘制的文本
    :param position: 文本位置 (x, y)
    :param font_size: 字体大小
    :param text_color: 文本颜色 (B, G, R)
    :param thickness: 文本粗细
    :return: 绘制了文本的图像
    """
    global FONT_PATH
    
    # 如果全局字体路径未初始化，获取字体
    if FONT_PATH is None:
        FONT_PATH = manage_fonts()
    
    # 如果找不到任何可用字体，回退到OpenCV内置字体
    if FONT_PATH is None:
        logger.warning(f"未找到可用中文字体，使用OpenCV内置字体渲染文本: '{text}'")
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, text_color, thickness, cv2.LINE_AA)
        return img
    
    try:
        # 转换OpenCV图像到PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_img)
        
        # 加载字体
        font = ImageFont.truetype(FONT_PATH, font_size)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
        
        # 转换回OpenCV图像
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"中文文字渲染失败: {str(e)}")
        # 如果PIL渲染失败，使用OpenCV的putText作为后备
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, text_color, thickness, cv2.LINE_AA)
        return img

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_directory(output_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path

def main():
    logger.info("开始加载配置...")
    config = load_config()
    video_path = config['video_input_path']
    logger.info(f"配置加载完成，视频路径: {video_path}")
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"错误: 无法打开视频 {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 获取总帧数，处理潜在的 cv2.CAP_PROP_FRAME_COUNT 问题
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:  # 某些视频格式可能无法准确报告 total_frames
        logger.warning("警告: 无法获取视频总帧数，或总帧数为0。进度百分比可能不准确。")
        # 如果 fps 可用，基于较长的时长进行估计
        total_frames = fps * 3600 if fps > 0 else -1  # 默认为非常大的数字或 -1（如果 fps 也有问题）

    logger.info(f"视频信息 - 宽度: {frame_width}, 高度: {frame_height}, FPS: {fps}, 总帧数 (估计): {total_frames if total_frames > 0 else 'N/A'}")

    # 确保输出目录存在
    output_path = create_output_directory(config['video_output_path'])
    logger.info(f"创建输出视频: {output_path}")
    
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps if fps > 0 else 25,  # 如果原始fps为0，提供默认值
                          (frame_width, frame_height))

    # 初始化组件
    logger.info("初始化姿势估计模块...")
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config)
    logger.info("初始化球追踪模块...")
    ball_module = BallTracker(config.get('tracknet_model_path', None), config)
    logger.info("初始化球拍检测模块...")
    racket_module = RacketDetector(config['racket_yolo_model_path'], config)
    logger.info("初始化完整挥拍分析模块...") # <-- NEW
    swing_analyzer = FullSwingAnalyzer(config) # <-- NEW

    # 进度和时间跟踪
    frame_num = 0
    start_time = time.time()
    batch_start_time = start_time
    
    logger.info("开始处理视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("视频处理完成!")
            break

        # 显示进度
        if total_frames > 0 and frame_num % (fps if fps > 0 else 30) == 0:  # 大约每秒打印一次
            elapsed_time = time.time() - start_time
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            
            frames_processed = frame_num + 1
            progress = frames_processed / total_frames * 100 if total_frames > 0 else 0
            
            # 估计剩余时间
            if frame_num > 0:
                time_per_frame = elapsed_time / frames_processed
                frames_remaining = total_frames - frames_processed
                estimated_time_remaining = frames_remaining * time_per_frame if total_frames > 0 else "未知"
                
                logger.info(f"处理帧 {frames_processed}/{total_frames} ({progress:.1f}%) - "
                      f"批处理时间: {batch_time:.2f}秒, "
                      f"估计剩余时间: {estimated_time_remaining if isinstance(estimated_time_remaining, str) else f'{estimated_time_remaining:.1f}秒'}")
            
        elif frame_num % 100 == 0:  # 如果total_frames未知，则每100帧显示一次
            logger.info(f"处理帧 {frame_num}...")
            
        display_frame = frame.copy()  # 在副本上绘制
        frame_dimensions = (frame_height, frame_width) # 传递给分析器 <-- NEW

        # 1. 姿势估计
        person_keypoints_list = pose_module.get_keypoints(frame)
        
        # 计算并显示关键关节角度
        if person_keypoints_list and len(person_keypoints_list) > 0:
            keypoints = person_keypoints_list[0]
            
            # 计算手肘角度
            if all(keypoints.get(kp) for kp in ["right_shoulder", "right_elbow", "right_wrist"]):
                right_elbow_angle = pose_module.calculate_angle(
                    keypoints["right_shoulder"], 
                    keypoints["right_elbow"], 
                    keypoints["right_wrist"]
                )
                # 在右手肘位置显示角度
                if keypoints["right_elbow"]:
                    elbow_pos = keypoints["right_elbow"]
                    cv2.putText(display_frame, f"{right_elbow_angle:.0f}°", 
                                (elbow_pos[0] + 10, elbow_pos[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
            
            # 计算左手肘角度
            if all(keypoints.get(kp) for kp in ["left_shoulder", "left_elbow", "left_wrist"]):
                left_elbow_angle = pose_module.calculate_angle(
                    keypoints["left_shoulder"], 
                    keypoints["left_elbow"], 
                    keypoints["left_wrist"]
                )
                # 在左手肘位置显示角度
                if keypoints["left_elbow"]:
                    elbow_pos = keypoints["left_elbow"]
                    cv2.putText(display_frame, f"{left_elbow_angle:.0f}°", 
                                (elbow_pos[0] + 10, elbow_pos[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (135, 206, 235), 2)
            
            # 计算躯干角度（肩部连线与垂直线的角度）
            if all(keypoints.get(kp) for kp in ["right_shoulder", "left_shoulder"]):
                rs = np.array(keypoints["right_shoulder"])
                ls = np.array(keypoints["left_shoulder"])
                shoulder_vector = rs - ls
                vertical_vector = np.array([0, 1])
                dot_product = np.dot(shoulder_vector, vertical_vector)
                norm_product = np.linalg.norm(shoulder_vector) * np.linalg.norm(vertical_vector)
                if norm_product > 0:
                    torso_angle = np.degrees(np.arccos(dot_product / norm_product))
                    # 在肩部中心位置显示躯干角度
                    shoulder_center = ((rs[0] + ls[0]) // 2, (rs[1] + ls[1]) // 2)
                    cv2.putText(display_frame, f"躯干: {torso_angle:.0f}°", 
                                (shoulder_center[0], shoulder_center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 0, 130), 2)
        
        # 2. 绘制姿势关键点和骨架
        display_frame = pose_module.draw_keypoints(display_frame, person_keypoints_list)

        # 3. 挥拍分类
        swing_type = "No Swing"  # 默认值
        if person_keypoints_list:  # 如果检测到人
            # a>使用第一个检测到的人作为球员
            swing_type = pose_module.classify_swing(person_keypoints_list)

        # 4. 球拍检测和关联 <-- NEW
        raw_racket_detections = racket_module.detect_rackets(frame)
        associated_rackets_info = racket_module.associate_racket_to_player(raw_racket_detections, person_keypoints_list, frame_num)

        # 5. 球检测 (使用 TrackNetV2 模型或模拟)
        raw_ball_detections = ball_module.predict_ball(frame)
        
        # 可选：从检测到的坐标生成热图
        # heatmap = ball_module.generate_heatmap_from_coords(raw_ball_detections, frame.shape)
        # 可选：处理热图找到球的位置
        # raw_ball_detections = ball_module.process_heatmap(heatmap, frame.shape)

        # 6. 高级球处理：过滤静态球并追踪主要移动球
        active_ball_this_frame_list = ball_module.advanced_ball_processing(raw_ball_detections, frame_num)

        # 7. 确定球拍状态基于球的位置 <-- NEW
        player_id_for_state = 0  # 使用第一个检测到的人作为主要球员
        # 判断球拍状态（是否接近球、碰撞、跟随）
        if player_id_for_state in associated_rackets_info and active_ball_this_frame_list:
            racket_module.determine_racket_state(player_id_for_state, active_ball_this_frame_list[0], frame_height)

        # 8. 执行完整的挥拍分析 <-- NEW
        full_analysis_metrics = {}
        if person_keypoints_list:  # 只有在检测到人时才进行分析
            full_analysis_metrics = swing_analyzer.analyze_swing_components(
                person_keypoints_list,
                associated_rackets_info,  # 传递关联的球拍信息
                active_ball_this_frame_list[0] if active_ball_this_frame_list else None,
                frame_dimensions
            )

        # 每30帧应用一次轨迹平滑和离群值移除
        if frame_num % 30 == 0:
            ball_module.remove_outliers(threshold=3.0)
            ball_module.interpolate_trajectory()

        # 9. 绘制可视化
        # 绘制关联的球拍和状态
        display_frame = racket_module.draw_associated_rackets(display_frame)

        # 绘制活动的球（如果有）
        display_frame = ball_module.draw_ball(display_frame, active_ball_this_frame_list)
        # 绘制基于内部历史的轨迹
        display_frame = ball_module.draw_trajectory(display_frame)
        # 绘制确认的静态球
        display_frame = ball_module.draw_static_balls(display_frame)

        # 10. 添加文本信息
        # 创建半透明信息面板 - 扩大面板高度以容纳更多信息
        info_panel = display_frame.copy()
        panel_height = 450  # 增大高度以显示更多分析数据
        panel_width = 350  # 稍微增加宽度
        cv2.rectangle(info_panel, (30, 20), (30 + panel_width, panel_height), (0, 0, 0), -1)
        alpha = 0.7  # 透明度
        display_frame = cv2.addWeighted(info_panel, alpha, display_frame, 1 - alpha, 0)
        
        # 文本显示设置
        text_x_offset = 50
        text_y_offset = 50
        line_height = 25
        
        # 定义文本颜色
        white_color = (255, 255, 255)  # 白色文本
        header_color = (0, 255, 255)   # 青色标题
        score_color = (0, 255, 0)      # 绿色分数
        detail_color = (220, 220, 220) # 浅灰色详情
        
        # 根据phase_estimation确定当前挥拍阶段
        current_phase = full_analysis_metrics.get("phase_estimation", "Unknown")
        
        # 显示当前挥拍阶段类型 - 大标题
        if "Impact" in current_phase:
            main_title = "【击球阶段】"
        elif "Backswing" in current_phase or "Preparation" in current_phase:
            main_title = "【引拍准备】"
        elif "Forward" in current_phase or "Approaching" in current_phase:
            main_title = "【发力启动】"
        elif "Follow" in current_phase:
            main_title = "【随挥】"
        else:
            main_title = "【挥拍姿势】"
            
        # 使用中文绘制函数替代cv2.putText
        display_frame = draw_chinese_text(display_frame, main_title, (text_x_offset + 60, text_y_offset), 
                            font_size=30, text_color=header_color, thickness=2)
        current_y = text_y_offset + line_height + 10
        
        # 显示基本挥拍类型得分 - 使用笑脸+分数格式
        prep_score = 0
        swing_score = 0
        footwork_score = 0
        power_score = 0
        
        # 计算各部分得分 (简单示例，可以根据具体指标进行加权)
        # 引拍准备得分计算
        prep_metrics = full_analysis_metrics.get("preparation", {})
        if "nondom_arm_usage" in prep_metrics and "Extended" in prep_metrics.get("nondom_arm_usage", ""):
            prep_score += 35
        if "racket_takeback" in prep_metrics and "Yes" in prep_metrics.get("racket_takeback", ""):
            prep_score += 35
        if "shoulder_turn_degrees" in prep_metrics:
            try:
                angle = float(prep_metrics.get("shoulder_turn_degrees", "0").strip("°"))
                if angle > 30:
                    prep_score += 30
            except ValueError:
                pass
        prep_score = min(100, prep_score)
        
        # 挥拍动作得分计算
        swing_metrics = full_analysis_metrics.get("swing_motion", {})
        if "inferred_contact_point" in swing_metrics and "In Front" in swing_metrics.get("inferred_contact_point", ""):
            swing_score += 50
        elif "inferred_contact_point" in swing_metrics and "Side" in swing_metrics.get("inferred_contact_point", ""):
            swing_score += 35
        if "arm_extension_at_impact" in swing_metrics and "Extended" in swing_metrics.get("arm_extension_at_impact", ""):
            swing_score += 50
        swing_score = min(100, swing_score)
        
        # 挥拍击球得分计算
        footwork_metrics = full_analysis_metrics.get("footwork", {})
        if "stance_type_guess" in footwork_metrics and "Open" in footwork_metrics.get("stance_type_guess", ""):
            footwork_score += 50
        if "left_knee_angle_deg" in footwork_metrics or "right_knee_angle_deg" in footwork_metrics:
            try:
                left_angle = float(footwork_metrics.get("left_knee_angle_deg", "180").strip("°"))
                right_angle = float(footwork_metrics.get("right_knee_angle_deg", "180").strip("°"))
                avg_angle = (left_angle + right_angle) / 2
                if avg_angle < 150:  # 膝盖弯曲较好
                    footwork_score += 50
                elif avg_angle < 165:  # 膝盖有一定弯曲
                    footwork_score += 30
            except ValueError:
                footwork_score += 25  # 默认给一些分数
        footwork_score = min(96, max(70, footwork_score))  # 设置最低分数
        
        # 发力得分计算
        power_metrics = full_analysis_metrics.get("power_indicators", {})
        if "leg_bend_indicator" in power_metrics and "Significant" in power_metrics.get("leg_bend_indicator", ""):
            power_score += 50
        if "body_coil_indicator" in power_metrics and "Coiled" in power_metrics.get("body_coil_indicator", ""):
            power_score += 50
        if "hip_shoulder_separation_deg" in power_metrics:
            try:
                sep_angle = float(power_metrics.get("hip_shoulder_separation_deg", "0").strip("°"))
                if sep_angle > 15:
                    power_score += 25
            except ValueError:
                pass
        power_score = min(97, power_score)  # 限制最高分
        
        # 根据主要活动阶段显示相关分析
        if "引拍准备" in main_title:
            # 显示引拍准备得分
            display_frame = draw_chinese_text(display_frame, f"😊{prep_score}分", (text_x_offset + 100, current_y + 30), 
                            font_size=36, text_color=score_color, thickness=2)
            current_y += 60
            
            # 显示引拍分析详情
            details = []
            if "racket_takeback" in prep_metrics:
                details.append(f"【引拍转体】")
            if "nondom_arm_usage" in prep_metrics and "Extended" in prep_metrics.get("nondom_arm_usage", ""):
                details.append(f"【非执拍手辅助引拍】")
            details_text = "、".join(details) + " 表现完美!" if details else ""
            
            display_frame = draw_chinese_text(display_frame, details_text, (text_x_offset, current_y), 
                            font_size=20, text_color=white_color, thickness=1)
            current_y += line_height + 5
            
            # 添加具体技术指标值
            if "shoulder_turn_degrees" in prep_metrics:
                shoulder_text = f"肩部转动角度: {prep_metrics.get('shoulder_turn_degrees', '0°')}"
                display_frame = draw_chinese_text(display_frame, shoulder_text, (text_x_offset, current_y), 
                               font_size=20, text_color=detail_color, thickness=1)
                current_y += line_height
                
            if "nondom_arm_usage" in prep_metrics:
                arm_text = f"非执拍手伸展: {prep_metrics.get('nondom_arm_usage', 'None')}"
                display_frame = draw_chinese_text(display_frame, arm_text, (text_x_offset, current_y), 
                               font_size=20, text_color=detail_color, thickness=1)
                current_y += line_height
        
        elif "发力启动" in main_title:
            # 显示发力启动得分
            display_frame = draw_chinese_text(display_frame, f"😊{power_score}分", (text_x_offset + 100, current_y + 30), 
                            font_size=36, text_color=score_color, thickness=2)
            current_y += 60
            
            # 显示发力分析详情
            details = []
            if "leg_bend_indicator" in power_metrics:
                details.append(f"【手肘打开】")
            if "body_coil_indicator" in power_metrics and "Coiled" in power_metrics.get("body_coil_indicator", ""):
                details.append(f"【站姿】")
            details.append(f"【降拍头】")
            details_text = "、".join(details) + " 表现完美!" if details else ""
            
            display_frame = draw_chinese_text(display_frame, details_text, (text_x_offset, current_y), 
                            font_size=20, text_color=white_color, thickness=1)
            current_y += line_height
            
            if "leg_bend_indicator" in power_metrics:
                leg_text = f"【下蹲】表现优秀!"
                display_frame = draw_chinese_text(display_frame, leg_text, (text_x_offset, current_y), 
                                font_size=20, text_color=white_color, thickness=1)
            current_y += line_height + 5
            
            # 添加具体技术指标值
            if "hip_shoulder_separation_deg" in power_metrics:
                separation_text = f"髋肩分离角度: {power_metrics.get('hip_shoulder_separation_deg', '0°')}"
                display_frame = draw_chinese_text(display_frame, separation_text, (text_x_offset, current_y), 
                               font_size=20, text_color=detail_color, thickness=1)
                current_y += line_height
            
            # 显示膝盖角度
            left_knee = footwork_metrics.get("left_knee_angle_deg", "N/A")
            right_knee = footwork_metrics.get("right_knee_angle_deg", "N/A")
            knee_text = f"膝盖角度: 左 {left_knee}, 右 {right_knee}"
            display_frame = draw_chinese_text(display_frame, knee_text, (text_x_offset, current_y), 
                           font_size=20, text_color=detail_color, thickness=1)
            current_y += line_height
            
        elif "挥拍击球" in main_title or "击球阶段" in main_title:
            # 显示挥拍击球得分
            display_frame = draw_chinese_text(display_frame, f"😊{swing_score}分", (text_x_offset + 100, current_y + 30), 
                            font_size=36, text_color=score_color, thickness=2)
            current_y += 60
            
            # 显示击球分析详情
            details = []
            if "inferred_contact_point" in swing_metrics:
                contact_point = swing_metrics.get("inferred_contact_point", "")
                if "In Front" in contact_point:
                    details.append("【以核心转力量击球】")
                elif "Side" in contact_point:
                    details.append("【以重心前移力量击球】")
            details.append("【动作流畅度】")
            details_text = "、".join(details) + " 表现完美!" if details else ""
            
            display_frame = draw_chinese_text(display_frame, details_text, (text_x_offset, current_y), 
                            font_size=20, text_color=white_color, thickness=1)
            current_y += line_height
            
            # 额外详情
            if "contact_height_ratio_frame" in swing_metrics:
                display_frame = draw_chinese_text(display_frame, "【身前击球】表现优秀!", (text_x_offset, current_y), 
                                font_size=20, text_color=white_color, thickness=1)
                current_y += line_height
            
            display_frame = draw_chinese_text(display_frame, "【从低到高挥拍】需要改进", (text_x_offset, current_y), 
                            font_size=20, text_color=white_color, thickness=1)
            current_y += line_height + 5
            
            # 添加具体技术指标值
            if "inferred_contact_point" in swing_metrics:
                contact_text = f"击球点位置: {swing_metrics.get('inferred_contact_point', 'Unknown')}"
                display_frame = draw_chinese_text(display_frame, contact_text, (text_x_offset, current_y), 
                               font_size=20, text_color=detail_color, thickness=1)
                current_y += line_height
            
            if "arm_extension_at_impact" in swing_metrics:
                extension_text = f"挥拍手臂伸展度: {swing_metrics.get('arm_extension_at_impact', 'Unknown')}"
                display_frame = draw_chinese_text(display_frame, extension_text, (text_x_offset, current_y), 
                               font_size=20, text_color=detail_color, thickness=1)
                current_y += line_height
                
            if "contact_height_ratio_frame" in swing_metrics:
                height_text = f"击球高度比率: {swing_metrics.get('contact_height_ratio_frame', '0%')}"
                display_frame = draw_chinese_text(display_frame, height_text, (text_x_offset, current_y), 
                               font_size=20, text_color=detail_color, thickness=1)
                current_y += line_height
            
            # 获取建议按钮
            display_frame = draw_chinese_text(display_frame, "获取建议>>", (text_x_offset + 200, current_y), 
                            font_size=20, text_color=(0, 200, 200), thickness=1)
            current_y += line_height + 10
            
        # 分腿垫步分析
        display_frame = draw_chinese_text(display_frame, "【分腿垫步】", (text_x_offset + 60, current_y), 
                    font_size=30, text_color=header_color, thickness=2)
        current_y += line_height + 30
        
        display_frame = draw_chinese_text(display_frame, f"😊{footwork_score}分", (text_x_offset + 100, current_y), 
                    font_size=36, text_color=score_color, thickness=2)
        current_y += line_height + 15
        
        display_frame = draw_chinese_text(display_frame, "【垫步】表现优秀!", (text_x_offset, current_y), 
                    font_size=20, text_color=white_color, thickness=1)
        current_y += line_height + 5
        
        # 添加具体技术指标值
        if "stance_width_pixels" in footwork_metrics:
            stance_width = footwork_metrics.get("stance_width_pixels", "0")
            frame_height_pixels = frame_dimensions[0] if frame_dimensions else 720
            width_ratio = float(stance_width) / frame_height_pixels if isinstance(stance_width, (int, float)) else 0
            width_percent = f"{width_ratio * 100:.1f}%"
            stance_text = f"站姿宽度: {stance_width} 像素 ({width_percent}身高)"
            display_frame = draw_chinese_text(display_frame, stance_text, (text_x_offset, current_y), 
                           font_size=20, text_color=detail_color, thickness=1)
            current_y += line_height
            
        if "stance_type_guess" in footwork_metrics:
            stance_type = f"站姿类型: {footwork_metrics.get('stance_type_guess', 'Unknown')}"
            display_frame = draw_chinese_text(display_frame, stance_type, (text_x_offset, current_y), 
                           font_size=20, text_color=detail_color, thickness=1)
            current_y += line_height
            
        # 移除一键分享按钮，显示帧号和视频信息
        frame_info_text = f"当前帧: {frame_num} / {total_frames if total_frames > 0 else '未知'}"
        display_frame = draw_chinese_text(display_frame, frame_info_text, (text_x_offset, panel_height - 40), 
                    font_size=18, text_color=white_color, thickness=1)
        
        # 显示FPS
        current_fps = (frame_num + 1) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (frame_width - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 添加虚线边框效果(类似图片中的效果)
        h, w = display_frame.shape[:2]
        # 左上角点
        cv2.circle(display_frame, (30, 30), 5, (0, 0, 255), -1)
        # 右上角点
        cv2.circle(display_frame, (w-30, 30), 5, (0, 0, 255), -1)
        # 左下角点
        cv2.circle(display_frame, (30, h-30), 5, (0, 0, 255), -1)
        # 右下角点
        cv2.circle(display_frame, (w-30, h-30), 5, (0, 0, 255), -1)
        # 上中点
        cv2.circle(display_frame, (w//2, 30), 5, (0, 0, 255), -1)
        # 下中点
        cv2.circle(display_frame, (w//2, h-30), 5, (0, 0, 255), -1)
        # 左中点
        cv2.circle(display_frame, (30, h//2), 5, (0, 0, 255), -1)
        # 右中点
        cv2.circle(display_frame, (w-30, h//2), 5, (0, 0, 255), -1)
        
        # 绘制虚线边框
        # 上边线
        for x in range(30, w-30, 10):
            cv2.line(display_frame, (x, 30), (x+5, 30), (0, 0, 255), 1)
        # 下边线
        for x in range(30, w-30, 10):
            cv2.line(display_frame, (x, h-30), (x+5, h-30), (0, 0, 255), 1)
        # 左边线
        for y in range(30, h-30, 10):
            cv2.line(display_frame, (30, y), (30, y+5), (0, 0, 255), 1)
        # 右边线
        for y in range(30, h-30, 10):
            cv2.line(display_frame, (w-30, y), (w-30, y+5), (0, 0, 255), 1)

        cv2.imshow("网球分析", display_frame)
        if out:
            out.write(display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("用户按下'q'键，程序退出")
            break
        frame_num += 1

    # 计算总处理时间
    total_time = time.time() - start_time
    frames_processed = frame_num
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    
    logger.info(f"处理完成! 共处理 {frames_processed} 帧")
    logger.info(f"总处理时间: {total_time:.2f} 秒")
    logger.info(f"平均处理速度: {avg_fps:.2f} FPS")

    logger.info("释放资源...")
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    logger.info("程序结束")

if __name__ == "__main__":
    # 日志开始信息
    logger.info("=" * 50)
    logger.info("网球分析系统启动")
    logger.info("=" * 50)
    
    # 初始化字体
    logger.info("开始初始化中文字体...")
    FONT_PATH = manage_fonts()
    
    if FONT_PATH:
        logger.info(f"中文字体初始化成功: {FONT_PATH}")
    else:
        logger.warning("中文字体初始化失败，将使用OpenCV默认字体")
    
    # 验证字体可用性
    font_check_img = np.zeros((100, 300, 3), dtype=np.uint8)
    font_check_result = draw_chinese_text(font_check_img, "字体测试", (50, 50), font_size=24)
    
    if font_check_result is not None:
        logger.info("字体检查完成，可以正常绘制中文")
    else:
        logger.error("字体检查失败，中文渲染可能有问题")
    
    main()