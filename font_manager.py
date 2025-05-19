# font_manager.py
import os
import sys
import subprocess
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
import platform
import logging

class FontManager:
    def __init__(self, config=None):
        self.config = config or {}
        self.font_dir = Path(self.config.get('font_dir', 'fonts'))
        self.font_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('FontManager')
        self.default_font = None
        self.pingfang_path = None
        self.system_os = platform.system()
        
        # 初始化日志
        self.setup_logger()
        
        # 初始化字体
        self.init_fonts()
    
    def setup_logger(self):
        """设置日志记录器"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    def is_mac(self):
        """检查是否在MacOS上运行"""
        return self.system_os == 'Darwin'
    
    def check_font_exists(self, font_name):
        """检查字体是否存在于系统中"""
        if self.is_mac():
            # 在Mac上检查特定字体是否存在
            possible_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/Library/Fonts/PingFang.ttc",
                str(Path.home() / "Library" / "Fonts" / "PingFang.ttc"),
                "/System/Library/Fonts/Core/PingFang.ttc",  # 部分macOS版本的路径
                "/System/Library/PrivateFrameworks/FontServices.framework/Resources/PingFang.ttc",  # 另一个可能的路径
                "/System/Library/PrivateFrameworks/FontServices.framework/Resources/Reserved/PingFang.ttc" # macOS 15的路径
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.logger.info(f"已找到PingFang字体：{path}")
                    self.pingfang_path = path
                    return True
                
            # 检查系统字体列表
            try:
                result = subprocess.run(
                    ["fc-list", ":lang=zh-cn"],
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                if "PingFang" in result.stdout:
                    self.logger.info("通过fc-list找到PingFang字体")
                    # 尝试找到具体路径
                    for line in result.stdout.splitlines():
                        if "PingFang" in line:
                            parts = line.split(":")
                            if parts and os.path.exists(parts[0]):
                                self.pingfang_path = parts[0]
                                self.logger.info(f"设置PingFang字体路径为: {self.pingfang_path}")
                                break
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                self.logger.warning("fc-list命令不可用，无法检查系统字体")
            
            return False
        else:
            # 其他系统上使用自带字体
            return False
    
    def download_pingfang_font(self):
        """尝试在Mac上下载PingFang字体"""
        if not self.is_mac():
            self.logger.warning("不是Mac系统，无法下载系统PingFang字体")
            return False
        
        self.logger.info("正在尝试使用Font Book下载PingFang字体...")
        try:
            # 使用osascript自动操作Font Book下载PingFang字体
            # 改进脚本以更好地处理不同macOS版本
            script = """
            tell application "Font Book"
                activate
                delay 2
                
                tell application "System Events" to tell process "Font Book"
                    # 点击搜索按钮
                    try
                        click button 1 of toolbar 1 of window 1
                        delay 1
                    end try
                    
                    # 输入搜索词
                    keystroke "PingFang"
                    delay 3
                    
                    # 尝试不同的UI元素来找到下载按钮
                    try
                        # 方法1：直接按名称寻找按钮
                        click button "下载" of window 1
                        delay 5
                        return "方法1成功：直接找到下载按钮"
                    on error
                        try
                            # 方法2：点击表格中的首个结果，然后找下载按钮
                            select row 1 of table 1 of scroll area 1 of window 1
                            delay 1
                            click button "下载" of window 1
                            delay 5
                            return "方法2成功：选择行后找到下载按钮"
                        on error
                            try
                                # 方法3：尝试使用AXDescription
                                set downloadButtons to buttons of window 1 whose description is "下载"
                                if length of downloadButtons > 0 then
                                    click item 1 of downloadButtons
                                    delay 5
                                    return "方法3成功：使用description找到下载按钮"
                                end if
                            on error
                                return "未找到下载按钮：请手动操作"
                            end try
                        end try
                    end try
                end tell
            end tell
            """
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )
            
            result_text = result.stdout.strip()
            self.logger.info(f"Font Book自动下载结果: {result_text}")
            
            # 等待一段时间，然后再次检查字体是否存在
            if "成功" in result_text:
                # 延迟检查，因为下载可能需要一些时间
                import time
                time.sleep(10)  # 增加等待时间，防止下载未完成
                if self.check_font_exists("PingFang"):
                    self.logger.info("PingFang字体下载并安装成功")
                    return True
            
            # 提示用户手动下载
            self.logger.warning("自动下载失败，请手动使用Font Book下载PingFang字体")
            self.logger.warning("1. 打开Font Book应用程序")
            self.logger.warning("2. 搜索 'PingFang'")
            self.logger.warning("3. 点击 '下载' 按钮")
            
            return False
            
        except Exception as e:
            self.logger.error(f"尝试下载PingFang字体时出错: {e}")
            return False
    
    def init_fonts(self):
        """初始化字体设置"""
        self.logger.info("初始化字体...")
        
        if self.is_mac():
            # 在Mac上初始化PingFang字体
            if self.check_font_exists("PingFang"):
                self.logger.info("已找到PingFang字体，将用于视频OSD文本")
                self.default_font = self.pingfang_path
            else:
                self.logger.warning("未找到PingFang字体，尝试下载...")
                if self.download_pingfang_font():
                    self.default_font = self.pingfang_path
                else:
                    self.logger.warning("无法获取PingFang字体，尝试使用系统其他中文字体")
                    # 尝试找到系统中其他可用的中文字体
                    fallback_fonts = self.find_fallback_chinese_fonts()
                    if fallback_fonts:
                        self.default_font = fallback_fonts[0]
                        self.logger.info(f"使用备用中文字体: {self.default_font}")
                    else:
                        self.logger.warning("未找到可用的中文字体，将使用系统默认字体")
        else:
            # 非Mac系统尝试找到中文字体
            fallback_fonts = self.find_fallback_chinese_fonts()
            if fallback_fonts:
                self.default_font = fallback_fonts[0]
                self.logger.info(f"非Mac系统，使用中文字体: {self.default_font}")
            else:
                self.logger.info("非Mac系统，未找到中文字体，将使用OpenCV默认字体")
    
    def find_fallback_chinese_fonts(self):
        """查找系统中可用的其他中文字体"""
        chinese_fonts = []
        
        # Mac常见中文字体
        mac_fonts = [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Songti.ttc",
            "/Library/Fonts/Kaiti.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # 韩文字体，但支持中文
            "/System/Library/Fonts/AdobeHeitiStd-Regular.otf"
        ]
        
        # Linux常见中文字体
        linux_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
        ]
        
        # Windows常见中文字体
        windows_fonts = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
            "C:/Windows/Fonts/msyh.ttc"
        ]
        
        # 根据系统选择可能的字体列表
        if self.is_mac():
            possible_fonts = mac_fonts
        elif self.system_os == "Linux":
            possible_fonts = linux_fonts
        elif self.system_os == "Windows":
            possible_fonts = windows_fonts
        else:
            possible_fonts = []
        
        # 检查字体是否存在
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                self.logger.info(f"找到可用的中文字体: {font_path}")
                chinese_fonts.append(font_path)
        
        # 如果没有找到系统字体，尝试使用fontconfig搜索
        if not chinese_fonts:
            try:
                # 搜索系统中的中文字体
                font_cmd = "fc-list :lang=zh"
                result = subprocess.run(font_cmd.split(), capture_output=True, text=True)
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.splitlines():
                        parts = line.split(":")
                        if parts and os.path.exists(parts[0]):
                            chinese_fonts.append(parts[0])
                            self.logger.info(f"通过fc-list找到中文字体: {parts[0]}")
                            break
            except (subprocess.SubprocessError, FileNotFoundError):
                self.logger.warning("fc-list命令不可用，无法搜索系统中文字体")
        
        return chinese_fonts
    
    def get_font(self):
        """获取字体路径"""
        return self.default_font
    
    def put_text_with_font(self, img, text, org, font_scale=0.7, color=(255, 255, 255), thickness=2, font_face=cv2.FONT_HERSHEY_SIMPLEX):
        """使用适当的字体绘制文本"""
        # 先检查是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        # 如果包含中文且系统支持PIL，则使用PIL绘制
        if has_chinese:
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                # 把OpenCV图像转换为PIL图像
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 获取合适的字体大小（基于font_scale转换）
                font_size = int(12 * font_scale * 2)  # 调整倍数可能需要根据实际情况修改
                
                # 尝试加载中文字体，按优先级尝试
                font = None
                try:
                    # 1. 首先尝试使用配置中的默认字体
                    if self.default_font and os.path.exists(self.default_font):
                        try:
                            font = ImageFont.truetype(self.default_font, font_size)
                            self.logger.debug(f"使用默认字体: {self.default_font}")
                        except Exception as e:
                            self.logger.warning(f"加载默认字体失败: {e}")
                    
                    # 2. 如果默认字体不可用，尝试使用系统字体名称
                    if font is None:
                        system_font_names = [
                            "PingFang SC", "STHeiti", "SimHei", "Microsoft YaHei", 
                            "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Hiragino Sans GB"
                        ]
                        for font_name in system_font_names:
                            try:
                                font = ImageFont.truetype(font_name, font_size)
                                self.logger.debug(f"使用系统字体: {font_name}")
                                break
                            except Exception:
                                pass
                    
                    # 3. 如果系统字体名称也不可用，尝试使用找到的备用字体
                    if font is None:
                        fallback_fonts = self.find_fallback_chinese_fonts()
                        for fb_font in fallback_fonts:
                            try:
                                font = ImageFont.truetype(fb_font, font_size)
                                self.logger.debug(f"使用备用字体: {fb_font}")
                                break
                            except Exception:
                                pass
                    
                    # 4. 最后，如果所有尝试都失败，使用默认字体
                    if font is None:
                        font = ImageFont.load_default()
                        self.logger.warning("无法加载任何中文字体，使用PIL默认字体")
                except Exception as e:
                    self.logger.error(f"字体加载过程中出错: {e}")
                    font = ImageFont.load_default()
                
                # 绘制文本
                text_color = (color[2], color[1], color[0])  # BGR转RGB
                
                # 对每行文本分别绘制（支持多行文本）
                lines = text.split('\n')
                y_offset = 0
                line_spacing = font_size * 0.3  # 行间距为字体大小的30%
                
                for line in lines:
                    draw.text((org[0], org[1] + y_offset), line, fill=text_color, font=font)
                    # 计算下一行的垂直偏移量
                    y_offset += font_size + line_spacing
                
                # 把PIL图像转换回OpenCV图像
                result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                return result_img
            
            except ImportError:
                self.logger.warning("PIL库不可用，使用OpenCV默认文本渲染")
                cv2.putText(img, text, org, font_face, font_scale, color, thickness, cv2.LINE_AA)
                return img
            except Exception as e:
                self.logger.error(f"使用PIL渲染文本出错: {e}")
                cv2.putText(img, text, org, font_face, font_scale, color, thickness, cv2.LINE_AA)
                return img
        else:
            # 不包含中文，使用OpenCV默认渲染
            # 对多行文本分别渲染
            lines = text.split('\n')
            result_img = img.copy()
            line_height = int(30 * font_scale)  # 估计的行高
            
            for i, line in enumerate(lines):
                y_position = org[1] + i * line_height
                cv2.putText(result_img, line, (org[0], y_position), 
                           font_face, font_scale, color, thickness, cv2.LINE_AA)
            
            return result_img

# 单例模式
_font_manager_instance = None

def get_font_manager(config=None):
    """获取字体管理器单例"""
    global _font_manager_instance
    if _font_manager_instance is None:
        _font_manager_instance = FontManager(config)
    return _font_manager_instance 