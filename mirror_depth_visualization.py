"""
mirror_depth_visualization.py
───────────────────────────────
算法 2.0 镜面深度立体空间几何与光路可视化引擎：
基于室内训练仓物理光学关系（正面高挂相机 + 选手站位 + 后墙平镜），
生成 2.5D Isometric 立体空间几何模型与光线追迹（Ray Optics）矢量图（SVG/HTML）。

核心物理参数与公式：
- 相机光心：位于 (X=0, Y=0, H_cam=3.3m)，俯仰角 θ_pitch ≈ 35°
- 选手站位：位于 Y_player ≈ 4.68m
- 后墙平镜：垂直位于地面深度 Y_wall ≈ 6.20m
- 镜中虚像：根据平面镜对称反射，位于 Y_virtual = 2 * Y_wall - Y_player = 7.72m
- 视差相对深度：ΔY = Y_virtual - Y_player = 3.04m
- 横向放大比：k_x = Y_virtual / Y_player ≈ 1.65
- 纵向透视尺度：k_y ≈ 1.27
- 各向异性比率：k_x / k_y ≈ 1.30（~30% 横纵投影尺度偏差）
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, Optional, Tuple, Union


def compute_mirror_depth_geometry(
    player_y: float = 4.68,
    wall_y: float = 6.20,
    cam_h: float = 3.30,
    cam_pitch_deg: float = 35.0,
    player_target_h: float = 1.20,
) -> Dict[str, Any]:
    """
    解算室内训练仓镜面光学空间几何参数。

    Args:
        player_y: 真实选手地面深度 (米)，默认 4.68m
        wall_y: 后墙平镜地面深度 (米)，默认 6.20m
        cam_h: 相机光心离地高度 (米)，默认 3.30m
        cam_pitch_deg: 相机向下俯倾角 (度)，默认 35°
        player_target_h: 选手躯干/持拍中心离地高度 (米)，默认 1.20m

    Returns:
        包含虚像深度、光线反射点、放大率与各向异性指标的字典
    """
    # 边界防御：选手必须位于相机与平镜之间
    safe_player_y = max(1.0, min(float(player_y), float(wall_y) - 0.2))
    safe_wall_y = max(safe_player_y + 0.2, float(wall_y))
    safe_cam_h = max(1.5, float(cam_h))
    safe_pitch = max(0.0, min(80.0, float(cam_pitch_deg)))
    safe_target_h = max(0.2, min(safe_cam_h - 0.1, float(player_target_h)))

    # 1. 虚像物理深度 Y_virtual
    virtual_y = 2.0 * safe_wall_y - safe_player_y
    delta_y = virtual_y - safe_player_y

    # 2. 光线在平镜上的反射点高度 Z_reflect
    # 直线方程从相机 (0, cam_h) 到虚像 (virtual_y, target_h)，在 Y=wall_y 处的值：
    reflect_z = safe_cam_h + (safe_target_h - safe_cam_h) * (safe_wall_y / max(0.01, virtual_y))

    # 3. 横向放大率 k_x 与纵向透视尺度 k_y
    scale_x = virtual_y / safe_player_y
    # 纵向透视缩放比受相机俯仰角调制
    # 几何模型中，沿光轴倾斜在像面投影具有 cos(pitch) 缩放效应
    pitch_rad = math.radians(safe_pitch)
    # 当 pitch=35° 时，cos(35°) ≈ 0.819，1.65 * 0.819 ≈ 1.35；经验校准值为 ~1.27
    scale_y = scale_x * math.cos(pitch_rad) * 0.94  # 结合像平面仰角校准
    if scale_y < 0.5:
        scale_y = 1.27
    anisotropy_ratio = scale_x / max(0.01, scale_y)

    # 4. 入射角与反射角
    # 光线在 Y-Z 垂直平面的入射角
    dz_incident = safe_cam_h - reflect_z
    dy_incident = safe_wall_y
    incident_angle_deg = math.degrees(math.atan2(dz_incident, dy_incident))

    return {
        "player_y": round(safe_player_y, 2),
        "wall_y": round(safe_wall_y, 2),
        "virtual_y": round(virtual_y, 2),
        "delta_y": round(delta_y, 2),
        "cam_h": round(safe_cam_h, 2),
        "cam_pitch_deg": round(safe_pitch, 1),
        "target_h": round(safe_target_h, 2),
        "reflect_z": round(reflect_z, 2),
        "scale_x": round(scale_x, 3),
        "scale_y": round(scale_y, 3),
        "anisotropy_ratio": round(anisotropy_ratio, 2),
        "incident_angle_deg": round(incident_angle_deg, 1),
    }


def render_mirror_depth_isometric_svg(
    geometry_data: Optional[Dict[str, Any]] = None,
    width: int = 860,
    height: int = 380,
    show_hud: bool = True,
) -> str:
    """
    纯矢量生成 2.5D Isometric 室内光路立体几何与镜面深度剖面图 (SVG)。
    零外部 CDN 依赖，完全内联自洽，适配暗黑科技主题。
    """
    geom = geometry_data or compute_mirror_depth_geometry()
    p_y = geom["player_y"]
    w_y = geom["wall_y"]
    v_y = geom["virtual_y"]
    c_h = geom["cam_h"]
    r_z = geom["reflect_z"]
    t_h = geom["target_h"]
    k_x = geom["scale_x"]
    k_y = geom["scale_y"]
    d_y = geom["delta_y"]
    pitch = geom["cam_pitch_deg"]

    # 2.5D 坐标映射系统 (Axonometric / Perspective Projection)
    # 原点位于画面左下方附近
    # 世界坐标: X (横向), Y (深度 0~9m), Z (高度 0~3.5m)
    # 屏幕投影:
    #   U = origin_u + Y * u_per_y + X * u_per_x
    #   V = origin_v - Y * v_per_y - Z * v_per_z
    origin_u = 80.0
    origin_v = 300.0

    u_per_y = 66.0   # 每米深度横向位移 (朝右)
    v_per_y = 12.0   # 每米深度纵向位移 (轻微倾斜透视)
    v_per_z = 62.0   # 每米高度纵向位移 (垂直向上)

    def to_screen(y_m: float, z_m: float, x_m: float = 0.0) -> Tuple[float, float]:
        u = origin_u + y_m * u_per_y + x_m * 22.0
        v = origin_v + y_m * v_per_y - z_m * v_per_z
        return round(u, 1), round(v, 1)

    # 关键坐标点
    # 1. 地面基准点
    cam_ground_u, cam_ground_v = to_screen(0.0, 0.0)
    player_ground_u, player_ground_v = to_screen(p_y, 0.0)
    wall_ground_u, wall_ground_v = to_screen(w_y, 0.0)
    virtual_ground_u, virtual_ground_v = to_screen(v_y, 0.0)

    # 2. 相机光心
    cam_u, cam_v = to_screen(0.0, c_h)

    # 3. 选手中心 (真像)
    player_target_u, player_target_v = to_screen(p_y, t_h)
    player_head_u, player_head_v = to_screen(p_y, 1.75)

    # 4. 镜面反射点与镜面四角
    reflect_u, reflect_v = to_screen(w_y, r_z)
    mirror_top_u, mirror_top_v = to_screen(w_y, 3.1)
    mirror_bot_u, mirror_bot_v = to_screen(w_y, 0.2)

    # 5. 虚像中心
    virtual_target_u, virtual_target_v = to_screen(v_y, t_h)
    virtual_head_u, virtual_head_v = to_screen(v_y, 1.75)

    # 生成 SVG 矢量代码
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" class="mirror-depth-svg" style="width:100%;height:auto;display:block;background:radial-gradient(ellipse at 40% 30%, #151e2e 0%, #0a0f18 100%);border-radius:10px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;user-select:none;">
  <defs>
    <!-- 发光滤镜与渐变 -->
    <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="glow-amber" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <linearGradient id="grad-mirror" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.25"/>
      <stop offset="50%" stop-color="#f5d04c" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="#00f0ff" stop-opacity="0.05"/>
    </linearGradient>
    <linearGradient id="grad-virtual-space" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="rgba(168,85,247,0.18)"/>
      <stop offset="100%" stop-color="rgba(168,85,247,0.01)"/>
    </linearGradient>
    <marker id="arrow-cyan" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#00f0ff"/>
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#f5d04c"/>
    </marker>
  </defs>

  <!-- 1. 背景立体网格：地面深度透视基线 -->
  <g class="court-floor-grid" stroke="#223048" stroke-width="1" opacity="0.65">
    <!-- 地面纵向导轨线 (0m -> 8.5m) -->
    <line x1="{to_screen(0, 0, -1.8)[0]}" y1="{to_screen(0, 0, -1.8)[1]}" x2="{to_screen(8.8, 0, -1.8)[0]}" y2="{to_screen(8.8, 0, -1.8)[1]}" stroke-dasharray="3,3"/>
    <line x1="{to_screen(0, 0, 0)[0]}" y1="{to_screen(0, 0, 0)[1]}" x2="{to_screen(8.8, 0, 0)[0]}" y2="{to_screen(8.8, 0, 0)[1]}" stroke="#2c3e58" stroke-width="1.5"/>
    <line x1="{to_screen(0, 0, 1.8)[0]}" y1="{to_screen(0, 0, 1.8)[1]}" x2="{to_screen(8.8, 0, 1.8)[0]}" y2="{to_screen(8.8, 0, 1.8)[1]}" stroke-dasharray="3,3"/>

    <!-- 地面横向刻度线 -->
    <line x1="{to_screen(0, 0, -1.8)[0]}" y1="{to_screen(0, 0, -1.8)[1]}" x2="{to_screen(0, 0, 1.8)[0]}" y2="{to_screen(0, 0, 1.8)[1]}" stroke="#00f0ff" stroke-width="1.5"/>
    <line x1="{to_screen(p_y, 0, -1.8)[0]}" y1="{to_screen(p_y, 0, -1.8)[1]}" x2="{to_screen(p_y, 0, 1.8)[0]}" y2="{to_screen(p_y, 0, 1.8)[1]}" stroke="#40d68b" stroke-width="1.5"/>
    <line x1="{to_screen(w_y, 0, -1.8)[0]}" y1="{to_screen(w_y, 0, -1.8)[1]}" x2="{to_screen(w_y, 0, 1.8)[0]}" y2="{to_screen(w_y, 0, 1.8)[1]}" stroke="#f5d04c" stroke-width="2"/>
    <line x1="{to_screen(v_y, 0, -1.8)[0]}" y1="{to_screen(v_y, 0, -1.8)[1]}" x2="{to_screen(v_y, 0, 1.8)[0]}" y2="{to_screen(v_y, 0, 1.8)[1]}" stroke="#a855f7" stroke-width="1.5" stroke-dasharray="4,4"/>
  </g>

  <!-- 2. 后墙平镜与虚拟反射空间背景 -->
  <!-- 镜后虚像反射空间区域 -->
  <polygon points="{wall_ground_u},{wall_ground_v} {mirror_top_u},{mirror_top_v} {to_screen(8.8, 3.1)[0]},{to_screen(8.8, 3.1)[1]} {to_screen(8.8, 0)[0]},{to_screen(8.8, 0)[1]}" fill="url(#grad-virtual-space)"/>
  
  <!-- 平面镜实体结构 (后墙 Y = {w_y}m) -->
  <g class="mirror-wall-group">
    <!-- 镜面边框与反射玻璃体 -->
    <polygon points="{to_screen(w_y, 0.2, -1.5)[0]},{to_screen(w_y, 0.2, -1.5)[1]} {to_screen(w_y, 3.1, -1.5)[0]},{to_screen(w_y, 3.1, -1.5)[1]} {to_screen(w_y, 3.1, 1.5)[0]},{to_screen(w_y, 3.1, 1.5)[1]} {to_screen(w_y, 0.2, 1.5)[0]},{to_screen(w_y, 0.2, 1.5)[1]}"
             fill="url(#grad-mirror)" stroke="#00f0ff" stroke-width="1.5" stroke-dasharray="none"/>
    <line x1="{mirror_bot_u}" y1="{mirror_bot_v}" x2="{mirror_top_u}" y2="{mirror_top_v}" stroke="#f5d04c" stroke-width="2.5" opacity="0.9"/>
    
    <!-- 镜面反光光芒线条 -->
    <line x1="{to_screen(w_y, 2.6, -1.1)[0]}" y1="{to_screen(w_y, 2.6, -1.1)[1]}" x2="{to_screen(w_y, 1.2, 0.8)[0]}" y2="{to_screen(w_y, 1.2, 0.8)[1]}" stroke="#ffffff" stroke-width="1" opacity="0.25"/>
    <line x1="{to_screen(w_y, 2.8, -0.6)[0]}" y1="{to_screen(w_y, 2.8, -0.6)[1]}" x2="{to_screen(w_y, 1.8, 0.9)[0]}" y2="{to_screen(w_y, 1.8, 0.9)[1]}" stroke="#ffffff" stroke-width="0.8" opacity="0.2"/>

    <!-- 镜面铭牌标签 -->
    <rect x="{wall_ground_u - 48}" y="{wall_ground_v + 12}" width="96" height="20" rx="4" fill="#182336" stroke="#f5d04c" stroke-width="1"/>
    <text x="{wall_ground_u}" y="{wall_ground_v + 26}" text-anchor="middle" fill="#f5d04c" font-size="11" font-weight="600">🪞 后墙平镜 Y={w_y}m</text>
  </g>

  <!-- 3. 光线追踪光路 (Ray Optics) -->
  <!-- A. 直射视线 (Camera -> Player Torso): 青色实线 -->
  <g class="ray-direct" filter="url(#glow-cyan)">
    <line x1="{cam_u}" y1="{cam_v}" x2="{player_target_u}" y2="{player_target_v}" stroke="#00f0ff" stroke-width="2.2" marker-end="url(#arrow-cyan)"/>
  </g>

  <!-- B. 镜面入射视线 (Camera -> Mirror Reflection Point): 琥珀色实线 -->
  <g class="ray-incident" filter="url(#glow-amber)">
    <line x1="{cam_u}" y1="{cam_v}" x2="{reflect_u}" y2="{reflect_v}" stroke="#f5d04c" stroke-width="2.2" marker-end="url(#arrow-amber)"/>
  </g>

  <!-- C. 镜面反射视线 (Mirror Reflection Point -> Player Back): 琥珀色/珊瑚色实线 -->
  <g class="ray-reflected">
    <line x1="{reflect_u}" y1="{reflect_v}" x2="{player_target_u}" y2="{player_target_v}" stroke="#f5d04c" stroke-width="2.2" stroke-dasharray="4,2"/>
  </g>

  <!-- D. 虚像延伸视线 (Mirror Reflection Point -> Virtual Player): 紫色/青色虚线 -->
  <g class="ray-virtual">
    <line x1="{reflect_u}" y1="{reflect_v}" x2="{virtual_target_u}" y2="{virtual_target_v}" stroke="#a855f7" stroke-width="2" stroke-dasharray="5,4" opacity="0.85"/>
  </g>

  <!-- 镜面反射交点标记 -->
  <circle cx="{reflect_u}" cy="{reflect_v}" r="4.5" fill="#f5d04c" stroke="#ffffff" stroke-width="1.5"/>
  <text x="{reflect_u + 8}" y="{reflect_v - 6}" fill="#f5d04c" font-size="10.5" font-weight="600">反射点 Z={r_z}m</text>

  <!-- 4. 真实选手 (Real Player, Y={p_y}m) -->
  <g class="player-real">
    <!-- 地面阴影 -->
    <ellipse cx="{player_ground_u}" cy="{player_ground_v}" rx="14" ry="5" fill="rgba(64,214,139,0.3)"/>
    <!-- 垂直身躯辅助线 -->
    <line x1="{player_ground_u}" y1="{player_ground_v}" x2="{player_head_u}" y2="{player_head_v}" stroke="#40d68b" stroke-width="1.5" stroke-dasharray="2,2"/>
    
    <!-- 简练人体解剖轮廓 (正面) -->
    <circle cx="{player_head_u}" cy="{player_head_v}" r="6.5" fill="#40d68b"/>
    <line x1="{player_head_u}" y1="{player_head_v + 6.5}" x2="{player_target_u}" y2="{player_target_v + 10}" stroke="#40d68b" stroke-width="3.5" stroke-linecap="round"/>
    <!-- 双腿 -->
    <line x1="{player_target_u}" y1="{player_target_v + 10}" x2="{player_ground_u - 7}" y2="{player_ground_v}" stroke="#40d68b" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="{player_target_u}" y1="{player_target_v + 10}" x2="{player_ground_u + 7}" y2="{player_ground_v}" stroke="#40d68b" stroke-width="2.5" stroke-linecap="round"/>
    <!-- 持拍手臂 (向右前伸) -->
    <line x1="{player_target_u}" y1="{player_target_v - 6}" x2="{player_target_u + 14}" y2="{player_target_v - 12}" stroke="#40d68b" stroke-width="2"/>
    <ellipse cx="{player_target_u + 18}" cy="{player_target_v - 16}" rx="5" ry="3" fill="none" stroke="#00f0ff" stroke-width="1.5" transform="rotate(-30 {player_target_u + 18} {player_target_v - 16})"/>

    <!-- 真实选手铭牌 -->
    <rect x="{player_ground_u - 50}" y="{player_ground_v + 12}" width="100" height="20" rx="4" fill="#14241d" stroke="#40d68b" stroke-width="1"/>
    <text x="{player_ground_u}" y="{player_ground_v + 26}" text-anchor="middle" fill="#40d68b" font-size="11" font-weight="600">👤 真实选手 Y={p_y}m</text>
  </g>

  <!-- 5. 镜中虚像 (Virtual Image, Y={v_y}m) -->
  <g class="player-virtual" opacity="0.82">
    <!-- 地面虚影 -->
    <ellipse cx="{virtual_ground_u}" cy="{virtual_ground_v}" rx="16" ry="5" fill="rgba(168,85,247,0.25)"/>
    <!-- 垂直身躯辅助线 -->
    <line x1="{virtual_ground_u}" y1="{virtual_ground_v}" x2="{virtual_head_u}" y2="{virtual_head_v}" stroke="#a855f7" stroke-width="1.5" stroke-dasharray="3,3"/>
    
    <!-- 背面虚像人体轮廓 (水平镜像反转) -->
    <circle cx="{virtual_head_u}" cy="{virtual_head_v}" r="6.5" fill="#a855f7" stroke="#ffffff" stroke-width="1"/>
    <line x1="{virtual_head_u}" y1="{virtual_head_v + 6.5}" x2="{virtual_target_u}" y2="{virtual_target_v + 10}" stroke="#a855f7" stroke-width="3.5" stroke-dasharray="4,2" stroke-linecap="round"/>
    <!-- 双腿 -->
    <line x1="{virtual_target_u}" y1="{virtual_target_v + 10}" x2="{virtual_ground_u - 8}" y2="{virtual_ground_v}" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="{virtual_target_u}" y1="{virtual_target_v + 10}" x2="{virtual_ground_u + 8}" y2="{virtual_ground_v}" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round"/>
    <!-- 镜像持拍手 (向左后展) -->
    <line x1="{virtual_target_u}" y1="{virtual_target_v - 6}" x2="{virtual_target_u - 15}" y2="{virtual_target_v - 13}" stroke="#a855f7" stroke-width="2"/>
    <ellipse cx="{virtual_target_u - 19}" cy="{virtual_target_v - 17}" rx="5" ry="3" fill="none" stroke="#f5d04c" stroke-width="1.5" transform="rotate(30 {virtual_target_u - 19} {virtual_target_v - 17})"/>

    <!-- 虚像铭牌 -->
    <rect x="{virtual_ground_u - 54}" y="{virtual_ground_v + 12}" width="108" height="20" rx="4" fill="#20162e" stroke="#a855f7" stroke-width="1"/>
    <text x="{virtual_ground_u}" y="{virtual_ground_v + 26}" text-anchor="middle" fill="#c084fc" font-size="11" font-weight="600">🪞 镜中虚像 Y={v_y}m</text>
  </g>

  <!-- 6. 相机支架与光心 (Camera, Y=0m, H={c_h}m) -->
  <g class="camera-rig">
    <!-- 立柱 -->
    <line x1="{cam_ground_u}" y1="{cam_ground_v}" x2="{cam_u}" y2="{cam_v}" stroke="#4a5f78" stroke-width="2.5"/>
    <circle cx="{cam_ground_u}" cy="{cam_ground_v}" r="4" fill="#4a5f78"/>
    
    <!-- 摄像机壳体与镜头 -->
    <rect x="{cam_u - 16}" y="{cam_v - 11}" width="26" height="18" rx="3" fill="#182336" stroke="#00f0ff" stroke-width="1.5"/>
    <!-- 镜头倾角指示器 (指向斜下方 pitch={pitch}°) -->
    <polygon points="{cam_u + 10},{cam_v - 5} {cam_u + 20},{cam_v - 1} {cam_u + 20},{cam_v + 11} {cam_u + 10},{cam_v + 7}" fill="#00f0ff" opacity="0.85"/>
    <circle cx="{cam_u - 2}" cy="{cam_v - 2}" r="3" fill="#ff6e73"/>

    <!-- 相机铭牌 -->
    <rect x="{cam_u - 54}" y="{cam_v - 32}" width="108" height="20" rx="4" fill="#0f1926" stroke="#00f0ff" stroke-width="1"/>
    <text x="{cam_u}" y="{cam_v - 18}" text-anchor="middle" fill="#00f0ff" font-size="11" font-weight="600">📷 相机 H={c_h}m (俯角{pitch}°)</text>
  </g>

  <!-- 7. 深度几何双箭头尺寸标注线 (Dimension Annotations) -->
  <!-- A. 真实深度到虚像深度跨度 (ΔY = {d_y}m) -->
  <g class="dimension-delta" stroke="#8d9caf" stroke-width="1">
    <line x1="{player_ground_u}" y1="{origin_v + 44}" x2="{virtual_ground_u}" y2="{origin_v + 44}"/>
    <line x1="{player_ground_u}" y1="{origin_v + 38}" x2="{player_ground_u}" y2="{origin_v + 50}"/>
    <line x1="{virtual_ground_u}" y1="{origin_v + 38}" x2="{virtual_ground_u}" y2="{origin_v + 50}"/>
    <rect x="{(player_ground_u + virtual_ground_u)/2 - 58}" y="{origin_v + 34}" width="116" height="18" rx="3" fill="#0e1420" stroke="#8d9caf" stroke-width="0.8"/>
    <text x="{(player_ground_u + virtual_ground_u)/2}" y="{origin_v + 47}" text-anchor="middle" fill="#eef4fb" font-size="10.5" font-weight="600">虚实相对深度 ΔY = {d_y}m</text>
  </g>

  <!-- 8. 右上角：光学空间与各向异性尺度 HUD 核心数据看板 -->
  {_render_hud_box(k_x, k_y, p_y, w_y, v_y, d_y) if show_hud else ""}
</svg>"""
    return svg


def _render_hud_box(k_x: float, k_y: float, p_y: float, w_y: float, v_y: float, d_y: float) -> str:
    """生成右上角半透明光学参数 HUD 面板。"""
    hud_x = 570
    hud_y = 16
    hud_w = 274
    hud_h = 132

    anisotropy_pct = round((k_x / max(0.01, k_y) - 1.0) * 100, 1)

    return f"""  <g class="optical-hud-panel" transform="translate({hud_x}, {hud_y})">
    <rect x="0" y="0" width="{hud_w}" height="{hud_h}" rx="8" fill="rgba(14, 20, 32, 0.88)" stroke="#273449" stroke-width="1"/>
    
    <!-- 标题栏 -->
    <text x="12" y="20" fill="#00f0ff" font-size="11.5" font-weight="700" letter-spacing="0.5">🪞 算法 2.0 镜面光学空间几何</text>
    <rect x="186" y="8" width="76" height="16" rx="3" fill="rgba(64, 214, 139, 0.15)" stroke="#40d68b" stroke-width="0.8"/>
    <text x="224" y="20" text-anchor="middle" fill="#40d68b" font-size="9.5" font-weight="600">仿射解耦已启用</text>
    
    <line x1="12" y1="28" x2="{hud_w - 12}" y2="28" stroke="#223046" stroke-width="1"/>

    <!-- 指标网格 -->
    <!-- 行 1: 横向放大率 k_x 与 纵向透视尺度 k_y -->
    <text x="12" y="47" fill="#8d9caf" font-size="11">横向放大率 k_x:</text>
    <text x="120" y="47" fill="#f5d04c" font-size="12" font-weight="700">×{k_x:.2f}</text>
    <text x="160" y="47" fill="#8d9caf" font-size="11">纵向透视 k_y:</text>
    <text x="240" y="47" fill="#00f0ff" font-size="12" font-weight="700">×{k_y:.2f}</text>

    <!-- 行 2: 虚实深度比率与视差跨度 -->
    <text x="12" y="69" fill="#8d9caf" font-size="11">虚像深度 Y_v:</text>
    <text x="120" y="69" fill="#a855f7" font-size="12" font-weight="700">{v_y:.2f}m</text>
    <text x="160" y="69" fill="#8d9caf" font-size="11">相对视差 ΔY:</text>
    <text x="240" y="69" fill="#eef4fb" font-size="12" font-weight="700">{d_y:.2f}m</text>

    <!-- 行 3: 各向异性投影偏差与解耦说明 -->
    <rect x="12" y="79" width="{hud_w - 24}" height="42" rx="5" fill="rgba(24, 32, 48, 0.7)" stroke="#273449" stroke-width="0.8"/>
    <text x="20" y="95" fill="#eef4fb" font-size="10.5">📐 <b>各向异性偏差:</b> +{anisotropy_pct}% (横纵比 {k_x/k_y:.2f})</text>
    <text x="20" y="111" fill="#8d9caf" font-size="9.5">避免单一身长缩放导致引拍横向位移严重低估</text>
  </g>"""


def render_mirror_depth_card_html(
    geometry_data: Optional[Dict[str, Any]] = None,
    measured_relative_z: Optional[float] = None,
    ground_calibration: Optional[Dict[str, Any]] = None,
    card_id: str = "mirror-depth-card",
) -> str:
    """
    渲染嵌入在分析报告或控制面板中的镜面深度立体可视化卡片 (完整 HTML 模块)。
    """
    geom = geometry_data or compute_mirror_depth_geometry()
    svg_code = render_mirror_depth_isometric_svg(geom, width=860, height=360, show_hud=True)

    rel_z_text = f"×{measured_relative_z:.3f}" if measured_relative_z is not None else "待击球点计算"
    rel_z_badge = "badge-optimal" if measured_relative_z is not None else "badge-pending"

    # 地面标定绑定信息
    ground_info = "未绑定独立地面单应性"
    if ground_calibration and isinstance(ground_calibration, dict):
        views = ground_calibration.get("views") or {}
        if "front" in views and "back" in views:
            ground_info = "已融合双视角地面单应性网格 (A-D / A′-D′)"

    card_html = f"""
<div class="mirror-depth-card" id="{html.escape(card_id)}">
  <div class="mirror-depth-header">
    <div class="title-wrap">
      <span class="icon">🪞</span>
      <div>
        <h3 class="title">室内光路立体几何与镜面深度剖面</h3>
        <p class="subtitle">3D Mirror Optical Depth &amp; Anisotropic Perspective Profile (算法 2.0 虚拟双机位)</p>
      </div>
    </div>
    <div class="status-tags">
      <span class="tag tag-cyan">平镜深度 {geom['wall_y']}m</span>
      <span class="tag tag-purple">虚像深度 {geom['virtual_y']}m</span>
      <span class="tag tag-amber">各向异性比 {geom['anisotropy_ratio']}</span>
    </div>
  </div>

  <div class="mirror-depth-body">
    <div class="svg-container">
      {svg_code}
    </div>

    <div class="metrics-grid">
      <div class="metric-item">
        <span class="label">📷 相机光心与俯角</span>
        <strong class="val val-cyan">H={geom['cam_h']}m / θ={geom['cam_pitch_deg']}°</strong>
        <span class="desc">俯角导致像面纵向缩放 k_y={geom['scale_y']}</span>
      </div>
      <div class="metric-item">
        <span class="label">👤 真实选手基准深度</span>
        <strong class="val val-green">Y={geom['player_y']}m</strong>
        <span class="desc">击球区前视角正面解剖坐标系</span>
      </div>
      <div class="metric-item">
        <span class="label">🪞 虚像距离与视差跨度</span>
        <strong class="val val-purple">Y_v={geom['virtual_y']}m (ΔY={geom['delta_y']}m)</strong>
        <span class="desc">实测视差比率: <span class="{rel_z_badge}">{rel_z_text}</span></span>
      </div>
      <div class="metric-item">
        <span class="label">📐 横纵仿射尺度解耦</span>
        <strong class="val val-amber">k_x={geom['scale_x']} / k_y={geom['scale_y']}</strong>
        <span class="desc">{html.escape(ground_info)}</span>
      </div>
    </div>
  </div>
</div>
"""
    return card_html
