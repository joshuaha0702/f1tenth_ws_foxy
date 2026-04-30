#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1tenth 데이터 로그 시각화 도구
- 타임스탬프마다 라이다 스캔을 극좌표로 렌더링
- 차량 상태: 조향각에 따라 기울어진 직사각형 박스
- 출력: MP4 영상
"""

import argparse
import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# 색상 팔레트 (다크 테마)
# ─────────────────────────────────────────────
BG        = '#0d1117'
PANEL_BG  = '#161b22'
ACCENT    = '#58a6ff'
LIDAR_FAR = '#1e6fa8'
LIDAR_MID = '#39d353'
LIDAR_NEAR= '#f78166'
CAR_COLOR = '#c9d1d9'
CAR_EDGE  = '#58a6ff'
GRID_CLR  = '#21262d'
TEXT_CLR  = '#8b949e'
WHITE     = '#e6edf3'

# ─────────────────────────────────────────────
# LiDAR 색상 매핑 (거리 기반)
# ─────────────────────────────────────────────
def lidar_color_map(distances, max_range=10.0):
    """거리에 따라 색상을 반환합니다 (가까울수록 붉은색)."""
    norm = np.clip(distances / max_range, 0, 1)
    colors = []
    for n in norm:
        if n < 0.33:
            # 빨강 → 노랑
            t = n / 0.33
            r = 1.0
            g = t * 0.85
            b = 0.4 * t
        elif n < 0.66:
            # 노랑 → 초록
            t = (n - 0.33) / 0.33
            r = 1.0 - t * 0.8
            g = 0.85
            b = 0.4 + t * 0.3
        else:
            # 초록 → 파랑
            t = (n - 0.66) / 0.34
            r = 0.2 - t * 0.2
            g = 0.85 - t * 0.6
            b = 0.7 + t * 0.3
        colors.append((r, g, b, 0.85))
    return colors


def build_figure(n_beams):
    """Figure와 축을 구성합니다."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        2, 4,
        figure=fig,
        left=0.05, right=0.97,
        top=0.93, bottom=0.06,
        wspace=0.35, hspace=0.40,
        width_ratios=[2.5, 1, 1, 1],
        height_ratios=[1.6, 1]
    )

    # 메인 라이다 극좌표 축
    ax_lidar = fig.add_subplot(gs[:, 0], projection='polar')
    ax_lidar.set_facecolor(PANEL_BG)

    # 차량 상태 축 (직사각형 + 조향각)
    ax_car = fig.add_subplot(gs[0, 1])
    ax_car.set_facecolor(PANEL_BG)

    # 조향각 게이지 축
    ax_steer = fig.add_subplot(gs[0, 2])
    ax_steer.set_facecolor(PANEL_BG)

    # 속도 게이지 축
    ax_speed = fig.add_subplot(gs[0, 3])
    ax_speed.set_facecolor(PANEL_BG)

    # 라이다 거리 분포 막대
    ax_hist = fig.add_subplot(gs[1, 1:])
    ax_hist.set_facecolor(PANEL_BG)

    for ax in [ax_car, ax_steer, ax_speed, ax_hist]:
        ax.spines['top'].set_color(GRID_CLR)
        ax.spines['right'].set_color(GRID_CLR)
        ax.spines['left'].set_color(GRID_CLR)
        ax.spines['bottom'].set_color(GRID_CLR)

    return fig, ax_lidar, ax_car, ax_steer, ax_speed, ax_hist


def draw_lidar(ax, angles_rad, distances, steering_rad, max_range=10.0):
    """극좌표에 라이다 데이터를 그립니다."""
    ax.cla()
    ax.set_facecolor(PANEL_BG)

    # 배경 동심원 그리드
    for r in [2, 4, 6, 8, 10]:
        circle = plt.Circle((0, 0), r, transform=ax.transData._b,
                             fill=False, color=GRID_CLR, linewidth=0.5, linestyle='--')

    # 월드가 고정되도록 각도를 조향각만큼 역으로 회전
    angles_rad = angles_rad - steering_rad

    colors = lidar_color_map(distances, max_range)

    # 스캔 영역 채우기 (바닥 - 원점을 포함하여 부채꼴 모양으로)
    fan_angles = np.concatenate([[angles_rad[0]], angles_rad, [angles_rad[-1]]])
    fan_dists  = np.concatenate([[0], distances, [0]])
    ax.fill(fan_angles, fan_dists, color=ACCENT, alpha=0.06)

    # 라이다 빔 scatter
    ax.scatter(angles_rad, distances, c=colors, s=2.5, zorder=3, linewidths=0)

    # 외곽선 (270도 FOV이므로 끝을 연결하지 않음)
    ax.plot(angles_rad, distances, color=ACCENT, linewidth=0.6, alpha=0.6, zorder=2)

    # 원점 (차량 위치)
    ax.scatter([0], [0], c=[CAR_EDGE], s=40, zorder=5, marker='D')

    # 축 설정
    ax.set_ylim(0, max_range)
    ax.set_theta_zero_location('N')   # 위쪽이 0도 (앞 방향)
    ax.set_theta_direction(-1)         # 시계 방향
    ax.set_rlabel_position(45)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)

    # 거리 눈금 색상
    for label in ax.get_yticklabels():
        label.set_color(TEXT_CLR)
        label.set_fontsize(7)
    for label in ax.get_xticklabels():
        label.set_color(TEXT_CLR)
        label.set_fontsize(7)

    ax.set_title('LiDAR Scan', color=WHITE, fontsize=11, fontweight='bold', pad=12)


def draw_car(ax, steering_rad, speed):
    """
    조향각에 따라 기울어진 직사각형 차량 박스를 그립니다.
    steering_rad: 조향각 (라디안). 양수=좌회전, 음수=우회전
    """
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    # 차체 크기
    car_w, car_h = 1.0, 1.8

    # 기울기: 차량이 실제 조향하는 방향으로 회전 (월드가 고정되므로 차가 회전해야 함)
    # 조향각이 양수(좌회전)이면 반시계방향(CCW) 회전이므로 그대로 양수 적용
    tilt_deg = math.degrees(steering_rad)

    # 차체 변환 (중앙 기준 회전)
    transform = (
        Affine2D().rotate_deg(tilt_deg) +
        ax.transData
    )

    # 차체 본체 (직사각형)
    body = patches.FancyBboxPatch(
        (-car_w / 2, -car_h / 2),
        car_w, car_h,
        boxstyle="round,pad=0.08",
        linewidth=2,
        edgecolor=CAR_EDGE,
        facecolor='#1f2937',
        transform=transform,
        zorder=3
    )
    ax.add_patch(body)

    # 앞/뒤 구분선
    front_bar = patches.FancyBboxPatch(
        (-car_w / 2, car_h / 2 - 0.35),
        car_w, 0.30,
        boxstyle="round,pad=0.03",
        linewidth=0,
        facecolor=ACCENT,
        alpha=0.6,
        transform=transform,
        zorder=4
    )
    ax.add_patch(front_bar)

    rear_bar = patches.FancyBboxPatch(
        (-car_w / 2, -car_h / 2 + 0.05),
        car_w, 0.20,
        boxstyle="round,pad=0.03",
        linewidth=0,
        facecolor='#f78166',
        alpha=0.5,
        transform=transform,
        zorder=4
    )
    ax.add_patch(rear_bar)

    # 타이어 4개
    wheel_positions = [
        (-car_w / 2 - 0.18, -car_h / 2 + 0.18),
        ( car_w / 2 - 0.08, -car_h / 2 + 0.18),
        (-car_w / 2 - 0.18,  car_h / 2 - 0.18),
        ( car_w / 2 - 0.08,  car_h / 2 - 0.18),
    ]
    for (wx, wy) in wheel_positions:
        wheel = patches.FancyBboxPatch(
            (wx, wy), 0.26, 0.15,
            boxstyle="round,pad=0.04",
            linewidth=1,
            edgecolor='#444',
            facecolor='#1a1a1a',
            transform=transform,
            zorder=5
        )
        ax.add_patch(wheel)

    # 조향각 화살표 (차 앞 방향)
    arrow_len = 0.8
    arrow_angle = math.pi / 2 + steering_rad  # 앞 방향 + 조향
    dx = arrow_len * math.cos(arrow_angle)
    dy = arrow_len * math.sin(arrow_angle)
    ax.annotate(
        '', xy=(dx, car_h / 2 * 0.85 + dy),
        xytext=(0, car_h / 2 * 0.85),
        arrowprops=dict(
            arrowstyle='->', color=ACCENT,
            lw=2.0, mutation_scale=16
        ),
        zorder=6
    )

    # 텍스트 (조향각 및 속도)
    steer_deg = math.degrees(steering_rad)
    direction = 'L' if steer_deg > 0 else 'R' if steer_deg < 0 else 'C'
    ax.text(
        0, -2.3,
        f'Steering: {steer_deg:+.1f}° ({direction})',
        ha='center', va='center', fontsize=9,
        color=WHITE, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_BG)]
    )
    ax.text(
        0, -2.7,
        f'Speed: {speed:.2f} m/s',
        ha='center', va='center', fontsize=10,
        color=LIDAR_MID, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_BG)]
    )

    ax.set_title('Vehicle State', color=WHITE, fontsize=11, fontweight='bold')


def draw_steering_gauge(ax, steering_rad, max_steer_rad=0.4189):
    """반원형 조향각 게이지를 그립니다."""
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Steering Gauge', color=WHITE, fontsize=11, fontweight='bold')

    # 반원 배경
    theta = np.linspace(0, math.pi, 200)
    ax.fill_between(np.cos(theta), np.sin(theta) * 0,
                    np.sin(theta), color=GRID_CLR, alpha=0.5)

    # 눈금
    for deg in range(-90, 91, 15):
        rad = math.radians(deg) + math.pi / 2
        x0, y0 = math.cos(rad) * 0.82, math.sin(rad) * 0.82
        x1, y1 = math.cos(rad) * 1.0, math.sin(rad) * 1.0
        ax.plot([x0, x1], [y0, y1], color=TEXT_CLR, lw=0.8)

    # 현재 조향각 → 게이지 각도 (PI/2 중심, -max ~ +max → 0 ~ PI)
    norm_steer = np.clip(steering_rad / max_steer_rad, -1.0, 1.0)
    gauge_angle = math.pi / 2 + norm_steer * (math.pi / 2 * 0.95)

    # 색상 선택
    if abs(norm_steer) < 0.2:
        needle_color = LIDAR_MID
    elif abs(norm_steer) < 0.6:
        needle_color = ACCENT
    else:
        needle_color = LIDAR_NEAR

    # 채워진 호 (조향 정도 표시)
    center_angle = math.pi / 2
    t_start = min(center_angle, gauge_angle)
    t_end   = max(center_angle, gauge_angle)
    t_arc = np.linspace(t_start, t_end, 80)
    ax.fill_between(
        np.cos(t_arc) * 0.82, np.sin(t_arc) * 0.0,
        np.sin(t_arc) * 0.82,
        alpha=0.3, color=needle_color
    )
    ax.plot(np.cos(t_arc) * 0.91, np.sin(t_arc) * 0.91,
            color=needle_color, lw=3, alpha=0.8)

    # 바늘
    nx = math.cos(gauge_angle) * 0.95
    ny = math.sin(gauge_angle) * 0.95
    ax.annotate(
        '', xy=(nx, ny), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=needle_color, lw=2.5, mutation_scale=14)
    )

    # 중앙 허브
    hub = plt.Circle((0, 0), 0.07, color=TEXT_CLR, zorder=5)
    ax.add_patch(hub)

    # 텍스트
    steer_deg = math.degrees(steering_rad)
    ax.text(0, -0.20, f'{steer_deg:+.2f}°',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=needle_color,
            path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_BG)])

    # L / R 레이블
    ax.text(-1.1, 0.05, 'L', ha='center', color=TEXT_CLR, fontsize=9)
    ax.text( 1.1, 0.05, 'R', ha='center', color=TEXT_CLR, fontsize=9)


def draw_speed_gauge(ax, speed, max_speed=5.0):
    """반원형 속도 게이지를 그립니다."""
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Speedometer', color=WHITE, fontsize=11, fontweight='bold')

    # 반원 배경
    theta = np.linspace(0, math.pi, 200)
    ax.fill_between(np.cos(theta), np.sin(theta) * 0,
                    np.sin(theta), color=GRID_CLR, alpha=0.5)

    # 눈금 (0 ~ max_speed)
    for pct in np.linspace(0, 1, 6):
        rad = math.pi - pct * math.pi
        x0, y0 = math.cos(rad) * 0.82, math.sin(rad) * 0.82
        x1, y1 = math.cos(rad) * 1.0, math.sin(rad) * 1.0
        ax.plot([x0, x1], [y0, y1], color=TEXT_CLR, lw=0.8)

    # 현재 속도 → 게이지 각도 (좌측 0, 우측 max_speed)
    norm_speed = np.clip(speed / max_speed, 0.0, 1.0)
    gauge_angle = math.pi - norm_speed * math.pi

    # 색상 선택
    if norm_speed < 0.5:
        needle_color = LIDAR_MID
    elif norm_speed < 0.8:
        needle_color = ACCENT
    else:
        needle_color = LIDAR_NEAR

    # 채워진 호 (속도 표시)
    start_angle = math.pi
    t_arc = np.linspace(start_angle, gauge_angle, 80)
    ax.fill_between(
        np.cos(t_arc) * 0.82, np.sin(t_arc) * 0.0,
        np.sin(t_arc) * 0.82,
        alpha=0.3, color=needle_color
    )
    ax.plot(np.cos(t_arc) * 0.91, np.sin(t_arc) * 0.91,
            color=needle_color, lw=3, alpha=0.8)

    # 바늘
    nx = math.cos(gauge_angle) * 0.95
    ny = math.sin(gauge_angle) * 0.95
    ax.annotate(
        '', xy=(nx, ny), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=needle_color, lw=2.5, mutation_scale=14)
    )

    # 중앙 허브
    hub = plt.Circle((0, 0), 0.07, color=TEXT_CLR, zorder=5)
    ax.add_patch(hub)

    # 텍스트
    ax.text(0, -0.20, f'{speed:.2f}',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=needle_color,
            path_effects=[pe.withStroke(linewidth=2, foreground=PANEL_BG)])

    # min / max 레이블
    ax.text(-1.1, 0.05, '0', ha='center', color=TEXT_CLR, fontsize=9)
    ax.text( 1.1, 0.05, f'{max_speed:.0f}', ha='center', color=TEXT_CLR, fontsize=9)


def draw_hist(ax, distances, max_range=10.0):
    """라이다 거리 분포 바 차트를 그립니다."""
    ax.cla()
    ax.set_facecolor(PANEL_BG)

    bins = np.arange(0, max_range + 0.5, 0.5)
    counts, edges = np.histogram(distances, bins=bins)

    # 각 빈의 색상을 거리 기반으로 결정
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = lidar_color_map(centers, max_range)

    ax.bar(centers, counts, width=0.45, color=bar_colors, edgecolor='none', alpha=0.85)

    ax.set_xlabel('Distance (m)', color=TEXT_CLR, fontsize=8)
    ax.set_ylabel('Beam Count', color=TEXT_CLR, fontsize=8)
    ax.set_title('LiDAR Distance Distribution', color=WHITE, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.set_xlim(0, max_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_CLR)
    ax.spines['bottom'].set_color(GRID_CLR)
    ax.set_facecolor(PANEL_BG)

    # 평균/최솟값 표시
    mean_d = np.mean(distances)
    min_d  = np.min(distances)
    ax.axvline(mean_d, color=ACCENT, linestyle='--', lw=1.2, alpha=0.8, label=f'Avg: {mean_d:.2f}m')
    ax.axvline(min_d,  color=LIDAR_NEAR, linestyle='--', lw=1.2, alpha=0.8, label=f'Min: {min_d:.2f}m')
    legend = ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_CLR, loc='upper right')


def render_video(csv_path: str, output_path: str, fps: int = 20, max_frames: int = None):
    """CSV 파일을 읽어 영상으로 렌더링합니다."""

    print(f"[INFO] 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)

    # 라이다 컬럼 추출
    scan_cols = [c for c in df.columns if c.startswith('lidar_')]
    n_beams   = len(scan_cols)

    print(f"[INFO] 총 프레임: {len(df)}, 라이다 빔 수: {n_beams}")

    if max_frames is not None:
        df = df.iloc[:max_frames]

    # 각도 배열 생성 (라이다는 전방 0° 기준, 시계방향 -180°~+180° 전체)
    angles_deg = np.linspace(-180, 180, n_beams, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    # 270도 FOV 마스킹 (전방 기준 -135도 ~ +135도)
    fov_mask = np.abs(angles_deg) <= 135.0
    valid_angles = angles_rad[fov_mask]

    n_frames = len(df)
    t0       = df['time'].iloc[0]

    # Figure 빌드
    fig, ax_lidar, ax_car, ax_steer, ax_speed, ax_hist = build_figure(n_beams)

    # 헤더 텍스트
    title_text = fig.text(
        0.5, 0.975,
        'F1Tenth Data Log Visualizer',
        ha='center', va='top',
        fontsize=16, fontweight='bold', color=WHITE,
        fontfamily='monospace'
    )
    info_text = fig.text(
        0.5, 0.955,
        '', ha='center', va='top',
        fontsize=9, color=TEXT_CLR, fontfamily='monospace'
    )

    def update(frame_idx):
        row       = df.iloc[frame_idx]
        timestamp = row['time']
        steering  = float(row['steer'])   # rad
        speed     = float(row['desired_speed'])      # m/s
        distances = row[scan_cols].values.astype(float)
        
        # FOV 마스킹 적용
        valid_distances = distances[fov_mask]

        # ─ 라이다 ─
        draw_lidar(ax_lidar, valid_angles, valid_distances, steering)

        # ─ 차량 박스 ─
        draw_car(ax_car, steering, speed)

        # ─ 조향 게이지 ─
        draw_steering_gauge(ax_steer, steering)

        # ─ 속도 게이지 ─
        draw_speed_gauge(ax_speed, speed)

        # ─ 거리 분포 ─
        draw_hist(ax_hist, valid_distances)

        # ─ 헤더 업데이트 ─
        elapsed = timestamp - t0
        info_text.set_text(
            f'|  Frame: {frame_idx+1:04d}/{n_frames:04d}  '
            f'|  Min dist: {np.min(valid_distances):.2f}m  '
            f'|  Beams: {len(valid_distances)}'
        )

        pct = (frame_idx + 1) / n_frames * 100
        print(f'\r[렌더링] {pct:5.1f}% ({frame_idx+1}/{n_frames})', end='', flush=True)

        return []

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False
    )

    print(f"\n[INFO] 영상 저장 중: {output_path}")
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={'title': 'F1Tenth Log Visualization'},
        bitrate=4000,
        extra_args=['-vcodec', 'libx264', '-preset', 'fast', '-crf', '20']
    )
    ani.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"\n[완료] 저장됨: {output_path}")


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1Tenth 데이터 로그 시각화')
    parser.add_argument('csv', help='입력 CSV 파일 경로')
    parser.add_argument('-o', '--output', default=None,
                        help='출력 영상 경로 (기본: <csv명>.mp4)')
    parser.add_argument('--fps', type=int, default=20,
                        help='출력 FPS (기본: 20)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='렌더링할 최대 프레임 수 (미지정 시 전체)')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[오류] 파일을 찾을 수 없습니다: {args.csv}")
        sys.exit(1)

    output = args.output or os.path.splitext(args.csv)[0] + '.mp4'
    if not output.endswith('.mp4'):
        output += '.mp4'
    
    render_video(args.csv, output, fps=args.fps, max_frames=args.max_frames)
