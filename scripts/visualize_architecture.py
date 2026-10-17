"""
scripts/visualize_architecture.py
Generates high-resolution 3D volumetric tensor block diagrams for Get-Thermal-Jetson-nano:
1. docs/figures/1_camera_acquisition_3d.png
2. docs/figures/2_radiometric_matrix_decode_3d.png
3. docs/figures/3_jetson_to_ai_service_3d.png

Canvas: 16x10 inches, 300 DPI.
Isometric 3D volumetric tensor blocks focusing on Sensing & Radiometric Extraction.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THEMES = {
    "camera": ["#0284C7", "#38BDF8", "#0369A1"],      # Cyan / Sky
    "preprocess": ["#475569", "#94A3B8", "#334155"],  # Slate Gray
    "matrix": ["#059669", "#34D399", "#047857"],      # Emerald Green
    "backend": ["#4F46E5", "#818CF8", "#3730A3"],     # Indigo
    "dark_bg": "#0B1120",
    "card_bg": "#1E293B",
    "card_border": "#334155",
    "text_white": "#F8FAFC",
    "text_muted": "#94A3B8",
    "text_yellow": "#FEF08A"
}

def draw_3d_block(ax, x, y, w, h, depth=0.35, theme="camera", label="", sublabel="", tensor_shape="", desc=""):
    front_c, top_c, side_c = THEMES[theme]
    dx = depth * 0.7
    dy = depth * 0.7

    # 1. Right Side facet
    side_pts = [[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]]
    ax.add_patch(Polygon(side_pts, closed=True, facecolor=side_c, edgecolor="#0F172A", lw=1.2, zorder=2))

    # 2. Top facet
    top_pts = [[x, y + h], [x + dx, y + h + dy], [x + w + dx, y + h + dy], [x + w, y + h]]
    ax.add_patch(Polygon(top_pts, closed=True, facecolor=top_c, edgecolor="#0F172A", lw=1.2, zorder=2))

    # 3. Front facet
    front_pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    ax.add_patch(Polygon(front_pts, closed=True, facecolor=front_c, edgecolor="#0F172A", lw=1.2, zorder=3))

    cx = x + w / 2.0
    cy = y + h / 2.0

    if label:
        ax.text(cx, cy + h * 0.22, label, ha="center", va="center", color=THEMES["text_white"], fontsize=9.5, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(cx, cy + h * 0.02, sublabel, ha="center", va="center", color="#F1F5F9", fontsize=8.0, zorder=4)
    if tensor_shape:
        ax.text(cx, cy - h * 0.22, tensor_shape, ha="center", va="center", color=THEMES["text_yellow"], fontsize=8.0, fontweight="bold", family="monospace", zorder=4)
    if desc:
        ax.text(cx + dx/2, y - 0.35, desc, ha="center", va="top", color=THEMES["text_muted"], fontsize=7.8, zorder=4)

def draw_3d_arrow(ax, x1, y1, x2, y2, color="#38BDF8", text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=color, edgecolor=color, width=1.8, headwidth=6, headlength=7, shrink=0.06), zorder=5)
    if text:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.25, text, ha="center", va="center", color="#FFFFFF", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=THEMES["card_bg"], edgecolor=THEMES["card_border"], alpha=0.95), zorder=6)


# Figure 1: Camera Acquisition Flow
def generate_fig1():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor(THEMES["dark_bg"])
    ax.set_facecolor(THEMES["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    ax.text(8.0, 8.8, "Hikvision ISAPI Radiometric Camera Acquisition Flow", ha="center", va="center",
            color=THEMES["text_white"], fontsize=18, fontweight="bold")
    ax.text(8.0, 8.35, "Physical Sensor Handshake -> HTTP Digest Request -> Multipart MIME Splitting -> Raw Buffer Extraction",
            ha="center", va="center", color=THEMES["text_muted"], fontsize=10.5)

    stages = [
        {"x": 0.5, "w": 2.5, "h": 2.8, "theme": "camera", "label": "Hikvision Sensor", "sub": "Bi-Spectrum CH2", "shape": "LWIR Sensor", "desc": "IP: 192.168.10.152\nUncompressed Stream"},
        {"x": 4.0, "w": 2.6, "h": 2.8, "theme": "preprocess", "label": "HTTP Digest Auth", "sub": "ISAPI Client", "shape": "Port 80 TCP", "desc": "Encrypted handshake\nGET /jpegPicWithAppendData"},
        {"x": 7.6, "w": 2.6, "h": 2.8, "theme": "preprocess", "label": "MIME Demuxer", "sub": "parse_multipart", "shape": "--boundary", "desc": "Separates JSON header\nfrom raw binary stream"},
        {"x": 11.2, "w": 2.5, "h": 2.8, "theme": "matrix", "label": "Binary Buffer", "sub": "raw_data", "shape": "p2pDataLen Bytes", "desc": "Little-endian float32\nPacked radiometric payload"},
        {"x": 14.5, "w": 1.6, "h": 2.8, "theme": "matrix", "label": "Output", "sub": "To Matrix Decoder", "shape": "Raw Bytes", "desc": "In-Memory Buffer"}
    ]

    for s in stages:
        draw_3d_block(ax, s["x"], 3.2, s["w"], s["h"], theme=s["theme"],
                      label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], desc=s["desc"])

    draw_3d_arrow(ax, 3.0, 4.6, 4.0, 4.6)
    draw_3d_arrow(ax, 6.6, 4.6, 7.6, 4.6)
    draw_3d_arrow(ax, 10.2, 4.6, 11.2, 4.6)
    draw_3d_arrow(ax, 13.7, 4.6, 14.5, 4.6)

    fig.savefig(OUTPUT_DIR / "1_camera_acquisition_3d.png", dpi=300)
    plt.close(fig)
    print("Get-Thermal-Jetson-nano: Generated 1_camera_acquisition_3d.png")


# Figure 2: Radiometric Matrix Decode
def generate_fig2():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor(THEMES["dark_bg"])
    ax.set_facecolor(THEMES["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    ax.text(8.0, 8.8, "Radiometric 384x288 Float32 Matrix Transformation", ha="center", va="center",
            color=THEMES["text_white"], fontsize=18, fontweight="bold")
    ax.text(8.0, 8.35, "Binary Byte Stream -> Float32 Array -> (288, 384) Spatial Grid -> Normalized Coordinate Mapping",
            ha="center", va="center", color=THEMES["text_muted"], fontsize=10.5)

    stages = [
        {"x": 0.6, "w": 2.6, "h": 2.8, "theme": "preprocess", "label": "P2P Byte Stream", "sub": "raw_data", "shape": "110,592 Floats", "desc": "384 x 288 x 4 bytes\n442,368 bytes chunk"},
        {"x": 4.2, "w": 2.8, "h": 2.8, "theme": "matrix", "label": "np.frombuffer", "sub": "dtype=np.float32", "shape": "[110592] f32", "desc": "Direct memory casting\nNo quantization loss"},
        {"x": 8.0, "w": 2.8, "h": 2.8, "theme": "matrix", "label": "Spatial Reshape", "sub": "reshape(288, 384)", "shape": "(288, 384) f32", "desc": "Height=288, Width=384\nTrue Celsius grid"},
        {"x": 11.8, "w": 2.8, "h": 2.8, "theme": "camera", "label": "Coordinate Query", "sub": "(x_norm * 384)", "shape": "Points & Zones", "desc": "Look up spot value\nMatrix[my, mx] in Celsius"}
    ]

    for s in stages:
        draw_3d_block(ax, s["x"], 3.2, s["w"], s["h"], theme=s["theme"],
                      label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], desc=s["desc"])

    draw_3d_arrow(ax, 3.2, 4.6, 4.2, 4.6)
    draw_3d_arrow(ax, 7.0, 4.6, 8.0, 4.6)
    draw_3d_arrow(ax, 10.8, 4.6, 11.8, 4.6)

    fig.savefig(OUTPUT_DIR / "2_radiometric_matrix_decode_3d.png", dpi=300)
    plt.close(fig)
    print("Get-Thermal-Jetson-nano: Generated 2_radiometric_matrix_decode_3d.png")


# Figure 3: Jetson to AI Service Relay
def generate_fig3():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor(THEMES["dark_bg"])
    ax.set_facecolor(THEMES["dark_bg"])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    ax.text(8.0, 8.8, "Edge Sensor Data Delivery to Downstream AI Service", ha="center", va="center",
            color=THEMES["text_white"], fontsize=18, fontweight="bold")
    ax.text(8.0, 8.35, "Get-Thermal-Jetson-nano Extraction -> Local IPC / HTTP Ingestion -> AI-Thermal-detect Forecasting Service",
            ha="center", va="center", color=THEMES["text_muted"], fontsize=10.5)

    stages = [
        {"x": 0.5, "w": 3.0, "h": 2.8, "theme": "matrix", "label": "Thermal Sensor Daemon", "sub": "Get-Thermal-Jetson", "shape": "Extraction Layer", "desc": "Continuous 10Hz sampling\nLocal thread coordination"},
        {"x": 4.5, "w": 3.0, "h": 2.8, "theme": "preprocess", "label": "JSON Payload Formatter", "sub": "thermal_sender.py", "shape": "Structured Telemetry", "desc": "Timestamp + Array of Points\n[{id: 'P1', temp: 42.5}, ...]"},
        {"x": 8.5, "w": 3.2, "h": 2.8, "theme": "backend", "label": "FastAPI Receiver", "sub": "AI-Thermal-detect", "shape": "Port 8080 TCP", "desc": "POST /api/thermal-data\nSchema check & CSV append"},
        {"x": 12.7, "w": 3.0, "h": 2.8, "theme": "matrix", "label": "Forecasting Pipeline", "sub": "LinearReg / MTKAN", "shape": "5-Min Ahead Model", "desc": "Feeds sliding window buffer\nfor future trend inference"}
    ]

    for s in stages:
        draw_3d_block(ax, s["x"], 3.2, s["w"], s["h"], theme=s["theme"],
                      label=s["label"], sublabel=s["sub"], tensor_shape=s["shape"], desc=s["desc"])

    draw_3d_arrow(ax, 3.5, 4.6, 4.5, 4.6)
    draw_3d_arrow(ax, 7.5, 4.6, 8.5, 4.6, text="HTTP POST")
    draw_3d_arrow(ax, 11.7, 4.6, 12.7, 4.6)

    fig.savefig(OUTPUT_DIR / "3_jetson_to_ai_service_3d.png", dpi=300)
    plt.close(fig)
    print("Get-Thermal-Jetson-nano: Generated 3_jetson_to_ai_service_3d.png")

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    print("Get-Thermal-Jetson-nano: All 3 3D figures generated successfully.")
