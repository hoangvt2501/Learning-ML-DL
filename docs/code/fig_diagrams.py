"""Sơ đồ khối (vẽ bằng matplotlib để không phụ thuộc công cụ ngoài)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "savefig.dpi": 110, "savefig.bbox": "tight"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs") + os.sep
os.makedirs(OUT, exist_ok=True)

def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#4a78c2", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(ax, p, q, text=None, color="#333", off=(0, 0.12), fs=8, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", fontsize=fs, color=color)

# ------------------------------------------------------------ Hình 13: một lớp Linear/Conv chạy toàn số nguyên
fig, ax = plt.subplots(figsize=(12, 4.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off")
G, O, B = ("#eaf6ee", "#4caf6e"), ("#fdf1e4", "#e0923a"), ("#eef3fb", "#4a78c2")
box(ax, 1.0, 3.2, 1.6, 0.7, "q_x\nuint8", *G)
box(ax, 1.0, 1.6, 1.6, 0.7, "q_w\nint8 (Z_w = 0)", *G)
box(ax, 3.4, 2.4, 2.0, 1.0, "Nhân int8 × int8\ncộng dồn int32\nΣ q_w·(q_x − Z_x)", *B)
box(ax, 5.9, 2.4, 1.7, 0.8, "+ q_b\n(bias int32)", *B)
box(ax, 8.2, 2.4, 2.2, 1.0, "Requantize\n(acc · M₀) >> (31 + n)\n[số nguyên 64 bit]", *B)
box(ax, 10.6, 2.4, 1.6, 0.8, "+ Z_y, clamp\n[0, 255]", *B)
box(ax, 10.6, 0.8, 1.6, 0.7, "q_y  uint8\n→ lớp tiếp theo", *G)
arrow(ax, (1.8, 3.2), (2.4, 2.7)); arrow(ax, (1.8, 1.6), (2.4, 2.1))
arrow(ax, (4.4, 2.4), (5.05, 2.4), "int32"); arrow(ax, (6.75, 2.4), (7.1, 2.4), "int32")
arrow(ax, (9.3, 2.4), (9.8, 2.4)); arrow(ax, (10.6, 2.0), (10.6, 1.15))
box(ax, 5.9, 4.0, 9.0, 0.55, "Tính TRƯỚC khi convert (offline):   Z_x·Σ q_w ,   q_b = round(b / (S_w·S_x)) ,   M = S_w·S_x / S_y = M₀·2^−(31+n)", *O, fs=8.5)
ax.text(6, 0.25, "Toàn bộ đường đi chỉ dùng số nguyên → chạy được trên NPU / DSP / vi điều khiển không có FPU.",
        ha="center", fontsize=9, style="italic", color="#444")
fig.savefig(OUT + "fig09_integer_pipeline.png"); plt.close(fig)

# ------------------------------------------------------------ Hình 16: đồ thị có nút fake-quant (dùng cho PTQ mô phỏng & QAT)
fig, ax = plt.subplots(figsize=(12, 2.9)); ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
FQ = ("#fdecea", "#d9534f")
box(ax, 0.6, 1.3, 0.9, 0.6, "x (FP32)", *G)
box(ax, 1.9, 1.3, 1.1, 0.6, "FakeQuant\n(act, 8 bit)", *FQ, fs=8)
box(ax, 3.5, 2.5, 1.0, 0.5, "W (FP32)", *G); box(ax, 3.5, 1.75, 1.1, 0.5, "FakeQuant\n(weight)", *FQ, fs=8)
box(ax, 5.0, 1.3, 1.6, 0.7, "Conv + BN đã gộp\n(tính bằng FP32)", *B)
box(ax, 6.7, 1.3, 0.9, 0.6, "ReLU", *B)
box(ax, 8.1, 1.3, 1.1, 0.6, "FakeQuant\n(act)", *FQ, fs=8)
box(ax, 9.6, 1.3, 1.1, 0.6, "Lớp kế tiếp …", *B)
arrow(ax, (1.05, 1.3), (1.35, 1.3)); arrow(ax, (2.45, 1.3), (4.2, 1.3))
arrow(ax, (3.5, 2.25), (3.5, 2.0)); arrow(ax, (3.5, 1.5), (4.3, 1.45))
arrow(ax, (5.8, 1.3), (6.25, 1.3)); arrow(ax, (7.15, 1.3), (7.55, 1.3)); arrow(ax, (8.65, 1.3), (9.05, 1.3))
ax.text(6, 0.3, "FakeQuant(x) = S·(clamp(round(x/S) + Z, q_min, q_max) − Z): giá trị vẫn là FP32 nhưng chỉ nhận các mức lượng tử.\n"
        "Lượt ngược dùng STE. Khi convert, mỗi cặp FakeQuant được thay bằng phép quantize / dequantize thật.",
        ha="center", fontsize=8.5, color="#444")
fig.savefig(OUT + "fig10_fake_quant.png"); plt.close(fig)

# ------------------------------------------------------------ Hình 17: quy trình chọn phương pháp
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
Y_ = ("#fff8e1", "#c9a227")
box(ax, 5.5, 5.5, 4.4, 0.6, "Mô hình FP32 đã huấn luyện + baseline accuracy", *G, bold=True)
box(ax, 5.5, 4.5, 4.6, 0.65, "Phần cứng đích hỗ trợ kiểu gì?\n(GPU: FP16/BF16/FP8/INT8; CPU/NPU/MCU: INT8)", *Y_, fs=8.5)
box(ax, 1.7, 3.4, 3.1, 0.7, "Chỉ cần nhẹ/nhanh vừa phải:\nFP16 / BF16 (gần như không mất acc)", *B, fs=8.5)
box(ax, 5.5, 3.4, 2.8, 0.7, "Weight-only INT8 / INT4\nhoặc dynamic quantization\n(không cần dữ liệu calibration)", *B, fs=8.5)
box(ax, 9.3, 3.4, 2.6, 0.7, "Static full-integer INT8\n+ calibration 100–1000 mẫu", *B, fs=8.5)
box(ax, 7.4, 2.1, 3.4, 0.75, "Acc giảm quá ngưỡng cho phép?\n→ per-channel, đổi calibration,\nCLE / bias correction / AdaRound", *Y_, fs=8.5)
box(ax, 7.4, 0.8, 3.4, 0.75, "Vẫn chưa đạt?\n→ mixed precision cho lớp nhạy cảm\n→ QAT (fine-tune có fake-quant)", *FQ, fs=8.5)
box(ax, 2.6, 1.4, 3.0, 0.8, "Đo lại TRÊN THIẾT BỊ ĐÍCH:\nkích thước, độ trễ, độ chính xác", *G, fs=8.5, bold=True)
arrow(ax, (5.5, 5.2), (5.5, 4.8))
arrow(ax, (4.2, 4.3), (2.4, 3.75)); arrow(ax, (5.5, 4.2), (5.5, 3.75)); arrow(ax, (6.8, 4.3), (8.7, 3.75))
arrow(ax, (9.3, 3.05), (8.4, 2.45)); arrow(ax, (5.9, 3.05), (6.6, 2.45))
arrow(ax, (7.4, 1.72), (7.4, 1.18), "có", off=(0.25, -0.05))
arrow(ax, (5.7, 2.1), (4.1, 1.6), "không", off=(0, 0.1))
arrow(ax, (1.6, 3.05), (2.2, 1.8)); arrow(ax, (5.7, 0.8), (4.1, 1.2))
fig.savefig(OUT + "fig17_decision.png"); plt.close(fig)
print("ok")
