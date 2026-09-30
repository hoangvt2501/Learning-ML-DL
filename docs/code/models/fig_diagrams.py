"""Sinh Hình 1, 3, 6, 8, 11 và 12 của giáo trình "Học sâu": các sơ đồ.

Đây là hình vẽ tay bằng matplotlib chứ không phải kết quả thí nghiệm, nhưng vẫn để
trong mã nguồn để sửa được và để mọi chi tiết khớp với chữ trong bài.

    python code/models/fig_diagrams.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = "figs/"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

INK, DIM = "#1f1d1a", "#8a857c"
TEAL, TEAL_BG = "#1f6f68", "#e0efed"
AMBER, AMBER_BG = "#8a5300", "#fbf0db"
RED, RED_BG = "#a13328", "#fbe9e6"
BLUE, BLUE_BG = "#1a5aa8", "#dceafb"
GREEN, GREEN_BG = "#2c6b39", "#e6f2e8"
PURPLE, PURPLE_BG = "#6b3b8a", "#efe4f7"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.5, weight="normal", lw=1.1, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, p, q, color=INK, lw=1.2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color,
                                 lw=lw, linestyle=ls, shrinkA=2, shrinkB=2, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xlim=(0, 12), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


# ==================================================== Hình 1 — bản đồ họ mô hình
fig, ax = canvas(8.6, 4.0, (0, 12), (0, 7.4))
ax.text(6, 7.1, "Bốn câu hỏi thiết kế, và các họ mô hình trả lời chúng khác nhau",
        ha="center", fontsize=9.5, weight="bold", color=INK)
cols = [
    ("Mô hình tuyến tính", BLUE_BG, BLUE, 0.2,
     ["giả thiết: quan hệ tuyến tính", "tham số: ít, giải thích được",
      "mạnh: dữ liệu ít, cần diễn giải", "yếu: quan hệ phi tuyến"]),
    ("Cây và tập hợp", GREEN_BG, GREEN, 3.15,
     ["giả thiết: chia trục toạ độ", "tham số: nhiều, phi tham số",
      "mạnh: dữ liệu bảng, đặc trưng lẫn lộn", "yếu: ngoại suy, dữ liệu thô"]),
    ("Mạng nơ-ron", AMBER_BG, AMBER, 6.1,
     ["giả thiết: hợp thành nhiều tầng", "tham số: rất nhiều",
      "mạnh: dữ liệu thô có cấu trúc", "yếu: cần nhiều dữ liệu"]),
    ("Transformer", PURPLE_BG, PURPLE, 9.05,
     ["giả thiết: quan hệ cặp giữa vị trí", "tham số: rất nhiều, song song tốt",
      "mạnh: chuỗi dài, mọi phương thức", "yếu: chi phí bậc hai theo độ dài"]),
]
for name, fc, ec, x, items in cols:
    box(ax, x, 0.5, 2.75, 5.9, "", fc=fc, ec=ec, lw=1.3)
    ax.text(x + 1.375, 5.95, name, ha="center", fontsize=9, weight="bold", color=ec)
    for i, it in enumerate(items):
        ax.text(x + 0.18, 5.25 - i * 1.28, "•", fontsize=9, color=ec, va="top")
        ax.text(x + 0.45, 5.25 - i * 1.28, it, fontsize=7.8, va="top", color=INK,
                linespacing=1.4, wrap=True)
for x in (2.98, 5.93, 8.88):
    arrow(ax, (x, 3.4), (x + 0.15, 3.4), color=DIM)
ax.text(6, 0.1, "Trục chung: càng sang phải thì càng ít giả thiết về dạng hàm, "
                "càng cần nhiều dữ liệu và càng khó diễn giải.",
        ha="center", fontsize=8, color=DIM)
fig.tight_layout(); fig.savefig(OUT + "models01_families.png"); plt.close(fig)


# ============================================ Hình 3 — cây chia không gian
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
rng = np.random.default_rng(4)
X = rng.uniform(0, 1, (260, 2))
y = ((X[:, 0] > 0.45) & (X[:, 1] > 0.55)) | ((X[:, 0] <= 0.45) & (X[:, 1] > 0.25))
ax = axes[0]
ax.scatter(X[~y, 0], X[~y, 1], s=11, color=RED, alpha=.7, label="lớp 0")
ax.scatter(X[y, 0], X[y, 1], s=11, color=TEAL, alpha=.7, label="lớp 1")
ax.axvline(0.45, color=INK, lw=1.4)
ax.plot([0, 0.45], [0.25, 0.25], color=INK, lw=1.4)
ax.plot([0.45, 1], [0.55, 0.55], color=INK, lw=1.4)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Cây chia không gian bằng các nhát cắt\nvuông góc với trục", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="lower left")

ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
box(ax, 3.4, 8.2, 3.2, 1.2, "$x_1 \\leq 0{,}45$ ?", fc=BLUE_BG, ec=BLUE, fs=9)
box(ax, 0.6, 5.4, 3.0, 1.2, "$x_2 \\leq 0{,}25$ ?", fc=BLUE_BG, ec=BLUE, fs=9)
box(ax, 6.4, 5.4, 3.0, 1.2, "$x_2 \\leq 0{,}55$ ?", fc=BLUE_BG, ec=BLUE, fs=9)
for x, lab, col, bg in [(0.2, "lớp 0", RED, RED_BG), (2.4, "lớp 1", TEAL, TEAL_BG),
                        (5.9, "lớp 0", RED, RED_BG), (8.1, "lớp 1", TEAL, TEAL_BG)]:
    box(ax, x, 2.9, 1.8, 1.0, lab, fc=bg, ec=col, fs=8.5)
for a, b, lab in [((4.4, 8.2), (2.1, 6.6), "đúng"), ((5.6, 8.2), (7.9, 6.6), "sai"),
                  ((1.4, 5.4), (1.1, 3.9), "đúng"), ((2.8, 5.4), (3.3, 3.9), "sai"),
                  ((7.2, 5.4), (6.8, 3.9), "đúng"), ((8.6, 5.4), (9.0, 3.9), "sai")]:
    arrow(ax, a, b, color=DIM)
    ax.text((a[0] + b[0]) / 2 - 0.35, (a[1] + b[1]) / 2, lab, fontsize=7, color=DIM)
ax.set_title("Cùng một mô hình, viết dưới dạng cây", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "models03_tree.png"); plt.close(fig)


# =============================== Hình 6 — đồ thị tính toán và lan truyền ngược
fig, ax = canvas(8.6, 3.4, (0, 12), (0, 6.2))
ax.text(6, 5.9, "Lan truyền ngược = quy tắc dây chuyền áp dụng theo thứ tự ngược",
        ha="center", fontsize=9.5, weight="bold", color=INK)
nodes = [("$x$", 0.3), ("$z_1 = W_1 x$", 2.0), ("$a_1 = \\sigma(z_1)$", 4.5),
         ("$z_2 = W_2 a_1$", 7.0), ("$\\hat{y}$", 9.5), ("$L$", 11.0)]
for t, x in nodes:
    w = 2.1 if len(t) > 6 else 1.0
    box(ax, x, 3.3, w, 1.0, t, fc=TEAL_BG, ec=TEAL, fs=8.5)
xs = [0.3 + 1.0, 2.0 + 2.1, 4.5 + 2.1, 7.0 + 2.1, 9.5 + 1.0]
tos = [2.0, 4.5, 7.0, 9.5, 11.0]
for a, b in zip(xs, tos):
    arrow(ax, (a, 3.8), (b, 3.8), color=TEAL, lw=1.4)
ax.text(0.3, 4.6, "lượt xuôi: tính và GHI LẠI mọi giá trị trung gian", fontsize=8, color=TEAL)

for t, x, w in [("$\\frac{\\partial L}{\\partial x}$", 0.3, 1.0),
                ("$\\frac{\\partial L}{\\partial z_1}$", 2.0, 2.1),
                ("$\\frac{\\partial L}{\\partial a_1}$", 4.5, 2.1),
                ("$\\frac{\\partial L}{\\partial z_2}$", 7.0, 2.1),
                ("$\\frac{\\partial L}{\\partial \\hat{y}}$", 9.5, 1.0)]:
    box(ax, x, 1.2, w, 1.0, t, fc=RED_BG, ec=RED, fs=8.5)
for a, b in zip([2.0, 4.5, 7.0, 9.5], [0.3 + 1.0, 2.0 + 2.1, 4.5 + 2.1, 7.0 + 2.1]):
    arrow(ax, (a, 1.7), (b, 1.7), color=RED, lw=1.4)
arrow(ax, (11.0, 3.3), (10.5, 2.2), color=RED, lw=1.4, rad=-0.2)
ax.text(0.3, 0.55, "lượt ngược: nhân dần các đạo hàm địa phương, "
                   "dùng lại đúng những giá trị đã ghi ở lượt xuôi", fontsize=8, color=RED)
fig.tight_layout(); fig.savefig(OUT + "models06_backprop.png"); plt.close(fig)


# ===================================== Hình 8 — tích chập và trường tiếp nhận
fig, ax = canvas(8.6, 3.4, (0, 12), (0, 6.4))
ax.text(6, 6.1, "Mỗi lớp tích chập 3×3 nới trường tiếp nhận thêm 2 ô",
        ha="center", fontsize=9.5, weight="bold", color=INK)
for li, (x0, rf, col, bg) in enumerate([(0.3, 1, BLUE, BLUE_BG), (4.2, 3, GREEN, GREEN_BG),
                                        (8.1, 5, AMBER, AMBER_BG)]):
    n = 7
    cell = 0.42
    ox, oy = x0 + 0.35, 1.5
    for i in range(n):
        for j in range(n):
            inside = abs(i - 3) <= rf // 2 and abs(j - 3) <= rf // 2
            ax.add_patch(Rectangle((ox + j * cell, oy + i * cell), cell * .92, cell * .92,
                                   fc=bg if inside else "white", ec=DIM, lw=.6, zorder=2))
    ax.add_patch(Rectangle((ox + 3 * cell, oy + 3 * cell), cell * .92, cell * .92,
                           fc=col, ec=col, lw=.6, zorder=3))
    ax.text(x0 + 1.8, oy - 0.45, f"sau {li} lớp: trường tiếp nhận {rf}×{rf}",
            ha="center", fontsize=8.2, color=col, weight="bold")
    ax.text(x0 + 1.8, oy + n * cell + 0.25, ["đầu vào", "một lớp 3×3", "hai lớp 3×3"][li],
            ha="center", fontsize=8.5, color=INK)
for x in (3.85, 7.75):
    arrow(ax, (x, 3.0), (x + 0.3, 3.0), color=DIM)
ax.text(6, 0.45, "Trường tiếp nhận lý thuyết tăng TUYẾN TÍNH theo số lớp: $r_L = 1 + L(k-1)$ với bước nhảy 1.\n"
                 "Muốn nó tăng nhanh hơn thì phải giảm độ phân giải (bước nhảy > 1, gộp) hoặc giãn hạt nhân.",
        ha="center", fontsize=8, color=DIM, linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "models08_cnn.png"); plt.close(fig)


# ======================================= Hình 11 — một khối Transformer đầy đủ
fig, ax = canvas(7.6, 6.4, (0, 12), (0, 14.0))
ax.text(5.6, 13.65, "Một khối Transformer: hai khối con, mỗi khối con có kết nối tắt",
        ha="center", fontsize=9.5, weight="bold", color=INK)

CX, CW = 2.4, 6.6            # cot chinh
RES = 9.9                     # duong tat ben phai
mid = CX + CW / 2

def up(y0, y1, color=INK, lw=1.3, x=None):
    arrow(ax, (x if x else mid, y0), (x if x else mid, y1), color=color, lw=lw)

# --- dau vao
box(ax, CX + 1.5, 0.3, 3.6, 0.8, "$x \\in \\mathbb{R}^{T \\times d}$", fc="white", ec=DIM, fs=9)

# --- khoi con 1 (doc tu duoi len)
box(ax, CX - 0.45, 1.9, CW + 0.9, 5.3, "", fc=TEAL_BG, ec=TEAL, lw=1.3)
ax.text(CX - 0.25, 6.95, "Khối con 1 — trộn thông tin GIỮA các vị trí",
        fontsize=8.2, color=TEAL, weight="bold")
box(ax, CX + 1.9, 2.15, 2.8, 0.7, "LayerNorm", fc="white", ec=TEAL, fs=8.2)
for lab, x in [("$W_Q$", CX + 0.1), ("$W_K$", CX + 2.4), ("$W_V$", CX + 4.7)]:
    box(ax, x, 3.45, 1.8, 0.65, lab, fc="white", ec=TEAL, fs=8.5)
    arrow(ax, (mid, 2.85), (x + 0.9, 3.45), color=TEAL, rad=-0.06)
    up(4.1, 4.55, color=TEAL, x=x + 0.9)
box(ax, CX, 4.55, CW, 0.95,
    "softmax$\\left(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\right)V$   ·   $h$ đầu song song",
    fc="white", ec=TEAL, fs=8.5)
box(ax, CX + 1.9, 6.0, 2.8, 0.7, "$W_O$", fc="white", ec=TEAL, fs=8.5)
up(5.5, 6.0, color=TEAL)
up(1.1, 2.15, color=DIM)                      # x -> LayerNorm

# --- cong tat 1
ax.add_patch(Circle((mid, 8.0), 0.34, fc="white", ec=INK, lw=1.3, zorder=5))
ax.text(mid, 8.0, "+", ha="center", va="center", fontsize=13, color=INK, zorder=6)
up(6.7, 7.66, color=TEAL)

arrow(ax, (mid + 1.9, 0.7), (RES, 0.7), color=DIM, ls=(0, (4, 2)))
arrow(ax, (RES, 0.7), (RES, 8.0), color=DIM, ls=(0, (4, 2)))
arrow(ax, (RES, 8.0), (mid + 0.36, 8.0), color=DIM)

# --- khoi con 2 (doc tu duoi len)
box(ax, CX - 0.45, 9.1, CW + 0.9, 2.9, "", fc=AMBER_BG, ec=AMBER, lw=1.3)
ax.text(CX - 0.25, 11.75, "Khối con 2 — biến đổi TỪNG vị trí, độc lập nhau",
        fontsize=8.2, color=AMBER, weight="bold")
box(ax, CX + 1.9, 9.35, 2.8, 0.7, "LayerNorm", fc="white", ec=AMBER, fs=8.2)
box(ax, CX, 10.45, CW, 0.95,
    "$W_2\\,\\phi(W_1 x)$   ·   $d_{\\mathrm{ff}} = 4d$   ·   $\\phi$ = ReLU / GELU / SwiGLU",
    fc="white", ec=AMBER, fs=8.2)
up(10.05, 10.45, color=AMBER)
up(8.34, 9.35, color=DIM)

# --- cong tat 2
ax.add_patch(Circle((mid, 12.6), 0.34, fc="white", ec=INK, lw=1.3, zorder=5))
ax.text(mid, 12.6, "+", ha="center", va="center", fontsize=13, color=INK, zorder=6)
up(11.4, 12.26, color=AMBER)
arrow(ax, (mid + 1.2, 8.55), (RES, 8.55), color=DIM, ls=(0, (4, 2)))
arrow(ax, (RES, 8.55), (RES, 12.6), color=DIM, ls=(0, (4, 2)))
arrow(ax, (RES, 12.6), (mid + 0.36, 12.6), color=DIM)
ax.text(RES + 0.2, 6.0, "kết nối tắt", fontsize=7.8, color=DIM, rotation=90, va="center")

box(ax, CX + 1.5, 12.95, 3.6, 0.55, "sang khối tiếp theo", fc="white", ec=DIM, fs=8.2)
up(12.94, 12.95, color=INK)

ax.text(0.05, 3.6,
        "Thứ tự ở đây\nlà pre-LN:\nchuẩn hoá TRƯỚC\nkhối con, nên\nđường tắt đi\nthẳng từ đầu vào\ntới đầu ra mà\nkhông qua phép\nchuẩn hoá nào.",
        fontsize=7.4, color=INK, va="center", linespacing=1.7)
fig.tight_layout(); fig.savefig(OUT + "models11_block.png"); plt.close(fig)


# ============================ Hình 14 — Transformer 2017 so với Transformer nay
fig, ax = canvas(8.6, 4.2, (0, 12), (0, 8.0))
ax.text(6, 7.7, "Cùng một bộ khung, năm chi tiết đã đổi", ha="center",
        fontsize=9.5, weight="bold", color=INK)
rows = [
    ("Vị trí trong khối", "Post-LN: $x + \\mathrm{Sublayer}(x)$ rồi mới LN",
     "Pre-LN: LN trước, tắt đi thẳng", "huấn luyện ổn định hơn, ít cần warmup"),
    ("Chuẩn hoá", "LayerNorm (trừ trung bình, chia độ lệch chuẩn)",
     "RMSNorm (chỉ chia chuẩn bậc hai)", "rẻ hơn, gần như không mất chất lượng"),
    ("Vị trí token", "Sin/cos cộng vào embedding",
     "RoPE: xoay $Q$, $K$ theo vị trí", "ngoại suy độ dài tốt hơn, quan hệ tương đối"),
    ("Đầu attention", "MHA: mỗi đầu một cặp $K$, $V$",
     "GQA/MQA: nhiều đầu dùng chung $K$, $V$", "KV cache nhỏ đi theo đúng tỉ lệ nhóm"),
    ("Hàm kích hoạt FFN", "ReLU, $d_{\\mathrm{ff}} = 4d$",
     "SwiGLU, $d_{\\mathrm{ff}} \\approx \\frac{2}{3}\\cdot 4d$", "ba ma trận thay vì hai, bù bằng bề rộng nhỏ hơn"),
]
box(ax, 0.2, 6.3, 2.6, 0.85, "Thành phần", fc="#f4f2ec", ec=DIM, fs=8.5, weight="bold")
box(ax, 2.9, 6.3, 3.4, 0.85, "Transformer 2017", fc=RED_BG, ec=RED, fs=8.5, weight="bold")
box(ax, 6.4, 6.3, 3.4, 0.85, "Phổ biến hiện nay", fc=GREEN_BG, ec=GREEN, fs=8.5, weight="bold")
box(ax, 9.9, 6.3, 1.9, 0.85, "Vì sao đổi", fc="#f4f2ec", ec=DIM, fs=8.5, weight="bold")
for i, (comp, old, new, why) in enumerate(rows):
    y = 5.1 - i * 1.12
    box(ax, 0.2, y, 2.6, 1.0, comp, fc="white", ec=DIM, fs=7.8)
    box(ax, 2.9, y, 3.4, 1.0, old, fc=RED_BG, ec=RED, fs=7.6)
    box(ax, 6.4, y, 3.4, 1.0, new, fc=GREEN_BG, ec=GREEN, fs=7.6)
    box(ax, 9.9, y, 1.9, 1.0, why, fc="white", ec=DIM, fs=7.0)
fig.tight_layout(); fig.savefig(OUT + "models14_modern.png"); plt.close(fig)

print("Đã lưu Hình 1, 3, 6, 8, 11 và 12 vào " + OUT)
