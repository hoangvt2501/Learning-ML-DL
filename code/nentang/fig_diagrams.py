"""Các sơ đồ khái niệm cho giáo trình "Nền tảng học máy".

Đây là hình vẽ tay bằng matplotlib chứ không phải kết quả thí nghiệm, để trong mã
nguồn cho sửa được và cho khớp với chữ trong bài.

    python code/nentang/fig_diagrams.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figs/"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"
C_SOFT, C_LINE = "#eef4f3", "#cfd8d6"


def hop(ax, x, y, w, h, text, fc=C_SOFT, ec=C_MAIN, fs=8.5, weight="normal", tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, linespacing=1.45)


def mui_ten(ax, p0, p1, color=C_DIM, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 color=color, linewidth=lw,
                                 shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------------
# Hình 1 — Ba thành phần của mọi thuật toán học có giám sát
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(.5, .955, "Các thành phần của một thuật toán học có giám sát",
        ha="center", fontsize=11.5, weight="bold", color="#1a1a1a")

hop(ax, .04, .70, .27, .16,
    "1. MÔ HÌNH\n$f_\\theta(x)$\nhọ hàm được phép chọn", fs=8.5, weight="bold")
hop(ax, .365, .70, .27, .16,
    "2. HÀM MẤT MÁT\n$L(f_\\theta(x),\\, y)$\nđo mức sai của dự đoán", fs=8.5, weight="bold")
hop(ax, .69, .70, .27, .16,
    "3. THUẬT TOÁN TỐI ƯU\n$\\min_\\theta \\sum_i L$\ncách tìm $\\theta$", fs=8.5, weight="bold")

vd = [("Hồi quy tuyến tính", "$w^\\top x + b$", "bình phương sai số",
       "phương trình chuẩn\nhoặc gradient descent"),
      ("Hồi quy logistic", "$\\sigma(w^\\top x + b)$", "cross-entropy", "gradient descent"),
      ("SVM lề mềm", "$w^\\top x + b$", "hinge $+\\;\\lambda\\|w\\|^2$",
       "quy hoạch toàn phương\n(bài toán đối ngẫu)"),
      ("Cây quyết định", "hàm hằng từng khúc", "Gini / entropy",
       "tham lam, chia từng nút"),
      ("Mạng nơ-ron", "hợp nhiều lớp affine\nvà phi tuyến", "tuỳ bài toán",
       "SGD + lan truyền ngược")]

y0, dy = .575, .102
ax.text(.04, y0 + .055, "Năm thuật toán, mô tả theo ba thành phần:", fontsize=9,
        color=C_DIM, style="italic")
for i, (ten, mh, mm, tu) in enumerate(vd):
    y = y0 - i * dy
    ax.add_patch(FancyBboxPatch((.04, y - .045), .92, .088,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor="#fbfbfa" if i % 2 else "#f4f6f5",
                                edgecolor=C_LINE, linewidth=.7))
    ax.text(.065, y, ten, fontsize=8.6, weight="bold", va="center", color=C_MAIN)
    ax.text(.30, y, mh, fontsize=8.2, va="center", ha="center")
    ax.text(.545, y, mm, fontsize=8.2, va="center", ha="center")
    ax.text(.82, y, tu, fontsize=8.2, va="center", ha="center", linespacing=1.3)

for x in (.175, .50, .825):
    mui_ten(ax, (x, .695), (x, .645), color=C_LINE, lw=1.0)

ax.text(.5, .022,
        "Thay một thành phần là được một thuật toán khác.",
        ha="center", fontsize=8.6, color=C_DIM, style="italic", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "nt01_bando.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 5 — Bốn hàm mất mát cho phân loại: chúng đều thay cho mất mát 0–1
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3))
m = np.linspace(-3, 3, 600)

loss = [("mất mát 0–1", np.where(m <= 0, 1.0, 0.0), "k", 2.0, "-"),
        ("perceptron: $\\max(0, -m)$", np.maximum(0, -m), C_WARN, 1.6, "-"),
        ("hinge (SVM): $\\max(0, 1-m)$", np.maximum(0, 1 - m), C_BAD, 1.6, "-"),
        ("logistic: $\\log_2(1+e^{-m})$", np.log2(1 + np.exp(-m)), C_MAIN, 1.6, "-")]
for lab, v, c, lw, ls in loss:
    ax[0].plot(m, v, label=lab, color=c, lw=lw, ls=ls)
ax[0].set_xlabel("lề $m = y\\,(w^\\top x + b)$")
ax[0].set_ylabel("mất mát")
ax[0].set_ylim(-.1, 3.1); ax[0].axvline(0, color=C_LINE, lw=.9)
ax[0].grid(alpha=.25)
ax[0].legend(frameon=False, fontsize=7.6, loc="upper right")
ax[0].set_title("Các hàm mất mát theo lề", fontsize=9)
ax[0].spines[["top", "right"]].set_visible(False)

ax[1].axis("off")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
rows = [("mất mát 0–1", "không lồi, đạo hàm bằng 0", "không dùng gradient được"),
        ("perceptron", "lồi, bằng 0 khi $m \\geq 0$", "dừng khi vừa phân loại đúng"),
        ("hinge", "lồi, bằng 0 khi $m \\geq 1$", "đòi hỏi lề, nghiệm thưa"),
        ("logistic", "lồi, trơn, luôn dương", "cho ra xác suất")]
ax[1].text(.5, .93, "Tính chất và hệ quả", ha="center", fontsize=10, weight="bold")
for i, (a, b, c) in enumerate(rows):
    y = .78 - i * .168
    ax[1].add_patch(FancyBboxPatch((.02, y - .065), .96, .13,
                                   boxstyle="round,pad=0.008,rounding_size=0.02",
                                   facecolor="#fbfbfa" if i % 2 else "#f4f6f5",
                                   edgecolor=C_LINE, linewidth=.7))
    ax[1].text(.06, y + .028, a, fontsize=8.6, weight="bold", color=C_MAIN, va="center")
    ax[1].text(.06, y - .03, b, fontsize=7.9, color="#333", va="center")
    ax[1].text(.96, y, c, fontsize=7.9, color=C_DIM, va="center", ha="right", style="italic")
ax[1].text(.5, .075,
           "Hinge và logistic (log cơ số 2) là chặn trên của mất mát 0–1;\n"
           "mất mát perceptron thì không, vì bằng 0 tại $m = 0$.",
           ha="center", fontsize=8.2, color=C_DIM, style="italic", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "nt05_matmat.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 13 — Sáu giáo trình của lộ trình kỹ sư AI và thứ tự đọc
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.2, 5.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .965, "Lộ trình sáu giáo trình", ha="center", fontsize=12.5, weight="bold")

phan = [
    ("1 · Nền tảng", C_MAIN,
     ["Đại số tuyến tính, xác suất", "Hồi quy, phân loại", "Gradient descent",
      "Đánh giá, regularization", "Tối ưu lồi, SVM", "PCA, K-means, gợi ý"]),
    ("2 · Học sâu", "#2f6f9f",
     ["Độ chệch – phương sai", "Cây, học tập hợp", "MLP, lan truyền ngược",
      "Khởi tạo, chuẩn hoá", "CNN, RNN", "Transformer, giải mã"]),
    ("3 · Biểu diễn & Căn chỉnh", "#b5651d",
     ["Embedding", "Học chuyển giao, LoRA", "VAE, GAN, khuếch tán",
      "Học tăng cường", "RLHF", "DPO"]),
    ("4 · Ứng dụng LLM", "#b0413e",
     ["Prompt, ngữ cảnh", "Chọn, triển khai mô hình", "Tìm kiếm vector",
      "RAG", "Agent, MCP", "Đánh giá, an toàn"]),
    ("5 · Quantization", "#8a5fa8",
     ["Biểu diễn số", "Affine quantization", "Suy luận số nguyên",
      "PTQ, QAT", "GPTQ, AWQ", "KV cache"]),
    ("6 · MLOps", C_WARN,
     ["Nợ kỹ thuật", "Dữ liệu, kiểm thử", "Phục vụ, ra mắt",
      "Giám sát, dịch chuyển", "Huấn luyện lại", "LLMOps"]),
]
W, G = .148, .0184
x0 = (1 - (6 * W + 5 * G)) / 2
for k, (ten, col, muc) in enumerate(phan):
    x = x0 + k * (W + G)
    ax.add_patch(FancyBboxPatch((x, .33), W, .55,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                facecolor="#fcfcfb", edgecolor=col, linewidth=1.4))
    ax.add_patch(FancyBboxPatch((x, .785), W, .095,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                facecolor=col, edgecolor=col, linewidth=1.4))
    ax.text(x + W / 2, .832, ten, ha="center", va="center", fontsize=7.9,
            weight="bold", color="white")
    for i, m_ in enumerate(muc):
        ax.text(x + .01, .705 - i * .066, "· " + m_, fontsize=6.9, va="center", color="#2a2a2a")
    if k < 5:
        mui_ten(ax, (x + W + .001, .56), (x + W + G - .001, .56), color="#6b7a78", lw=1.6)

# ba chặng lớn bên dưới
chang = [(0, 1, "Kiến thức nền", C_MAIN), (2, 3, "Mô hình ngôn ngữ và ứng dụng", "#b0413e"),
         (4, 5, "Tối ưu và vận hành", C_WARN)]
for a_, b_, ten, col in chang:
    xa = x0 + a_ * (W + G); xb = x0 + b_ * (W + G) + W
    ax.plot([xa + .004, xb - .004], [.285, .285], color=col, lw=2.2, solid_capstyle="round")
    ax.text((xa + xb) / 2, .245, ten, ha="center", va="center", fontsize=8.6, color=col, weight="bold")

ax.text(.5, .12,
        "Giáo trình 1 và 2 là kiến thức nền mà roadmap AI Engineer giả định người học đã có; giáo trình 3 và 4 đi theo\n"
        "các chặng của roadmap từ cách mô hình ngôn ngữ được huấn luyện tới cách xây ứng dụng trên nó; giáo trình 5 và 6\n"
        "trình bày cách làm mô hình chạy hiệu quả và vận hành ổn định trong sản xuất.",
        ha="center", fontsize=7.9, color="#333", linespacing=1.6)
fig.tight_layout(); fig.savefig(OUT + "nt13_mach.png"); plt.close(fig)

print("Da luu 3 hinh: nt01_bando, nt05_matmat, nt13_mach")
