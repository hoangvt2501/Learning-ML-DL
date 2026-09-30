"""Các sơ đồ khái niệm cho giáo trình "Nền tảng Machine Learning".

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

ax.text(.5, .955, "Mọi thuật toán học có giám sát đều gồm đúng ba thứ",
        ha="center", fontsize=11.5, weight="bold", color="#1a1a1a")

hop(ax, .04, .70, .27, .16,
    "1. MÔ HÌNH\n$f_\\theta(x)$\ndạng hàm được phép dùng", fs=8.5, weight="bold")
hop(ax, .365, .70, .27, .16,
    "2. HÀM MẤT MÁT\n$L(f_\\theta(x),\\, y)$\nthế nào là đoán sai", fs=8.5, weight="bold")
hop(ax, .69, .70, .27, .16,
    "3. THUẬT TOÁN TỐI ƯU\n$\\min_\\theta \\sum_i L$\ntìm $\\theta$ thế nào", fs=8.5, weight="bold")

vd = [("Hồi quy tuyến tính", "$w^\\top x + b$", "bình phương sai lệch",
       "nghiệm giải tích\nhoặc GD"),
      ("Hồi quy logistic", "$\\sigma(w^\\top x + b)$", "entropy chéo", "GD (hàm lồi)"),
      ("SVM lề mềm", "$w^\\top x + b$", "hinge $+\\;\\lambda\\|w\\|^2$",
       "quy hoạch toàn phương\n(bài toán đối ngẫu)"),
      ("Cây quyết định", "hàm hằng từng khúc", "Gini / entropy",
       "tham lam, chia từng nút"),
      ("Mạng nơ-ron", "hợp nhiều lớp affine\nvà phi tuyến", "tuỳ bài toán",
       "SGD + lan truyền ngược")]

y0, dy = .575, .102
ax.text(.04, y0 + .055, "Năm ví dụ đọc theo đúng ba cột ấy:", fontsize=9,
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
        "Đổi một trong ba cột là ra một thuật toán khác. Phần lớn \"thuật toán mới\"\n"
        "chỉ là một tổ hợp mới của ba lựa chọn này.",
        ha="center", fontsize=8.6, color=C_DIM, style="italic", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "nt01_bando.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 5 — Bốn hàm mất mát cho phân loại: chúng đều thay cho mất mát 0–1
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3))
m = np.linspace(-3, 3, 600)

loss = [("mất mát 0–1 (thứ ta THỰC SỰ muốn)", np.where(m <= 0, 1.0, 0.0), "k", 2.0, "-"),
        ("perceptron: $\\max(0, -m)$", np.maximum(0, -m), C_WARN, 1.6, "-"),
        ("hinge (SVM): $\\max(0, 1-m)$", np.maximum(0, 1 - m), C_BAD, 1.6, "-"),
        ("logistic: $\\log_2(1+e^{-m})$", np.log2(1 + np.exp(-m)), C_MAIN, 1.6, "-")]
for lab, v, c, lw, ls in loss:
    ax[0].plot(m, v, label=lab, color=c, lw=lw, ls=ls)
ax[0].set_xlabel("biên $m = y\\,(w^\\top x + b)$")
ax[0].set_ylabel("mất mát")
ax[0].set_ylim(-.1, 3.1); ax[0].axvline(0, color=C_LINE, lw=.9)
ax[0].grid(alpha=.25)
ax[0].legend(frameon=False, fontsize=7.6, loc="upper right")
ax[0].set_title("Ba hàm mất mát là ba cách làm trơn cùng một thứ", fontsize=9)
ax[0].spines[["top", "right"]].set_visible(False)

ax[1].axis("off")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
rows = [("mất mát 0–1", "không lồi, đạo hàm 0 khắp nơi", "không tối ưu trực tiếp được"),
        ("perceptron", "lồi, nhưng phẳng ở $m>0$", "dừng ngay khi vừa đúng"),
        ("hinge", "lồi, phẳng ở $m>1$", "ép có lề, cho nghiệm thưa"),
        ("logistic", "lồi, trơn, không bao giờ phẳng hẳn", "cho ra xác suất")]
ax[1].text(.5, .93, "Vì sao lại cần thay?", ha="center", fontsize=10, weight="bold")
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
           "Cả ba đều là chặn trên của mất mát 0–1, nên giảm chúng là\n"
           "gián tiếp giảm tỉ lệ đoán sai.",
           ha="center", fontsize=8.2, color=C_DIM, style="italic", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "nt05_matmat.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 13 — Bản đồ toàn bộ bốn giáo trình: cái gì dẫn tới cái gì
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .962, "Bốn giáo trình đọc như một mạch",
        ha="center", fontsize=12, weight="bold")

phan = [
    (.035, "I · Nền tảng", C_MAIN,
     ["Đại số tuyến tính, xác suất",
      "Hồi quy tuyến tính & logistic",
      "Gradient descent",
      "Đánh giá, phạt chuẩn, MAP",
      "Tối ưu lồi → SVM",
      "PCA, K-means, hệ gợi ý"]),
    (.275, "II · Mô hình & Kiến trúc", "#2f6f9f",
     ["Thiên lệch – phương sai",
      "Cây và học tập hợp",
      "MLP, lan truyền ngược",
      "Khởi tạo, chuẩn hoá, kết nối tắt",
      "CNN, RNN, Attention",
      "Transformer hiện đại"]),
    (.515, "III · Quantization", "#8a5fa8",
     ["Biểu diễn số thực",
      "Uniform affine, S và Z",
      "Suy luận số nguyên",
      "Calibration, PTQ, QAT",
      "GPTQ, AWQ, NF4",
      "KV cache"]),
    (.755, "IV · MLOps", "#b8860b",
     ["Nợ kỹ thuật, CACE",
      "Dữ liệu và kiểm thử",
      "Phục vụ và ra mắt",
      "Giám sát, dịch chuyển",
      "Thí nghiệm A/B",
      "Huấn luyện lại"]),
]

for x, ten, col, muc in phan:
    ax.add_patch(FancyBboxPatch((x, .28), .21, .60,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#fcfcfb", edgecolor=col, linewidth=1.4))
    ax.add_patch(FancyBboxPatch((x, .785), .21, .095,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=col, edgecolor=col, linewidth=1.4))
    ax.text(x + .105, .832, ten, ha="center", va="center", fontsize=9.2,
            weight="bold", color="white")
    for i, m_ in enumerate(muc):
        ax.text(x + .014, .715 - i * .073, "· " + m_, fontsize=7.3,
                va="center", color="#2a2a2a")

for x in (.248, .488, .728):
    mui_ten(ax, (x, .56), (x + .027, .56), color="#6b7a78", lw=2.0)

ax.text(.5, .175,
        "Phần I trả lời \"học là gì và tối ưu thế nào\"; phần II thay mô hình tuyến tính bằng mạng sâu;\n"
        "phần III làm mô hình chạy được trên phần cứng thật; phần IV giữ cho nó sống trong sản xuất.",
        ha="center", fontsize=8.4, color="#333", linespacing=1.6)
ax.text(.5, .065,
        "Đọc ngang cũng được: mỗi phần tự đứng vững. Nhưng mũi tên là thứ tự ít phải quay lại nhất.",
        ha="center", fontsize=8.2, color=C_DIM, style="italic")
fig.tight_layout(); fig.savefig(OUT + "nt13_mach.png"); plt.close(fig)

print("Da luu 3 hinh: nt01_bando, nt05_matmat, nt13_mach")
