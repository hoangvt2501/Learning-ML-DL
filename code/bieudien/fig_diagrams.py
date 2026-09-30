"""Sơ đồ khái niệm cho giáo trình "Biểu diễn & Căn chỉnh".

Hình vẽ tay bằng matplotlib, để trong mã nguồn cho sửa được và cho khớp với chữ.

    python code/bieudien/fig_diagrams.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = "figs/"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"
C_SOFT, C_LINE = "#eef4f3", "#cfd8d6"


def hop(ax, x, y, w, h, text, fc=C_SOFT, ec=C_MAIN, fs=8.4, weight="normal", tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                                facecolor=fc, edgecolor=ec, linewidth=1.15))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, linespacing=1.45)


def ten(ax, p0, p1, color=C_DIM, lw=1.2, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 color=color, linewidth=lw, shrinkA=3, shrinkB=3,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------------
# Hình 1 — Bản đồ: bốn câu hỏi của phần này
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .955, "Bốn câu hỏi, bốn nhóm kỹ thuật",
        ha="center", fontsize=12, weight="bold")

nhom = [
    (.025, "BIỂU DIỄN", C_MAIN,
     "Làm sao biến\nchữ, ảnh, người dùng\nthành vector?",
     ["Word2Vec, GloVe", "embedding ngữ cảnh", "học tương phản", "tìm kiếm vector"]),
    (.265, "CHUYỂN GIAO", "#2f6f9f",
     "Đã có mô hình lớn rồi,\nlàm sao dùng lại cho\nviệc của mình?",
     ["tiền huấn luyện", "đóng băng / tinh chỉnh", "LoRA và PEFT", "chưng cất"]),
    (.505, "SINH", "#8a5fa8",
     "Làm sao sinh ra\nmẫu mới giống\ndữ liệu thật?",
     ["VAE", "GAN", "khuếch tán", "tự hồi quy"]),
    (.745, "CĂN CHỈNH", C_WARN,
     "Mô hình biết nói rồi,\nlàm sao bắt nó nói\nthứ ta muốn?",
     ["học tăng cường", "mô hình thưởng", "RLHF", "DPO"]),
]

for x, tieu_de, col, cau_hoi, muc in nhom:
    ax.add_patch(FancyBboxPatch((x, .13), .21, .72,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#fcfcfb", edgecolor=col, linewidth=1.5))
    ax.add_patch(FancyBboxPatch((x, .755), .21, .095,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=col, edgecolor=col, linewidth=1.5))
    ax.text(x + .105, .802, tieu_de, ha="center", va="center", fontsize=9.6,
            weight="bold", color="white")
    ax.text(x + .105, .63, cau_hoi, ha="center", va="center", fontsize=8.1,
            color="#333", style="italic", linespacing=1.5)
    for i, m in enumerate(muc):
        ax.text(x + .015, .47 - i * .072, "· " + m, fontsize=7.9, va="center", color="#2a2a2a")

for x in (.243, .483, .723):
    ten(ax, (x, .49), (x + .017, .49), color="#6b7a78", lw=1.9)

ax.text(.5, .065,
        "Đọc từ trái sang phải là đọc theo đúng thứ tự một mô hình ngôn ngữ ra đời: học biểu diễn từ\n"
        "dữ liệu thô, tái dùng cho việc mới, học sinh ra nội dung, rồi cuối cùng mới học nói cho phải.",
        ha="center", fontsize=8.4, color="#333", linespacing=1.6)
fig.tight_layout(); fig.savefig(OUT + "bd01_bando.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 5 — Bốn cách tái dùng một mô hình đã huấn luyện
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.4, 3.7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .95, "Bốn cách tái dùng một mô hình đã huấn luyện",
        ha="center", fontsize=11.5, weight="bold")

cach = [
    (.02, "Dùng thẳng", ["đóng băng", "đóng băng", "đóng băng", "đóng băng"],
     "0 tham số\nhuấn luyện", "không cần nhãn"),
    (.26, "Đóng băng + đầu mới", ["đóng băng", "đóng băng", "đóng băng", "HỌC"],
     "chỉ lớp cuối", "rất ít nhãn"),
    (.50, "Tinh chỉnh toàn phần", ["HỌC", "HỌC", "HỌC", "HỌC"],
     "toàn bộ", "nhiều nhãn\n+ nhiều bộ nhớ"),
    (.74, "LoRA", ["+ LoRA", "+ LoRA", "+ LoRA", "HỌC"],
     "< 1% tham số", "ít nhãn\nít bộ nhớ"),
]
mau_tt = {"đóng băng": ("#e9ecec", C_DIM), "HỌC": ("#d6e9e6", C_MAIN),
          "+ LoRA": ("#f2e6f5", "#8a5fa8")}

for x, ten_c, lop, tham_so, khi_nao in cach:
    ax.text(x + .11, .84, ten_c, ha="center", fontsize=9, weight="bold", color="#222")
    for i, l in enumerate(lop):
        fc, ec = mau_tt[l]
        y = .68 - i * .115
        ax.add_patch(FancyBboxPatch((x, y), .22, .095,
                                    boxstyle="round,pad=0.005,rounding_size=0.012",
                                    facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(x + .11, y + .048, l, ha="center", va="center", fontsize=7.4,
                color=ec, weight="bold" if l != "đóng băng" else "normal")
    ax.text(x + .11, .175, tham_so, ha="center", fontsize=7.8, weight="bold", color=C_MAIN)
    ax.text(x + .11, .095, khi_nao, ha="center", fontsize=7.4, color=C_DIM,
            style="italic", linespacing=1.4)

ax.text(.005, .49, "lớp\nđầu", ha="right", fontsize=7, color=C_DIM, va="center")
ax.text(.5, .015, "Lớp trên cùng là lớp gần đầu ra. Càng ít lớp phải học thì càng ít nhãn và ít bộ nhớ,\n"
                  "nhưng cũng càng ít khả năng sửa lại phần đặc trưng cho khớp việc mới.",
        ha="center", fontsize=8, color="#444", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "bd05_taidung.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 7 — LoRA hoạt động thế nào
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 3.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .95, "LoRA: học phần cộng thêm có hạng thấp", ha="center", fontsize=11.5, weight="bold")

ax.add_patch(FancyBboxPatch((.06, .35), .2, .38,
                            boxstyle="round,pad=0.008,rounding_size=0.02",
                            facecolor="#e9ecec", edgecolor=C_DIM, linewidth=1.3))
ax.text(.16, .60, "$W_0$", ha="center", fontsize=15)
ax.text(.16, .50, "$d \\times d$", ha="center", fontsize=8.5, color="#444")
ax.text(.16, .42, "đóng băng", ha="center", fontsize=7.6, color=C_DIM, weight="bold")
ax.text(.16, .30, "$d^2$ tham số", ha="center", fontsize=8, color=C_DIM)

ax.add_patch(FancyBboxPatch((.42, .55), .085, .18,
                            boxstyle="round,pad=0.006,rounding_size=0.015",
                            facecolor="#f2e6f5", edgecolor="#8a5fa8", linewidth=1.3))
ax.text(.4625, .64, "$B$", ha="center", fontsize=13, color="#8a5fa8")
ax.text(.4625, .50, "$d\\times r$", ha="center", fontsize=8, color="#8a5fa8")

ax.add_patch(FancyBboxPatch((.55, .55), .085, .18,
                            boxstyle="round,pad=0.006,rounding_size=0.015",
                            facecolor="#f2e6f5", edgecolor="#8a5fa8", linewidth=1.3))
ax.text(.5925, .64, "$A$", ha="center", fontsize=13, color="#8a5fa8")
ax.text(.5925, .50, "$r\\times d$", ha="center", fontsize=8, color="#8a5fa8")
ax.text(.5275, .40, "học, chỉ $2dr$ tham số", ha="center", fontsize=8,
        color="#8a5fa8", weight="bold")

ax.text(.345, .64, "+", ha="center", fontsize=20, color="#333")
ax.text(.70, .64, "=", ha="center", fontsize=18, color="#333")

ax.add_patch(FancyBboxPatch((.74, .35), .2, .38,
                            boxstyle="round,pad=0.008,rounding_size=0.02",
                            facecolor=C_SOFT, edgecolor=C_MAIN, linewidth=1.3))
ax.text(.84, .60, "$W_0 + BA$", ha="center", fontsize=13)
ax.text(.84, .48, "gộp lại khi\nsuy luận", ha="center", fontsize=8,
        color=C_MAIN, linespacing=1.4)

ax.text(.5, .22, "Giả thiết: phần cần sửa khi chuyển sang nhiệm vụ mới có hạng thấp.",
        ha="center", fontsize=8.8, color="#222")
ax.text(.5, .13, "Với $d = 4096$ và $r = 8$: $2dr / d^2 = 2r/d = 0{,}39\\%$ số tham số của lớp.",
        ha="center", fontsize=8.5, color="#333")
ax.text(.5, .04, "Khi suy luận, cộng $BA$ vào $W_0$ một lần, nên LoRA không làm chậm mô hình.",
        ha="center", fontsize=8.2, color=C_DIM, style="italic")
fig.tight_layout(); fig.savefig(OUT + "bd07_lora.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 11 — Ba họ mô hình sinh, so sánh trên cùng một khung
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 4.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .955, "Ba cách trả lời cùng một câu hỏi: sinh mẫu mới thế nào",
        ha="center", fontsize=11.5, weight="bold")

ho = [
    (.02, "VAE", "#2f6f9f",
     "$z \\sim N(0,I) \\;\\to\\; $ giải mã $\\to x$",
     ["huấn luyện: tối đa chặn dưới ELBO",
      "ưu: ổn định, có không gian ẩn dùng được",
      "nhược: mẫu sinh ra thường **mờ**",
      "vì sao mờ: tối thiểu MSE = đoán trung bình"]),
    (.34, "GAN", "#8a5fa8",
     "$z \\sim N(0,I) \\;\\to\\; G \\to x$, có $D$ chấm",
     ["huấn luyện: trò chơi hai bên",
      "ưu: mẫu **sắc nét**",
      "nhược: khó huấn luyện, dễ sụp chế độ",
      "không có hàm hợp lý để so sánh"]),
    (.66, "Khuếch tán", C_MAIN,
     "$x_T \\sim N(0,I) \\;\\to\\; T$ bước khử nhiễu",
     ["huấn luyện: đoán nhiễu đã thêm vào",
      "ưu: sắc nét VÀ ổn định, phủ hết chế độ",
      "nhược: lấy mẫu **chậm** (nhiều bước)",
      "huấn luyện là hồi quy — không có trò chơi"]),
]
for x, ten_h, col, cong_thuc, muc in ho:
    ax.add_patch(FancyBboxPatch((x, .17), .3, .68,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#fcfcfb", edgecolor=col, linewidth=1.5))
    ax.add_patch(FancyBboxPatch((x, .755), .3, .095,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=col, edgecolor=col, linewidth=1.5))
    ax.text(x + .15, .802, ten_h, ha="center", va="center", fontsize=10,
            weight="bold", color="white")
    ax.text(x + .15, .69, cong_thuc, ha="center", fontsize=8.2, color="#222")
    for i, m in enumerate(muc):
        ax.text(x + .015, .59 - i * .095,
                "· " + m.replace("**", ""), fontsize=7.7, va="center", color="#2a2a2a")

ax.text(.5, .085,
        "Cả ba đều biến nhiễu Gauss thành mẫu. Khác nhau ở cách huấn luyện:\n"
        "VAE tối đa một chặn dưới của hợp lý, GAN chơi một trò chơi, khuếch tán giải một bài hồi quy.",
        ha="center", fontsize=8.4, color="#333", linespacing=1.6)
fig.tight_layout(); fig.savefig(OUT + "bd11_mohinhsinh.png"); plt.close(fig)


# ---------------------------------------------------------------------
# Hình 14 — Ba giai đoạn huấn luyện một trợ lý, và DPO cắt bớt chỗ nào
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(.5, .955, "Từ mô hình ngôn ngữ thô tới trợ lý: ba giai đoạn",
        ha="center", fontsize=11.5, weight="bold")

gd = [
    (.03, "1 · Tiền huấn luyện", C_DIM,
     "đoán token tiếp theo", "hàng nghìn tỉ token\nkhông cần nhãn",
     "→ biết ngôn ngữ,\nnhưng không biết\nlàm theo yêu cầu"),
    (.27, "2 · Tinh chỉnh có giám sát", "#2f6f9f",
     "bắt chước câu trả lời mẫu", "hàng chục nghìn cặp\n(câu hỏi, trả lời tốt)",
     "→ biết làm theo yêu cầu,\nnhưng chất lượng\nchỉ bằng người viết mẫu"),
    (.51, "3a · Mô hình thưởng", C_WARN,
     "học từ SO SÁNH", "hàng trăm nghìn cặp\n(trả lời tốt hơn / kém hơn)",
     "→ một hàm chấm điểm"),
    (.75, "3b · Tối ưu có KL", C_BAD,
     "tối đa thưởng,\ncó phạt KL", "không cần dữ liệu mới",
     "→ trợ lý"),
]
for x, ten_g, col, lam_gi, du_lieu, ket_qua in gd:
    ax.add_patch(FancyBboxPatch((x, .30), .22, .55,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#fcfcfb", edgecolor=col, linewidth=1.4))
    ax.add_patch(FancyBboxPatch((x, .755), .22, .095,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=col, edgecolor=col, linewidth=1.4))
    ax.text(x + .11, .802, ten_g, ha="center", va="center", fontsize=8.4,
            weight="bold", color="white")
    ax.text(x + .11, .665, lam_gi, ha="center", fontsize=8, color="#222", linespacing=1.4)
    ax.text(x + .11, .555, du_lieu, ha="center", fontsize=7.4, color=C_DIM,
            style="italic", linespacing=1.4)
    ax.text(x + .11, .40, ket_qua, ha="center", fontsize=7.5, color="#333", linespacing=1.45)

for x in (.253, .493, .733):
    ten(ax, (x, .56), (x + .014, .56), color="#6b7a78", lw=1.8)

ax.add_patch(FancyBboxPatch((.50, .14), .47, .12,
                            boxstyle="round,pad=0.008,rounding_size=0.02",
                            facecolor="#fdf4e6", edgecolor=C_WARN, linewidth=1.4))
ax.text(.735, .20, "DPO gộp 3a và 3b thành một bước, không cần mô hình thưởng",
        ha="center", va="center", fontsize=9.0, weight="bold", color="#7a5a10")
ten(ax, (.62, .29), (.62, .265), color=C_WARN, lw=1.5)
ten(ax, (.86, .29), (.86, .265), color=C_WARN, lw=1.5)

ax.text(.5, .055,
        "Bước 3 là bước duy nhất mô hình học từ so sánh thay vì từ ví dụ mẫu, và đó là lý do nó vượt được\n"
        "chất lượng của người viết mẫu: nhận ra câu nào hay hơn dễ hơn nhiều so với tự viết ra câu hay nhất.",
        ha="center", fontsize=8.3, color="#333", linespacing=1.6)
fig.tight_layout(); fig.savefig(OUT + "bd14_rlhf_quytrinh.png"); plt.close(fig)

print("Da luu 5 hinh: bd01_bando, bd05_taidung, bd07_lora, bd11_mohinhsinh, bd14_rlhf_quytrinh")
