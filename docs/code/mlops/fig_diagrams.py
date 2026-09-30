"""Sinh Hình 1–5 và Hình 8 của giáo trình MLOps: các sơ đồ khái niệm.

Đây là hình vẽ tay bằng matplotlib chứ không phải kết quả thí nghiệm, nhưng vẫn
để trong mã nguồn để sửa được và để mọi chi tiết khớp với chữ trong bài.

    python code/mlops/fig_diagrams.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figs/"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

INK, DIM = "#1f1d1a", "#8a857c"
TEAL, TEAL_BG = "#1f6f68", "#e0efed"
AMBER, AMBER_BG = "#8a5300", "#fbf0db"
RED, RED_BG = "#a13328", "#fbe9e6"
BLUE, BLUE_BG = "#1a5aa8", "#dceafb"
GREEN, GREEN_BG = "#2c6b39", "#e6f2e8"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.5, weight="normal", lw=1.1, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, p, q, color=INK, lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=4))


def canvas(w, h, xlim=(0, 10), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


# ===================================================== Hình 1 — hộp mã ML bé xíu
fig, ax = canvas(8.0, 3.6, (0, 12), (0, 7))
around = [
    ("Thu thập\ndữ liệu", 0.2, 4.6, 1.9, 1.5, BLUE_BG, BLUE),
    ("Kiểm định\ndữ liệu", 2.3, 5.2, 1.9, 0.9, BLUE_BG, BLUE),
    ("Trích xuất\nđặc trưng", 0.2, 2.6, 1.9, 1.7, GREEN_BG, GREEN),
    ("Quản lý\ncấu hình", 4.4, 5.2, 1.7, 0.9, AMBER_BG, AMBER),
    ("Hạ tầng\nphục vụ", 8.3, 4.2, 1.9, 1.9, TEAL_BG, TEAL),
    ("Quản lý\ntài nguyên máy", 10.4, 4.2, 1.5, 1.9, TEAL_BG, TEAL),
    ("Công cụ\nphân tích", 6.3, 5.2, 1.8, 0.9, AMBER_BG, AMBER),
    ("Quản lý\nquy trình", 8.3, 2.0, 1.9, 1.9, AMBER_BG, AMBER),
    ("Giám sát", 10.4, 2.0, 1.5, 1.9, RED_BG, RED),
    ("Hạ tầng\nphục vụ dữ liệu", 0.2, 0.3, 3.0, 2.0, GREEN_BG, GREEN),
    ("Kiểm thử và gỡ lỗi", 3.4, 0.3, 4.6, 1.5, RED_BG, RED),
    ("Xác minh\nmô hình", 8.3, 0.3, 3.6, 1.4, RED_BG, RED),
]
for t, x, y, w, h, fc, ec in around:
    box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=8)
box(ax, 4.4, 2.6, 3.6, 2.3, "", fc="#f4f2ec", ec=DIM)
ax.text(6.2, 4.6, "Thu thập đặc trưng", ha="center", va="center", fontsize=8, color=INK, zorder=3)
box(ax, 5.55, 3.35, 1.3, 0.8, "Mã\nML", fc=INK, ec=INK, fs=9, weight="bold")
ax.text(6.2, 3.75, "Mã\nML", ha="center", va="center", fontsize=9.5,
        color="white", weight="bold", zorder=5, linespacing=1.3)
ax.text(6.0, 6.55, "Chỉ một phần rất nhỏ của hệ thống ML thật là mã học máy",
        ha="center", fontsize=9.5, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT + "mlops01_debt.png"); plt.close(fig)


# ===================================================== Hình 2 — vòng đời là vòng lặp
fig, ax = canvas(8.4, 3.4, (0, 12), (0, 6))
stages = [
    ("Bài toán\nnghiệp vụ", 0.2, TEAL_BG, TEAL),
    ("Dữ liệu và\nnhãn", 2.15, BLUE_BG, BLUE),
    ("Đặc trưng và\nthí nghiệm", 4.1, BLUE_BG, BLUE),
    ("Đánh giá và\nra mắt", 6.05, GREEN_BG, GREEN),
    ("Phục vụ", 8.0, GREEN_BG, GREEN),
    ("Giám sát", 9.95, RED_BG, RED),
]
for t, x, fc, ec in stages:
    box(ax, x, 3.0, 1.75, 1.3, t, fc=fc, ec=ec, fs=8.2)
for i in range(len(stages) - 1):
    arrow(ax, (stages[i][1] + 1.75, 3.65), (stages[i + 1][1], 3.65))
# ba vong quay lai
loops = [(10.85, 5.05, "dịch chuyển phân phối → huấn luyện lại", RED),
         (10.85, 3.05, "lỗi dữ liệu → sửa pipeline", AMBER),
         (10.85, 2.25, "chỉ số sản phẩm không nhúc nhích → đổi bài toán", TEAL)]
targets = [(4.98, 3.0), (3.03, 3.0), (1.08, 3.0)]
for (sx, sy, label, col), (tx, ty) in zip(loops, targets):
    ax.add_patch(FancyArrowPatch((sx, 3.0), (tx, ty - 0.02),
                                 connectionstyle=f"arc3,rad={0.16 + 0.07 * loops.index((sx, sy, label, col))}",
                                 arrowstyle="-|>", mutation_scale=10, color=col, lw=1.2,
                                 linestyle=(0, (4, 2)), shrinkA=2, shrinkB=2, zorder=1))
    ax.text(tx + (sx - tx) / 2, 1.35 - 0.42 * loops.index((sx, sy, label, col)),
            label, ha="center", fontsize=7.6, color=col)
ax.text(6.0, 5.5, "Vòng đời ML là một vòng lặp, không phải một đường thẳng",
        ha="center", fontsize=9.5, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT + "mlops02_lifecycle.png"); plt.close(fig)


# ===================================================== Hình 3 — ba mức tự động hoá
fig, ax = canvas(8.6, 3.5, (0, 12), (0, 6))
levels = [
    ("Mức 0 — thủ công", 0.2, ["mọi bước làm tay", "bàn giao mô hình đã\nhuấn luyện cho kỹ sư",
                               "vài tháng một lần", "không giám sát"], RED_BG, RED),
    ("Mức 1 — tự động hoá pipeline", 4.1, ["pipeline huấn luyện\nchạy tự động",
                                           "kiểm định dữ liệu và\nmô hình trong pipeline",
                                           "bàn giao cả PIPELINE", "có huấn luyện lại (CT)"], AMBER_BG, AMBER),
    ("Mức 2 — tự động hoá CI/CD", 8.0, ["CI/CD cho chính pipeline",
                                        "thử nghiệm ý tưởng mới\nrồi triển khai nhanh",
                                        "nhiều pipeline song song", "giám sát khép kín"], GREEN_BG, GREEN),
]
for title, x, items, fc, ec in levels:
    box(ax, x, 0.5, 3.7, 4.5, "", fc=fc, ec=ec, lw=1.3)
    ax.text(x + 1.85, 4.55, title, ha="center", fontsize=9, weight="bold", color=ec)
    for i, it in enumerate(items):
        ax.text(x + 0.25, 3.85 - i * 0.92, "•", fontsize=9, color=ec, va="top")
        ax.text(x + 0.6, 3.85 - i * 0.92, it, fontsize=8, va="top", color=INK, linespacing=1.4)
for x in (3.95, 7.85):
    arrow(ax, (x, 2.7), (x + 0.3, 2.7), color=DIM)
ax.text(6.0, 5.5, "Càng lên mức cao, thứ được bàn giao càng lớn: mô hình → pipeline → hệ thống",
        ha="center", fontsize=9.3, weight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT + "mlops03_levels.png"); plt.close(fig)


# ===================================================== Hình 4 — phổ lỗi dữ liệu
fig, ax = canvas(8.4, 3.2, (0, 12), (0, 5.4))
cells = [
    ("Lỗi cứng\n(hard)", "đảo cột, tuổi âm,\nkiểu sai, tệp rỗng",
     "Dự đoán sai rõ ràng.\nDễ bắt bằng ràng buộc schema.", "Chặn pipeline ngay", RED_BG, RED, 0.2),
    ("Lỗi mềm\n(soft)", "vài trường null,\nđơn vị đổi, mã lạ",
     "Vẫn ra dự đoán hợp lý,\nnên trôi qua kiểm tra.", "Cảnh báo + theo dõi tỉ lệ", AMBER_BG, AMBER, 4.1),
    ("Dịch chuyển\n(drift)", "phân phối đổi dần\ntheo tuần, theo mùa",
     "Không có gì “sai”,\nchỉ là thế giới đã khác.", "Điều tra, có thể huấn luyện lại", BLUE_BG, BLUE, 8.0),
]
for title, ex, why, act, fc, ec, x in cells:
    box(ax, x, 0.4, 3.7, 4.2, "", fc=fc, ec=ec, lw=1.3)
    ax.text(x + 1.85, 4.15, title, ha="center", fontsize=9.5, weight="bold", color=ec, linespacing=1.3)
    ax.text(x + 1.85, 3.25, ex, ha="center", fontsize=8, color=INK, style="italic", linespacing=1.4)
    ax.text(x + 1.85, 2.25, why, ha="center", fontsize=8, color=INK, linespacing=1.5)
    box(ax, x + 0.3, 0.65, 3.1, 0.65, act, fc="white", ec=ec, fs=8)
arrow(ax, (0.2, 4.95), (11.7, 4.95), color=DIM)
ax.text(0.3, 5.12, "dễ phát hiện, hại ngay", fontsize=8, color=DIM)
ax.text(11.6, 5.12, "khó phát hiện, hại chậm", fontsize=8, color=DIM, ha="right")
fig.tight_layout(); fig.savefig(OUT + "mlops04_errors.png"); plt.close(fig)


# ============================================ Hình 5 — tính đúng theo thời điểm
fig, ax = canvas(8.6, 4.1, (0, 12), (0, 8.0))
ax.text(6.0, 7.6, "Cùng một dòng huấn luyện, hai cách ghép cho hai giá trị khác nhau",
        ha="center", fontsize=9.5, weight="bold", color=INK)

for row, (lab, col, used, src) in enumerate(
        [("ĐÚNG — ghép theo thời điểm", GREEN, "31", 4.0),
         ("SAI — ghép giá trị hiện tại", RED, "480", 10.4)]):
    y = 5.3 - row * 3.35
    ax.text(0.5, y + 1.25, lab, fontsize=9, weight="bold", color=col)
    ax.plot([0.5, 11.5], [y, y], color=DIM, lw=1.0, zorder=1)
    for t, x in [("1/3", 1.6), ("15/3", 4.0), ("1/4", 6.6), ("hôm nay", 10.4)]:
        ax.plot([x], [y], "o", ms=4.5, color=DIM, zorder=3)
        ax.text(x, y - 0.34, t, ha="center", fontsize=7.5, color=DIM, va="top")
    ax.plot([6.6, 6.6], [y - 0.12, y + 0.95], color=AMBER, lw=1.3, ls="--", zorder=2)
    ax.text(6.75, y + 0.9, "thời điểm cần dự đoán", fontsize=7.6, color=AMBER, va="top")
    for x, v in [(1.6, "12"), (4.0, "31"), (10.4, "480")]:
        hot = abs(x - src) < 0.01
        box(ax, x - 0.42, y + 0.18, 0.84, 0.46, v,
            fc=(GREEN_BG if row == 0 else RED_BG) if hot else "white",
            ec=col if hot else DIM, fs=8, weight="bold" if hot else "normal")
    ax.add_patch(FancyArrowPatch((src, y + 0.14), (2.2, y - 0.88), arrowstyle="-|>",
                                 mutation_scale=11, color=col, lw=1.5,
                                 connectionstyle="arc3,rad=-0.09" if row else "arc3,rad=0.3",
                                 shrinkA=3, shrinkB=3, zorder=4))
    box(ax, 0.5, y - 1.48, 5.8, 0.6,
        f"giá trị đưa vào tập huấn luyện = {used}", fc="white", ec=col, fs=8.4)
    if row == 1:
        ax.text(6.6, y - 1.18, "← thông tin của tương lai, lúc dự đoán thật không hề có",
                fontsize=8, color=RED, va="center")
fig.tight_layout(); fig.savefig(OUT + "mlops05_pit.png"); plt.close(fig)


# ===================================================== Hình 8 — chiến lược ra mắt
fig, ax = canvas(8.6, 4.0, (0, 12), (0, 7.4))
ax.text(6.0, 7.05, "Bốn cách đưa mô hình mới ra, khác nhau ở chỗ ai chịu rủi ro",
        ha="center", fontsize=9.5, weight="bold", color=INK)
strategies = [
    ("Shadow", 0.2, 3.9, "100% lưu lượng chạy qua cả hai,\nchỉ bản cũ trả lời người dùng",
     "rủi ro cho người dùng: 0\nnhưng không đo được phản ứng thật", TEAL_BG, TEAL),
    ("Canary", 6.2, 3.9, "1% → 5% → 25% → 100%,\ndừng lại nếu chỉ số xấu đi",
     "hạn chế bán kính thiệt hại\ncần chỉ số đủ nhạy ở 1%", GREEN_BG, GREEN),
    ("Blue–green", 0.2, 0.5, "hai môi trường song song,\nđổi toàn bộ trong một nhịp",
     "quay lui tức thì\nnhưng thiệt hại là 100% nếu hỏng", AMBER_BG, AMBER),
    ("A/B test", 6.2, 0.5, "chia người dùng ngẫu nhiên,\nđo chỉ số sản phẩm có ý nghĩa thống kê",
     "câu trả lời nhân quả duy nhất\nnhưng tốn thời gian và lưu lượng", BLUE_BG, BLUE),
]
for name, x, y, how, tradeoff, fc, ec in strategies:
    box(ax, x, y, 5.6, 2.9, "", fc=fc, ec=ec, lw=1.3)
    ax.text(x + 0.3, y + 2.5, name, fontsize=9.5, weight="bold", color=ec)
    ax.text(x + 0.3, y + 1.55, how, fontsize=8.2, color=INK, va="center", linespacing=1.5)
    box(ax, x + 0.3, y + 0.25, 5.0, 0.85, tradeoff, fc="white", ec=ec, fs=7.8)
fig.tight_layout(); fig.savefig(OUT + "mlops08_rollout.png"); plt.close(fig)

print("Đã lưu Hình 1–5 và Hình 8 vào " + OUT)
