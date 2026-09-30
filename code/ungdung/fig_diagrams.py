"""Các sơ đồ khái niệm cho giáo trình "Ứng dụng LLM".

Hình vẽ bằng matplotlib, không phải kết quả thí nghiệm; để trong mã nguồn cho dễ sửa
và cho khớp với chữ trong bài.

    python code/ungdung/fig_diagrams.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figs") + os.sep
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
C_MAIN, C_WARN, C_BAD, C_DIM, C_BLUE, C_PURP = "#1f6f68", "#b8860b", "#b0413e", "#8a857c", "#2f5f9f", "#7a4fa0"
C_SOFT, C_LINE = "#eef4f3", "#cfd8d6"


def hop(ax, x, y, w, h, text, fc=C_SOFT, ec=C_MAIN, fs=8.2, weight="normal", tc="#1a1a1a", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            weight=weight, linespacing=1.4)


def ten(ax, p0, p1, color=C_DIM, lw=1.2, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def khung(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------
# ud01 — Các thành phần của một ứng dụng dùng LLM
# ---------------------------------------------------------------------
fig, ax = khung((9.4, 4.8))
ax.text(.5, .96, "Các thành phần của một ứng dụng dùng mô hình ngôn ngữ lớn", ha="center",
        fontsize=11, weight="bold")
hop(ax, .02, .45, .13, .14, "Người dùng", fc="#f6f4ef", ec=C_DIM)
hop(ax, .22, .30, .30, .44, "", fc="#fbfbfa", ec=C_MAIN, lw=1.4)
ax.text(.37, .705, "Ứng dụng", ha="center", fontsize=9.5, weight="bold", color=C_MAIN)
hop(ax, .245, .56, .25, .09, "Lớp ngữ cảnh: prompt, lịch sử,\ntài liệu, kết quả tool (Chương 4, 5)", fs=7.4)
hop(ax, .245, .445, .25, .085, "Điều phối: gọi mô hình, gọi tool,\nvòng lặp agent (Chương 9)", fs=7.4)
hop(ax, .245, .33, .25, .085, "Rào chắn: kiểm tra đầu vào\nvà đầu ra (Chương 11)", fs=7.4, fc="#fbeceb", ec=C_BAD)
hop(ax, .62, .60, .16, .12, "Mô hình ngôn ngữ\n(API hoặc tự triển khai)\nChương 2, 3", fs=7.6, fc="#eaf0f8", ec=C_BLUE)
hop(ax, .62, .42, .16, .12, "Truy xuất: embedding,\nCSDL vector\nChương 6, 7, 8", fs=7.6, fc="#fdf5e6", ec=C_WARN)
hop(ax, .62, .24, .16, .12, "Tool và dữ liệu ngoài\n(API, MCP server)\nChương 9, 10", fs=7.6, fc="#f3edf8", ec=C_PURP)
hop(ax, .84, .42, .14, .30, "Đánh giá\nvà quan sát\n\nbộ kiểm thử,\ntrace, chi phí\n\nChương 12, 13", fs=7.4,
    fc="#f4f6f5", ec=C_DIM)
ten(ax, (.15, .54), (.22, .54), color=C_DIM); ten(ax, (.22, .50), (.15, .50), color=C_DIM)
for y in (.66, .48, .30):
    ten(ax, (.52, .52), (.62, y), color=C_LINE, lw=1.1)
    ten(ax, (.62, y - .02), (.52, .50), color=C_LINE, lw=1.1)
ax.text(.5, .12, "Mô hình chỉ là một thành phần. Phần lớn công việc của kỹ sư AI nằm ở lớp ngữ cảnh, "
        "truy xuất, tool,\nrào chắn và vòng đánh giá bao quanh mô hình.", ha="center", fontsize=8.2,
        color="#333", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "ud01_ungdung.png"); plt.close(fig)


# ---------------------------------------------------------------------
# ud03 — Ngân sách cửa sổ ngữ cảnh của một yêu cầu
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 2.3))
phan = [("chỉ dẫn hệ thống", 600, C_MAIN), ("định nghĩa tool", 900, C_PURP),
        ("tài liệu truy xuất", 1750, C_WARN), ("lịch sử hội thoại", 800, C_BLUE),
        ("câu hỏi", 60, C_BAD), ("dành cho đầu ra", 1000, "#bbbbbb")]
trai = 0
for ten_, v, col in phan:
    ax.barh(0, v, left=trai, color=col, edgecolor="white", height=.5)
    if v > 250:
        ax.text(trai + v / 2, 0, f"{ten_}\n{v:,}".replace(",", " "), ha="center", va="center",
                fontsize=7.4, color="white" if col != "#bbbbbb" else "#222")
    trai += v
ax.annotate("câu hỏi (60)", xy=(600 + 900 + 1750 + 800 + 30, .26), xytext=(4300, .62), fontsize=7.4,
            arrowprops=dict(arrowstyle="-", color=C_DIM, lw=.8), color="#333")
ax.set_xlim(0, trai * 1.02); ax.set_ylim(-.6, .9); ax.set_yticks([])
ax.set_xlabel("số token trong cửa sổ ngữ cảnh")
ax.set_title("Một yêu cầu điển hình: câu hỏi của người dùng chỉ chiếm khoảng 1% số token", fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud03_ngucanh.png"); plt.close(fig)


# ---------------------------------------------------------------------
# ud04 — Quy trình RAG: lập chỉ mục và trả lời
# ---------------------------------------------------------------------
fig, ax = khung((9.6, 4.4))
ax.text(.5, .955, "Retrieval-Augmented Generation", ha="center", fontsize=11, weight="bold")
ax.text(.02, .84, "Lập chỉ mục (làm trước, ngoại tuyến)", fontsize=8.8, weight="bold", color=C_WARN)
buoc1 = ["Tài liệu\n(PDF, web, wiki)", "Tách văn bản,\nlàm sạch", "Chia đoạn\n(chunking)", "Embedding\nmỗi đoạn",
         "CSDL vector\n+ chỉ mục từ khoá"]
for i, t in enumerate(buoc1):
    hop(ax, .02 + i * .195, .63, .16, .15, t, fc="#fdf5e6", ec=C_WARN, fs=7.8)
    if i:
        ten(ax, (.02 + i * .195 - .035, .705), (.02 + i * .195, .705))
ax.text(.02, .5, "Trả lời (mỗi yêu cầu, trực tuyến)", fontsize=8.8, weight="bold", color=C_BLUE)
buoc2 = ["Câu hỏi", "Viết lại truy vấn\n(tuỳ chọn)", "Truy xuất\ntop-k đoạn", "Xếp lại\n(rerank)",
         "Dựng prompt:\nchỉ dẫn + đoạn\n+ câu hỏi", "LLM sinh câu trả\nlời có trích dẫn"]
for i, t in enumerate(buoc2):
    hop(ax, .02 + i * .163, .26, .135, .17, t, fc="#eaf0f8", ec=C_BLUE, fs=7.6)
    if i:
        ten(ax, (.02 + i * .163 - .028, .345), (.02 + i * .163, .345))
ten(ax, (.02 + 4 * .195 + .08, .63), (.02 + 2 * .163 + .07, .43), color=C_WARN, ls="--", lw=1.0)
ax.text(.5, .1, "Chất lượng câu trả lời bị chặn bởi chất lượng truy xuất: nếu đoạn văn đúng không nằm trong\n"
        "top-k, mô hình chỉ còn cách dựa vào kiến thức sẵn có hoặc đoán.", ha="center", fontsize=8.1,
        color="#333", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "ud04_rag.png"); plt.close(fig)


# ---------------------------------------------------------------------
# ud09 — Vòng lặp của một agent (ReAct / function calling)
# ---------------------------------------------------------------------
fig, ax = khung((9.0, 4.4))
ax.text(.5, .955, "Vòng lặp của một agent", ha="center", fontsize=11, weight="bold")
hop(ax, .03, .45, .15, .14, "Nhiệm vụ\ncủa người dùng", fc="#f6f4ef", ec=C_DIM)
hop(ax, .30, .60, .20, .15, "LLM suy nghĩ:\ncần làm gì tiếp?", fc="#eaf0f8", ec=C_BLUE)
hop(ax, .60, .60, .22, .15, "Yêu cầu gọi tool\n{\"name\": ..., \"arguments\": ...}", fc="#f3edf8", ec=C_PURP, fs=7.6)
hop(ax, .60, .28, .22, .15, "Ứng dụng thực thi tool\n(kiểm tra quyền, giới hạn)", fc="#fbeceb", ec=C_BAD, fs=7.6)
hop(ax, .30, .28, .20, .15, "Kết quả tool được\nthêm vào ngữ cảnh", fc=C_SOFT, ec=C_MAIN, fs=7.8)
hop(ax, .86, .45, .12, .14, "Câu trả lời\ncuối cùng", fc="#f6f4ef", ec=C_DIM)
ten(ax, (.18, .52), (.30, .66)); ten(ax, (.50, .675), (.60, .675), color=C_PURP)
ten(ax, (.71, .60), (.71, .43), color=C_BAD); ten(ax, (.60, .355), (.50, .355), color=C_MAIN)
ten(ax, (.40, .43), (.40, .60), color=C_BLUE)
ten(ax, (.50, .72), (.86, .56), color=C_DIM, ls="--", rad=-.25)
ax.text(.66, .83, "khi đủ thông tin", fontsize=7.6, color=C_DIM, style="italic")
ax.text(.5, .12, "Mô hình chỉ đề xuất lời gọi tool dưới dạng dữ liệu có cấu trúc; chính ứng dụng quyết định có "
        "thực thi hay không.\nMỗi vòng lặp là một lần gọi mô hình, nên chi phí và rủi ro tăng theo số bước.",
        ha="center", fontsize=8.1, color="#333", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "ud09_agent.png"); plt.close(fig)


# ---------------------------------------------------------------------
# ud10 — Kiến trúc Model Context Protocol
# ---------------------------------------------------------------------
fig, ax = khung((9.4, 4.6))
ax.text(.5, .955, "Model Context Protocol: host, client, server", ha="center", fontsize=11, weight="bold")
hop(ax, .03, .22, .36, .64, "", fc="#fbfbfa", ec=C_BLUE, lw=1.4)
ax.text(.21, .81, "Host (ứng dụng: IDE, trợ lý chat, agent)", ha="center", fontsize=8.4, weight="bold", color=C_BLUE)
hop(ax, .06, .62, .30, .1, "LLM và phần điều phối\n(giữ toàn bộ hội thoại)", fc="#eaf0f8", ec=C_BLUE, fs=7.6)
for i, y in enumerate([.47, .36, .25]):
    hop(ax, .09, y, .24, .08, f"MCP client {i + 1}", fc=C_SOFT, ec=C_MAIN, fs=7.8)
ser = [("MCP server: tệp và Git\n(chạy cục bộ, qua stdio)", .62), ("MCP server: cơ sở dữ liệu\n(chạy cục bộ, qua stdio)", .43),
       ("MCP server: API bên ngoài\n(từ xa, qua Streamable HTTP)", .24)]
for (t, y), yc in zip(ser, [.51, .40, .29]):
    hop(ax, .52, y, .26, .13, t, fc="#f3edf8", ec=C_PURP, fs=7.5)
    ten(ax, (.33, yc), (.52, y + .065), color=C_MAIN, style="<|-|>")
hop(ax, .83, .30, .15, .40, "Server cung cấp:\n\n• tools\n(hàm để gọi)\n\n• resources\n(dữ liệu để đọc)\n\n• prompts\n(mẫu có sẵn)",
    fc="#fbfbfa", ec=C_DIM, fs=7.4)
ax.text(.5, .09, "Mỗi client nối với đúng một server bằng thông điệp JSON-RPC 2.0. Server chỉ thấy phần ngữ cảnh "
        "được gửi cho nó,\nkhông đọc được toàn bộ hội thoại và không nhìn thấy server khác.", ha="center",
        fontsize=8.1, color="#333", linespacing=1.5)
fig.tight_layout(); fig.savefig(OUT + "ud10_mcp.png"); plt.close(fig)


# ---------------------------------------------------------------------
# ud13 — Một trace của yêu cầu RAG có gọi tool
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 3.0))
spans = [("yêu cầu /chat", 0, 3150, 0, C_DIM), ("kiểm tra đầu vào", 0, 60, 1, C_BAD),
         ("embedding câu hỏi", 60, 95, 1, C_WARN), ("truy xuất vector", 95, 150, 1, C_WARN),
         ("xếp lại", 150, 330, 1, C_WARN), ("gọi LLM lần 1", 330, 1320, 1, C_BLUE),
         ("tool: tra đơn hàng", 1320, 1710, 1, C_PURP), ("gọi LLM lần 2", 1710, 3080, 1, C_BLUE),
         ("kiểm tra đầu ra", 3080, 3150, 1, C_BAD)]
row = 0
for i, (t, a, b, lv, col) in enumerate(spans):
    y = -i
    ax.barh(y, b - a, left=a, color=col, height=.62, alpha=.9)
    ax.text(-40, y, t, ha="right", va="center", fontsize=7.6)
    ax.text(b + 25, y, f"{b - a} ms", va="center", fontsize=7, color="#444")
ax.set_xlim(-900, 3700); ax.set_yticks([])
ax.set_xlabel("thời gian (ms)")
ax.set_title("Một trace: mỗi thanh là một span, lồng trong span của cả yêu cầu", fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.axvline(0, color=C_LINE, lw=.8)
fig.tight_layout(); fig.savefig(OUT + "ud13_trace.png"); plt.close(fig)


# ---------------------------------------------------------------------
# ud16 — Số token của một ảnh theo độ phân giải (mô hình kiểu ViT, ô 14 và 28 điểm ảnh)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 2.8))
canh = np.array([224, 336, 448, 672, 896, 1344])
for o, col, nhan in [(14, C_BAD, "ô 14 × 14 điểm ảnh"), (28, C_BLUE, "ô 28 × 28 (gộp 2 × 2 ô 14)")]:
    tok = (canh // o) ** 2
    ax.plot(canh, tok, "-o", ms=4, color=col, label=nhan)
    for c_, t_ in zip(canh, tok):
        if c_ in (448, 1344):
            ax.text(c_, t_ * 1.15, f"{t_:,}".replace(",", " "), ha="center", fontsize=7, color=col)
ax.set_yscale("log"); ax.set_xlabel("cạnh ảnh vuông (điểm ảnh)"); ax.set_ylabel("số token của ảnh (log)")
ax.set_title("Số token tăng theo bình phương độ phân giải", fontsize=9)
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud16_anh.png"); plt.close(fig)

print("Da luu 7 hinh: ud01, ud03, ud04, ud09, ud10, ud13, ud16")
