"""Số liệu và hình cho Mục 7.5–7.6 của giáo trình "Học sâu": IoU, NMS, AP và Dice.

Không huấn luyện mô hình nào: một bộ phát hiện được mô phỏng bằng cách làm nhiễu
hộp thật, để cô lập đúng phần cần học là cách ghép hộp, cách lọc trùng và cách tính AP.
Hạt giống cố định nên chạy lại cho kết quả y hệt.

    python code/models/thi_giac.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "figs/"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"


def head(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


def iou(a, b):
    """Hộp dạng (x1, y1, x2, y2)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, nguong):
    """Giữ hộp điểm cao nhất, bỏ mọi hộp còn lại trùng với nó quá ngưỡng, lặp lại."""
    thu_tu = list(np.argsort(-np.asarray(scores)))
    giu = []
    while thu_tu:
        i = thu_tu.pop(0)
        giu.append(i)
        thu_tu = [j for j in thu_tu if iou(boxes[i], boxes[j]) <= nguong]
    return giu


# =====================================================================
# (A) IoU tính tay
# =====================================================================
head("(A) IoU cua hai cap hop")
for a, b in [((0, 0, 4, 4), (2, 2, 6, 6)), ((0, 0, 4, 4), (1, 0, 5, 4)), ((0, 0, 4, 4), (0, 0, 4, 2))]:
    print(f"  {a} va {b}: IoU = {iou(a, b):.4f}")


# =====================================================================
# (B) NMS từng bước trên sáu hộp
# =====================================================================
head("(B) NMS tren sau hop du doan, nguong IoU 0.5")
VD = {  # tên: (x1, y1, x2, y2, điểm)
    "A": (10, 10, 50, 50, 0.92), "B": (12, 8, 52, 48, 0.85), "C": (16, 16, 56, 56, 0.60),
    "D": (60, 60, 90, 95, 0.80), "E": (63, 58, 93, 92, 0.55), "F": (30, 30, 70, 70, 0.40),
}
ten = list(VD)
hop = [VD[k][:4] for k in ten]
diem = [VD[k][4] for k in ten]
print("IoU giua cac hop:")
print("     " + "".join(f"{k:>7}" for k in ten))
for i, k in enumerate(ten):
    print(f"  {k}  " + "".join(f"{iou(hop[i], hop[j]):7.3f}" for j in range(len(ten))))
giu = nms(hop, diem, 0.5)
print("Giu lai:", [ten[i] for i in giu])


# =====================================================================
# (C) AP trên một bộ dữ liệu mô phỏng
# =====================================================================
head("(C) AP cua mot bo phat hien mo phong: 300 anh, mot lop doi tuong")
rng = np.random.default_rng(1207)
N_ANH = 300
gt, det = [], []      # gt[i]: các hộp thật của ảnh i; det: (ảnh, hộp, điểm)
for i in range(N_ANH):
    hops = []
    for _ in range(rng.integers(1, 5)):
        w, h = rng.uniform(15, 45, 2)
        x, y = rng.uniform(0, 100 - w), rng.uniform(0, 100 - h)
        hops.append((x, y, x + w, y + h))
    gt.append(hops)
    for (x1, y1, x2, y2) in hops:
        s = np.array([x2 - x1, y2 - y1, x2 - x1, y2 - y1])
        if rng.random() < 0.9:                               # phát hiện được
            b = np.array([x1, y1, x2, y2]) + rng.normal(0, 0.06, 4) * s
            det.append((i, tuple(b), float(np.clip(rng.normal(0.75, 0.12), 0.01, 0.99))))
        if rng.random() < 0.3:                               # thêm một hộp trùng, lệch hơn
            b = np.array([x1, y1, x2, y2]) + rng.normal(0, 0.12, 4) * s
            det.append((i, tuple(b), float(np.clip(rng.normal(0.55, 0.15), 0.01, 0.99))))
    for _ in range(rng.poisson(0.6)):                        # báo nhầm ở chỗ không có gì
        w, h = rng.uniform(10, 40, 2)
        x, y = rng.uniform(0, 100 - w), rng.uniform(0, 100 - h)
        det.append((i, (x, y, x + w, y + h), float(np.clip(rng.normal(0.35, 0.15), 0.01, 0.99))))
n_gt = sum(len(g) for g in gt)


def qua_nms(dets, nguong=0.5):
    ra = []
    for i in range(N_ANH):
        cua = [d for d in dets if d[0] == i]
        if not cua:
            continue
        giu_i = nms([d[1] for d in cua], [d[2] for d in cua], nguong)
        ra.extend(cua[k] for k in giu_i)
    return ra


def duong_pr(dets, nguong_iou):
    """Ghép kiểu COCO: xét hộp theo điểm giảm dần; mỗi hộp ghép với hộp thật chưa ghép
    có IoU lớn nhất; đủ ngưỡng thì là dương tính thật, không thì là dương tính giả."""
    da_ghep = [np.zeros(len(g), bool) for g in gt]
    tp = []
    for (i, b, s) in sorted(dets, key=lambda d: -d[2]):
        best, j_best = 0.0, -1
        for j, g in enumerate(gt[i]):
            if not da_ghep[i][j]:
                v = iou(b, g)
                if v > best:
                    best, j_best = v, j
        if best >= nguong_iou:
            da_ghep[i][j_best] = True
            tp.append(1)
        else:
            tp.append(0)
    tp = np.array(tp)
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(1 - tp)
    return tp_cum / n_gt, tp_cum / np.maximum(tp_cum + fp_cum, 1)


def ap_moi_diem(rec, prec):
    """Diện tích dưới đường bao của precision (cách tính của PASCAL VOC từ năm 2010)."""
    r = np.r_[0, rec, 1]
    p = np.r_[0, prec, 0]
    for k in range(len(p) - 2, -1, -1):
        p[k] = max(p[k], p[k + 1])
    idx = np.flatnonzero(r[1:] != r[:-1])
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def ap_101(rec, prec):
    """Trung bình precision nội suy tại 101 mức recall 0; 0,01; ...; 1 (cách tính của COCO)."""
    p = prec.copy()
    for k in range(len(p) - 2, -1, -1):
        p[k] = max(p[k], p[k + 1])
    out = []
    for t in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, t, side="left")
        out.append(p[k] if k < len(p) else 0.0)
    return float(np.mean(out))


sau_nms = qua_nms(det)
print(f"So hop that: {n_gt}; so hop du doan: {len(det)} truoc NMS, {len(sau_nms)} sau NMS (nguong 0,5)")
r0, p0 = duong_pr(det, 0.5)
r5, p5 = duong_pr(sau_nms, 0.5)
r75, p75 = duong_pr(sau_nms, 0.75)
print(f"{'':28}{'AP (moi diem)':>15}{'AP (101 diem)':>15}{'recall cuoi':>13}")
print(f"{'IoU 0,5, khong NMS':28}{ap_moi_diem(r0, p0):15.3f}{ap_101(r0, p0):15.3f}{r0[-1]:13.3f}")
print(f"{'IoU 0,5, co NMS':28}{ap_moi_diem(r5, p5):15.3f}{ap_101(r5, p5):15.3f}{r5[-1]:13.3f}")
print(f"{'IoU 0,75, co NMS':28}{ap_moi_diem(r75, p75):15.3f}{ap_101(r75, p75):15.3f}{r75[-1]:13.3f}")
aps = []
for t in np.arange(0.5, 0.96, 0.05):
    rt, pt = duong_pr(sau_nms, t)
    aps.append(ap_101(rt, pt))
print("AP 101 diem theo nguong IoU:", " ".join(f"{t:.2f}:{a:.3f}" for t, a in zip(np.arange(0.5, 0.96, 0.05), aps)))
print(f"AP@[0,5:0,95] (trung binh 10 nguong, kieu COCO) = {np.mean(aps):.3f}")


# =====================================================================
# (D) Dice và IoU
# =====================================================================
head("(D) Dice = 2 IoU / (1 + IoU)")
for v in [0.3, 0.5, 0.7, 0.9]:
    print(f"  IoU {v:.1f} -> Dice {2 * v / (1 + v):.3f}")


# =====================================================================
# Hình
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
a0 = ax[0]
for i, k in enumerate(ten):
    x1, y1, x2, y2 = hop[i]
    giu_k = i in giu
    a0.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           lw=2.0 if giu_k else 1.0, ls="-" if giu_k else "--",
                           ec=C_MAIN if giu_k else C_BAD))
    # vị trí nhãn chọn tay để các nhãn không đè lên nhau
    vt = {"A": (x1 + 1, y1 + 1, "left", "top"), "B": (x2 - 1, y1 - 1, "right", "bottom"),
          "C": (x2 - 1, y2 - 1, "right", "bottom"), "D": (x1 + 1, y1 + 1, "left", "top"),
          "E": (x2 - 1, y1 - 1, "right", "bottom"), "F": (x1 + 1, y2 - 1, "left", "bottom")}[k]
    a0.annotate(f"{k} {diem[i]:.2f}".replace(".", ","), vt[:2], fontsize=7.5,
                ha=vt[2], va=vt[3], color=C_MAIN if giu_k else C_BAD)
a0.set_xlim(0, 100); a0.set_ylim(100, 0); a0.set_aspect("equal")
a0.set_title("NMS, ngưỡng IoU 0,5: nét liền giữ lại, nét đứt bị loại", fontsize=8.5)
a0.grid(False)

a1 = ax[1]
a1.plot(r0, p0, color=C_DIM, lw=1.3, ls="--", label=f"IoU 0,5, không NMS (AP {ap_moi_diem(r0, p0):.2f})".replace(".", ",", 1))
a1.plot(r5, p5, color=C_MAIN, lw=1.8, label=f"IoU 0,5, có NMS (AP {ap_moi_diem(r5, p5):.2f})".replace(".", ",", 1))
a1.plot(r75, p75, color=C_BAD, lw=1.6, label=f"IoU 0,75, có NMS (AP {ap_moi_diem(r75, p75):.2f})".replace(".", ",", 1))
a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
a1.set_xlabel("recall"); a1.set_ylabel("precision")
a1.set_title("Đường precision–recall của bộ phát hiện mô phỏng", fontsize=8.5)
a1.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "models15_detect.png"); plt.close(fig)
head("Xong. Da luu figs/models15_detect.png")
