"""Sinh các hình minh họa phần nền tảng. Mọi số liệu trên hình đều được TÍNH TRỰC TIẾP bởi code này
(trừ Hình 1 lấy số liệu từ Horowitz, ISSCC 2014)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 110,
                     "savefig.bbox": "tight"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs") + os.sep
os.makedirs(OUT, exist_ok=True)
C_SIGN, C_EXP, C_MAN, C_INT = "#d9534f", "#4a78c2", "#4caf6e", "#9e9e9e"
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Hình 1: năng lượng
ops = [("Cộng INT8", 0.03), ("Cộng INT32", 0.1), ("Cộng FP16", 0.4), ("Cộng FP32", 0.9),
       ("Nhân INT8", 0.2), ("Nhân INT32", 3.1), ("Nhân FP16", 1.1), ("Nhân FP32", 3.7),
       ("Đọc cache 8 KB (64 bit)", 10), ("Đọc cache 1 MB (64 bit)", 100),
       ("Đọc DRAM (64 bit)", 1300)]
colors = ["#4caf6e"] * 2 + ["#4a78c2"] * 2 + ["#4caf6e"] * 2 + ["#4a78c2"] * 2 + ["#e0923a"] * 3
fig, ax = plt.subplots(figsize=(8, 4.6))
y = np.arange(len(ops))[::-1]
ax.barh(y, [v for _, v in ops], color=colors)
ax.set_yticks(y, [n for n, _ in ops]); ax.set_xscale("log")
for yi, (n, v) in zip(y, ops):
    lab = "1300–2600" if "DRAM" in n else f"{v:g}"
    ax.text((2600 if "DRAM" in n else v) * 1.15, yi, lab + " pJ", va="center", fontsize=9)
ax.set_xlim(0.01, 2e4); ax.set_xlabel("Năng lượng mỗi phép toán (pJ, thang log)")
ax.set_title("Năng lượng phép toán và truy cập bộ nhớ (công nghệ 45 nm, 0.9 V)")
ax.errorbar([1950], [y[-1]], xerr=[[650], [650]], fmt="none", ecolor="k", capsize=3)
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in ["#4caf6e", "#4a78c2", "#e0923a"]],
          labels=["Số nguyên", "Dấu phẩy động", "Bộ nhớ"], loc="upper right", frameon=False)
fig.text(0.01, -0.02, "Nguồn số liệu: M. Horowitz, “Computing's energy problem”, ISSCC 2014.", fontsize=8)
fig.savefig(OUT + "fig01_energy.png"); plt.close(fig)

# ---------------------------------------------------------------- Hình 2: bố cục bit
fmts = [("FP32", [("S", 1, C_SIGN), ("Exponent (8)", 8, C_EXP), ("Mantissa (23)", 23, C_MAN)]),
        ("FP16", [("S", 1, C_SIGN), ("Exp (5)", 5, C_EXP), ("Mantissa (10)", 10, C_MAN)]),
        ("BF16", [("S", 1, C_SIGN), ("Exponent (8)", 8, C_EXP), ("Man (7)", 7, C_MAN)]),
        ("FP8 E4M3", [("S", 1, C_SIGN), ("E (4)", 4, C_EXP), ("M (3)", 3, C_MAN)]),
        ("FP8 E5M2", [("S", 1, C_SIGN), ("E (5)", 5, C_EXP), ("M (2)", 2, C_MAN)]),
        ("INT8", [("Số nguyên bù 2 (8)", 8, C_INT)]),
        ("INT4", [("INT (4)", 4, C_INT)])]
fig, ax = plt.subplots(figsize=(9, 3.8))
for r, (name, parts) in enumerate(fmts):
    x0, yy = 0, len(fmts) - 1 - r
    for lab, nb, c in parts:
        ax.add_patch(Rectangle((x0, yy + 0.1), nb, 0.8, facecolor=c, edgecolor="white", lw=1.5))
        if nb >= 2 or lab == "S":
            ax.text(x0 + nb / 2, yy + 0.5, lab, ha="center", va="center", color="white", fontsize=8.5,
                    fontweight="bold")
        x0 += nb
    ax.text(-0.6, yy + 0.5, name, ha="right", va="center", fontweight="bold")
    ax.text(x0 + 0.5, yy + 0.5, f"{x0} bit", va="center", fontsize=9, color="#555")
ax.set_xlim(-6, 36); ax.set_ylim(0, len(fmts)); ax.axis("off")
ax.set_title("Bố cục bit của các kiểu dữ liệu thường gặp (S = bit dấu)")
fig.savefig(OUT + "fig02_bit_layout.png"); plt.close(fig)

# ---------------------------------------------------------------- Hình 3: lưới giá trị FP8 vs INT8
def e4m3_values():
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:      # mã NaN duy nhất của E4M3 (biến thể OCP "fn")
                continue
            vals.append((m / 8) * 2.0 ** (1 - 7) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return np.array(vals)
v8 = e4m3_values()
assert np.isclose(v8.max(), 448.0)
s_int8 = 448 / 127
vi8 = np.arange(0, 128) * s_int8
fig, axes = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True, gridspec_kw={"hspace": 0.6})
for ax, vals, title, c in [(axes[0], v8, "FP8 E4M3 (các giá trị dương biểu diễn được)", C_EXP),
                           (axes[1], vi8, f"INT8 đối xứng, cùng giá trị lớn nhất 448 (bước S = 448/127 ≈ {s_int8:.2f})", C_INT)]:
    vv = vals[vals <= 40]
    ax.vlines(vv, 0, 1, color=c, lw=1)
    ax.set_yticks([]); ax.set_title(title, fontsize=9.5, loc="left"); ax.spines["left"].set_visible(False)
axes[1].set_xlabel("Giá trị (chỉ hiển thị đoạn [0, 40])")
fig.suptitle("Lưới dấu phẩy động dày gần 0, thưa ở xa; lưới số nguyên cách đều", y=1.03)
fig.savefig(OUT + "fig03_fp_grid.png"); plt.close(fig)

# ---------------------------------------------------------------- Hình 4: bậc thang quantize
def quant_params(alpha, beta, qmin, qmax):
    S = (beta - alpha) / (qmax - qmin)
    Z = int(np.clip(np.round(qmin - alpha / S), qmin, qmax))
    return S, Z
def fake_quant(x, S, Z, qmin, qmax):
    q = np.clip(np.round(x / S) + Z, qmin, qmax)
    return S * (q - Z)
S, Z = quant_params(-1.0, 3.0, 0, 7)
x = np.linspace(-2, 4, 4000)
xh = fake_quant(x, S, Z, 0, 7)
lo, hi = S * (0 - Z), S * (7 - Z)
fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
ax.plot(x, x, "--", color="#999", label="Lý tưởng: x̂ = x")
ax.plot(x, xh, color=C_EXP, lw=2, label="x̂ = S·(q − Z)")
for a in axes:
    a.axvspan(-2, lo, color=C_SIGN, alpha=0.08); a.axvspan(hi, 4, color=C_SIGN, alpha=0.08)
ax.text(-1.95, 2.9, "vùng clipping", color=C_SIGN, fontsize=9)
ax.text(hi + 0.05, -1.6, "vùng clipping", color=C_SIGN, fontsize=9)
ax.set_ylabel("x̂ (sau quantize → dequantize)"); ax.legend(loc="upper left", bbox_to_anchor=(0.2, 1), frameon=False)
ax.set_title(f"Quantize 3 bit không dấu trên [−1, 3]:  S = 4/7 ≈ {S:.4f},  Z = {Z}\n"
             f"Dải biểu diễn thực tế sau khi làm tròn Z: [{lo:.3f}, {hi:.3f}]")
err = xh - x
axes[1].plot(x, err, color=C_SIGN)
axes[1].axhline(S / 2, ls=":", color="#555"); axes[1].axhline(-S / 2, ls=":", color="#555")
axes[1].text(-1.9, S / 2 + 0.05, "±S/2", fontsize=9)
axes[1].set_ylim(-1.3, 1.3); axes[1].set_ylabel("Sai số x̂ − x"); axes[1].set_xlabel("x")
fig.savefig(OUT + "fig04_staircase.png"); plt.close(fig)

# ---------------------------------------------------------------- Hình 5: đối xứng vs bất đối xứng trên dữ liệu sau ReLU
act = np.maximum(rng.normal(0.3, 1.0, 200_000), 0)
amax = act.max()
bits = 3
# đối xứng có dấu, dải hạn chế [-(2^{b-1}-1), 2^{b-1}-1]
qs = 2 ** (bits - 1) - 1
S_sym = amax / qs
lv_sym = np.arange(-qs, qs + 1) * S_sym
# bất đối xứng không dấu [0, 2^b - 1]
S_asym, Z_asym = quant_params(0.0, amax, 0, 2 ** bits - 1)
lv_asym = (np.arange(0, 2 ** bits) - Z_asym) * S_asym
def mse_sym(x, b):
    q = 2 ** (b - 1) - 1; s = np.abs(x).max() / q
    return np.mean((np.clip(np.round(x / s), -q, q) * s - x) ** 2)
def mse_asym(x, b):
    s, z = quant_params(x.min(), x.max(), 0, 2 ** b - 1)
    return np.mean((fake_quant(x, s, z, 0, 2 ** b - 1) - x) ** 2)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, lv, t in [(axes[0], lv_sym, f"Đối xứng ({bits} bit có dấu): {np.sum(lv_sym >= 0)}/{len(lv_sym)} mức dùng được"),
                  (axes[1], lv_asym, f"Bất đối xứng ({bits} bit không dấu): {len(lv_asym)}/{len(lv_asym)} mức dùng được")]:
    ax.hist(act, bins=120, range=(-amax, amax), color="#bbb", density=True)
    for l in lv:
        ax.axvline(l, color=C_SIGN if l < -1e-9 else C_EXP, lw=1.2, ls="--" if l < -1e-9 else "-")
    ax.set_title(t, fontsize=9.5); ax.set_xlabel("Giá trị activation sau ReLU"); ax.set_ylim(0, 0.6)
    ax.annotate(f"đỉnh tại 0 ({np.mean(act == 0):.0%} giá trị)\nbị cắt bớt cho dễ nhìn", xy=(0, 0.58), xytext=(-4.8, 0.45),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
axes[0].set_ylabel("Mật độ")
fig.suptitle("Mức lượng tử (đường dọc) đặt trên histogram activation; mức màu đỏ nét đứt bị lãng phí", y=1.04)
fig.savefig(OUT + "fig05_sym_asym.png"); plt.close(fig)
sym_asym_mse = {b: (mse_sym(act, b), mse_asym(act, b)) for b in (4, 8)}

# ---------------------------------------------------------------- Hình 6: đánh đổi rounding - clipping
w = rng.laplace(0, 1.0, 1_000_000)
bits = 4; q = 2 ** (bits - 1) - 1
cs = np.linspace(0.5, np.abs(w).max(), 400)
r_err, c_err = [], []
for c in cs:
    s = c / q
    inside = np.abs(w) <= c
    wq = np.clip(np.round(w / s), -q, q) * s
    e2 = (wq - w) ** 2
    r_err.append(e2[inside].sum() / w.size); c_err.append(e2[~inside].sum() / w.size)
r_err, c_err = np.array(r_err), np.array(c_err); tot = r_err + c_err
c_opt = cs[np.argmin(tot)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cs, r_err, label="Sai số làm tròn (rounding)", color=C_EXP)
ax.plot(cs, c_err, label="Sai số cắt (clipping)", color=C_SIGN)
ax.plot(cs, tot, label="Tổng MSE", color="k", lw=2)
ax.axvline(c_opt, ls="--", color="k"); ax.axvline(cs[-1], ls=":", color="#777")
ax.text(c_opt + 0.2, 2e-4, f"ngưỡng tối ưu c* ≈ {c_opt:.2f}\nMSE = {tot.min():.4f}", fontsize=9)
ax.text(cs[-1] - 0.2, 5e-3, f"min–max: c = max|w| = {cs[-1]:.2f}\nMSE = {tot[-1]:.4f}", ha="right", fontsize=9)
ax.set_yscale("log"); ax.set_xlabel("Ngưỡng cắt c  (dải quantize [−c, c])"); ax.set_ylabel("MSE (thang log)")
ax.set_title("INT4 đối xứng trên 10⁶ trọng số phân phối Laplace(0, 1)"); ax.legend(frameon=False)
fig.savefig(OUT + "fig07_round_clip.png"); plt.close(fig)
clip_stats = (c_opt, tot.min(), cs[-1], tot[-1])

# ---------------------------------------------------------------- Hình 7: SQNR theo số bit
def sqnr_db(x, xh): return 10 * np.log10(np.mean(x ** 2) / np.mean((x - xh) ** 2))
def q_sym(x, b, c):
    qq = 2 ** (b - 1) - 1; s = c / qq
    return np.clip(np.round(x / s), -qq, qq) * s
bs = np.arange(2, 11)
u = rng.uniform(-1, 1, 1_000_000); g = rng.normal(0, 1, 1_000_000)
s_uni = [sqnr_db(u, q_sym(u, b, 1.0)) for b in bs]
s_gmm = [sqnr_db(g, q_sym(g, b, np.abs(g).max())) for b in bs]
def best_clip(x, b):
    cands = np.linspace(0.5, np.abs(x).max(), 300)
    return max(sqnr_db(x, q_sym(x, b, c)) for c in cands)
g_small = g[:200_000]
s_gopt = [best_clip(g_small, b) for b in bs]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(bs, 20 * np.log10(2 ** bs - 2), "k--", label="Lý thuyết cho phân phối đều: 20·log₁₀(2ᵇ−2) ≈ 6.02·b dB")
ax.plot(bs, s_uni, "o", color=C_MAN, label="Thực nghiệm: dữ liệu đều U(−1, 1)")
ax.plot(bs, s_gmm, "s-", color=C_SIGN, label="Gauss N(0,1), dải min–max")
ax.plot(bs, s_gopt, "^-", color=C_EXP, label="Gauss N(0,1), ngưỡng cắt tối ưu")
ax.set_xlabel("Số bit b"); ax.set_ylabel("SQNR (dB)"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Mỗi bit thêm vào tăng SQNR khoảng 6 dB")
fig.savefig(OUT + "fig06_sqnr.png"); plt.close(fig)
sqnr_tab = list(zip(bs, s_uni, s_gmm, s_gopt))

# ---------------------------------------------------------------- Hình 8: STE
x = np.linspace(-1.6, 1.6, 3000)
qq = 3; s = 1 / qq
fq = np.clip(np.round(x / s), -qq, qq) * s
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(x, x, "--", color="#999", label="x"); axes[0].plot(x, fq, color=C_EXP, lw=2, label="FakeQuant(x)")
axes[0].set_title("Lượt xuôi (forward): hàm bậc thang"); axes[0].legend(frameon=False); axes[0].set_xlabel("x")
axes[1].plot(x, np.zeros_like(x), color=C_SIGN, lw=2, label="Đạo hàm thật: 0 hầu khắp nơi\n(không xác định tại các bước nhảy)")
axes[1].plot(x, ((x >= -1) & (x <= 1)).astype(float) + 0.02, color=C_MAN, lw=2, label="STE: 1 trong [α, β], 0 ngoài")
axes[1].set_ylim(-0.3, 1.4); axes[1].set_title("Lượt ngược (backward): đạo hàm dùng để cập nhật")
axes[1].legend(frameon=False, fontsize=8.5, loc="upper right"); axes[1].set_xlabel("x")
fig.savefig(OUT + "fig12_ste.png"); plt.close(fig)

# ---------------------------------------------------------------- Hình 9: NF4 vs INT4
offset = 0.9677083
v1 = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
v3 = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
nf4 = np.sort(np.concatenate([v1, [0.0], v3])); nf4 /= nf4.max()
int4 = np.arange(-7, 8) / 7.0
W = rng.normal(0, 1, (20_000, 64))
Wn = (W / np.abs(W).max(axis=1, keepdims=True)).ravel()      # chuẩn hóa absmax theo khối 64
def nearest(x, grid):
    idx = np.abs(x[:, None] - grid[None, :]).argmin(1); return grid[idx]
sub = Wn[:400_000]
mse_nf4 = np.mean((nearest(sub, nf4) - sub) ** 2); mse_int4 = np.mean((nearest(sub, int4) - sub) ** 2)
fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
for ax, grid, t, c in [(axes[0], int4, f"INT4 đối xứng (15 mức cách đều) — MSE = {mse_int4:.2e}", C_INT),
                       (axes[1], nf4, f"NF4 (16 mức theo phân vị phân phối chuẩn) — MSE = {mse_nf4:.2e}", C_EXP)]:
    ax.hist(Wn, bins=200, color="#ddd", density=True)
    for l in grid: ax.axvline(l, color=c, lw=1.3)
    ax.set_title(t, fontsize=9.5, loc="left"); ax.set_yticks([])
axes[1].set_xlabel("Trọng số sau khi chia cho absmax của khối 64 phần tử (w ~ N(0,1))")
fig.savefig(OUT + "fig16_nf4.png"); plt.close(fig)

# ---------------------------------------------------------------- in số liệu dùng trong bài
print("S,Z (3bit,[-1,3]):", S, Z, "range", lo, hi)
print("sym/asym MSE on ReLU act:", {b: (f"{a:.3e}", f"{c:.3e}") for b, (a, c) in sym_asym_mse.items()})
print("clip stats c_opt, mse_opt, c_max, mse_max:", clip_stats)
print("SQNR table (b, uniform, gauss-minmax, gauss-opt):")
for r in sqnr_tab: print("  %d  %.2f  %.2f  %.2f" % r)
print("NF4 levels:", np.round(nf4, 4)); print("mse nf4 / int4:", mse_nf4, mse_int4, mse_int4 / mse_nf4)
print("E4M3 #pos finite values:", len(v8), "min subnormal", v8[v8 > 0].min(), "max", v8.max())
