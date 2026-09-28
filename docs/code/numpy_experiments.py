"""Thí nghiệm numpy: (A) lớp Linear chạy hoàn toàn bằng số nguyên, (B) GPTQ so với làm tròn thường (RTN),
(C) mô phỏng outlier activation trong LLM và cách SmoothQuant / LLM.int8() xử lý."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs") + os.sep
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ============================================================ (A) Integer-only Linear
def affine_params(lo, hi, qmin=0, qmax=255):
    lo, hi = min(lo, 0.0), max(hi, 0.0)            # dải phải chứa 0 để 0 biểu diễn chính xác
    S = (hi - lo) / (qmax - qmin)
    Z = int(np.clip(round(qmin - lo / S), qmin, qmax))
    return S, Z

def quantize_multiplier(M):
    """Viết M (0 < M < 1) dưới dạng M0 * 2^(-31-shift), M0 là int32 trong [2^30, 2^31)."""
    assert 0 < M < 1
    shift = 0
    while M < 0.5:
        M *= 2; shift += 1
    M0 = int(round(M * (1 << 31)))
    if M0 == (1 << 31):                          # làm tròn lên chạm 2^31
        M0 //= 2; shift -= 1
    return M0, shift

def rounding_right_shift(x, n):                   # chia cho 2^n, làm tròn gần nhất (x là int64)
    return (x + (1 << (n - 1))) >> n

K, N_OUT, BATCH = 256, 64, 1000
X = rng.normal(0.8, 1.0, (BATCH, K)).astype(np.float32)          # activation lệch dương (ví dụ sau GELU/Add)
W = rng.normal(0, 0.05, (N_OUT, K)).astype(np.float32)
b = rng.normal(0, 0.1, N_OUT).astype(np.float32)
Y = X @ W.T + b                                                          # tham chiếu FP32

S_x, Z_x = affine_params(X.min(), X.max())                   # activation: uint8 bất đối xứng
S_w = np.abs(W).max() / 127; Z_w = 0                        # weight: int8 đối xứng
S_y, Z_y = affine_params(Y.min(), Y.max())                   # output: uint8 bất đối xứng

q_x = np.clip(np.round(X / S_x) + Z_x, 0, 255).astype(np.int32)
q_w = np.clip(np.round(W / S_w), -127, 127).astype(np.int32)
q_b = np.round(b / (S_w * S_x)).astype(np.int32)            # bias int32, scale S_w*S_x, zero-point 0

# Lõi int8 x int8 -> cộng dồn int32 ; số hạng Z_x * sum(q_w) tính trước lúc convert
acc = q_x @ q_w.T - Z_x * q_w.sum(axis=1) + q_b            # int32
M = S_w * S_x / S_y
M0, shift = quantize_multiplier(M)
q_y = Z_y + rounding_right_shift(acc.astype(np.int64) * M0, 31 + shift)
q_y = np.clip(q_y, 0, 255)
Y_int = S_y * (q_y - Z_y)                                    # dequantize chỉ để so sánh

# Đường "mô phỏng" (fake-quant bằng float) để kiểm tra đường số nguyên tính đúng
Y_sim = (S_x * (q_x - Z_x)) @ (S_w * q_w).T + S_w * S_x * q_b
q_y_sim = np.clip(np.round(Y_sim / S_y) + Z_y, 0, 255)
rel = lambda a, r: np.linalg.norm(a - r) / np.linalg.norm(r)
print("=== (A) Integer-only Linear ===")
print(f"S_x={S_x:.6f} Z_x={Z_x} | S_w={S_w:.6f} | S_y={S_y:.6f} Z_y={Z_y}")
print(f"M = {M:.10f} = {M0} * 2^-(31+{shift}) -> {M0 / 2 ** (31 + shift):.10f}")
print("max |q_y(integer) - q_y(float sim)| =", int(np.abs(q_y - q_y_sim).max()), "LSB;",
      "tỉ lệ khác nhau:", f"{np.mean(q_y != q_y_sim):.4%}")
print(f"sai số tương đối so với FP32: {rel(Y_int, Y):.4%}")
print("max |acc| =", int(np.abs(acc).max()), "; giới hạn int16 =", 2 ** 15 - 1,
      "; cận trên lý thuyết 255*127*K =", 255 * 127 * K)

# ============================================================ (B) GPTQ vs RTN
def quant_rtn_rows(W, scales, qmax=7):
    return np.clip(np.round(W / scales[:, None]), -qmax, qmax) * scales[:, None]

def gptq(W, H, scales, qmax=7, damp=0.01):
    """GPTQ tối giản (không lazy-batch): lượng tử từng cột, bù sai số vào các cột chưa lượng tử."""
    W = W.copy(); d = W.shape[1]
    H = H + damp * np.mean(np.diag(H)) * np.eye(d)
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T            # Cholesky trên (upper) của H^-1
    Q = np.zeros_like(W)
    for i in range(d):
        w = W[:, i]
        q = np.clip(np.round(w / scales), -qmax, qmax) * scales
        Q[:, i] = q
        err = (w - q) / Hinv[i, i]
        W[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])
    return Q

d_in, d_out, n_cal = 256, 128, 4096
A = rng.normal(0, 1, (d_in, d_in)) * (np.linspace(1, 0.05, d_in) ** 2)[None, :]   # trộn -> đặc trưng tương quan
Xc = (A @ rng.normal(0, 1, (d_in, n_cal))).astype(np.float64)
Xt = (A @ rng.normal(0, 1, (d_in, n_cal))).astype(np.float64)
Wg = rng.normal(0, 0.02, (d_out, d_in))
scales = np.abs(Wg).max(axis=1) / 7                           # INT4 đối xứng, scale theo hàng
H = 2 * Xc @ Xc.T / n_cal
Q_rtn = quant_rtn_rows(Wg, scales)
Q_gptq = gptq(Wg, H, scales)
out_err = lambda Q: np.linalg.norm((Wg - Q) @ Xt) ** 2 / np.linalg.norm(Wg @ Xt) ** 2
w_err = lambda Q: np.mean((Wg - Q) ** 2)
print("\n=== (B) GPTQ vs RTN, INT4 theo hàng ===")
print(f"RTN : sai số trọng số MSE={w_err(Q_rtn):.3e}, sai số output tương đối={out_err(Q_rtn):.4%}")
print(f"GPTQ: sai số trọng số MSE={w_err(Q_gptq):.3e}, sai số output tương đối={out_err(Q_gptq):.4%}")
gptq_res = (w_err(Q_rtn), out_err(Q_rtn), w_err(Q_gptq), out_err(Q_gptq))

# ============================================================ (C) Outlier activation, LLM.int8(), SmoothQuant
T, Kc, Mo = 2048, 512, 512
Xl = rng.normal(0, 1, (T, Kc))
outlier_ch = rng.choice(Kc, 6, replace=False)
Xl[:, outlier_ch] *= 60                                          # vài kênh có biên độ lớn gấp hàng chục lần
Wl = rng.normal(0, 0.02, (Mo, Kc))
Yl = Xl @ Wl.T

def q_sym(x, S, qmax=127):
    return np.clip(np.round(x / S), -qmax, qmax) * S
def w8_per_channel(W):                                           # per-output-channel
    return q_sym(W, np.abs(W).max(axis=1, keepdims=True) / 127)
def a8_per_tensor(X):
    return q_sym(X, np.abs(X).max() / 127)
def a8_per_token(X):
    return q_sym(X, np.abs(X).max(axis=1, keepdims=True) / 127)

res = {}
res["W8A8, activation per-tensor"] = rel(a8_per_tensor(Xl) @ w8_per_channel(Wl).T, Yl)
res["W8A8, activation per-token"] = rel(a8_per_token(Xl) @ w8_per_channel(Wl).T, Yl)
# LLM.int8(): cột outlier (|x| > 6) tính FP16, phần còn lại int8 per-token x per-channel
mask = (np.abs(Xl) > 6.0).any(axis=0)
X_reg = Xl.copy(); X_reg[:, mask] = 0
W_reg = Wl.copy(); W_reg[:, mask] = 0
Y_mix = a8_per_token(X_reg) @ w8_per_channel(W_reg).T + Xl[:, mask] @ Wl[:, mask].T
res["LLM.int8() (tách outlier)"] = rel(Y_mix, Yl)
# SmoothQuant alpha = 0.5
alpha = 0.5
s = np.abs(Xl).max(axis=0) ** alpha / np.abs(Wl).max(axis=0) ** (1 - alpha)
Xs, Ws = Xl / s, Wl * s
assert np.allclose(Xs @ Ws.T, Yl)                                 # phép biến đổi tương đương toán học
res["SmoothQuant (α=0.5) + per-tensor"] = rel(a8_per_tensor(Xs) @ w8_per_channel(Ws).T, Yl)
print("\n=== (C) Outlier activation ===")
print("số cột được LLM.int8 tách:", int(mask.sum()), "; kênh outlier thật:", sorted(outlier_ch.tolist()))
for k, v in res.items(): print(f"{k:38s}: {v:.4%}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1.3]})
ch = np.arange(Kc)
axes[0].vlines(ch, 0, np.abs(Xl).max(0), color="#d9534f", lw=1.2)
axes[0].set_title("max|X| theo kênh — TRƯỚC khi làm mịn"); axes[0].set_xlabel("Kênh (chiều ẩn)")
axes[1].vlines(ch, 0, np.abs(Xs).max(0), color="#4a78c2", lw=1.2, label="max|X/s| (activation)")
axes[1].vlines(ch + 0.5, 0, -np.abs(Ws).max(0), color="#4caf6e", lw=1.2, label="−max|W·s| (trọng số, vẽ ngược xuống)")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_title("SAU SmoothQuant (α = 0.5)"); axes[1].set_xlabel("Kênh (chiều ẩn)"); axes[1].legend(frameon=False, fontsize=8)
names = list(res.keys()); vals = [res[n] * 100 for n in names]
axes[2].barh(range(len(names))[::-1], vals, color=["#d9534f", "#e0923a", "#4caf6e", "#4a78c2"])
axes[2].set_yticks(range(len(names))[::-1], [n.replace(", ", "\n").replace(" + ", "\n+ ") for n in names], fontsize=8.5)
axes[2].set_xscale("log"); axes[2].set_xlabel("Sai số output tương đối (%, thang log)")
for i, v in zip(range(len(names))[::-1], vals): axes[2].text(v * 1.1, i, f"{v:.2f}%", va="center", fontsize=8.5)
axes[2].set_xlim(0.1, max(vals) * 6); axes[2].set_title("W8A8 trên dữ liệu có outlier")
fig.suptitle("Mô phỏng: 6/512 kênh activation có biên độ gấp 60 lần (dữ liệu tổng hợp)", y=1.03)
fig.savefig(OUT + "fig14_llm_outliers.png"); plt.close(fig)

# Hình GPTQ
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
lab = ["RTN", "GPTQ"]
axes[0].bar(lab, [gptq_res[0], gptq_res[2]], color=["#9e9e9e", "#4a78c2"]); axes[0].set_title("Sai số TRỌNG SỐ (MSE)")
axes[1].bar(lab, [gptq_res[1] * 100, gptq_res[3] * 100], color=["#9e9e9e", "#4a78c2"]); axes[1].set_title("Sai số OUTPUT tương đối (%) trên dữ liệu kiểm tra")
for ax in axes: ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
fig.suptitle("INT4 cho một lớp Linear 256→128, dữ liệu đầu vào có tương quan (tổng hợp)", y=1.04)
fig.savefig(OUT + "fig15_gptq.png"); plt.close(fig)
