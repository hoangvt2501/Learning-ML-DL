"""Bài 8 — quét tham số α của SmoothQuant và đo sai số đầu ra.

Script này dựng lại đúng thí nghiệm (C) của `numpy_experiments.py`: cùng hạt giống 42
và cùng thứ tự rút số ngẫu nhiên, nên các mốc đã in trong giáo trình (per-tensor
7,7435% và SmoothQuant α=0,5 cho 1,4430%) được tái lập y hệt.

Chỉ cần NumPy. Nếu có matplotlib thì script vẽ thêm đường cong ra figs/fig18_alpha.png.

    python code/sweep_alpha.py
"""

import numpy as np

rng = np.random.default_rng(42)
rel = lambda a, r: np.linalg.norm(a - r) / np.linalg.norm(r)

# --- Rút lại đúng các mẫu của phần (A) và (B) để trạng thái RNG khớp với ------
# --- numpy_experiments.py khi bước vào phần (C). Không dùng tới kết quả. ------
K, N_OUT, BATCH = 256, 64, 1000
rng.normal(0.8, 1.0, (BATCH, K)).astype(np.float32)
rng.normal(0, 0.05, (N_OUT, K)).astype(np.float32)
rng.normal(0, 0.1, N_OUT).astype(np.float32)

d_in, d_out, n_cal = 256, 128, 4096
A = rng.normal(0, 1, (d_in, d_in)) * (np.linspace(1, 0.05, d_in) ** 2)[None, :]
(A @ rng.normal(0, 1, (d_in, n_cal))).astype(np.float64)
(A @ rng.normal(0, 1, (d_in, n_cal))).astype(np.float64)
rng.normal(0, 0.02, (d_out, d_in))

# --- Phần (C): activation có outlier theo kênh -------------------------------
T, Kc, Mo = 2048, 512, 512
Xl = rng.normal(0, 1, (T, Kc))
outlier_ch = rng.choice(Kc, 6, replace=False)
Xl[:, outlier_ch] *= 60
Wl = rng.normal(0, 0.02, (Mo, Kc))
Yl = Xl @ Wl.T


def q_sym(x, S, qmax=127):
    return np.clip(np.round(x / S), -qmax, qmax) * S


def w8_per_channel(W):                      # trọng số: mỗi kênh ĐẦU RA một scale
    return q_sym(W, np.abs(W).max(axis=1, keepdims=True) / 127)


def a8_per_tensor(X):                       # activation: một scale cho cả tensor
    return q_sym(X, np.abs(X).max() / 127)


ax = np.abs(Xl).max(axis=0)                 # biên độ activation theo kênh đầu vào
aw = np.abs(Wl).max(axis=0)                 # biên độ trọng số theo kênh đầu vào

spread = lambda v: v.max() / v.min()
base = rel(a8_per_tensor(Xl) @ w8_per_channel(Wl).T, Yl)

print("=== Trước biến đổi ===")
print(f"activation: chênh lệch giữa các kênh = {spread(ax):6.1f} lần")
print(f"trọng số  : chênh lệch giữa các cột  = {spread(aw):6.2f} lần")
print(f"sai số đầu ra W8A8 khi KHÔNG biến đổi = {base:.4%}\n")

print("=== Quét alpha ===")
print(f"{'alpha':>6} {'sai số đầu ra':>14} {'chênh kênh X':>14} {'chênh cột W':>13}")
rows = []
for alpha in np.round(np.arange(0.0, 1.0001, 0.05), 2):
    s = ax ** alpha / aw ** (1 - alpha)
    Xs, Ws = Xl / s, Wl * s
    assert np.allclose(Xs @ Ws.T, Yl)       # biến đổi tương đương về mặt toán học
    err = rel(a8_per_tensor(Xs) @ w8_per_channel(Ws).T, Yl)
    nx, nw = np.abs(Xs).max(axis=0), np.abs(Ws).max(axis=0)
    rows.append((alpha, err, spread(nx), spread(nw)))
    if abs(round(alpha, 2) * 10 % 1) < 1e-9:
        print(f"{alpha:6.2f} {err:13.4%} {spread(nx):13.1f}x {spread(nw):12.1f}x")

best = min(rows, key=lambda r: r[1])
a0 = rows[0]
print(f"\ncực tiểu trên lưới: alpha = {best[0]:.2f}, sai số = {best[1]:.4%}")
print(f"alpha = 0 cho {a0[1]:.4%}, tức TỆ HƠN cả khi không biến đổi ({base:.4%}) "
      f"— vì chia cho s = 1/max|W| làm chênh lệch giữa các kênh activation "
      f"tăng từ {spread(ax):.1f} lên {a0[2]:.1f} lần.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    al = [r[0] for r in rows]
    er = [r[1] * 100 for r in rows]
    fig, ax_ = plt.subplots(figsize=(6.4, 3.6))
    ax_.plot(al, er, "-o", ms=3.5, color="#1f6f68", label="SmoothQuant + activation per-tensor")
    ax_.axhline(base * 100, ls="--", lw=1.2, color="#b0413e", label=f"không biến đổi ({base:.2%})")
    ax_.axvline(best[0], ls=":", lw=1.2, color="#666")
    ax_.annotate(f"α* ≈ {best[0]:.2f}", (best[0], best[1] * 100),
                 textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax_.set_xlabel("α (mức chuyển độ khó từ activation sang trọng số)")
    ax_.set_ylabel("Sai số đầu ra tương đối (%)")
    ax_.set_title("Bài 8 — sai số W8A8 theo α của SmoothQuant")
    ax_.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/fig18_alpha.png", dpi=150)
    print("\nĐã lưu figs/fig18_alpha.png")
except ImportError:
    print("\n(Không có matplotlib — bỏ qua phần vẽ hình.)")
