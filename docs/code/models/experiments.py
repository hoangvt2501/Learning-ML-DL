"""Thí nghiệm sinh số liệu và hình cho giáo trình "Học sâu".

Mọi con số thực nghiệm trong tài liệu đều in ra từ script này. Hạt giống cố định
nên chạy lại cho kết quả y hệt. Chỉ cần NumPy, SciPy, scikit-learn, matplotlib.

    python code/models/experiments.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

OUT = "figs/"
rng = np.random.default_rng(7)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"


def head(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


# =====================================================================
# (A) Hình 2 — Phân rã độ chệch / phương sai, đo bằng mô phỏng
# =====================================================================
head("(A) Phan ra do chech - phuong sai cua da thuc")

NOISE, N_TRAIN, N_SETS = 0.35, 40, 250
f_true = lambda x: np.sin(2.2 * x) + 0.35 * x
x_test = np.linspace(-3, 3, 200)
y_true = f_true(x_test)

degrees = list(range(1, 15))
bias2, var, total = [], [], []
for d in degrees:
    preds = np.empty((N_SETS, len(x_test)))
    for s in range(N_SETS):
        r = np.random.default_rng(1000 + s)
        xt = r.uniform(-3, 3, N_TRAIN)
        yt = f_true(xt) + r.normal(0, NOISE, N_TRAIN)
        # Polynomial.fit chuan hoa mien truoc khi khop nen on dinh so o bac cao
        pf = np.polynomial.Polynomial.fit(xt, yt, d)
        preds[s] = pf(x_test)
    mean_pred = preds.mean(axis=0)
    b2 = np.mean((mean_pred - y_true) ** 2)
    v = np.mean(preds.var(axis=0))
    bias2.append(b2); var.append(v); total.append(b2 + v + NOISE ** 2)

best = int(np.argmin(total))
print(f"{'bậc':>4}{'độ chệch²':>14}{'phương sai':>13}{'nhiễu²':>10}{'tổng MSE':>12}")
for i, d in enumerate(degrees):
    star = "  <- nhỏ nhất" if i == best else ""
    print(f"{d:4d}{bias2[i]:14.4f}{var[i]:13.4f}{NOISE**2:10.4f}{total[i]:12.4f}{star}")
print(f"\nBac toi uu = {degrees[best]}; tai do do chech^2={bias2[best]:.4f}, phuong sai={var[best]:.4f}")
print(f"Bac 1  : do chech^2 = {bias2[0]:.3f} (chiem {bias2[0]/total[0]*100:.0f}% MSE) -> UNDERFITTING")
print(f"Bac {degrees[-1]}: phuong sai  = {var[-1]:.3f} (chiem {var[-1]/total[-1]*100:.0f}% MSE) -> OVERFITTING")
# Ước lượng độ chệch² từ N_SETS lần khớp bị cộng thêm sai số Monte Carlo cỡ phương sai / N_SETS
print(f"Sai so Monte Carlo cua uoc luong do chech^2 (~ phuong sai / {N_SETS}): "
      f"bac 12 ~ {var[11]/N_SETS:.2f}, bac 13 ~ {var[12]/N_SETS:.2f}, bac 14 ~ {var[13]/N_SETS:.2f}")

fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(degrees, bias2, "-o", ms=4, color=C_MAIN, label="độ chệch²")
ax.plot(degrees, var, "-o", ms=4, color=C_BAD, label="phương sai")
ax.plot(degrees, total, "-o", ms=4, color=C_WARN, label="tổng MSE")
ax.axhline(NOISE ** 2, ls=":", color=C_DIM, lw=1.2)
ax.annotate("nhiễu không thể giảm", (1.2, NOISE ** 2 * 1.15), fontsize=8, color=C_DIM)
ax.axvline(degrees[best], ls="--", lw=1.1, color=C_DIM)
ax.set_yscale("log"); ax.set_xlabel("bậc đa thức"); ax.set_ylabel("sai số bình phương (log)")
ax.set_title("Phân rã đo được: MSE = độ chệch² + phương sai + nhiễu²")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "models02_biasvar.png"); plt.close(fig)


# =====================================================================
# (B) Hình 4 — Bagging giảm phương sai, boosting giảm độ chệch
# =====================================================================
head("(B) Bagging va boosting cat vao thanh phan nao")

def make(seed, n=120):
    r = np.random.default_rng(seed)
    x = r.uniform(-3, 3, n)
    return x.reshape(-1, 1), f_true(x) + r.normal(0, NOISE, n)

models = {
    "Một cây (sâu 8)": lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
    "Bagging 100 cây": lambda: BaggingRegressor(
        DecisionTreeRegressor(max_depth=8), n_estimators=100, random_state=0),
    "Boosting 100 gốc": lambda: GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.1, random_state=0),
}
Xte = x_test.reshape(-1, 1)
rows = []
for name, mk in models.items():
    preds = np.empty((40, len(x_test)))
    for s in range(40):
        X, y = make(2000 + s)
        preds[s] = mk().fit(X, y).predict(Xte)
    mp = preds.mean(axis=0)
    b2 = np.mean((mp - y_true) ** 2); v = np.mean(preds.var(axis=0))
    rows.append((name, b2, v, b2 + v))
    print(f"{name:<20} độ chệch²={b2:.4f}  phương sai={v:.4f}  tổng={b2+v:.4f}")
base = rows[0]
print(f"\nBagging giam phuong sai {base[2]/rows[1][2]:.1f} lan, do chech doi {rows[1][1]/base[1]:.2f} lan")
print(f"Boosting giam do chech {base[1]/rows[2][1]:.1f} lan, phuong sai doi {rows[2][2]/base[2]:.2f} lan")

fig, ax = plt.subplots(figsize=(5.6, 3.1))
xs = np.arange(len(rows)); w = .38
ax.bar(xs - w/2, [r[1] for r in rows], w, label="độ chệch²", color=C_MAIN)
ax.bar(xs + w/2, [r[2] for r in rows], w, label="phương sai", color=C_BAD)
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel("đóng góp vào MSE"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Độ chệch² và phương sai của ba mô hình")
fig.tight_layout(); fig.savefig(OUT + "models04_ensemble.png"); plt.close(fig)


# =====================================================================
# (C) Hình 5 — Độ sâu mua được sự gọn gàng: chứng minh bằng kiến tạo
# =====================================================================
head("(C) Sau mua duoc su gon gang (chung minh kien tao, khong phu thuoc toi uu)")

# Ham leu: g(x) = 2*relu(x) - 4*relu(x - 1/2). Dung DUNG 2 don vi ReLU.
def tent(x):
    return 2 * np.maximum(x, 0.0) - 4 * np.maximum(x - 0.5, 0.0)

def deep_sawtooth(x, k):
    """Hop k lan ham leu -> song rang cua co 2^k doan tuyen tinh, dung 2k don vi."""
    h = x
    for _ in range(k):
        h = tent(h)
    return h

xg = np.linspace(0, 1, 20001)

def one_layer_best_mse(y, width):
    """Mang MOT lop an tot nhat co the voi `width` don vi ReLU.

    Dat diem gay deu tren (0,1) roi giai binh phuong toi thieu cho lop ra.
    Day la CAN TREN cua sai so ma mang mot lop dat duoc, va vi diem gay dat deu
    la vi tri toi uu cho ham tuan hoan nay nen no rat sat can duoi.
    """
    b = np.linspace(0, 1, width + 2)[1:-1]
    Phi = np.concatenate([np.maximum(xg[:, None] - b[None, :], 0.0),
                          xg[:, None], np.ones((len(xg), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return float(np.mean((Phi @ coef - y) ** 2))

widths = [2, 4, 8, 16, 32, 64, 128, 256]
print(f"{'k lớp sâu':>10}{'đoạn 2^k':>10}{'tham số sâu':>13}   sai số của mạng MỘT lớp theo bề rộng")
print(f"{'':>33}   " + "".join(f"{w:>9}" for w in widths))
Crows = []
for k in [2, 3, 4, 5, 6, 7]:
    y = deep_sawtooth(xg, k)
    deep_params = 6 * k          # moi lop: 2 trong so vao + 2 do lech + 2 trong so ra
    errs = [one_layer_best_mse(y, w) for w in widths]
    # be rong nho nhat de tot hon DANG KE so voi du doan hang so
    baseline = float(np.var(y))
    need = next((w for w, e in zip(widths, errs) if e < 0.5 * baseline), None)
    Crows.append((k, 2 ** k, deep_params, need, errs))
    print(f"{k:>10}{2**k:>10}{deep_params:>13}   " + "".join(f"{e:>9.1e}" for e in errs))
print()
print(f"{'k':>3}{'đoạn 2^k':>10}{'tham số mạng sâu':>19}{'bề rộng 1 lớp cần':>20}{'tham số 1 lớp':>16}{'gấp':>7}")
for k, npieces, dp, need, _ in Crows:
    sp = need * 3 + 2 if need else float("nan")
    print(f"{k:>3}{npieces:>10}{dp:>19}{need if need else 0:>20}{sp:>16.0f}{sp/dp:>7.1f}x")
print("\nBe rong can thiet dung bang so DOAN TUYEN TINH = 2^k, tuc TANG THEO HAM MU cua do sau,")
print("trong khi mang sau chi can TUYEN TINH theo k. Day la 'do sau mua duoc su gon gang'.")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
for k, npieces, _, _, errs in Crows:
    ax[0].plot(widths, np.maximum(errs, 1e-12), "-o", ms=4, label=f"$k={k}$ ({npieces} đoạn)")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xlabel("bề rộng của mạng MỘT lớp ẩn"); ax[0].set_ylabel("MSE nhỏ nhất đạt được (log)")
ax[0].set_title("Sai số của mạng một lớp ẩn theo bề rộng")
ax[0].legend(frameon=False, fontsize=7.5)
ks = [c[0] for c in Crows]
ax[1].plot(ks, [c[2] for c in Crows], "-o", ms=5, color=C_MAIN, label="mạng sâu $k$ lớp")
ax[1].plot(ks, [c[3] * 3 + 2 for c in Crows], "-o", ms=5, color=C_BAD, label="mạng một lớp")
ax[1].set_yscale("log"); ax[1].set_xlabel("số lớp $k$ của mạng sâu")
ax[1].set_ylabel("số tham số cần (log)"); ax[1].set_xticks(ks)
ax[1].set_title("Tuyến tính theo $k$ so với hàm mũ theo $k$")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "models05_depth.png"); plt.close(fig)


# =====================================================================
# (D) Hình 7 — Gradient tiêu biến, và ba cách chữa
# =====================================================================
head("(D) Gradient truyen nguoc qua 40 lop: do bang |dL/da| tung lop")

def layernorm_fwd(z, eps=1e-5):
    mu = z.mean(-1, keepdims=True)
    sd = np.sqrt(z.var(-1, keepdims=True) + eps)
    return (z - mu) / sd, sd

def layernorm_bwd(dy, y, sd):
    """Dao ham chinh xac cua LayerNorm theo truc dac trung."""
    n = y.shape[-1]
    return (dy - dy.mean(-1, keepdims=True) - y * (dy * y).mean(-1, keepdims=True)) / sd

def act_grad_profile(depth=40, width=128, gain=np.sqrt(2.0), mode="plain", seed=0):
    """Tra ve |dL/da_i| tai tung lop, chuan hoa theo lop tren cung."""
    r = np.random.default_rng(seed)
    Ws = [r.normal(0, gain / np.sqrt(width), (width, width)) for _ in range(depth)]
    a = r.normal(0, 1, (128, width))
    cache = []
    for W in Ws:
        z = a @ W
        y, sd = (None, None)
        if mode in ("layernorm", "prenorm"):
            z, sd = layernorm_fwd(z)
            y = z
        h = np.maximum(z, 0)
        if mode in ("residual", "prenorm"):
            h = h + a
        cache.append((a, z, y, sd)); a = h
    g = r.normal(0, 1, a.shape); g /= np.linalg.norm(g)
    norms = [np.linalg.norm(g)]
    for i in range(depth - 1, -1, -1):
        a_in, z, y, sd = cache[i]
        g_skip = g if mode in ("residual", "prenorm") else None
        gz = g * (z > 0)
        if mode in ("layernorm", "prenorm"):
            gz = layernorm_bwd(gz, y, sd)
        g = gz @ Ws[i].T
        if g_skip is not None:
            g = g + g_skip
        norms.append(np.linalg.norm(g))
    return np.array(norms[::-1]) / norms[0]

cases = [("gain 0,5 — khởi tạo quá nhỏ", dict(gain=0.5, mode="plain")),
         ("gain √2 — khởi tạo He", dict(gain=np.sqrt(2.0), mode="plain")),
         ("gain 2,0 — khởi tạo quá lớn", dict(gain=2.0, mode="plain")),
         ("He + LayerNorm", dict(gain=np.sqrt(2.0), mode="layernorm")),
         ("He + kết nối tắt, KHÔNG chuẩn hoá", dict(gain=np.sqrt(2.0), mode="residual")),
         ("He + chuẩn hoá + kết nối tắt (pre-LN)", dict(gain=np.sqrt(2.0), mode="prenorm"))]
print(f"{'cách dựng lớp':<40}{'|∇a| lớp 40':>14}{'lớp 20':>13}{'lớp 1':>13}{'lớp1/lớp40':>14}")
Dprof = {}
for lab, kw in cases:
    p = np.mean([act_grad_profile(seed=s, **kw) for s in range(5)], axis=0)
    Dprof[lab] = p
    print(f"{lab:<40}{p[-1]:14.3e}{p[20]:13.3e}{p[0]:13.3e}{p[0]/p[-1]:14.3e}")
print("\nDoc bang: ti le lop1/lop40 << 1 la TIEU BIEN, >> 1 la BUNG NO, ~1 la on dinh.")

fig, ax = plt.subplots(figsize=(6.2, 3.2))
for lab, p in Dprof.items():
    ax.plot(np.arange(len(p)), p, lw=1.7, label=lab)
ax.set_yscale("log"); ax.set_xlabel("chỉ số lớp (0 = gần đầu vào, 40 = gần hàm mất mát)")
ax.set_ylabel("$\\|\\partial L/\\partial a\\|$ tương đối (log)")
ax.set_title("Cùng độ sâu 40, chỉ khác khởi tạo và cách dựng lớp")
ax.legend(frameon=False, fontsize=7.8)
fig.tight_layout(); fig.savefig(OUT + "models07_vanish.png"); plt.close(fig)


# =====================================================================
# (E) Hình 10 — Vì sao attention phải chia cho căn d_k
# =====================================================================
head("(E) Vi sao chia cho can d_k")

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

print(f"{'d_k':>6}{'Var(q·k)':>12}{'Var lý thuyết':>15}{'entropy KHÔNG chia':>21}{'entropy CÓ chia':>18}{'max softmax':>13}")
dks = [4, 16, 64, 256, 1024]
E = []
for dk in dks:
    r = np.random.default_rng(3)
    q = r.normal(0, 1, (4000, dk)); k = r.normal(0, 1, (64, dk))
    s = q @ k.T
    ent_raw = -np.sum(softmax(s) * np.log(softmax(s) + 1e-12), axis=-1).mean()
    ps = softmax(s / np.sqrt(dk))
    ent_sc = -np.sum(ps * np.log(ps + 1e-12), axis=-1).mean()
    mx = softmax(s).max(axis=-1).mean()
    E.append((dk, s.var(), dk, ent_raw, ent_sc, mx))
    print(f"{dk:6d}{s.var():12.2f}{dk:15d}{ent_raw:21.4f}{ent_sc:18.4f}{mx:13.4f}")
print(f"\nEntropy toi da khi deu tren 64 khoa: ln(64) = {np.log(64):.4f}")
print("Khong chia: entropy sup ve 0 -> softmax bao hoa -> gradient gan bang 0.")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
ax[0].plot(dks, [e[1] for e in E], "-o", ms=4, color=C_MAIN, label="đo được")
ax[0].plot(dks, dks, ":", color=C_DIM, lw=1.4, label="lý thuyết $d_k$")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("$d_k$"); ax[0].set_ylabel("phương sai của $q\\cdot k$")
ax[0].set_title("Phương sai của tích vô hướng theo $d_k$"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(dks, [e[3] for e in E], "-o", ms=4, color=C_BAD, label="không chia")
ax[1].plot(dks, [e[4] for e in E], "-o", ms=4, color=C_MAIN, label="chia $\\sqrt{d_k}$")
ax[1].axhline(np.log(64), ls=":", color=C_DIM, lw=1.2)
ax[1].annotate("entropy tối đa ln 64", (dks[0], np.log(64) * 0.93), fontsize=7.5, color=C_DIM)
ax[1].set_xscale("log"); ax[1].set_xlabel("$d_k$"); ax[1].set_ylabel("entropy của softmax (nat)")
ax[1].set_title("Entropy của phân phối attention"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "models10_scaling.png"); plt.close(fig)


# =====================================================================
# (F) Hình 9 — RNN quên, LSTM nhớ
# =====================================================================
head("(F) Gradient theo thoi gian: RNN so voi LSTM")

def rnn_grad(T=100, n=64, gain=1.0, seed=0):
    r = np.random.default_rng(seed)
    W = r.normal(0, gain / np.sqrt(n), (n, n))
    h = r.normal(0, 1, n) * 0.1
    hs, zs = [h], []
    for _ in range(T):
        z = W @ h; h = np.tanh(z); zs.append(z); hs.append(h)
    g = r.normal(0, 1, n); g /= np.linalg.norm(g)
    out = []
    for t in range(T - 1, -1, -1):
        g = (W.T @ (g * (1 - np.tanh(zs[t]) ** 2)))
        out.append(np.linalg.norm(g))
    return np.array(out[::-1])

def lstm_grad(T=100, n=64, f_bias=2.0, seed=0):
    """Chi theo duong trang thai o nho c_t: dc_{t-1} = dc_t * f_t."""
    r = np.random.default_rng(seed)
    g = 1.0; out = []
    for _ in range(T):
        f = 1 / (1 + np.exp(-(r.normal(0, 0.5) + f_bias)))   # cong quen mo san
        g = g * f; out.append(g)
    return np.array(out[::-1])

F = {}
for gain, lab in [(0.9, "RNN, gain 0,9"), (1.0, "RNN, gain 1,0"), (1.2, "RNN, gain 1,2")]:
    F[lab] = np.mean([rnn_grad(gain=gain, seed=s) for s in range(8)], axis=0)
for fb, lab in [(1.0, "LSTM, bias cổng quên = 1"), (4.0, "LSTM, bias cổng quên = 4")]:
    F[lab] = np.mean([lstm_grad(f_bias=fb, seed=s) for s in range(8)], axis=0)

print(f"{'mô hình':<26}{'|∇| sau 10 bước':>18}{'sau 50 bước':>14}{'sau 100 bước':>15}")
for lab, gs in F.items():
    g0 = gs[-1]
    print(f"{lab:<26}{gs[-10]/g0:18.3e}{gs[-50]/g0:14.3e}{gs[0]/g0:15.3e}")

fig, ax = plt.subplots(figsize=(6.0, 3.1))
for lab, gs in F.items():
    ax.plot(np.arange(len(gs))[::-1], gs / gs[-1], lw=1.6, label=lab)
ax.set_yscale("log"); ax.set_xlabel("số bước lùi về quá khứ")
ax.set_ylabel("chuẩn gradient tương đối (log)")
ax.set_title("Gradient truyền ngược qua thời gian")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "models09_rnn.png"); plt.close(fig)


# =====================================================================
# (G) Hình 12 — RoPE chỉ phụ thuộc khoảng cách tương đối
# =====================================================================
head("(G) RoPE: tich vo huong chi phu thuoc m - n")

def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    i = np.arange(0, d, 2)
    theta = base ** (-i / d)
    ang = pos * theta
    c, s = np.cos(ang), np.sin(ang)
    out = x.copy()
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out

d = 64
r = np.random.default_rng(11)
q, k = r.normal(0, 1, d), r.normal(0, 1, d)
print(f"{'m':>5}{'n':>5}{'m-n':>6}{'<R_m q, R_n k>':>18}")
vals = {}
for m, n in [(5, 2), (105, 102), (500, 497), (7, 4), (9, 3), (109, 103)]:
    v = float(rope(q, m) @ rope(k, n))
    vals.setdefault(m - n, []).append(v)
    print(f"{m:5d}{n:5d}{m-n:6d}{v:18.6f}")
ok = all(np.allclose(v, v[0], atol=1e-9) for v in vals.values())
print(f"\nCung m-n thi tich vo huong giong nhau: {ok}")

# Suy giam theo khoang cach: do bang CUNG mot vector o hai vi tri, chuan hoa theo |q|^2.
dists = np.arange(0, 256)
sims = []
for D_ in dists:
    acc = []
    for _ in range(300):
        v = r.normal(0, 1, d)
        acc.append(float(rope(v, D_) @ rope(v, 0)) / float(v @ v))
    sims.append(np.mean(acc))
sims = np.array(sims)
print(f"<R_m q, R_0 q>/|q|^2 trung binh: khoang cach 0 -> {sims[0]:.3f}; "
      f"8 -> {sims[8]:.3f}; 64 -> {sims[64]:.3f}; 255 -> {sims[255]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
ds = sorted(vals)
ax[0].bar([str(x) for x in ds], [vals[x][0] for x in ds], color=C_MAIN)
ax[0].set_xlabel("khoảng cách tương đối $m-n$"); ax[0].set_ylabel("$\\langle R_m q, R_n k\\rangle$")
ax[0].set_title("Cùng $m-n$ cho cùng một giá trị")
ax[1].plot(dists, sims, lw=1.4, color=C_MAIN)
ax[1].axhline(0, lw=.8, color=C_DIM)
ax[1].set_xlabel("khoảng cách tương đối"); ax[1].set_ylabel(r"$\langle R_m q, R_0 q\rangle / |q|^2$")
ax[1].set_title("Suy giảm theo khoảng cách")
fig.tight_layout(); fig.savefig(OUT + "models12_rope.png"); plt.close(fig)


# =====================================================================
# (H) Hình 14 — Các chiến lược giải mã thay đổi phân phối thế nào
# =====================================================================
head("(H) Nhiet do, top-k, top-p")

V = 50000
r = np.random.default_rng(1)
logits = np.sort(r.normal(0, 2.2, V))[::-1]        # phan phoi dang Zipf thoai
p0 = softmax(logits)
print(f"Phan phoi goc: entropy {-(p0*np.log(p0+1e-12)).sum():.3f} nat, "
      f"top-1 {p0[0]:.3f}, so token phu 90% xac suat: {np.searchsorted(np.cumsum(p0), 0.9)+1}")

def apply_temp(lg, T): return softmax(lg / T)
def apply_topk(lg, k):
    out = np.full_like(lg, -np.inf); out[:k] = lg[:k]; return softmax(out)
def apply_topp(lg, p):
    pr = softmax(lg); c = np.cumsum(pr); n = int(np.searchsorted(c, p)) + 1
    out = np.full_like(lg, -np.inf); out[:n] = lg[:n]; return softmax(out)

print(f"\n{'chiến lược':<22}{'entropy (nat)':>15}{'số token phủ 90%':>20}{'top-1':>10}")
H = []
rows2 = [("gốc (T = 1)", p0), ("T = 0,7", apply_temp(logits, 0.7)),
         ("T = 1,3", apply_temp(logits, 1.3)), ("top-k = 40", apply_topk(logits, 40)),
         ("top-p = 0,9", apply_topp(logits, 0.9)), ("top-p = 0,5", apply_topp(logits, 0.5))]
for name, p in rows2:
    ent = -(p * np.log(p + 1e-12)).sum()
    n90 = int(np.searchsorted(np.cumsum(p), 0.9)) + 1
    H.append((name, ent, n90, p[0]))
    print(f"{name:<22}{ent:15.4f}{n90:20d}{p[0]:10.4f}")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
for name, p in rows2[:4]:
    ax[0].plot(np.arange(1, 201), p[:200], lw=1.4, label=name)
ax[0].set_yscale("log"); ax[0].set_xlabel("hạng của token"); ax[0].set_ylabel("xác suất (log)")
ax[0].set_title("Hình dạng phân phối"); ax[0].legend(frameon=False, fontsize=7.5)
ax[1].barh([h[0] for h in H], [h[2] for h in H], color=C_MAIN)
ax[1].set_xscale("log"); ax[1].set_xlabel("số token phủ 90% xác suất (log)")
ax[1].set_title("Số token phủ 90% xác suất")
fig.tight_layout(); fig.savefig(OUT + "models13_decode.png"); plt.close(fig)


# =====================================================================
# (I) Đếm tham số và FLOP, đối chiếu với GPT-2 đã công bố
# =====================================================================
head("(I) Dem tham so Transformer, doi chieu GPT-2")

def gpt2_params(n_layer, d_model, d_ff, vocab, n_ctx, tie=True):
    emb = vocab * d_model
    pos = n_ctx * d_model
    attn = n_layer * (d_model * 3 * d_model + 3 * d_model      # W_qkv + bias
                      + d_model * d_model + d_model)          # W_o + bias
    ffn = n_layer * (d_model * d_ff + d_ff + d_ff * d_model + d_model)
    ln = n_layer * 2 * 2 * d_model + 2 * d_model               # 2 LN/lop + LN cuoi, moi LN co gamma & beta
    total = emb + pos + attn + ffn + ln + (0 if tie else vocab * d_model)
    non_emb = 12 * n_layer * d_model ** 2                      # cong thuc Kaplan, bo bias va LN
    return dict(emb=emb, pos=pos, attn=attn, ffn=ffn, ln=ln, total=total, non_emb=non_emb)

specs = [("GPT-2 small", 12, 768, 3072, 50257, 1024, 124_439_808),
         ("GPT-2 medium", 24, 1024, 4096, 50257, 1024, 354_823_168),
         ("GPT-2 large", 36, 1280, 5120, 50257, 1024, 774_030_080)]
print(f"{'mô hình':<15}{'công thức':>16}{'đã công bố':>16}{'lệch':>8}{'phi-embedding':>16}")
for name, L, dm, dff, V_, ctx, published in specs:
    p = gpt2_params(L, dm, dff, V_, ctx)
    print(f"{name:<15}{p['total']:16,d}{published:16,d}{p['total']-published:8d}{p['non_emb']:16,d}")

p = gpt2_params(12, 768, 3072, 50257, 1024)
print(f"\nGPT-2 small tach nho: embedding {p['emb']:,} ({p['emb']/p['total']*100:.1f}%), "
      f"vi tri {p['pos']:,}, attention {p['attn']:,}, FFN {p['ffn']:,}")
print(f"Kiem tra 12*L*d^2 = {p['non_emb']:,} ; FFN chiem {p['ffn']/(p['attn']+p['ffn'])*100:.0f}% khoi Transformer")
print(f"FLOP suy luan moi token ~ 2N = {2*p['non_emb']:,} ; huan luyen ~ 6N = {6*p['non_emb']:,}")


# =====================================================================
# (J) Kiến trúc kiểu Llama: tham số, FLOP huấn luyện, và tác dụng của GQA
# =====================================================================
head("(J) Kien truc kieu Llama: RoPE, RMSNorm, SwiGLU, GQA")

def llama_params(n_layer, d_model, d_ff, vocab, n_kv_heads, n_heads):
    """RoPE nen khong co embedding vi tri; RMSNorm chi co gamma; khong do lech;
    FFN SwiGLU dung BA ma tran; lop ra khong dung chung trong so voi embedding."""
    d_head = d_model // n_heads
    attn = n_layer * (d_model * d_model                      # W_Q
                      + 2 * d_model * n_kv_heads * d_head    # W_K, W_V (GQA)
                      + d_model * d_model)                   # W_O
    ffn = n_layer * 3 * d_model * d_ff                       # SwiGLU: W, V, W_2
    norms = n_layer * 2 * d_model + d_model
    emb = vocab * d_model
    head_ = vocab * d_model
    return dict(emb=emb, head=head_, attn=attn, ffn=ffn, norms=norms,
                total=emb + head_ + attn + ffn + norms, non_emb=attn + ffn)

print(f"{'mô hình':<16}{'công thức':>18}{'đã công bố':>18}{'lệch':>8}")
for name, L, dm, dff, V_, nkv, nh, published in [
        ("Llama-2 7B", 32, 4096, 11008, 32000, 32, 32, 6_738_415_616),
        ("Llama-2 13B", 40, 5120, 13824, 32000, 40, 40, 13_015_864_320)]:
    q = llama_params(L, dm, dff, V_, nkv, nh)
    print(f"{name:<16}{q['total']:18,d}{published:18,d}{q['total']-published:8d}")

q = llama_params(32, 4096, 11008, 32000, 32, 32)
N = q["non_emb"]
print(f"  d_ff = 11008 so voi 8/3*d = {8/3*4096:.0f}  -> lam tron len boi cua 256")
print(f"  Llama-2 7B phi-embedding N = {N:,}")

D_tok = 2e12
C = 6 * N * D_tok
print(f"\nHuan luyen tren {D_tok:.0e} token: C = 6ND = {C:.4e} FLOP")
for mfu in (0.30, 0.376, 0.40):
    hours = C / (312e12 * mfu) / 3600
    print(f"  o {mfu*100:5.1f}% MFU tren A100 (312 TFLOPS bf16): {hours:9,.0f} gio-GPU")
print("  Bao cao cua Llama 2 cho ban 7B: 184.320 gio-GPU")

def kv_bytes(L, n_kv, d_head, T, B, bpe=2):
    return 2 * L * n_kv * d_head * T * B * bpe

mha = kv_bytes(80, 64, 128, 4096, 8)
gqa = kv_bytes(80, 8, 128, 4096, 8)
print(f"\nKV cache mo hinh 70B (80 lop, d_head 128), ngu canh 4096, batch 8, FP16:")
print(f"  MHA, 64 KV head: {mha / 2**30:7.2f} GiB")
print(f"  GQA,  8 KV head: {gqa / 2**30:7.2f} GiB   -> giam dung {mha / gqa:.0f} lan")
print(f"  moi token moi chuoi voi GQA: {kv_bytes(80, 8, 128, 1, 1) / 2**20:.4f} MiB")

# Khi nao attention chi phoi chi phi
print(f"\nTi le chi phi attention tren phan con lai = T / (6d):")
for d_, T_ in [(768, 1024), (4096, 4096), (4096, 32768), (4096, 131072)]:
    print(f"  d={d_:5d}, T={T_:6d}: {T_ / (6 * d_):7.2f}   "
          f"({'attention chi phoi' if T_ > 6 * d_ else 'phan con lai chi phoi'})")

print("\nDa luu 8 hinh vao " + OUT)
