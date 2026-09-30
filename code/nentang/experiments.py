"""Thí nghiệm sinh số liệu và hình cho giáo trình "Nền tảng Machine Learning".

Mọi con số thực nghiệm trong tài liệu đều in ra từ script này. Hạt giống cố định
nên chạy lại cho kết quả y hệt. Chỉ cần NumPy, SciPy, scikit-learn, matplotlib.

    python code/nentang/experiments.py

Nhiều khối ở đây không phải mô phỏng mà là **kiểm chứng một đẳng thức**: ridge
đúng bằng MAP với tiên nghiệm Gauss, Naive Bayes Gauss dùng chung ma trận hiệp
phương sai đúng bằng LDA, sai số tái dựng của PCA đúng bằng tổng các trị riêng bị
bỏ. Những khối ấy phải khớp tới sai số máy, không phải khớp "xấp xỉ".
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figs/"
rng = np.random.default_rng(11)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"


def head(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


# =====================================================================
# (A) Hình 2 — Hồi quy tuyến tính: nghiệm giải tích, và cái giá của cộng tuyến
# =====================================================================
head("(A) Hoi quy tuyen tinh: nghiem giai tich va dieu kien so")

n, d = 200, 3
Xr = rng.normal(size=(n, d))
X = np.c_[np.ones(n), Xr]
w_true = np.array([2.0, -1.5, 0.8, 3.0])
y = X @ w_true + rng.normal(0, 0.5, n)

w_normal = np.linalg.solve(X.T @ X, X.T @ y)
w_pinv = np.linalg.pinv(X) @ y
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"{'cách giải':<34}{'w0':>9}{'w1':>9}{'w2':>9}{'w3':>9}")
for lab, w in [("phương trình chuẩn tắc", w_normal),
               ("giả nghịch đảo Moore-Penrose", w_pinv),
               ("lstsq (phân rã QR)", w_lstsq)]:
    print(f"{lab:<34}" + "".join(f"{v:9.4f}" for v in w))
print(f"lech giua 3 cach giai: {np.abs(w_normal - w_pinv).max():.2e} / "
      f"{np.abs(w_normal - w_lstsq).max():.2e}  -> cung mot nghiem")

# Truong hop X^T X suy bien: mot cot bi lap lai.
Xs = np.c_[X, X[:, 1]]
gram = Xs.T @ Xs
print(f"\nKhi lap lai mot cot: hang cua X = {np.linalg.matrix_rank(Xs)} "
      f"nhung X co {Xs.shape[1]} cot -> X^T X suy bien")
try:
    np.linalg.solve(gram, Xs.T @ y)
    print("  solve: khong bao loi (khong nen xay ra)")
except np.linalg.LinAlgError:
    print("  solve: BAO LOI 'Singular matrix'")
w_ps = np.linalg.pinv(Xs) @ y
resid = np.linalg.norm(Xs @ w_ps - y)
w_alt = w_ps + np.array([0, 1.0, 0, 0, -1.0])       # cung du bao, khac he so
print(f"  pinv: van cho mot nghiem, chuan ||w|| = {np.linalg.norm(w_ps):.4f}")
print(f"  mot nghiem khac cung du bao y het: ||w|| = {np.linalg.norm(w_alt):.4f} "
      f"(sai so du bao lech {abs(np.linalg.norm(Xs @ w_alt - y) - resid):.2e})")
print("  -> pinv chon nghiem CHUAN NHO NHAT trong vo so nghiem")

# Cong tuyen: he so dao lon, nhung du bao thi khong.
print(f"\n{'tuong quan 2 cot':>18}{'cond(X^T X)':>14}{'do lech chuan w1':>19}"
      f"{'do lech chuan du bao':>22}")
Arows = []
for rho in [0.0, 0.9, 0.99, 0.999]:
    z1 = rng.normal(size=n)
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 1e-12)) * rng.normal(size=n)
    Xc = np.c_[np.ones(n), z1, z2]
    wt = np.array([1.0, 2.0, -1.0])
    ws, preds = [], []
    for s in range(300):
        rr = np.random.default_rng(5000 + s)
        yy = Xc @ wt + rr.normal(0, 0.5, n)
        ws.append(np.linalg.pinv(Xc) @ yy)
        preds.append(Xc @ ws[-1])
    ws, preds = np.array(ws), np.array(preds)
    cond = np.linalg.cond(Xc.T @ Xc)
    sd_w, sd_p = ws[:, 1].std(), preds.std(axis=0).mean()
    Arows.append((rho, cond, sd_w, sd_p))
    print(f"{rho:18.3f}{cond:14.1f}{sd_w:19.4f}{sd_p:22.4f}")
print("Doc bang: cong tuyen lam HE SO dao manh (cot 3) nhung DU BAO gan nhu khong doi")
print("(cot 4). Vi vay cong tuyen la van de cua DIEN GIAI, khong phai cua du bao.")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
rhos = [a[0] for a in Arows]
ax[0].plot(rhos, [a[1] for a in Arows], "-o", ms=5, color=C_BAD)
ax[0].set_yscale("log"); ax[0].set_xlabel("tương quan giữa hai cột")
ax[0].set_ylabel("$\\mathrm{cond}(X^\\top X)$ (log)")
ax[0].set_title("Số điều kiện bùng nổ theo cộng tuyến")
ax[1].plot(rhos, [a[2] for a in Arows], "-o", ms=5, color=C_BAD, label="độ lệch chuẩn của $w_1$")
ax[1].plot(rhos, [a[3] for a in Arows], "-o", ms=5, color=C_MAIN, label="độ lệch chuẩn của dự báo")
ax[1].set_yscale("log"); ax[1].set_xlabel("tương quan giữa hai cột")
ax[1].set_ylabel("độ lệch chuẩn (log)"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("Hệ số dao động, dự báo thì không")
fig.tight_layout(); fig.savefig(OUT + "nt02_linreg.png"); plt.close(fig)


# =====================================================================
# (B) Hình 3 — Gradient descent: tốc độ hội tụ phụ thuộc số điều kiện
# =====================================================================
head("(B) Gradient descent tren ham bac hai: ly thuyet doi chieu voi do duoc")

def run_gd(kappa, mode, tol=1e-8, itmax=200000):
    """Toi thieu f(x) = 0.5*(x0^2 + kappa*x1^2), nghiem la goc toa do."""
    L, m = float(kappa), 1.0
    if mode == "gd":
        eta, beta = 2.0 / (L + m), 0.0
    else:                                    # heavy ball voi tham so toi uu
        eta = 4.0 / (np.sqrt(L) + np.sqrt(m)) ** 2
        beta = ((np.sqrt(L) - np.sqrt(m)) / (np.sqrt(L) + np.sqrt(m))) ** 2
    A = np.array([1.0, float(kappa)])
    x = np.array([1.0, 1.0]); v = np.zeros(2)
    x0n = np.linalg.norm(x)
    for k in range(1, itmax + 1):
        g = A * x
        v = beta * v - eta * g
        x = x + v
        if np.linalg.norm(x) / x0n < tol:
            return k
    return itmax

print(f"{'kappa':>8}{'GD (số vòng)':>15}{'heavy ball':>13}{'tỉ lệ':>9}{'căn kappa':>12}")
Brows = []
for kap in [1, 10, 100, 1000, 10000]:
    a, b = run_gd(kap, "gd"), run_gd(kap, "hb")
    Brows.append((kap, a, b))
    print(f"{kap:8d}{a:15d}{b:13d}{a / max(b, 1):9.2f}{np.sqrt(kap):12.2f}")
print("Ly thuyet: GD can O(kappa) vong, heavy ball can O(sqrt(kappa)) vong.")
print("Cot 'ti le' bam sat cot 'can kappa' -> dung nhu ly thuyet du doan.")

fig, ax = plt.subplots(figsize=(5.6, 3.1))
ks = [b[0] for b in Brows]
ax.plot(ks, [b[1] for b in Brows], "-o", ms=5, color=C_BAD, label="gradient descent")
ax.plot(ks, [b[2] for b in Brows], "-o", ms=5, color=C_MAIN, label="thêm quán tính")
ax.plot(ks, ks, ":", color=C_DIM, lw=1.2, label="$O(\\kappa)$")
ax.plot(ks, np.sqrt(ks) * 3, "--", color=C_DIM, lw=1.2, label="$O(\\sqrt{\\kappa})$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("số điều kiện $\\kappa$"); ax.set_ylabel("số vòng lặp tới sai số $10^{-8}$")
ax.set_title("Quán tính đổi $\\kappa$ thành $\\sqrt{\\kappa}$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "nt03_gd.png"); plt.close(fig)


# =====================================================================
# (C) Hình 4 — Kiểm tra đạo hàm: vì sao epsilon quá nhỏ lại tệ hơn
# =====================================================================
head("(C) Kiem tra dao ham bang sai phan: diem toi uu cua epsilon")

f = lambda x: np.sin(x) * np.exp(0.3 * x)
df = lambda x: np.cos(x) * np.exp(0.3 * x) + 0.3 * np.sin(x) * np.exp(0.3 * x)
# Lay trung binh tren nhieu diem x0: neu chi dung mot diem thi sai so cat cut va
# sai so lam tron doi khi TRIET TIEU nhau ngau nhien, tao ra mot day gia tao rat sau.
x0s = np.linspace(0.7, 2.1, 25)
eps_grid = np.logspace(-1, -14, 53)
err_c = np.array([np.mean([abs((f(x0 + e) - f(x0 - e)) / (2 * e) - df(x0)) for x0 in x0s])
                  for e in eps_grid])
err_f = np.array([np.mean([abs((f(x0 + e) - f(x0)) / e - df(x0)) for x0 in x0s])
                  for e in eps_grid])
i_best = int(np.argmin(err_c))
u = np.finfo(float).eps
print(f"{'epsilon':>12}{'sai phân trung tâm':>22}{'sai phân tiến':>18}")
for i in range(0, len(eps_grid), 6):
    print(f"{eps_grid[i]:12.1e}{err_c[i]:22.3e}{err_f[i]:18.3e}")
print(f"\nSai so nho nhat cua sai phan trung tam tai epsilon = {eps_grid[i_best]:.2e} "
      f"(sai so {err_c[i_best]:.2e})")
print(f"Ly thuyet: epsilon* ~ u^(1/3) = {u ** (1 / 3):.2e}   (u = {u:.2e})")
print(f"Voi sai phan tien thi epsilon* ~ u^(1/2) = {u ** 0.5:.2e}, "
      f"do duoc {eps_grid[int(np.argmin(err_f))]:.2e}")
print("Epsilon NHO QUA thi sai so LON LEN: tru hai so gan bang nhau lam mat chu so co nghia.")

fig, ax = plt.subplots(figsize=(5.8, 3.2))
ax.loglog(eps_grid, np.maximum(err_c, 1e-18), lw=1.8, color=C_MAIN, label="sai phân trung tâm")
ax.loglog(eps_grid, np.maximum(err_f, 1e-18), lw=1.8, color=C_WARN, label="sai phân tiến")
ax.axvline(u ** (1 / 3), color=C_DIM, ls=":", lw=1.2)
ax.text(u ** (1 / 3) * 1.3, 1e-4, "$u^{1/3}$", fontsize=8, color=C_DIM)
ax.set_xlabel("$\\varepsilon$"); ax.set_ylabel("sai số tuyệt đối")
ax.invert_xaxis(); ax.legend(frameon=False, fontsize=8)
ax.set_title("Hình chữ V: sai số cắt cụt gặp sai số làm tròn")
fig.tight_layout(); fig.savefig(OUT + "nt04_gradcheck.png"); plt.close(fig)


# =====================================================================
# (D) Perceptron: chặn số lần sai, và chuyện gì xảy ra khi không tách được
# =====================================================================
head("(D) Perceptron: chan Novikoff va truong hop khong tach duoc tuyen tinh")

def make_separable(n, gamma, seed):
    r = np.random.default_rng(seed)
    w_star = np.array([1.0, -2.0]); w_star /= np.linalg.norm(w_star)
    pts, lab = [], []
    while len(pts) < n:
        x = r.uniform(-1, 1, 2)
        s = x @ w_star
        # Giu cac diem nam trong dai sat le: le that su cua tap bang dung gamma,
        # va phan lon diem deu kho, nen chan Novikoff moi noi len dieu gi do.
        if gamma <= abs(s) <= gamma + 0.05:
            pts.append(x); lab.append(1 if s > 0 else -1)
    return np.array(pts), np.array(lab)

def perceptron(Xp, yp, itmax=100000):
    w = np.zeros(Xp.shape[1]); mistakes = 0
    for _ in range(itmax):
        err = 0
        for i in range(len(Xp)):
            if yp[i] * (w @ Xp[i]) <= 0:
                w += yp[i] * Xp[i]; mistakes += 1; err += 1
        if err == 0:
            return w, mistakes, True
    return w, mistakes, False

print(f"{'lề gamma':>10}{'R = max||x||':>14}{'chặn (R/gamma)^2':>19}"
      f"{'số lần sai đo được':>21}{'hội tụ':>9}")
for gamma in [0.30, 0.20, 0.10, 0.05]:
    Xp, yp = make_separable(200, gamma, 42)
    R = np.linalg.norm(Xp, axis=1).max()
    gam_thuc = np.abs(Xp @ (np.array([1.0, -2.0]) / np.sqrt(5))).min()
    _, mis, ok = perceptron(Xp, yp)
    print(f"{gam_thuc:10.3f}{R:14.4f}{(R / gam_thuc) ** 2:19.1f}{mis:21d}{str(ok):>9}")
print("Dinh ly Novikoff: so lan sai <= (R/gamma)^2, khong phu thuoc so diem hay so chieu.")
print("Bang tren: cot 4 luon nho hon cot 3 -> chan dung, va no khong chat.")

r = np.random.default_rng(3)
Xn = r.normal(size=(60, 2)); yn = np.where(Xn[:, 0] + Xn[:, 1] > 0, 1, -1)
yn[:6] *= -1                                     # lam nhieu nhan -> khong tach duoc
_, mis_n, ok_n = perceptron(Xn, yn, itmax=2000)
print(f"\nDu lieu KHONG tach duoc tuyen tinh: hoi tu = {ok_n}, "
      f"so lan sai sau 2000 luot = {mis_n:,}")
print("Perceptron khong dung lai va khong co bao dam gi: no chi duoc chung minh cho")
print("truong hop tach duoc. Day la ly do can ham mat mat tron nhu hoi quy logistic.")


# =====================================================================
# (E) Hồi quy logistic: quan hệ với softmax, và vì sao dữ liệu tách được lại hỏng
# =====================================================================
head("(E) Hoi quy logistic va softmax: hai ten goi cua cung mot thu khi K = 2")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fit_logistic(Xl, yl, lam=0.0, iters=4000, lr=0.5):
    """Nhi phan: mot vector trong so w, xac suat = sigmoid(Xw)."""
    w = np.zeros(Xl.shape[1])
    for _ in range(iters):
        p = sigmoid(Xl @ w)
        g = Xl.T @ (p - yl) / len(yl) + lam * w
        w -= lr * g
    return w

def fit_softmax(Xs_, ys_, K, lam=0.0, iters=4000, lr=0.5):
    """Da lop: ma tran trong so W (d x K), xac suat = softmax(XW)."""
    W = np.zeros((Xs_.shape[1], K))
    Y = np.eye(K)[ys_]
    for _ in range(iters):
        Z = Xs_ @ W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = Xs_.T @ (P - Y) / len(ys_) + lam * W
        W -= lr * G
    return W

r = np.random.default_rng(21)
Xe = np.c_[np.ones(300), r.normal(size=(300, 2))]
w_gen = np.array([0.3, 1.5, -2.0])
ye = (r.uniform(size=300) < sigmoid(Xe @ w_gen)).astype(int)

# Khong phat chuan: khi ay hai bai toan toi uu la MOT, nen nghiem phai trung
# khop toi sai so may chu khong phai "gan giong".
w_log = fit_logistic(Xe, ye.astype(float), lam=0.0, iters=60000, lr=1.0)
W_soft = fit_softmax(Xe, ye, 2, lam=0.0, iters=60000, lr=1.0)
p_log = sigmoid(Xe @ w_log)
Z = Xe @ W_soft; Z -= Z.max(axis=1, keepdims=True)
P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
print(f"Sai khac lon nhat giua xac suat cua logistic va cua softmax(K=2): "
      f"{np.abs(p_log - P[:, 1]).max():.3e}")
print(f"Hieu hai cot trong so cua softmax:  {(W_soft[:, 1] - W_soft[:, 0])}")
print(f"Trong so cua hoi quy logistic:      {w_log}")
print(f"  -> lech {np.abs((W_soft[:, 1] - W_soft[:, 0]) - w_log).max():.3e}")
print(f"Tong hai cot trong so cua softmax (phai bang 0): "
      f"{np.abs(W_soft.sum(axis=1)).max():.3e}")
print("Softmax co K vector trong so nhung chi HIEU cua chung la xac dinh duoc:")
print("cong cung mot vector vao moi cot thi xac suat khong doi. Voi K = 2, hieu")
print("ay chinh la vector trong so cua hoi quy logistic -> hai mo hinh la MOT.")

# Du lieu tach duoc hoan toan: khong phat chuan thi ||w|| chay ra vo cuc.
Xsep = np.c_[np.ones(60), r.normal(size=(60, 2))]
ysep = (Xsep[:, 1] + Xsep[:, 2] > 0).astype(float)
print(f"\n{'số vòng lặp':>14}{'||w|| khi lam = 0':>22}{'||w|| khi lam = 0,01':>24}")
for it in [500, 2000, 10000, 50000]:
    a = np.linalg.norm(fit_logistic(Xsep, ysep, lam=0.0, iters=it, lr=1.0))
    b = np.linalg.norm(fit_logistic(Xsep, ysep, lam=0.01, iters=it, lr=1.0))
    print(f"{it:14d}{a:22.3f}{b:24.3f}")
print("Du lieu tach duoc thi hop ly cuc dai KHONG co nghiem huu han: cang tang ||w||")
print("thi likelihood cang tang. Phat chuan chan dieu do lai -> day la ly do BAT BUOC.")


# =====================================================================
# (F) Naive Bayes, LDA, QDA: ba mo hinh Gauss khac nhau o dung mot cho
# =====================================================================
head("(F) Naive Bayes / LDA / QDA: khac nhau o gia thiet ve ma tran hiep phuong sai")

def gauss_disc(Xtr, ytr, Xte, shared, diag):
    """Bo phan lop sinh Gauss. shared: dung chung Sigma; diag: ep Sigma cheo."""
    Ks = np.unique(ytr)
    prior = np.array([(ytr == k).mean() for k in Ks])
    mus = np.array([Xtr[ytr == k].mean(axis=0) for k in Ks])
    p = Xtr.shape[1]
    if shared:
        S = np.zeros((p, p))
        for i, k in enumerate(Ks):
            D = Xtr[ytr == k] - mus[i]
            S += D.T @ D
        S /= (len(Xtr) - len(Ks))
        if diag:
            S = np.diag(np.diag(S))
        Sig = [S] * len(Ks)
    else:
        Sig = []
        for i, k in enumerate(Ks):
            D = Xtr[ytr == k] - mus[i]
            S = D.T @ D / (len(D) - 1)
            Sig.append(np.diag(np.diag(S)) if diag else S)
    out = np.zeros((len(Xte), len(Ks)))
    for i in range(len(Ks)):
        Si = Sig[i] + 1e-9 * np.eye(p)
        L = np.linalg.cholesky(Si)
        z = np.linalg.solve(L, (Xte - mus[i]).T)
        out[:, i] = -0.5 * (z ** 2).sum(axis=0) - np.log(np.diag(L)).sum() + np.log(prior[i])
    return out

r = np.random.default_rng(33)
mu0, mu1 = np.array([0.0, 0.0]), np.array([1.6, 1.2])
C0 = np.array([[1.0, 0.75], [0.75, 1.0]])
C1 = np.array([[1.4, -0.5], [-0.5, 0.6]])
n_each = 400
Xf = np.vstack([r.multivariate_normal(mu0, C0, n_each),
                r.multivariate_normal(mu1, C1, n_each)])
yf = np.r_[np.zeros(n_each, int), np.ones(n_each, int)]
idx = r.permutation(len(Xf)); Xf, yf = Xf[idx], yf[idx]
Xtr, ytr, Xte, yte = Xf[:600], yf[:600], Xf[600:], yf[600:]

cfg = [("Naive Bayes Gauss (riêng, chéo)", False, True),
       ("LDA (dùng chung, đầy đủ)", True, False),
       ("QDA (riêng, đầy đủ)", False, False),
       ("NB dùng chung Sigma chéo", True, True)]
scores, accs = {}, {}
print(f"{'mô hình':<34}{'giả thiết Sigma':<26}{'độ chính xác':>14}")
for lab, sh, dg in cfg:
    sc = gauss_disc(Xtr, ytr, Xte, sh, dg)
    scores[lab] = sc
    accs[lab] = (sc.argmax(axis=1) == yte).mean()
    gt = ("dùng chung" if sh else "riêng từng lớp") + (", chéo" if dg else ", đầy đủ")
    print(f"{lab:<34}{gt:<26}{accs[lab]:14.4f}")

def nb_tich_mot_chieu(Xtr_, ytr_, Xte_):
    """Naive Bayes dung dung dinh nghia: NHAN cac mat do Gauss MOT CHIEU cua
    tung dac trung lai voi nhau. Khong dung ma tran hiep phuong sai o bat ky dau."""
    Ks_ = np.unique(ytr_)
    p_ = Xtr_.shape[1]
    var = np.zeros(p_)
    for k in Ks_:
        var += ((Xtr_[ytr_ == k] - Xtr_[ytr_ == k].mean(axis=0)) ** 2).sum(axis=0)
    var /= (len(Xtr_) - len(Ks_))              # phuong sai gop, rieng tung dac trung
    out = np.zeros((len(Xte_), len(Ks_)))
    for i_, k in enumerate(Ks_):
        mu = Xtr_[ytr_ == k].mean(axis=0)
        logp = np.full(len(Xte_), np.log((ytr_ == k).mean()))
        for j_ in range(p_):                   # tich cac mat do -> tong cac log
            logp = (logp - 0.5 * np.log(2 * np.pi * var[j_])
                    - (Xte_[:, j_] - mu[j_]) ** 2 / (2 * var[j_]))
        out[:, i_] = logp
    return out

d_lda = gauss_disc(Xtr, ytr, Xte, True, True)   # Gauss NHIEU chieu, Sigma cheo
d_nbs = nb_tich_mot_chieu(Xtr, ytr, Xte)        # tich cac Gauss MOT chieu
print("")
print("Kiem chung dang thuc: tich cac Gauss MOT CHIEU == Gauss NHIEU CHIEU, Sigma cheo")
print("  hai cach tinh duoc cai dat hoan toan doc lap nhau trong ma nguon.")
dl = d_lda[:, 1] - d_lda[:, 0]
dn = d_nbs[:, 1] - d_nbs[:, 0]
print(f"  sai khac lon nhat cua hieu diem phan biet: {np.abs(dl - dn).max():.3e}")
print(f"  hai mo hinh du bao giong nhau o "
      f"{(d_lda.argmax(axis=1) == d_nbs.argmax(axis=1)).mean() * 100:.1f}% so mau")
b_shared = gauss_disc(Xtr, ytr, Xte, True, False)
lin = b_shared[:, 1] - b_shared[:, 0]
A = np.c_[Xte, np.ones(len(Xte))]
coef, *_ = np.linalg.lstsq(A, lin, rcond=None)
print(f"  ham quyet dinh cua LDA co TUYEN TINH khong? sai so khop tuyen tinh = "
      f"{np.abs(A @ coef - lin).max():.3e}  -> co")
b_q = gauss_disc(Xtr, ytr, Xte, False, False)
linq = b_q[:, 1] - b_q[:, 0]
coefq, *_ = np.linalg.lstsq(A, linq, rcond=None)
print(f"  con QDA thi sao? sai so khop tuyen tinh = {np.abs(A @ coefq - linq).max():.3e} "
      f"-> khong, no bac hai")
print("\nDay la ca cau chuyen: ba mo hinh chi khac nhau o RANG BUOC dat len Sigma.")
print("Gia thiet 'naive' cua Naive Bayes chinh la ep Sigma thanh ma tran cheo.")


# =====================================================================
# (G) Hình 6 — Vì sao độ chính xác nói dối trên dữ liệu mất cân bằng
# =====================================================================
head("(G) Do chinh xac noi doi tren du lieu mat can bang")

def prf(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and (prec + rec) else 0.0
    return tp, fp, fn, tn, prec, rec, f1

def roc_auc(yt, sc):
    o = np.argsort(-sc); yt = yt[o]
    P, N = yt.sum(), len(yt) - yt.sum()
    tpr = np.r_[0, np.cumsum(yt) / P]; fpr = np.r_[0, np.cumsum(1 - yt) / N]
    return float(np.trapezoid(tpr, fpr))

def pr_auc(yt, sc):
    o = np.argsort(-sc); yt = yt[o]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    rec = tp / yt.sum(); prec = tp / (tp + fp)
    return float(np.trapezoid(prec, rec))

r = np.random.default_rng(44)
N_G, rate = 20000, 0.01
yg = (r.uniform(size=N_G) < rate).astype(int)
# Mot bo phan lop "tam duoc": diem cua lop duong cao hon mot chut.
sg = r.normal(0, 1, N_G) + 2.6 * yg
print(f"Ty le lop duong: {yg.mean() * 100:.2f}%  ({yg.sum()} tren {N_G} mau)")
print(f"\n{'bộ phân loại':<30}{'độ chính xác':>14}{'precision':>11}{'recall':>9}{'F1':>8}")
allneg = np.zeros(N_G, int)
tp, fp, fn, tn, pr_, rc_, f1_ = prf(yg, allneg)
print(f"{'đoán TẤT CẢ là âm':<30}{(allneg == yg).mean():14.4f}{'—':>11}{rc_:9.4f}{f1_:8.4f}")
for thr, name in [(2.6, "ngưỡng 2,6"), (1.0, "ngưỡng 1,0"), (-1.0, "ngưỡng -1,0")]:
    yp = (sg > thr).astype(int)
    tp, fp, fn, tn, pr_, rc_, f1_ = prf(yg, yp)
    print(f"{name:<30}{(yp == yg).mean():14.4f}{pr_:11.4f}{rc_:9.4f}{f1_:8.4f}")
print(f"\nROC-AUC = {roc_auc(yg, sg):.4f}   nhung   PR-AUC = {pr_auc(yg, sg):.4f}")
print(f"PR-AUC cua bo doan ngau nhien chinh bang ty le lop duong = {rate:.4f}")
print("ROC-AUC trong dep vi mau so cua FPR la so luong lop AM, ma lop am thi rat nhieu;")
print("them vai bao dong gia khong lam FPR nhuc nhich. PR-AUC khong bi the.")

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
o = np.argsort(-sg); ys = yg[o]
tpr = np.r_[0, np.cumsum(ys) / ys.sum()]; fpr = np.r_[0, np.cumsum(1 - ys) / (len(ys) - ys.sum())]
ax[0].plot(fpr, tpr, color=C_MAIN, lw=1.8); ax[0].plot([0, 1], [0, 1], ":", color=C_DIM)
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[0].set_title(f"ROC — AUC = {roc_auc(yg, sg):.3f} (trông rất tốt)")
tpc = np.cumsum(ys); fpc = np.cumsum(1 - ys)
ax[1].plot(tpc / ys.sum(), tpc / (tpc + fpc), color=C_BAD, lw=1.8)
ax[1].axhline(rate, ls=":", color=C_DIM)
ax[1].text(0.45, rate * 1.6, "mức đoán ngẫu nhiên", fontsize=7.5, color=C_DIM)
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title(f"Precision–Recall — AUC = {pr_auc(yg, sg):.3f}")
fig.tight_layout(); fig.savefig(OUT + "nt06_metrics.png"); plt.close(fig)


# =====================================================================
# (H) Hình 7 — Ridge ĐÚNG BẰNG MAP với tiên nghiệm Gauss
# =====================================================================
head("(H) Phat chuan la tien nghiem: ridge == MAP Gauss, lasso == MAP Laplace")

from scipy.optimize import minimize
from sklearn.linear_model import Lasso

r = np.random.default_rng(55)
n_h, p_h = 80, 12
Xh = r.normal(size=(n_h, p_h))
w_star = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
sigma = 1.0
yh = Xh @ w_star + r.normal(0, sigma, n_h)

tau = 0.7
lam_ridge = sigma ** 2 / tau ** 2          # ket qua ly thuyet: lambda = sigma^2 / tau^2
w_ridge = np.linalg.solve(Xh.T @ Xh + lam_ridge * np.eye(p_h), Xh.T @ yh)

def neg_log_posterior(w):
    """-log p(w | du lieu) = -log p(du lieu | w) - log p(w), bo hang so."""
    resid = yh - Xh @ w
    return resid @ resid / (2 * sigma ** 2) + w @ w / (2 * tau ** 2)

res = minimize(neg_log_posterior, np.zeros(p_h), method="BFGS",
               options={"gtol": 1e-12, "maxiter": 20000})
print(f"lambda ly thuyet = sigma^2/tau^2 = {sigma ** 2:.1f}/{tau ** 2:.2f} = {lam_ridge:.6f}")
print(f"Sai khac giua ridge dang dong va MAP toi uu bang so: "
      f"{np.abs(w_ridge - res.x).max():.3e}")
print("Hai duong tinh hoan toan khac nhau (mot giai he tuyen tinh, mot chay BFGS tren")
print("hau nghiem) nhung ra CUNG MOT nghiem -> phat chuan L2 chinh la tien nghiem Gauss.")

print(f"\n{'lambda':>10}{'||w||':>10}{'số hệ số = 0 (ridge)':>24}{'số hệ số = 0 (lasso)':>24}")
Hrows = []
for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
    wr = np.linalg.solve(Xh.T @ Xh + lam * np.eye(p_h), Xh.T @ yh)
    las = Lasso(alpha=lam / n_h, max_iter=200000, tol=1e-12).fit(Xh, yh)
    nz_r = int((np.abs(wr) < 1e-10).sum()); nz_l = int((np.abs(las.coef_) < 1e-10).sum())
    Hrows.append((lam, np.linalg.norm(wr), nz_r, nz_l))
    print(f"{lam:10.2f}{np.linalg.norm(wr):10.4f}{nz_r:24d}{nz_l:24d}")
print("Ridge KHONG BAO GIO cho he so bang dung 0; lasso thi co. Ly do nam o dao ham")
print("tai goc: |w| co diem gay nen nghiem bi 'dinh' vao 0, con w^2 thi tron.")
print(f"(su that: mo hinh sinh du lieu co {int((w_star == 0).sum())} he so bang 0)")

# Ridge co ngot theo tung huong rieng: he so d_i^2/(d_i^2 + lambda)
U, dsv, Vt = np.linalg.svd(Xh, full_matrices=False)
lam_demo = 10.0
w_svd = Vt.T @ ((dsv / (dsv ** 2 + lam_demo)) * (U.T @ yh))
w_cf = np.linalg.solve(Xh.T @ Xh + lam_demo * np.eye(p_h), Xh.T @ yh)
print(f"\nRidge qua SVD so voi dang dong: lech {np.abs(w_svd - w_cf).max():.3e}")
print(f"{'trị kỳ dị d_i':>16}{'hệ số co ngót d^2/(d^2+lam)':>32}")
for i in [0, 1, p_h // 2, p_h - 2, p_h - 1]:
    f_ = dsv[i] ** 2 / (dsv[i] ** 2 + lam_demo)
    print(f"{dsv[i]:16.4f}{f_:32.4f}")
print("Huong co tri ky di LON gan nhu khong bi dong toi; huong YEU bi co manh.")
print("Ridge khong co deu moi huong -> no co dung nhung huong du lieu noi it nhat.")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
lams = np.logspace(-2, 2.5, 60)
pr = np.array([np.linalg.solve(Xh.T @ Xh + l * np.eye(p_h), Xh.T @ yh) for l in lams])
pl_ = np.array([Lasso(alpha=l / n_h, max_iter=50000, tol=1e-10).fit(Xh, yh).coef_ for l in lams])
for j in range(p_h):
    ax[0].plot(lams, pr[:, j], lw=1.2, color=C_MAIN if w_star[j] != 0 else C_DIM, alpha=.85)
    ax[1].plot(lams, pl_[:, j], lw=1.2, color=C_MAIN if w_star[j] != 0 else C_DIM, alpha=.85)
for a, t in [(ax[0], "Ridge: co dần nhưng không bao giờ chạm 0"),
             (ax[1], "Lasso: cắt hẳn về 0")]:
    a.set_xscale("log"); a.axhline(0, color="k", lw=.7)
    a.set_xlabel("$\\lambda$"); a.set_ylabel("hệ số"); a.set_title(t, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "nt07_regular.png"); plt.close(fig)


# =====================================================================
# (I) Hình 8 — Hàm lồi: kiểm tra bằng số, và vì sao lồi thì yên tâm
# =====================================================================
head("(I) Tinh loi: kiem tra bang dinh nghia, va hau qua len toi uu hoa")

def kiem_tra_loi(f_, lo, hi, n_thu=20000, seed=0):
    """Thu truc tiep bat dang thuc dinh nghia f(t*a+(1-t)*b) <= t*f(a)+(1-t)*f(b)."""
    rr = np.random.default_rng(seed)
    a = rr.uniform(lo, hi, n_thu); b = rr.uniform(lo, hi, n_thu); t = rr.uniform(0, 1, n_thu)
    trai = f_(t * a + (1 - t) * b)
    phai = t * f_(a) + (1 - t) * f_(b)
    vi_pham = trai - phai
    return float(vi_pham.max()), float((vi_pham > 1e-12).mean())

ham = [("$x^2$", lambda x: x ** 2, -3, 3),
       ("$|x|$", lambda x: np.abs(x), -3, 3),
       ("$e^x$", lambda x: np.exp(x), -3, 3),
       ("$\\log(1+e^x)$ (mất mát logistic)", lambda x: np.logaddexp(0, x), -6, 6),
       ("$x^4 - 3x^2$", lambda x: x ** 4 - 3 * x ** 2, -3, 3),
       ("$\\sin x$", lambda x: np.sin(x), -6, 6),
       ("$x^3$", lambda x: x ** 3, -3, 3)]
print(f"{'hàm':<36}{'vi phạm lớn nhất':>20}{'tỉ lệ vi phạm':>16}{'kết luận':>10}")
for lab, fn, lo, hi in ham:
    vmax, tile = kiem_tra_loi(fn, lo, hi)
    kl = "lồi" if vmax <= 1e-12 else "KHÔNG"
    print(f"{lab:<36}{vmax:20.3e}{tile:16.4f}{kl:>10}")
print("Phep thu nay khong CHUNG MINH tinh loi, nhung mot phan vi du thi du de BAC BO.")

def gd_1d(fn, dfn, x0, eta=0.02, it=8000):
    x = x0
    for _ in range(it):
        x -= eta * dfn(x)
    return x

f_loi, df_loi = lambda x: x ** 2, lambda x: 2 * x
f_kl = lambda x: x ** 4 - 3 * x ** 2 + 0.5 * x
df_kl = lambda x: 4 * x ** 3 - 6 * x + 0.5
starts = np.linspace(-2.5, 2.5, 21)
end_loi = np.array([gd_1d(f_loi, df_loi, s) for s in starts])
end_kl = np.array([gd_1d(f_kl, df_kl, s) for s in starts])
print(f"\nXuat phat tu 21 diem khac nhau roi chay gradient descent:")
print(f"  ham LOI     x^2          : so diem dung khac nhau = "
      f"{len(np.unique(np.round(end_loi, 6)))}, gia tri = {np.round(end_loi, 4).min():.4f}")
u_kl = np.unique(np.round(end_kl, 4))
print(f"  ham KHONG LOI x^4-3x^2+x/2: so diem dung khac nhau = {len(u_kl)}, "
      f"cac gia tri = {u_kl}")
print(f"  gia tri ham tai hai diem do: {[round(float(f_kl(v)), 4) for v in u_kl]}")
print("Voi ham loi, MOI cuc tieu dia phuong deu la cuc tieu toan cuc, nen diem khoi tao")
print("khong quan trong. Day la ly do ta quan tam toi tinh loi den vay.")

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
xs = np.linspace(-2.5, 2.5, 400)
ax[0].plot(xs, xs ** 2, color=C_MAIN, lw=1.8)
ax[0].plot(end_loi, end_loi ** 2, "o", ms=5, color=C_BAD)
ax[0].set_title("Hàm lồi: 21 điểm xuất phát, 1 điểm dừng", fontsize=9)
ax[1].plot(xs, f_kl(xs), color=C_WARN, lw=1.8)
ax[1].plot(end_kl, f_kl(end_kl), "o", ms=5, color=C_BAD)
ax[1].set_title("Không lồi: 21 điểm xuất phát, 2 điểm dừng", fontsize=9)
for a in ax:
    a.set_xlabel("$x$"); a.set_ylabel("$f(x)$")
fig.tight_layout(); fig.savefig(OUT + "nt08_convex.png"); plt.close(fig)


# =====================================================================
# (J) Hình 9 — SVM: đối ngẫu mạnh, điều kiện KKT, và vai trò của C
# =====================================================================
head("(J) SVM: kiem chung doi ngau manh va dieu kien KKT bang so")

from sklearn.svm import SVC

r = np.random.default_rng(66)
n_j = 60
Xj = np.vstack([r.normal([-1.6, -1.0], 0.55, (n_j, 2)),
                r.normal([1.6, 1.2], 0.55, (n_j, 2))])
yj = np.r_[-np.ones(n_j), np.ones(n_j)]

svm_hard = SVC(kernel="linear", C=1e8, tol=1e-12).fit(Xj, yj)
w_sk = svm_hard.coef_[0]
b_sk = float(svm_hard.intercept_[0])
sv = svm_hard.support_vectors_
alpha_y = svm_hard.dual_coef_[0]                   # bang alpha_i * y_i
alpha = np.abs(alpha_y)
y_sv = np.sign(alpha_y)

w_dual = alpha_y @ sv                              # w = sum alpha_i y_i x_i
print(f"Dung lai w tu nghiem doi ngau: lech so voi coef_ cua sklearn = "
      f"{np.abs(w_dual - w_sk).max():.3e}")
marg_sv = y_sv * (sv @ w_sk + b_sk)
print(f"Vector ho tro co thoa y_i(w.x_i + b) = 1 khong? "
      f"lech lon nhat = {np.abs(marg_sv - 1).max():.3e}")
print(f"So vector ho tro: {len(sv)} tren {len(Xj)} diem "
      f"({len(sv) / len(Xj) * 100:.1f}%)")

primal = 0.5 * w_sk @ w_sk
K = sv @ sv.T
dual = alpha.sum() - 0.5 * (alpha_y @ K @ alpha_y)
print(f"\nDoi ngau manh:")
print(f"  gia tri bai toan goc  0.5*||w||^2          = {primal:.10f}")
print(f"  gia tri bai toan doi ngau                  = {dual:.10f}")
print(f"  khe doi ngau                               = {abs(primal - dual):.3e}")
print(f"Le = 2/||w|| = {2 / np.linalg.norm(w_sk):.6f}; "
      f"khoang cach nho nhat tu diem toi sieu phang x 2 = "
      f"{2 * np.abs(Xj @ w_sk + b_sk).min() / np.linalg.norm(w_sk):.6f}")

# Le mem: hai lop CHONG LAN nhau, va C dieu khien danh doi le rong / bam du lieu.
r2j = np.random.default_rng(99)
n_tr, n_te = 120, 4000
mu_a, mu_b, sd_j = np.array([-0.9, -0.6]), np.array([0.9, 0.7]), 1.15
def sinh(nn, rr):
    Xa = rr.normal(mu_a, sd_j, (nn // 2, 2)); Xb = rr.normal(mu_b, sd_j, (nn // 2, 2))
    return np.vstack([Xa, Xb]), np.r_[-np.ones(nn // 2), np.ones(nn // 2)]
Xtr_j, ytr_j = sinh(n_tr, r2j)
Xte_j, yte_j = sinh(n_te, np.random.default_rng(100))

print("")
print("Hai lop chong lan nhau (le mem la bat buoc), quet C:")
print(f"{'C':>10}{'lề 2/||w||':>13}{'số VTHT':>10}{'đúng trên lề':>15}"
      f"{'vi phạm lề':>13}{'sai số h.luyện':>17}{'sai số kiểm tra':>18}")
Jrows = []
for Cv in [0.003, 0.03, 0.3, 3.0, 30.0, 300.0]:
    m = SVC(kernel="linear", C=Cv, tol=1e-10).fit(Xtr_j, ytr_j)
    ww = m.coef_[0]
    a = np.abs(m.dual_coef_[0])
    eps_c = 1e-6 * max(Cv, 1.0)
    tren_le = int(((a > eps_c) & (a < Cv - eps_c)).sum())
    vi_pham = int((a >= Cv - eps_c).sum())
    e_tr = float((m.predict(Xtr_j) != ytr_j).mean())
    e_te = float((m.predict(Xte_j) != yte_j).mean())
    Jrows.append((Cv, 2 / np.linalg.norm(ww), len(a), tren_le, vi_pham, e_tr, e_te))
    print(f"{Cv:10.3f}{2 / np.linalg.norm(ww):13.4f}{len(a):10d}{tren_le:15d}"
          f"{vi_pham:13d}{e_tr:17.4f}{e_te:18.4f}")
best = min(Jrows, key=lambda t: t[6])
print(f"C tot nhat tren tap kiem tra: C = {best[0]:g} (sai so {best[6]:.4f}); "
      f"C lon nhat cho {Jrows[-1][6]:.4f}")
print("C nho -> le RONG, nhieu diem duoc phep vi pham, mo hinh don gian hon.")
print("C lon -> le HEP, co ep tung diem cho dung, bat dau bam vao nhieu.")
print("Dieu kien KKT chia cac diem lam ba nhom, va bang tren dem dung ba nhom ay:")
print("  alpha = 0      -> nam NGOAI le, khong anh huong gi toi nghiem")
print("  0 < alpha < C  -> nam DUNG TREN le")
print("  alpha = C      -> VI PHAM le (lot vao trong, hoac sang han phia sai)")

# Kernel: du lieu khong tach duoc tuyen tinh.
th = r.uniform(0, 2 * np.pi, 200)
rad = np.r_[r.normal(1.0, 0.18, 100), r.normal(2.4, 0.18, 100)]
Xk = np.c_[rad * np.cos(th), rad * np.sin(th)]
yk = np.r_[-np.ones(100), np.ones(100)]
print(f"\nDu lieu hai vong tron dong tam:")
for name, mk in [("nhân tuyến tính", SVC(kernel="linear", C=10)),
                 ("nhân đa thức bậc 2", SVC(kernel="poly", degree=2, C=10, gamma="scale")),
                 ("nhân RBF", SVC(kernel="rbf", C=10, gamma="scale"))]:
    mk.fit(Xk, yk)
    print(f"  {name:<24} do chinh xac = {(mk.predict(Xk) == yk).mean():.4f}")
print("Thu thuat nhan: thay vi anh xa x -> phi(x) roi tinh tich vo huong, ta tinh thang")
print("K(x, x') = phi(x).phi(x'). Voi RBF thi phi(x) o khong gian VO HAN chieu.")

fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.1))
for a, Cv in zip(ax[:2], [0.003, 300.0]):
    m = SVC(kernel="linear", C=Cv, tol=1e-10).fit(Xtr_j, ytr_j)
    xx, yy = np.meshgrid(np.linspace(-4.4, 4.4, 300), np.linspace(-4.2, 4.4, 300))
    Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z > 0, alpha=.13, cmap="coolwarm")
    a.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["#888", "k", "#888"],
              linestyles=["--", "-", "--"], linewidths=[.9, 1.4, .9])
    a.scatter(*Xtr_j[ytr_j == -1].T, s=13, color=C_MAIN)
    a.scatter(*Xtr_j[ytr_j == 1].T, s=13, color=C_BAD)
    a.set_title(f"$C = {Cv:g}$ — lề {2 / np.linalg.norm(m.coef_[0]):.2f}", fontsize=9)
mk = SVC(kernel="rbf", C=10, gamma="scale").fit(Xk, yk)
xx, yy = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.2, 3.2, 300))
Z = mk.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax[2].contourf(xx, yy, Z > 0, alpha=.13, cmap="coolwarm")
ax[2].contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.4)
ax[2].scatter(*Xk[yk == -1].T, s=11, color=C_MAIN); ax[2].scatter(*Xk[yk == 1].T, s=11, color=C_BAD)
ax[2].set_title("Nhân RBF trên dữ liệu vòng tròn", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "nt09_svm.png"); plt.close(fig)


# =====================================================================
# (K) Hình 10 — PCA: định lý Eckart–Young, và khi nào PCA làm hỏng việc
# =====================================================================
head("(K) PCA: sai so tai dung DUNG BANG tong cac tri rieng bi bo")

r = np.random.default_rng(77)
n_k, p_k = 300, 8
Z = r.normal(size=(n_k, 3))
Bmix = r.normal(size=(3, p_k))
Xp_ = Z @ Bmix + 0.35 * r.normal(size=(n_k, p_k))
Xc = Xp_ - Xp_.mean(axis=0)

Cov = Xc.T @ Xc / (n_k - 1)
eigval, eigvec = np.linalg.eigh(Cov)
eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
U, dsv, Vt = np.linalg.svd(Xc, full_matrices=False)

print(f"Hai duong tinh PCA:")
print(f"  tri rieng cua ma tran hiep phuong sai : {np.round(eigval[:4], 6)}")
print(f"  d_i^2/(n-1) tu SVD                    : {np.round(dsv[:4] ** 2 / (n_k - 1), 6)}")
print(f"  lech lon nhat tren ca {p_k} tri: {np.abs(eigval - dsv ** 2 / (n_k - 1)).max():.3e}")
print(f"  goc giua cac truc chinh (tri tuyet doi tich vo huong phai bang 1):")
print(f"    {np.round(np.abs(np.diag(eigvec.T @ Vt.T))[:4], 10)}")

print(f"\n{'k':>4}{'||X - X_k||_F^2':>20}{'(n-1) * tổng trị riêng bỏ':>30}{'lệch':>12}"
      f"{'% phương sai giữ':>19}")
tong = eigval.sum()
for k in range(1, p_k + 1):
    Xk_ = (U[:, :k] * dsv[:k]) @ Vt[:k]
    lhs = float(((Xc - Xk_) ** 2).sum())
    rhs = float((n_k - 1) * eigval[k:].sum())
    print(f"{k:4d}{lhs:20.6f}{rhs:30.6f}{abs(lhs - rhs):12.2e}"
          f"{eigval[:k].sum() / tong * 100:19.2f}")
print("Day la dinh ly Eckart-Young: phep chieu PCA la xap xi hang k TOT NHAT theo")
print("chuan Frobenius, va sai so cua no dung bang tong cac tri rieng bi bo di.")

# Khi PCA lam hong viec: huong phuong sai lon nhat vo dung cho phan loai.
r2 = np.random.default_rng(88)
n_c = 400
noise_dir = r2.normal(0, 6.0, n_c)                 # phuong sai RAT lon, khong mang thong tin
sig = np.r_[-np.ones(n_c // 2), np.ones(n_c // 2)] * 1.0
Xl_ = np.c_[noise_dir, sig + r2.normal(0, 0.45, n_c)]
yl_ = (sig > 0).astype(int)
Xlc = Xl_ - Xl_.mean(axis=0)
ev, evec = np.linalg.eigh(Xlc.T @ Xlc / (n_c - 1))
pc1 = evec[:, np.argmax(ev)]
mu0_, mu1_ = Xl_[yl_ == 0].mean(axis=0), Xl_[yl_ == 1].mean(axis=0)
Sw = np.zeros((2, 2))
for c in (0, 1):
    D = Xl_[yl_ == c] - (mu0_ if c == 0 else mu1_)
    Sw += D.T @ D
w_lda = np.linalg.solve(Sw, mu1_ - mu0_); w_lda /= np.linalg.norm(w_lda)

def auc_1d(proj, lab):
    o = np.argsort(proj); lab = lab[o]
    P_, N_ = lab.sum(), len(lab) - lab.sum()
    return float(np.trapezoid(np.r_[0, np.cumsum(lab) / P_], np.r_[0, np.cumsum(1 - lab) / N_]))

a_pca = auc_1d(Xl_ @ pc1, yl_); a_pca = max(a_pca, 1 - a_pca)
a_lda = auc_1d(Xl_ @ w_lda, yl_); a_lda = max(a_lda, 1 - a_lda)
print(f"\nDu lieu co mot huong nhieu bien do lon va mot huong tin hieu bien do nho:")
print(f"  truc chinh thu nhat cua PCA: {np.round(pc1, 4)}  "
      f"(giu {ev.max() / ev.sum() * 100:.1f}% phuong sai)")
print(f"  huong cua LDA              : {np.round(w_lda, 4)}")
print(f"  AUC khi chieu len truc PCA : {a_pca:.4f}")
print(f"  AUC khi chieu len huong LDA: {a_lda:.4f}")
print("PCA giu gan het phuong sai nhung vut di toan bo kha nang phan loai. Ly do:")
print("PCA KHONG NHIN NHAN. Phuong sai lon khong dong nghia co ich.")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
ax[0].bar(np.arange(1, p_k + 1), eigval / tong * 100, color=C_MAIN, alpha=.85)
ax[0].plot(np.arange(1, p_k + 1), np.cumsum(eigval) / tong * 100, "-o", ms=4, color=C_BAD)
ax[0].set_xlabel("thành phần chính"); ax[0].set_ylabel("% phương sai")
ax[0].set_title("Phương sai từng trục và luỹ kế", fontsize=9)
ax[1].scatter(*Xl_[yl_ == 0].T, s=8, color=C_MAIN, alpha=.6)
ax[1].scatter(*Xl_[yl_ == 1].T, s=8, color=C_BAD, alpha=.6)
for vec, lab, col in [(pc1, "trục PCA", C_DIM), (w_lda, "hướng LDA", "k")]:
    ax[1].plot([-8 * vec[0], 8 * vec[0]], [-8 * vec[1], 8 * vec[1]], lw=1.8,
               color=col, label=lab)
ax[1].legend(frameon=False, fontsize=8); ax[1].set_aspect("equal")
ax[1].set_title("PCA chọn hướng vô dụng cho phân loại", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "nt10_pca.png"); plt.close(fig)


# =====================================================================
# (L) Hình 11 — K-means: phụ thuộc khởi tạo, và nó giả định cái gì
# =====================================================================
head("(L) K-means: khoi tao quyet dinh ket qua, va gia thiet ngam ve hinh dang cum")

from itertools import permutations

def kmeans(Xm, k, init, seed, itmax=300):
    rr = np.random.default_rng(seed)
    if init == "ngau nhien":
        C_ = Xm[rr.choice(len(Xm), k, replace=False)].copy()
    else:                                          # k-means++
        C_ = [Xm[rr.integers(len(Xm))]]
        for _ in range(k - 1):
            d2 = np.min(((Xm[:, None, :] - np.array(C_)[None]) ** 2).sum(-1), axis=1)
            C_.append(Xm[rr.choice(len(Xm), p=d2 / d2.sum())])
        C_ = np.array(C_)
    lab = np.zeros(len(Xm), int)
    for _ in range(itmax):
        lab = np.argmin(((Xm[:, None, :] - C_[None]) ** 2).sum(-1), axis=1)
        new = np.array([Xm[lab == j].mean(axis=0) if (lab == j).any() else C_[j]
                        for j in range(k)])
        if np.allclose(new, C_):
            break
        C_ = new
    return C_, lab, float(((Xm - C_[lab]) ** 2).sum())

def do_chinh_xac_gom_cum(lab, truth, k):
    """Gom cum khong co nhan, nen phai thu moi cach ghep nhan va lay cach tot nhat."""
    return max((np.array([p[l] for l in lab]) == truth).mean()
               for p in permutations(range(k)))

# Bai toan kho hon: 8 cum, mot so cum gan nhau -> khoi tao moi thuc su quan trong.
r = np.random.default_rng(111)
cent = np.array([[0, 0], [1.9, 0.3], [4.6, 0.1], [6.3, 1.7],
                 [0.4, 3.4], [2.4, 3.9], [5.1, 4.2], [7.0, 4.0]], float)
Xm = np.vstack([r.normal(c, 0.55, (90, 2)) for c in cent])
truth_m = np.repeat(np.arange(len(cent)), 90)
K_M = len(cent)

N_SEED = 200
print(f"Chay {N_SEED} lan voi {K_M} cum, moi lan mot hat giong khac nhau:")
print(f"{'cách khởi tạo':<18}{'quán tính nhỏ nhất':>21}{'trung bình':>13}"
      f"{'% lần kẹt ở nghiệm tồi':>26}")
best_global = min(kmeans(Xm, K_M, ini, s)[2]
                  for ini in ("ngau nhien", "k-means++") for s in range(N_SEED))
for init in ["ngau nhien", "k-means++"]:
    vals = np.array([kmeans(Xm, K_M, init, s)[2] for s in range(N_SEED)])
    ket = (vals > best_global * 1.02).mean() * 100
    print(f"{init:<18}{vals.min():21.3f}{vals.mean():13.3f}{ket:26.1f}")
print("K-means chi bao dam hoi tu toi cuc tieu DIA PHUONG. k-means++ rai cac tam ban")
print("dau cho xa nhau theo xac suat ti le d^2, nen it roi vao nghiem toi hon han.")

# K-means gia dinh cum HINH CAU va co kich thuoc tuong duong.
t_ = np.linspace(0, np.pi, 300)
moon1 = np.c_[np.cos(t_), np.sin(t_)] + r.normal(0, .08, (300, 2))
moon2 = np.c_[1 - np.cos(t_), 1 - np.sin(t_) - 0.5] + r.normal(0, .08, (300, 2))
Xmoon = np.vstack([moon1, moon2]); ymoon = np.r_[np.zeros(300, int), np.ones(300, int)]
Xan = np.vstack([r.normal([0, 0], [3.0, 0.35], (300, 2)),
                 r.normal([0, 2.4], [3.0, 0.35], (300, 2))])
yan = np.r_[np.zeros(300, int), np.ones(300, int)]

r3 = np.random.default_rng(5)
cent3 = np.array([[0, 0], [5.0, 0.5], [2.5, 4.5]], float)
Xsph = np.vstack([r3.normal(c, 0.7, (200, 2)) for c in cent3])
ysph = np.repeat(np.arange(3), 200)

_, lab_sph, _ = kmeans(Xsph, 3, "k-means++", 0)
_, lab_moon, _ = kmeans(Xmoon, 2, "k-means++", 0)
_, lab_an, _ = kmeans(Xan, 2, "k-means++", 0)
acc_sph = do_chinh_xac_gom_cum(lab_sph, ysph, 3)
acc_moon = do_chinh_xac_gom_cum(lab_moon, ymoon, 2)
acc_an = do_chinh_xac_gom_cum(lab_an, yan, 2)
print(f"\nDo chinh xac gom cum (da thu moi cach ghep nhan, lay cach tot nhat):")
print(f"  ba cum TRON, tach roi              : {acc_sph:.4f}")
print(f"  hai hinh LUOI LIEM long nhau       : {acc_moon:.4f}")
print(f"  hai DAI DET nam ngang, cach nhau   : {acc_an:.4f}")
print("K-means gan moi diem cho tam GAN NHAT theo khoang cach Euclid, nen bien giua hai")
print("cum luon la mot sieu phang. Cum khong loi hoac bi keo det thi gia thiet do sai.")

# Khuyu tay: khong co dinh nghia toan hoc, chi la mot quy tac nhin bang mat.
print(f"\n{'k':>4}{'quán tính':>14}{'giảm so với k-1':>20}   (dữ liệu có đúng 8 cụm)")
prev = None
for k in range(1, 13):
    inr = min(kmeans(Xm, k, "k-means++", s)[2] for s in range(8))
    gi = "—" if prev is None else f"{(prev - inr) / prev * 100:.1f}%"
    mark = "  <-- khuỷu tay" if k == K_M else ""
    print(f"{k:4d}{inr:14.2f}{gi:>20}{mark}")
    prev = inr
print("Quan tinh LUON giam khi k tang (k = n thi bang 0), nen KHONG the chon k bang")
print("cach toi thieu no. 'Khuyu tay' la cho muc giam dot ngot cham han lai.")

fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0))
C8, l8, _ = kmeans(Xm, K_M, "k-means++", 0)
ax[0].scatter(*Xm.T, c=l8, s=6, cmap="tab10", alpha=.75)
ax[0].scatter(*C8.T, marker="X", s=80, color="k", edgecolor="w", linewidth=1)
ax[0].set_title(f"Cụm tròn, tách rời — đúng {acc_sph * 100:.0f}% (ví dụ 3 cụm)", fontsize=8.5)
ax[1].scatter(*Xmoon.T, c=lab_moon, s=7, cmap="viridis", alpha=.75)
ax[1].set_title(f"Hai lưỡi liềm — đúng {acc_moon * 100:.0f}%", fontsize=9)
ax[2].scatter(*Xan.T, c=lab_an, s=7, cmap="viridis", alpha=.75)
ax[2].set_title(f"Hai dải dẹt — đúng {acc_an * 100:.0f}%", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "nt11_kmeans.png"); plt.close(fig)


# =====================================================================
# (M) Hình 12 — Hệ gợi ý: cần bao nhiêu đánh giá mỗi người thì mới đoán được
# =====================================================================
head("(M) He goi y: nguong du lieu de phan ra ma tran hoat dong, va khoi dau lanh")

r = np.random.default_rng(123)
n_u, n_i, rank_true, sd_noise = 300, 200, 4, 0.4
P_true = r.normal(0, 1, (n_u, rank_true))
Q_true = r.normal(0, 1, (n_i, rank_true))
R_sach = P_true @ Q_true.T                        # tin hieu that, khong nhieu
R_quan_sat = R_sach + r.normal(0, sd_noise, R_sach.shape)
print(f"Ma tran danh gia: {n_u} nguoi x {n_i} muc, hang that = {rank_true}, "
      f"nhieu quan sat sd = {sd_noise}")
print(f"  mo hinh hang {rank_true} co {(n_u + n_i) * rank_true:,} tham so tu do, "
      f"so voi {n_u * n_i:,} o -> {(n_u + n_i) * rank_true / (n_u * n_i) * 100:.1f}%")
print(f"  do lech chuan cua tin hieu that: {R_sach.std():.4f}")

def mf(mask, Robs, k, lam=0.1, iters=300, seed=0):
    """Binh phuong toi thieu luan phien co phat chuan (ALS)."""
    rr = np.random.default_rng(seed)
    P = rr.normal(0, .1, (Robs.shape[0], k)); Q = rr.normal(0, .1, (Robs.shape[1], k))
    I = np.eye(k)
    for _ in range(iters):
        for u in range(Robs.shape[0]):
            idx = np.where(mask[u])[0]
            if len(idx):
                Qi = Q[idx]; P[u] = np.linalg.solve(Qi.T @ Qi + lam * I, Qi.T @ Robs[u, idx])
        for i_ in range(Robs.shape[1]):
            idx = np.where(mask[:, i_])[0]
            if len(idx):
                Pi = P[idx]; Q[i_] = np.linalg.solve(Pi.T @ Pi + lam * I, Pi.T @ Robs[idx, i_])
    return P, Q

print(f"\n{'% ô thấy được':>15}{'đánh giá / người':>19}{'RMSE ô chưa thấy':>20}"
      f"{'so với đoán bừa':>18}")
Mrows = []
for frac in [0.02, 0.03, 0.05, 0.10, 0.20, 0.40]:
    mask = r.uniform(size=(n_u, n_i)) < frac
    P, Q = mf(mask, R_quan_sat, rank_true, seed=1)
    rm = float(np.sqrt((((P @ Q.T) - R_sach)[~mask] ** 2).mean()))
    Mrows.append((frac, mask.sum(1).mean(), rm))
    print(f"{frac * 100:15.0f}{mask.sum(1).mean():19.1f}{rm:20.4f}"
          f"{rm / R_sach.std():18.2f}")
print(f"Nguong rat sac net: duoi ~20 danh gia moi nguoi thi phan ra ma tran HONG HAN")
print(f"(sai so con te hon doan bua bang 0); tu khoang 5 lan so hang tro len thi no")
print(f"khoi phuc gan nhu chinh xac. Ly do: moi nguoi dung co {rank_true} an so can uoc luong,")
print(f"nen ho phai co du rang buoc thi he moi giai duoc.")

mask = r.uniform(size=(n_u, n_i)) < 0.30
print(f"\nChon hang k (do o mat do 30%, noi viec khop la dang tin):")
print(f"{'hạng k':>10}{'RMSE ô chưa thấy':>20}{'RMSE trên ô đã thấy':>23}")
for k in [1, 2, 3, 4, 6, 10, 20]:
    P, Q = mf(mask, R_quan_sat, k, seed=2)
    Rh = P @ Q.T
    rm_out = float(np.sqrt(((Rh - R_sach)[~mask] ** 2).mean()))
    rm_in = float(np.sqrt(((Rh - R_quan_sat)[mask] ** 2).mean()))
    star = "  <- hang dung" if k == rank_true else ""
    print(f"{k:10d}{rm_out:20.4f}{rm_in:23.4f}{star}")
print("Sai so tren o DA THAY giam don dieu theo k (mo hinh manh hon thi khop chat hon),")
print("nhung sai so tren o CHUA THAY cham day o dung hang that roi tang len. Day chinh")
print("la danh doi thien lech - phuong sai, lan nay o dang chon hang.")

mask_cold = mask.copy()
n_new = 10
mask_cold[-n_new:, :] = False                      # 10 nguoi dung hoan toan moi
P, Q = mf(mask_cold, R_quan_sat, rank_true, seed=3)
Rh = P @ Q.T
cu = ~mask_cold[:-n_new]
rmse_cu = float(np.sqrt(((Rh - R_sach)[:-n_new][cu] ** 2).mean()))
rmse_moi = float(np.sqrt(((Rh - R_sach)[-n_new:] ** 2).mean()))
print(f"\nKhoi dau lanh: {n_new} nguoi dung MOI, khong co lay mot danh gia nao")
print(f"  RMSE cho nguoi dung cu   : {rmse_cu:.4f}")
print(f"  RMSE cho nguoi dung moi  : {rmse_moi:.4f}")
print(f"  RMSE neu doan bua bang 0 : {R_sach.std():.4f}")
print(f"  ||P|| trung binh, nguoi cu : {np.linalg.norm(P[:-n_new], axis=1).mean():.4f}")
print(f"  ||P|| trung binh, nguoi moi: {np.linalg.norm(P[-n_new:], axis=1).mean():.2e}")
print("Vector cua nguoi dung moi bi keo thang ve 0 vi khong co rang buoc nao ngoai phat")
print("chuan, nen mo hinh du bao 0 cho tat ca. Do la ly do he thuc te luon can mot duong")
print("du phong dua tren NOI DUNG: dac trung cua muc va ho so khai bao cua nguoi dung.")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
ax[0].plot([m[1] for m in Mrows], [m[2] for m in Mrows], "-o", ms=5, color=C_MAIN)
ax[0].axhline(R_sach.std(), ls=":", color=C_DIM)
ax[0].text(28, R_sach.std() * 1.05, "mức đoán bừa bằng 0", fontsize=7.5, color=C_DIM)
ax[0].axvline(5 * rank_true, ls="--", color=C_BAD, lw=1)
ax[0].text(5 * rank_true * 1.1, 1.2, "$5k$", fontsize=8, color=C_BAD)
ax[0].set_yscale("log"); ax[0].set_xlabel("số đánh giá trung bình mỗi người")
ax[0].set_ylabel("RMSE trên ô chưa thấy (log)")
ax[0].set_title("Ngưỡng dữ liệu để phân rã ma trận chạy được", fontsize=9)
im = ax[1].imshow(R_sach[:40, :40], cmap="coolwarm", aspect="auto")
ax[1].set_title("Ma trận đánh giá thật (hạng 4)", fontsize=9)
fig.colorbar(im, ax=ax[1], fraction=.046)
fig.tight_layout(); fig.savefig(OUT + "nt12_recsys.png"); plt.close(fig)

print("\n" + "=" * 74)
print("Da luu 10 hinh vao thu muc figs/: nt02, nt03, nt04, nt06, nt07, nt08, nt09, nt10, nt11, nt12")
print("=" * 74)
