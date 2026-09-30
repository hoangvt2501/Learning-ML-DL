"""Thí nghiệm sinh số liệu và Hình 6, 7, 9–14 của giáo trình MLOps.

Mọi con số thực nghiệm trong tài liệu đều in ra từ script này. Hạt giống cố định
nên chạy lại cho kết quả y hệt.

    python code/mlops/experiments.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

OUT = "figs/"
SEED = 2026
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"


def head(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# =====================================================================
# (A) Hình 7 — Đuôi độ trễ khi một yêu cầu toả ra nhiều dịch vụ
# =====================================================================
head("(A) Ngân sach do tre: p99 cua max-of-k (tail at scale)")

P50, P99 = 20.0, 100.0                      # ms, do tre mot dich vu
sigma = np.log(P99 / P50) / stats.norm.ppf(0.99)
mu = np.log(P50)
print(f"lognormal: median={P50} ms, p99={P99} ms -> mu={mu:.4f}, sigma={sigma:.4f}")

def q_of_max(q, k):
    """Phan vi q cua max cua k lan goi doc lap."""
    return np.exp(mu + sigma * stats.norm.ppf(q ** (1.0 / k)))

ks = np.array([1, 2, 5, 10, 20, 50, 100])
p99_max = np.array([q_of_max(0.99, k) for k in ks])
slow_frac = 1 - 0.99 ** ks                   # ti le yeu cau cham it nhat 1 nhanh

print(f"{'k':>5} {'p99 cua max (ms)':>18} {'% yeu cau co >=1 nhanh vuot p99':>34}")
for k, p, s in zip(ks, p99_max, slow_frac):
    print(f"{k:5d} {p:18.1f} {s*100:33.1f}%")
A = dict(ks=ks, p99=p99_max, slow=slow_frac, sigma=sigma, mu=mu)

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
ax[0].plot(ks, p99_max, "-o", ms=4, color=C_MAIN)
ax[0].axhline(P99, ls="--", lw=1.1, color=C_DIM)
ax[0].annotate(f"p99 một dịch vụ = {P99:.0f} ms", (1, P99), xytext=(4, -14),
               textcoords="offset points", fontsize=8, color=C_DIM)
ax[0].set_xscale("log"); ax[0].set_xlabel("số nhánh k gọi song song")
ax[0].set_ylabel("p99 của thời gian trả lời (ms)")
ax[0].set_title("Đuôi dài lên rất nhanh theo số nhánh")
ax[1].plot(ks, slow_frac * 100, "-o", ms=4, color=C_BAD)
ax[1].set_xscale("log"); ax[1].set_xlabel("số nhánh k gọi song song")
ax[1].set_ylabel("% yêu cầu chạm ít nhất một nhánh chậm")
ax[1].set_title("Xác suất chạm đuôi = $1-0{,}99^k$")
fig.tight_layout(); fig.savefig(OUT + "mlops07_tail.png"); plt.close(fig)


# =====================================================================
# (B) Hình 6 — Chỉ số gộp tăng trong khi một lát cắt tụt
# =====================================================================
head("(B) Danh gia theo lat cat: chi so gop che mat su that")

def make_slices(n=8000, seed=1):
    """Nhóm B là người dùng mới: đặc trưng x2 (lịch sử mua hàng) chỉ có nghĩa với nhóm A."""
    r = np.random.default_rng(seed)
    is_b = r.random(n) < 0.15                    # 15% la nguoi dung moi
    x1 = r.normal(0, 1, n)                       # dac trung ai cung co
    y = (r.random(n) < 1 / (1 + np.exp(-1.1 * x1))).astype(int)
    # x2 manh nhung chi co tin hieu voi nhom A; voi nhom B no la nhieu thuan tuy
    x2 = np.where(is_b, r.normal(0, 1, n), 2.2 * (y - 0.5) + r.normal(0, 0.7, n))
    return np.c_[x1, x2], y, is_b.astype(int)

Xtr, ytr, gtr = make_slices(seed=1)
Xte, yte, gte = make_slices(seed=2)

v1 = LogisticRegression().fit(Xtr[:, :1], ytr)          # chi dung x1
v2 = LogisticRegression().fit(Xtr, ytr)                  # them x2

ok1 = v1.predict(Xte[:, :1]) == yte
ok2 = v2.predict(Xte) == yte
rows = []
for name, mask in [("Tổng thể", np.ones(len(yte), bool)),
                   ("Nhóm A (85%)", gte == 0), ("Nhóm B (15%)", gte == 1)]:
    rows.append((name, ok1[mask].mean(), ok2[mask].mean()))
print("v2 = v1 + đặc trưng x2, thứ chỉ có tín hiệu với nhóm A")
print(f"{'Lát cắt':<14}{'v1':>9}{'v2':>9}{'thay đổi':>11}")
for n_, a, b in rows:
    print(f"{n_:<14}{a*100:8.2f}%{b*100:8.2f}%{(b-a)*100:+10.2f} đ")
print(f"trọng số v2 học được: x1={v2.coef_[0,0]:.2f}, x2={v2.coef_[0,1]:.2f}")
B = rows

fig, ax = plt.subplots(figsize=(5.2, 3.0))
names = [r[0] for r in rows]; xs = np.arange(len(rows)); w = .36
ax.bar(xs - w/2, [r[1]*100 for r in rows], w, label="mô hình v1", color=C_DIM)
ax.bar(xs + w/2, [r[2]*100 for r in rows], w, label="mô hình v2", color=C_MAIN)
for i, r in enumerate(rows):
    d = (r[2] - r[1]) * 100
    ax.annotate(f"{d:+.1f}đ", (i + w/2, r[2]*100 + .6), ha="center", fontsize=8,
                color=C_BAD if d < 0 else C_MAIN)
ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylabel("Độ chính xác (%)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Gộp lại thì v2 tốt hơn — nhóm B thì không")
fig.tight_layout(); fig.savefig(OUT + "mlops06_slice.png"); plt.close(fig)


# =====================================================================
# (C) Hình 9 — Cỡ mẫu cần cho A/B test
# =====================================================================
head("(C) Co mau A/B test cho ti le chuyen doi")

def n_per_arm(p, rel_lift, alpha=0.05, power=0.80):
    """Hai phia, hai nhanh bang nhau, xap xi chuan."""
    p2 = p * (1 + rel_lift)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) ** 2 * (p * (1 - p) + p2 * (1 - p2)) / (p2 - p) ** 2

base = 0.05
lifts = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
print(f"CTR nen = {base:.0%}, alpha=0.05 hai phia, power=80%")
print(f"{'lift tuong doi':>16}{'n moi nhanh':>14}{'ngay (100k/ngay/nhanh)':>26}")
C = []
for l in lifts:
    n = n_per_arm(base, l)
    C.append((l, n, n / 100_000))
    print(f"{l:15.0%}{n:14,.0f}{n/100_000:25.1f}")

fine = np.linspace(0.005, 0.25, 200)
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for p0, c in [(0.01, C_BAD), (0.05, C_MAIN), (0.20, C_WARN)]:
    ax.plot(fine * 100, [n_per_arm(p0, l) for l in fine], color=c, label=f"CTR nền {p0:.0%}")
ax.set_yscale("log"); ax.set_xlabel("Mức cải thiện tương đối cần phát hiện (%)")
ax.set_ylabel("Cỡ mẫu mỗi nhánh (log)")
ax.set_title("Muốn bắt cải thiện nhỏ thì cỡ mẫu tăng theo $1/\\Delta^2$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "mlops09_abtest.png"); plt.close(fig)


# =====================================================================
# (D) Hình 10 — Ba loại dịch chuyển và cái nào dò được bằng X
# =====================================================================
head("(D) Ba loai dich chuyen: cai nao do duoc ma khong can nhan")

n = 4000
r = np.random.default_rng(7)
Xs = r.normal(0, 1, (n, 2))
w_true = np.array([1.5, -1.0])
def label(X, w, noise=0.0, r=r):
    p = 1 / (1 + np.exp(-(X @ w)))
    return (r.random(len(X)) < p).astype(int)
ys = label(Xs, w_true)
clf = LogisticRegression().fit(Xs, ys)

scen = {}
# 1. Covariate shift: P(X) doi, P(Y|X) giu nguyen
Xc = r.normal(1.2, 1.0, (n, 2)); yc = label(Xc, w_true)
scen["Covariate shift"] = (Xc, yc)
# 2. Label shift: P(Y) doi, P(X|Y) giu nguyen  -> lay mau lai theo nhan
Xl_pool, yl_pool = r.normal(0, 1, (4 * n, 2)), None
yl_pool = label(Xl_pool, w_true)
idx1 = np.flatnonzero(yl_pool == 1); idx0 = np.flatnonzero(yl_pool == 0)
take = np.r_[r.choice(idx1, int(.8 * n)), r.choice(idx0, int(.2 * n))]
scen["Label shift"] = (Xl_pool[take], yl_pool[take])
# 3. Concept drift: P(X) giu nguyen, P(Y|X) doi
Xd = r.normal(0, 1, (n, 2)); yd = label(Xd, np.array([-1.0, 1.5]))
scen["Concept drift"] = (Xd, yd)

print(f"{'Kịch bản':<18}{'Độ chính xác':>14}{'KS trên x1 (p)':>17}{'KS trên x2 (p)':>17}{'Dò được bằng X?':>19}")
base_acc = clf.score(Xs, ys)
print(f"{'Không dịch chuyển':<18}{base_acc*100:13.1f}%{'—':>17}{'—':>17}{'—':>19}")
D = []
for name, (X2, y2) in scen.items():
    acc = clf.score(X2, y2)
    p1 = stats.ks_2samp(Xs[:, 0], X2[:, 0]).pvalue
    p2 = stats.ks_2samp(Xs[:, 1], X2[:, 1]).pvalue
    seen = "CÓ" if min(p1, p2) < 1e-3 else "KHÔNG"
    D.append((name, acc, p1, p2, seen))
    print(f"{name:<18}{acc*100:13.1f}%{p1:17.2e}{p2:17.2e}{seen:>19}")

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharex=True, sharey=True)
for ax_, (name, (X2, y2)) in zip(axes, scen.items()):
    ax_.scatter(Xs[:600, 0], Xs[:600, 1], s=5, alpha=.35, color=C_DIM, label="nguồn")
    ax_.scatter(X2[:600, 0], X2[:600, 1], s=5, alpha=.45, color=C_MAIN, label="đích")
    xx = np.linspace(-4, 5, 50)
    ax_.plot(xx, -(clf.coef_[0, 0] * xx + clf.intercept_[0]) / clf.coef_[0, 1],
             color=C_BAD, lw=1.4, label="biên mô hình")
    acc = clf.score(X2, y2)
    ax_.set_title(f"{name}\nđộ chính xác {acc*100:.1f}%", fontsize=9)
    ax_.set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$"); axes[0].set_xlim(-4, 5); axes[0].set_ylim(-4, 5)
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "mlops10_shifts.png"); plt.close(fig)


# =====================================================================
# (E) Hình 11 — PSI dưới giả thuyết không phụ thuộc n và số bin
# =====================================================================
head("(E) PSI khi KHONG he co dich chuyen: E[PSI] ~ 2(k-1)/n")

def psi(a, b, bins):
    edges = np.quantile(a, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    eps = 1e-6
    pa, pb = np.clip(pa, eps, None), np.clip(pb, eps, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))

print(f"{'n mỗi mẫu':>11}{'k=5':>11}{'k=10':>11}{'k=20':>11}   | lý thuyết 2(k-1)/n với k=10")
E = []
for n_ in [200, 500, 1000, 5000, 20000]:
    row = []
    for k in [5, 10, 20]:
        vals = [psi(rng.normal(size=n_), rng.normal(size=n_), k) for _ in range(300)]
        row.append(np.mean(vals))
    E.append((n_, *row, 2 * 9 / n_))
    print(f"{n_:11d}{row[0]:11.4f}{row[1]:11.4f}{row[2]:11.4f}   | {2*9/n_:.4f}")

fig, ax = plt.subplots(figsize=(5.6, 3.1))
ns = np.array([e[0] for e in E])
for j, k in enumerate([5, 10, 20]):
    ax.plot(ns, [e[1 + j] for e in E], "-o", ms=4, label=f"đo được, k={k} bin")
    ax.plot(ns, 2 * (k - 1) / ns, ":", lw=1.2, color="k", alpha=.5)
ax.axhline(0.1, ls="--", lw=1.1, color=C_WARN)
ax.axhline(0.25, ls="--", lw=1.1, color=C_BAD)
ax.annotate("ngưỡng 0,25 “dịch chuyển mạnh”", (ns[0], 0.26), fontsize=7.5, color=C_BAD)
ax.annotate("ngưỡng 0,10", (ns[0], 0.105), fontsize=7.5, color=C_WARN)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("cỡ mẫu n mỗi phía"); ax.set_ylabel("PSI trung bình khi hai mẫu CÙNG phân phối")
ax.set_title("Đường chấm là $2(k-1)/n$")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "mlops11_psi_null.png"); plt.close(fig)


# =====================================================================
# (F) Hình 12 — Tỉ lệ báo động giả của ngưỡng PSI cố định
# =====================================================================
head("(F) Ti le bao dong gia va do nhay: PSI 0.25 so voi kiem dinh KS")

def rates(n_, delta, trials=400, k=10, alpha=0.01):
    fp_psi = fp_ks = 0
    for _ in range(trials):
        a = rng.normal(0, 1, n_); b = rng.normal(delta, 1, n_)
        if psi(a, b, k) > 0.25: fp_psi += 1
        if stats.ks_2samp(a, b).pvalue < alpha: fp_ks += 1
    return fp_psi / trials, fp_ks / trials

print(f"{'n':>7}{'δ':>6}{'PSI>0,25 báo':>15}{'KS p<0,01 báo':>16}")
F = []
for n_ in [50, 100, 200, 1000, 10000]:
    for d in [0.0, 0.1, 0.3]:
        rp, rk = rates(n_, d)
        F.append((n_, d, rp, rk))
        print(f"{n_:7d}{d:6.1f}{rp*100:14.0f}%{rk*100:15.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
deltas = np.linspace(0, 0.4, 9)
for n_, c in [(100, C_BAD), (1000, C_WARN), (10000, C_MAIN)]:
    rp = [rates(n_, d, trials=250)[0] for d in deltas]
    rk = [rates(n_, d, trials=250)[1] for d in deltas]
    ax[0].plot(deltas, np.array(rp) * 100, "-o", ms=3.5, color=c, label=f"n={n_}")
    ax[1].plot(deltas, np.array(rk) * 100, "-o", ms=3.5, color=c, label=f"n={n_}")
ax[0].set_title("Quy tắc PSI > 0,25"); ax[1].set_title("Kiểm định KS, p < 0,01")
for a_ in ax:
    a_.set_xlabel("độ dịch chuyển thật $\\delta$ (đơn vị $\\sigma$)"); a_.legend(frameon=False, fontsize=8)
ax[0].set_ylabel("% lần báo có dịch chuyển")
fig.tight_layout(); fig.savefig(OUT + "mlops12_detect.png"); plt.close(fig)


# =====================================================================
# (G) Hình 13 — Mô hình cũ đi và nhịp huấn luyện lại
# =====================================================================
head("(G) Do cu cua mo hinh va nhip huan luyen lai")

WEEKS, DRIFT = 52, 0.035          # radian moi tuan, bien quyet dinh xoay dan
def world(week, m=4000, r=None):
    r = r or np.random.default_rng(1000 + week)
    X = r.normal(0, 1, (m, 2))
    th = DRIFT * week
    w = np.array([np.cos(th), np.sin(th)]) * 2.0
    return X, label(X, w, r=r)

def run(cadence):
    """cadence = so tuan giua hai lan huan luyen lai; None = khong bao gio."""
    X0, y0 = world(0); model = LogisticRegression().fit(X0, y0)
    acc = []
    for wk in range(WEEKS):
        Xw, yw = world(wk)
        acc.append(model.score(Xw, yw))
        if cadence and (wk + 1) % cadence == 0:
            model = LogisticRegression().fit(Xw, yw)
    return np.array(acc)

G = {}
for cad, lab in [(None, "không huấn luyện lại"), (12, "mỗi quý"), (4, "mỗi tháng"), (1, "mỗi tuần")]:
    a = run(cad); G[lab] = a
    print(f"{lab:<24} trung bình {a.mean()*100:5.2f}%   thấp nhất {a.min()*100:5.2f}%   tuần 52: {a[-1]*100:5.2f}%")
stale_cost = (G["mỗi tuần"].mean() - G["không huấn luyện lại"].mean()) * 100
print(f"\nGia cua viec khong huan luyen lai: {stale_cost:.2f} diem phan tram do chinh xac trung binh")

fig, ax = plt.subplots(figsize=(6.0, 3.1))
for lab, a in G.items():
    ax.plot(np.arange(WEEKS), a * 100, lw=1.6, label=lab)
ax.set_xlabel("tuần kể từ lần huấn luyện đầu"); ax.set_ylabel("độ chính xác (%)")
ax.set_title("Cùng một mô hình, chỉ khác nhịp huấn luyện lại")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "mlops13_staleness.png"); plt.close(fig)


# =====================================================================
# (H) Hình 14 — Vòng phản hồi thoái hoá
# =====================================================================
head("(H) Vong phan hoi thoai hoa: do da dang sup the nao")

N_ITEM, ROUNDS, USERS, TOPK, WARMUP = 2000, 30, 3000, 10, 200
true_q = rng.beta(1.6, 9.0, N_ITEM)   # it mon that tot, phan lon tam thuong          # chat luong that, he thong khong biet

def simulate(eps, seed=11):
    """Mô phỏng hệ gợi ý tự huấn luyện lại trên log click của chính nó."""
    r = np.random.default_rng(seed)
    shown = np.zeros(N_ITEM); clicks = np.zeros(N_ITEM)
    # Vòng khởi động: hiển thị ngẫu nhiên một ít, nên ước lượng ban đầu rất nhiễu.
    imp = r.choice(N_ITEM, WARMUP)
    np.add.at(shown, imp, 1)
    np.add.at(clicks, imp, r.random(WARMUP) < true_q[imp])
    ever = set(imp.tolist())

    cov, quality, found = [], [], []
    elite = set(np.argsort(-true_q)[:20].tolist())   # 1% danh muc tot nhat
    for _ in range(ROUNDS):
        score = np.where(shown > 0, clicks / np.maximum(shown, 1), 0.0)
        top = np.argsort(-score)[:TOPK]
        pool = top
        if eps > 0:                               # thay eps ti le bang muc ngau nhien
            n_rand = max(1, int(round(eps * TOPK)))
            pool = np.r_[top[: TOPK - n_rand], r.choice(N_ITEM, n_rand, replace=False)]
        imp = r.choice(pool, USERS)
        cl = r.random(USERS) < true_q[imp]
        np.add.at(shown, imp, 1); np.add.at(clicks, imp, cl)
        ever.update(np.unique(imp).tolist())
        cov.append(len(ever) / N_ITEM)
        quality.append(true_q[top].mean())
        found.append(len(elite & ever) / len(elite))
    return np.array(cov), np.array(quality), np.array(found)

H = {}
print(f"{'Chiến lược':<24}{'phủ danh mục':>14}{'chất lượng top-10':>20}{'đã thấy top-1% tốt nhất':>24}")
for eps, lab in [(0.0, "không ngẫu nhiên hoá"), (0.1, "ε = 10% ngẫu nhiên"), (0.3, "ε = 30% ngẫu nhiên")]:
    covs, quals, fnds = [], [], []
    for sd in range(20):                          # trung binh 20 lan chay
        c, q, fd = simulate(eps, seed=11 + sd)
        covs.append(c); quals.append(q); fnds.append(fd)
    cov, qual, fnd = np.mean(covs, 0), np.mean(quals, 0), np.mean(fnds, 0)
    H[lab] = (cov, qual, fnd)
    print(f"{lab:<24}{cov[-1]*100:13.1f}%{qual[-1]:20.3f}{fnd[-1]*100:23.0f}%")
ceil10 = np.sort(true_q)[-10:].mean()
print(f"Tran co the dat (top-10 that): {ceil10:.3f} | trung binh danh muc: {true_q.mean():.3f}")

fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.9))
for lab, (cov, qual, fnd) in H.items():
    ax[0].plot(cov * 100, lw=1.7, label=lab)
    ax[1].plot(qual, lw=1.7, label=lab)
    ax[2].plot(fnd * 100, lw=1.7, label=lab)
ax[1].axhline(true_q.max(), ls="--", lw=1.1, color=C_DIM)
ax[1].annotate("món tốt nhất thật sự", (0.4, true_q.max() - .012), fontsize=7.5, color=C_DIM)
ax[0].set_ylabel("% danh mục từng được hiển thị"); ax[0].set_title("Độ đa dạng")
ax[1].set_ylabel("chất lượng thật của top-10"); ax[1].set_title("Chất lượng")
ax[2].set_ylabel("% nhóm 1% tốt nhất đã từng được hiện"); ax[2].set_title("Khả năng khám phá")
for a_ in ax: a_.set_xlabel("vòng huấn luyện lại")
ax[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT + "mlops14_feedback.png"); plt.close(fig)

print("\nDa luu 8 hinh vao " + OUT)
