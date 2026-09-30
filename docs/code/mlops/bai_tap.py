"""
Số liệu cho lời giải các bài tập Chương 14 của giáo trình MLOps.

Chạy:  python code/mlops/bai_tap.py
Mọi nguồn ngẫu nhiên đều có hạt giống cố định, nên chạy lại cho kết quả y hệt.

  Bài 1  PSI khi không có dịch chuyển: công thức tiệm cận và mô phỏng
  Bài 2  Cỡ mẫu A/B test và các cách rút ngắn thời gian
  Bài 3  Đuôi độ trễ khi toả nhánh 25 dịch vụ
  Bài 4  Ghép theo thời điểm (as-of join) trên bảng nhỏ của đề bài
  Bài 7  Nhịp huấn luyện lại theo tốc độ dịch chuyển
  Bài 8  Quét epsilon trong mô phỏng vòng phản hồi, 30 và 300 vòng
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression


def head(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# =====================================================================
# Bài 1 — PSI khi không có dịch chuyển, n = 500, k = 10
# =====================================================================
head("Bai 1. PSI khi khong co dich chuyen, n = 500, k = 10")

def psi(a, b, bins):
    """Giống hệt hàm psi trong experiments.py: bin theo phân vị của mẫu tham chiếu."""
    edges = np.quantile(a, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    eps = 1e-6
    pa, pb = np.clip(pa, eps, None), np.clip(pb, eps, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))

n1, k1 = 500, 10
rng = np.random.default_rng(2026)
sim = np.array([psi(rng.normal(size=n1), rng.normal(size=n1), k1) for _ in range(20000)])
print(f"Ky vong theo cong thuc 2(k-1)/n     : {2 * (k1 - 1) / n1:.4f}")
print(f"Trung binh mo phong (20 000 lan)    : {sim.mean():.4f}")
print(f"{'phan vi':>10}{'cong thuc chi2':>18}{'mo phong':>12}")
for q in [0.95, 0.99, 0.999]:
    print(f"{q:10.3f}{2 / n1 * stats.chi2.ppf(q, k1 - 1):18.4f}{np.quantile(sim, q):12.4f}")
p_formula = stats.chi2.sf(0.10 * n1 / 2, k1 - 1)
p_sim = (sim > 0.10).mean()
print(f"P(PSI > 0,10) theo cong thuc        : {p_formula:.5f}")
print(f"P(PSI > 0,10) theo mo phong         : {p_sim:.5f}")
print(f"So bao dong gia moi ngay, 200 dac trung: {200 * p_formula:.2f}")
print(f"n nho nhat de E[PSI] < 0,01         : n > {2 * (k1 - 1) / 0.01:.0f}")


# =====================================================================
# Bài 2 — Cỡ mẫu A/B test
# =====================================================================
head("Bai 2. Co mau A/B test: nen 2%, cai thien tuong doi 5%")

def n_per_arm(p, rel_lift, alpha=0.05, power=0.80):
    """Giống hàm n_per_arm trong experiments.py: hai phía, hai nhánh bằng nhau."""
    p2 = p * (1 + rel_lift)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) ** 2 * (p * (1 - p) + p2 * (1 - p2)) / (p2 - p) ** 2

n2 = n_per_arm(0.02, 0.05)
print(f"z_(0,975) = {stats.norm.ppf(0.975):.5f}, z_(0,80) = {stats.norm.ppf(0.80):.5f}")
print(f"Co mau moi nhanh                    : {n2:,.0f}")
print(f"So ngay voi 40 000 luot/nhanh/ngay  : {n2 / 40000:.2f}")
print(f"Giam phuong sai 40% (CUPED)         : {0.6 * n2:,.0f} mau, {0.6 * n2 / 40000:.2f} ngay")
lift_half = brentq(lambda l: n_per_arm(0.02, l) - n2 / 2, 0.05, 0.2)
print(f"Muc cai thien de co mau giam mot nua: {lift_half * 100:.2f}% "
      f"({n_per_arm(0.02, lift_half):,.0f} mau)")
print(f"Luc kiem dinh neu giu 5% va chi chay {n2 / 2 / 40000:.2f} ngay: "
      f"{stats.norm.cdf(np.sqrt(n2 / 2 * (0.02 * 0.05) ** 2 / (0.02 * 0.98 + 0.021 * 0.979)) - stats.norm.ppf(0.975)):.3f}")


# =====================================================================
# Bài 3 — Toả nhánh 25 dịch vụ
# =====================================================================
head("Bai 3. Toa nhanh 25 dich vu, moi dich vu p99 = 80 ms")

k3 = 25
q_each = 0.99 ** (1 / k3)
print(f"Ti le yeu cau gap it nhat mot nhanh cham: {1 - 0.99 ** k3:.4f}")
print(f"Phan vi moi nhanh can dat               : {q_each:.6f} "
      f"(1 trong {1 / (1 - q_each):,.0f} loi goi duoc phep cham)")
for k in [25, 10, 5, 1]:
    print(f"  k = {k:3d}: 1 - 0,99^k = {1 - 0.99 ** k:.4f}")


# =====================================================================
# Bài 4 — Ghép theo thời điểm
# =====================================================================
head("Bai 4. Ghep theo thoi diem tren bang cua de bai")

features = [  # (user_id, valid_from, tong_don_hang)
    (7, "2026-01-01", 3),
    (7, "2026-03-01", 19),
    (7, "2026-06-01", 52),
]
labels = [  # (user_id, event_time, nhan)
    (7, "2026-02-10", 0),
    (7, "2026-04-15", 1),
]

def as_of(user, t):
    """Bản ghi mới nhất có valid_from <= t (ngày dạng ISO so sánh được như chuỗi)."""
    rows = [f for f in features if f[0] == user and f[1] <= t]
    return max(rows, key=lambda f: f[1]) if rows else None

print("Dung (as-of join):")
for u, t, y in labels:
    f = as_of(u, t)
    print(f"  user {u}  {t}  nhan={y}  tong_don_hang={f[2]}  (tu ban ghi {f[1]})")
latest = max(features, key=lambda f: f[1])
print("Sai (ghep vao gia tri hien tai):")
for u, t, y in labels:
    print(f"  user {u}  {t}  nhan={y}  tong_don_hang={latest[2]}")
print(f"Sai (JOIN vao bang lich su, quen dieu kien thoi gian): "
      f"{len(labels) * len(features)} dong thay vi {len(labels)}")


# =====================================================================
# Bài 7 — Nhịp huấn luyện lại theo tốc độ dịch chuyển
# =====================================================================
head("Bai 7. Nhip huan luyen lai theo toc do dich chuyen")

WEEKS = 52

def label(X, w, r):
    p = 1 / (1 + np.exp(-(X @ w)))
    return (r.random(len(X)) < p).astype(int)

def world(week, drift, m=4000):
    """Giống world() trong experiments.py, thêm tham số drift."""
    r = np.random.default_rng(1000 + week)
    X = r.normal(0, 1, (m, 2))
    th = drift * week
    w = np.array([np.cos(th), np.sin(th)]) * 2.0
    return X, label(X, w, r)

def run(cadence, drift):
    X0, y0 = world(0, drift); model = LogisticRegression().fit(X0, y0)
    acc = []
    for wk in range(WEEKS):
        Xw, yw = world(wk, drift)
        acc.append(model.score(Xw, yw))
        if cadence and (wk + 1) % cadence == 0:
            model = LogisticRegression().fit(Xw, yw)
    return np.mean(acc)

CADENCES = [(None, "khong bao gio"), (12, "moi quy"), (4, "moi thang"), (2, "moi 2 tuan"), (1, "moi tuan")]
print(f"{'drift (rad/tuan)':>17}" + "".join(f"{lab:>15}" for _, lab in CADENCES) + "   nhip thua nhat trong 1 diem")
for drift in [0.010, 0.035, 0.080]:
    accs = [run(c, drift) * 100 for c, _ in CADENCES]
    best = accs[-1]
    ok = [lab for (c, lab), a in zip(CADENCES, accs) if c and best - a <= 1.0]
    print(f"{drift:17.3f}" + "".join(f"{a:14.2f}%" for a in accs) + f"   {ok[0]}")


# =====================================================================
# Bài 8 — Quét epsilon trong mô phỏng vòng phản hồi
# =====================================================================
head("Bai 8. Quet epsilon: chat luong he thong biet va chat luong giao toi nguoi dung")

N_ITEM, USERS, TOPK, WARMUP = 2000, 3000, 10, 200
true_q = np.random.default_rng(88).beta(1.6, 9.0, N_ITEM)   # danh muc rieng cua bai tap
ceil10 = np.sort(true_q)[-10:].mean()
print(f"Danh muc cua bai tap: tran top-10 = {ceil10:.3f}, trung binh = {true_q.mean():.3f}")

def simulate(eps, rounds, seed):
    """Như simulate() trong experiments.py, thêm chất lượng thật của các món đã hiển thị."""
    r = np.random.default_rng(seed)
    shown = np.zeros(N_ITEM); clicks = np.zeros(N_ITEM)
    imp = r.choice(N_ITEM, WARMUP)
    np.add.at(shown, imp, 1)
    np.add.at(clicks, imp, r.random(WARMUP) < true_q[imp])
    believed, served = [], []
    for _ in range(rounds):
        score = np.where(shown > 0, clicks / np.maximum(shown, 1), 0.0)
        top = np.argsort(-score)[:TOPK]
        pool = top
        if eps > 0:
            n_rand = max(1, int(round(eps * TOPK)))
            pool = np.r_[top[: TOPK - n_rand], r.choice(N_ITEM, n_rand, replace=False)]
        imp = r.choice(pool, USERS)
        cl = r.random(USERS) < true_q[imp]
        np.add.at(shown, imp, 1); np.add.at(clicks, imp, cl)
        believed.append(true_q[top].mean())
        served.append(true_q[imp].mean())
    return np.array(believed), np.array(served)

for rounds in [30, 300]:
    print(f"\n{rounds} vong, trung binh 20 lan chay")
    print(f"{'epsilon':>8}{'top-10 vong cuoi':>18}{'% tran':>8}{'giao, ca ky':>13}{'giao, 5 vong cuoi':>19}")
    # epsilon * 10 là số vị trí ngẫu nhiên trong danh sách 10 món, nên chỉ xét bội của 0,1
    for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        bs, ss = [], []
        for sd in range(20):
            b, s = simulate(eps, rounds, seed=500 + sd)
            bs.append(b); ss.append(s)
        b, s = np.mean(bs, 0), np.mean(ss, 0)
        print(f"{eps:8.2f}{b[-1]:18.3f}{b[-1] / ceil10 * 100:7.0f}%{s.mean():13.4f}{s[-5:].mean():19.4f}")
