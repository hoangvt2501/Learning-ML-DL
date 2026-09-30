"""Thí nghiệm cho giáo trình "Biểu diễn, Sinh và Căn chỉnh".

Mọi con số thực nghiệm trong tài liệu đều in ra từ script này. Hạt giống cố định
nên chạy lại cho kết quả y hệt.

    python code/bieudien/experiments.py

Một số khối không phải mô phỏng mà là **kiểm chứng một đẳng thức**: nghiệm tối ưu
của bài toán RL có ràng buộc KL đúng bằng softmax có trọng số, và hàm mục tiêu của
DPO đạt cực tiểu đúng tại nghiệm ấy. Những khối đó phải khớp tới sai số máy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figs/"
rng = np.random.default_rng(17)
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
})
C_MAIN, C_WARN, C_BAD, C_DIM = "#1f6f68", "#b8860b", "#b0413e", "#8a857c"


def head(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# =====================================================================
# (A) Hình 2 — Embedding học được gì, và nó thực ra đang phân rã cái gì
# =====================================================================
head("(A) Tu dong hien den PMI den embedding: quan he deu dan noi len the nao")

# Kho ngu lieu nhan tao co CAU TRUC hai truc: moi tu la (chu de, vai tro).
CHU_DE = ["vua", "thanh", "mua", "mon", "cay"]
VAI_TRO = ["nam", "nu", "nho", "lon"]
tu = [f"{c}_{v}" for c in CHU_DE for v in VAI_TRO]
tu2id = {w: i for i, w in enumerate(tu)}
V = len(tu)

def sinh_kho(n_cau=40000, seed=0):
    """Hai loai cau, chia doi.

    Loai 1 gom cac tu CUNG CHU DE; loai 2 gom cac tu CUNG VAI TRO. Khong cau nao
    noi rang hai truc ay doc lap nhau, va khong o nao trong du lieu ghi nhan
    "chu de" hay "vai tro" la khai niem. Neu embedding tach duoc chung thanh hai
    huong CONG DUOC VOI NHAU thi do la dieu mo hinh tu rut ra.
    """
    r = np.random.default_rng(seed)
    cau = []
    for i in range(n_cau):
        if i % 2 == 0:
            c = CHU_DE[r.integers(len(CHU_DE))]
            cau.append([f"{c}_{v}" for v in r.permutation(VAI_TRO)[:3]])
        else:
            v = VAI_TRO[r.integers(len(VAI_TRO))]
            cau.append([f"{c}_{v}" for c in r.permutation(CHU_DE)[:3]])
    return cau

kho = sinh_kho()
dong_hien = np.zeros((V, V))
for cau in kho:
    ids = [tu2id[w] for w in cau]
    for i, a in enumerate(ids):
        for j in range(len(ids)):
            if i != j:
                dong_hien[a, ids[j]] += 1
print(f"Kho ngu lieu: {len(kho):,} cau, tu vung {V} tu "
      f"({len(CHU_DE)} chu de x {len(VAI_TRO)} vai tro)")
print(f"So cap dong hien: {int(dong_hien.sum()):,}")

# ---- Buoc 1: tu dem sang PMI --------------------------------------
P = dong_hien / dong_hien.sum()
p_w = P.sum(1, keepdims=True)
p_c = P.sum(0, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    PMI = np.log(P / (p_w * p_c))
PMI[~np.isfinite(PMI)] = 0.0
PPMI = np.maximum(PMI, 0.0)

print(f"\nPMI(w,c) = log[ p(w,c) / (p(w)p(c)) ] — do 'dong hien nhieu hon ngau nhien'")
print(f"  PMI giua hai tu cung chu de, khac vai tro : "
      f"{PMI[tu2id['vua_nam'], tu2id['vua_nu']]:+.4f}")
print(f"  PMI giua hai tu cung vai tro, khac chu de : "
      f"{PMI[tu2id['vua_nam'], tu2id['mua_nam']]:+.4f}")
print(f"  PMI giua hai tu khac ca hai               : "
      f"{PMI[tu2id['vua_nam'], tu2id['mua_nu']]:+.4f}")
print("Ba con so nay la toan bo tin hieu ma mo hinh co. Khong co gi khac.")

# ---- Buoc 2: phan ra PPMI bang SVD --------------------------------
U_, s_, Vt_ = np.linalg.svd(PPMI)
DIM = 8
E = U_[:, :DIM] * np.sqrt(s_[:DIM])
En = E / np.linalg.norm(E, axis=1, keepdims=True)

def tuong_tu_vec(v, loai_tru=()):
    v = v / np.linalg.norm(v)
    s = En @ v
    for x in loai_tru:
        s[tu2id[x]] = -9
    return tu[int(np.argmax(s))]

def do_loai_suy(En_):
    dung = tong = 0
    for c1 in CHU_DE:
        for c2 in CHU_DE:
            if c1 == c2:
                continue
            for v1 in VAI_TRO:
                for v2 in VAI_TRO:
                    if v1 == v2:
                        continue
                    a, b, c = f"{c1}_{v1}", f"{c1}_{v2}", f"{c2}_{v1}"
                    q = En_[tu2id[b]] - En_[tu2id[a]] + En_[tu2id[c]]
                    q = q / np.linalg.norm(q)
                    sc = En_ @ q
                    for x in (a, b, c):
                        sc[tu2id[x]] = -9
                    dung += tu[int(np.argmax(sc))] == f"{c2}_{v2}"
                    tong += 1
    return dung / tong, tong

acc, tong = do_loai_suy(En)
nn = 1.0 / (V - 3)
print(f"\nPhan ra PPMI bang SVD, giu {DIM} chieu:")
print(f"  vua_nam : vua_nu :: mua_nho : ?  ->  "
      f"{tuong_tu_vec(En[tu2id['vua_nu']] - En[tu2id['vua_nam']] + En[tu2id['mua_nho']], ('vua_nam', 'vua_nu', 'mua_nho'))}")
print(f"  Do chinh xac tren {tong} phep loai suy: {acc:.4f}   "
      f"(doan ngau nhien: {nn:.4f})")

print(f"\n{'số chiều giữ lại':>18}{'độ chính xác loại suy':>25}")
for dim in [2, 4, 6, 8, 12, 16, 20]:
    Ed = U_[:, :dim] * np.sqrt(s_[:dim])
    Ed = Ed / np.maximum(np.linalg.norm(Ed, axis=1, keepdims=True), 1e-12)
    a_, _ = do_loai_suy(Ed)
    sao = "  <- dung o day" if dim == DIM else ""
    print(f"{dim:18d}{a_:25.4f}{sao}")
print("Bang nay KHONG phai mot duong cong dep: giu 20 chieu (hang day du) lai dat 1,0")
print("tro lai, vi khi do ta tai dung PPMI chinh xac. Voi tu vung chi 20 tu thi khong")
print("du cho to ra mot danh doi that su; muon thay duong cong that phai chay tren kho")
print("ngu lieu that. Bang dua vao day dung de noi ro dieu ay chu khong de ket luan.")

# ---- Buoc 3: skip-gram that su dang toi uu cai gi -----------------
cw = dong_hien.sum(1)
cc = dong_hien.sum(0)
Ntong = dong_hien.sum()

def sgns(dim, k=1, epochs=4000, lr=0.5, seed=1):
    """Skip-gram voi lay mau am, xuong doc tren KY VONG cua ham muc tieu.

        L = -sum_wc [ #(w,c) log sigma(<w,c>) + k #(w) P_n(c) log sigma(-<w,c>) ]

    Dung truc tiep bang dem nen gradient la ky vong chinh xac chu khong phai uoc
    luong nhieu. P_n lay la phan phoi unigram (khong nang luy thua 3/4) de doi
    chieu duoc voi ket qua ly thuyet.
    """
    r = np.random.default_rng(seed)
    W = r.normal(0, .1, (V, dim))
    C = r.normal(0, .1, (V, dim))
    am = k * np.outer(cw, cc / Ntong)
    for _ in range(epochs):
        S = W @ C.T
        sig = 1 / (1 + np.exp(-S))
        G = (-dong_hien * (1 - sig) + am * sig) / Ntong
        W -= lr * (G @ C)
        C -= lr * (G.T @ W)
    return W, C

o = dong_hien > 0
print(f"\nLevy & Goldberg (2014) chung minh: skip-gram voi lay mau am AM THAM phan ra")
print(f"ma tran PMI dich mot luong log k. Cho dao ham bang 0 se ra dung")
print(f"    <w_i, c_j> = log[ #(i,j) N / (#(i) #(j)) ] - log k = PMI(i,j) - log k")
print(f"\n{'số chiều':>10}{'tương quan với PMI':>22}{'|lệch| trung bình':>20}{'|lệch| lớn nhất':>18}")
for dim in (8, V):
    W_sg, C_sg = sgns(dim)
    S_sg = W_sg @ C_sg.T
    rho = np.corrcoef(S_sg[o].ravel(), PMI[o].ravel())[0, 1]
    dif = np.abs(S_sg - PMI)[o]
    ghi = "  <- hang day du" if dim == V else "  <- hang thap"
    print(f"{dim:10d}{rho:22.4f}{dif.mean():20.4f}{dif.max():18.4f}{ghi}")
    if dim == V:
        S_full, rho_full = S_sg, rho
    else:
        W_low = W_sg

print("O HANG DAY DU, xuong doc hoi tu ve dung ma tran PMI — day la kiem chung bang so")
print("cho ket qua ly thuyet o tren, va hai duong tinh khong lien quan gi toi nhau.")
print("O HANG THAP thi no khong the khop chinh xac, va phai chon giu lai cai gi. Chinh")
print("su ep buoc ay tao ra embedding co ich: mo hinh buoc phai gom cac tu giong nhau.")

E_sg = W_low / np.linalg.norm(W_low, axis=1, keepdims=True)
a_sg, _ = do_loai_suy(E_sg)
print(f"\nDo chinh xac loai suy cua embedding skip-gram hang 8: {a_sg:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1))
im = ax[0].imshow(PMI, cmap="coolwarm", vmin=-2, vmax=2)
ax[0].set_title("Ma trận PMI (20×20 từ)", fontsize=9)
ax[0].set_xticks([]); ax[0].set_yticks([])
fig.colorbar(im, ax=ax[0], fraction=.046)

xy = (U_[:, :2] * np.sqrt(s_[:2]))
mau = {c: plt.cm.tab10(i) for i, c in enumerate(CHU_DE)}
mk = {"nam": "o", "nu": "s", "nho": "^", "lon": "D"}
for i, w in enumerate(tu):
    c, v = w.split("_")
    ax[1].scatter(xy[i, 0], xy[i, 1], color=mau[c], marker=mk[v], s=54,
                  edgecolor="w", linewidth=.6)
for c in CHU_DE:
    ax[1].scatter([], [], color=mau[c], label=c, s=36)
ax[1].legend(frameon=False, fontsize=6.5, ncol=2, loc="best")
ax[1].set_title("Màu = chủ đề, hình = vai trò", fontsize=9)
ax[1].set_xlabel("chiều 1"); ax[1].set_ylabel("chiều 2")

ax[2].scatter(PMI[o].ravel(), S_full[o].ravel(), s=9, alpha=.55, color=C_MAIN)
lo, hi = PMI[o].min(), PMI[o].max()
ax[2].plot([lo, hi], [lo, hi], ":", color=C_DIM, lw=1.2)
ax[2].set_xlabel("PMI$(i,j)$")
ax[2].set_ylabel("$\\langle w_i, c_j\\rangle$ học được")
ax[2].set_title(f"Skip-gram hội tụ về PMI (r = {rho_full:.4f})", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "bd02_word2vec.png"); plt.close(fig)


# =====================================================================
# (B) Vì sao phải có lấy mẫu âm: đếm phép tính
# =====================================================================
head("(B) Softmax day du so voi lay mau am: dem phep tinh")

print(f"{'từ vựng V':>12}{'chiều d':>10}{'softmax đầy đủ':>18}{'lấy mẫu âm (k=5)':>20}{'rẻ hơn':>10}")
Brows = []
for V_, d_ in [(1_000, 300), (50_000, 300), (400_000, 300), (2_000_000, 300)]:
    full = 2 * V_ * d_          # mot phep nhan + mot phep cong cho moi trong so
    negs = 2 * (5 + 1) * d_
    Brows.append((V_, full, negs, full / negs))
    print(f"{V_:12,}{d_:10}{full:18,}{negs:20,}{full / negs:9.0f}x")
print("Chi phi cua softmax day du ti le TUYEN TINH voi co tu vung; cua lay mau am thi")
print("KHONG PHU THUOC tu vung. Day la ly do word2vec huan luyen duoc tren kho tu ty tu")
print("nam 2013, khi phan cung con yeu hon bay gio rat nhieu.")

fig, ax = plt.subplots(figsize=(5.4, 3.0))
Vs = [b[0] for b in Brows]
ax.plot(Vs, [b[1] for b in Brows], "-o", ms=5, color=C_BAD, label="softmax đầy đủ")
ax.plot(Vs, [b[2] for b in Brows], "-o", ms=5, color=C_MAIN, label="lấy mẫu âm, $k=5$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("kích thước từ vựng $V$"); ax.set_ylabel("phép tính mỗi cặp (log)")
ax.set_title("Lấy mẫu âm cắt đứt phụ thuộc vào $V$", fontsize=9)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "bd03_negsampling.png"); plt.close(fig)


# =====================================================================
# (C) Hình 4 — Bất đẳng hướng: vì sao cosine thô hay đánh lừa
# =====================================================================
head("(C) Bat dang huong cua khong gian embedding")

def do_bat_dang_huong(X, ten):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    cos_tb = G[iu].mean()
    Xc = X - X.mean(0)
    Xcn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    Gc = Xcn @ Xcn.T
    cos_tb_c = Gc[iu].mean()
    # Do lech tam: ti le giua do dai vector trung binh va do dai trung binh.
    lech_tam = np.linalg.norm(X.mean(0)) / np.linalg.norm(X, axis=1).mean()
    return ten, cos_tb, cos_tb_c, lech_tam

print(f"{'phân bố embedding':<34}{'cosine TB':>12}{'sau khi trừ TB':>17}{'độ lệch tâm':>14}")
r = np.random.default_rng(5)
cases = [
    ("đều trên mặt cầu (lý tưởng)", r.normal(0, 1, (900, 64))),
    ("lệch tâm nhẹ", r.normal(0, 1, (900, 64)) + 0.8),
    ("lệch tâm mạnh (hay gặp thật)", r.normal(0, 1, (900, 64)) + 2.6),
]
for ten, X in cases:
    t, a, b, lt = do_bat_dang_huong(X, ten)
    print(f"{t:<34}{a:12.4f}{b:17.4f}{lt:14.4f}")
print("Khi dam may embedding LECH khoi goc, moi cap deu co cosine duong lon, va con so")
print("ay khong con noi len su giong nhau nua — no chi phan anh huong chung. Tru trung")
print("binh (hoac chuan hoa theo lop) tra lai y nghia cho cosine.")

fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
for a_, (ten, X) in zip(ax, cases):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    a_.hist(G[iu], bins=60, color=C_MAIN, alpha=.85)
    a_.axvline(G[iu].mean(), color=C_BAD, lw=1.4)
    a_.set_title(ten, fontsize=8.5)
    a_.set_xlabel("cosine giữa hai vector bất kỳ")
    a_.set_xlim(-1, 1)
ax[0].set_ylabel("số cặp")
fig.tight_layout(); fig.savefig(OUT + "bd04_anisotropy.png"); plt.close(fig)


# =====================================================================
# (D) LoRA: đếm tham số, và vì sao nó rẻ tới vậy
# =====================================================================
head("(D) LoRA: dem tham so va bo nho")

def lora_params(d, r_):
    return 2 * d * r_              # A (d x r) va B (r x d)

print(f"{'lớp':<26}{'d':>7}{'toàn phần d^2':>16}{'LoRA r=8':>12}{'LoRA r=64':>12}{'% (r=8)':>10}")
for ten, d in [("GPT-2 small", 768), ("GPT-2 large", 1280), ("Llama-2 7B", 4096),
               ("Llama-2 70B", 8192)]:
    full = d * d
    print(f"{ten:<26}{d:7}{full:16,}{lora_params(d, 8):12,}{lora_params(d, 64):12,}"
          f"{lora_params(d, 8) / full * 100:9.3f}%")

# Ap cho ca mo hinh: LoRA thuong chi gan vao W_Q va W_V.
L, d = 32, 4096
attn_full = L * 4 * d * d
N_7B = 6_476_005_376
for r_ in (4, 8, 16, 64):
    lora_all = L * 2 * lora_params(d, r_)        # W_Q va W_V moi lop
    print(f"\nLlama-2 7B, LoRA hang {r_} tren W_Q va W_V:")
    print(f"  tham so huan luyen : {lora_all:,}")
    print(f"  so voi {N_7B:,} tham so cua ca mo hinh: {lora_all / N_7B * 100:.4f}%")
    if r_ == 8:
        print(f"  trang thai Adam (fp32, 2 trang thai): "
              f"{lora_all * 2 * 4 / 2**20:.1f} MiB so voi "
              f"{N_7B * 2 * 4 / 2**30:.1f} GiB neu tinh chinh toan phan")
print("\nLoRA khong lam mo hinh chay nhanh hon khi SUY LUAN (co the gop B@A vao W).")
print("Cai no doi la BO NHO LUC HUAN LUYEN, va do la thu chan nguoi ta lai.")


# =====================================================================
# (E) Hình 6 — Học chuyển giao: khi nào đóng băng, khi nào tinh chỉnh hết
# =====================================================================
head("(E) Dong bang dac trung so voi tinh chinh toan phan theo co du lieu")

r = np.random.default_rng(23)
d_in, d_hid, K = 40, 24, 4

# "Mo hinh da tien huan luyen": mot phep chieu CO CAU TRUC, gan dung voi cau truc
# that cua nhiem vu dich nhung khong trung khop han — dung nhu tinh huong that.
W_that = r.normal(0, 1, (d_in, d_hid))
W_tien = W_that + 0.55 * r.normal(0, 1, (d_in, d_hid))
w_out_that = np.random.default_rng(99).normal(0, 1, (d_hid, K))

def sinh(n, seed):
    rr = np.random.default_rng(seed)
    X = rr.normal(0, 1, (n, d_in))
    H = np.tanh(X @ W_that)
    y = (H @ w_out_that + 0.35 * rr.normal(0, 1, (n, K))).argmax(1)
    return X, y

def huan_luyen(X, y, W1_khoi, dong_bang, vong=2500, lr=0.12, lam=1e-3, seed=0):
    """MLP mot lop an, tu cai de NAP DUOC trong so tien huan luyen.

    scikit-learn khong cho nap trong so ban dau, nen neu dung MLPClassifier thi
    "tinh chinh toan phan" that ra van la huan luyen tu dau — mot so sanh vo nghia.
    """
    rr = np.random.default_rng(seed)
    W1 = W1_khoi.copy()
    W2 = rr.normal(0, .1, (W1.shape[1], K))
    b2 = np.zeros(K)
    Y = np.eye(K)[y]
    n = len(X)
    for _ in range(vong):
        H = np.tanh(X @ W1)
        P = softmax(H @ W2 + b2)
        dZ = (P - Y) / n
        gW2 = H.T @ dZ + lam * W2
        gb2 = dZ.sum(0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        if not dong_bang:
            dH = (dZ @ W2.T) * (1 - H ** 2)
            W1 -= lr * (X.T @ dH + lam * W1)
    return W1, W2, b2

def cham(X, y, mo_hinh):
    W1, W2, b2 = mo_hinh
    return float(((np.tanh(X @ W1) @ W2 + b2).argmax(1) == y).mean())

X_te, y_te = sinh(4000, 777)

print(f"{'số mẫu có nhãn':>17}{'đóng băng + đầu tuyến tính':>30}"
      f"{'tinh chỉnh toàn phần':>24}{'huấn luyện từ đầu':>21}")
Erows = []
for n in [20, 50, 150, 500, 2000, 8000]:
    X_tr, y_tr = sinh(n, 4000 + n)
    W_ngau = np.random.default_rng(7).normal(0, 1 / np.sqrt(d_in), (d_in, d_hid))
    a_fr = cham(X_te, y_te, huan_luyen(X_tr, y_tr, W_tien, True, seed=1))
    a_ft = cham(X_te, y_te, huan_luyen(X_tr, y_tr, W_tien, False, seed=1))
    a_sc = cham(X_te, y_te, huan_luyen(X_tr, y_tr, W_ngau, False, seed=1))
    Erows.append((n, a_fr, a_ft, a_sc))
    print(f"{n:17,}{a_fr:30.4f}{a_ft:24.4f}{a_sc:21.4f}")
print("Ca ba cot dung CUNG mot cai dat mang, chi khac trong so ban dau cua lop an va")
print("viec co cap nhat no hay khong — nen so sanh moi co nghia.")
print("Voi RAT IT nhan, dac trung dong bang thang: no chi phai uoc luong mot lop tuyen")
print("tinh. Khi du lieu nhieu len, tinh chinh bat kip roi vuot len, vi luc do mo hinh")
print("du rang buoc de sua lai phan dac trung cho khop nhiem vu. Huan luyen tu dau luon")
print("di sau tinh chinh cho toi khi du lieu rat nhieu.")

fig, ax = plt.subplots(figsize=(5.8, 3.2))
ns = [e[0] for e in Erows]
ax.plot(ns, [e[1] for e in Erows], "-o", ms=5, color=C_MAIN, label="đóng băng + đầu tuyến tính")
ax.plot(ns, [e[2] for e in Erows], "-o", ms=5, color=C_WARN, label="tinh chỉnh toàn phần")
ax.plot(ns, [e[3] for e in Erows], "-o", ms=5, color=C_BAD, label="huấn luyện từ đầu")
ax.set_xscale("log"); ax.set_xlabel("số mẫu có nhãn (log)")
ax.set_ylabel("độ chính xác trên tập kiểm tra")
ax.set_title("Ít nhãn thì đóng băng thắng; nhiều nhãn thì ngược lại", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "bd06_transfer.png"); plt.close(fig)


# =====================================================================
# (F) Hình 8 — VAE: beta quyết định bao nhiêu chiều ẩn còn sống
# =====================================================================
head("(F) VAE: beta dieu khien so chieu an con mang thong tin")

def vae(X, d_z=6, beta=1.0, vong=4000, lr=0.006, seed=0):
    """VAE tuyến tính (bộ mã hoá và giải mã đều affine).

    Chọn tuyến tính có chủ ý: khi bộ giải mã tuyến tính và nhiễu Gauss, VAE có
    lời giải hiểu được và liên hệ thẳng với PCA xác suất, nên mọi hiện tượng
    quan sát được đều truy được về công thức chứ không phải về mẹo huấn luyện.
    """
    r = np.random.default_rng(seed)
    n, d = X.shape
    We = r.normal(0, .1, (d, d_z)); be = np.zeros(d_z)
    Wl = r.normal(0, .1, (d, d_z)); bl = np.full(d_z, -1.0)   # log phương sai
    Wd = r.normal(0, .1, (d_z, d)); bd = np.zeros(d)
    for _ in range(vong):
        mu = X @ We + be
        logv = np.clip(X @ Wl + bl, -9, 9)
        std = np.exp(0.5 * logv)
        eps = r.normal(0, 1, (n, d_z))
        z = mu + std * eps
        Xh = z @ Wd + bd
        dX = (Xh - X) / n
        gWd = z.T @ dX; gbd = dX.sum(0)
        dz = dX @ Wd.T
        gmu = dz + beta * mu / n
        glogv = 0.5 * std * (dz * eps) + beta * 0.5 * (np.exp(logv) - 1) / n
        gWe = X.T @ gmu; gbe = gmu.sum(0)
        gWl = X.T @ glogv; gbl = glogv.sum(0)
        for p_, g_ in ((We, gWe), (be, gbe), (Wl, gWl), (bl, gbl), (Wd, gWd), (bd, gbd)):
            p_ -= lr * np.clip(g_, -5, 5)
    mu = X @ We + be
    logv = np.clip(X @ Wl + bl, -9, 9)
    Xh = mu @ Wd + bd
    tai_dung = float(((Xh - X) ** 2).sum(1).mean())
    kl_chieu = 0.5 * (mu ** 2 + np.exp(logv) - logv - 1).mean(0)
    return tai_dung, float(kl_chieu.sum()), kl_chieu

r = np.random.default_rng(31)
n_f = 600
z_that = r.normal(0, 1, (n_f, 2))
A_f = np.array([[2.4, 0.3, 1.1, -0.6], [0.2, 2.1, -0.8, 1.4]])
X_f = z_that @ A_f + 0.25 * r.normal(0, 1, (n_f, 4))
D_Z = 6

print(f"Du lieu: {n_f} diem trong R^4, sinh tu DUNG 2 yeu to an cong nhieu nho.")
print(f"Ta co tinh cho VAE {D_Z} chieu an — nhieu hon can — de xem no lam gi voi phan thua.")
print(f"\n{'beta':>7}{'sai số tái dựng':>18}{'tổng KL':>10}{'số chiều còn sống':>20}   KL của từng chiều")
Frows = []
for beta in [0.0, 0.05, 0.2, 1.0, 4.0, 16.0]:
    td, kl, klc = vae(X_f, d_z=D_Z, beta=beta)
    song = int((klc > 0.01).sum())
    Frows.append((beta, td, kl, song))
    print(f"{beta:7.2f}{td:18.4f}{kl:10.4f}{song:20d}   {np.round(klc, 3)}")
print("\nMot chieu goi la CON SONG khi KL cua no lon hon 0 dang ke — tuc hau nghiem")
print("q(z_i|x) con khac tien nghiem N(0,1), nghia la chieu ay con mang thong tin ve x.")
print("beta = 0   : khong suc ep nao, ca 6 chieu deu song. Do la bo tu ma hoa thuong;")
print("             tai dung rat tot nhung khong gian an khong co cau truc de lay mau.")
print("beta = 1   : dung 2 chieu song lai — DUNG BANG so yeu to that sinh ra du lieu.")
print("             VAE tu tim ra so chieu can thiet, khong ai bao no.")
print("beta = 16  : KHONG chieu nao song. Day la SUP HAU NGHIEM: z khong con lien quan")
print("             gi toi x, va bo giai ma chi con biet doan gia tri trung binh.")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
bs = [f[0] for f in Frows]
ax[0].plot(bs, [f[1] for f in Frows], "-o", ms=5, color=C_BAD, label="sai số tái dựng")
ax[0].plot(bs, [f[2] for f in Frows], "-o", ms=5, color=C_MAIN, label="tổng KL")
ax[0].set_xscale("symlog", linthresh=.05); ax[0].set_yscale("log")
ax[0].set_xlabel("$\\beta$"); ax[0].set_ylabel("giá trị (log)")
ax[0].set_title("Tăng $\\beta$ đổi tái dựng lấy KL nhỏ", fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(bs, [f[3] for f in Frows], "-o", ms=6, color=C_WARN)
ax[1].axhline(2, ls=":", color=C_DIM)
ax[1].text(0.06, 2.15, "số yếu tố thật = 2", fontsize=7.5, color=C_DIM)
ax[1].set_xscale("symlog", linthresh=.05)
ax[1].set_xlabel("$\\beta$"); ax[1].set_ylabel("số chiều ẩn còn sống")
ax[1].set_yticks(range(0, D_Z + 1))
ax[1].set_title("Các chiều thừa chết dần khi $\\beta$ tăng", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "bd08_vae.png"); plt.close(fig)


# =====================================================================
# (G) Hình 9 — GAN: vì sao bài báo gốc phải đổi hàm mất mát của bộ sinh
# =====================================================================
head("(G) GAN: ham mat mat bao hoa so voi khong bao hoa")

def gan(X_that, vong=4000, d_z=8, h=48, lr=0.004, seed=0, bao_hoa=False, m=128):
    """GAN nhỏ sinh điểm 2 chiều, cài tay để đổi được hàm mất mát của bộ sinh.

    bao_hoa=True  : bộ sinh tối thiểu  log(1 - D(G(z)))  — đúng bài toán minimax gốc.
    bao_hoa=False : bộ sinh tối đa     log D(G(z))       — mẹo "không bão hoà".
    """
    r = np.random.default_rng(seed)
    kh = lambda a, b: r.normal(0, np.sqrt(2.0 / a), (a, b))
    G1, g1 = kh(d_z, h), np.zeros(h)
    G2, g2 = kh(h, 2), np.zeros(2)
    D1, d1 = kh(2, h), np.zeros(h)
    D2, d2 = kh(h, 1), np.zeros(1)
    for _ in range(vong):
        z = r.normal(0, 1, (m, d_z))
        hg = np.maximum(z @ G1 + g1, 0)
        Xg = hg @ G2 + g2
        Xr = X_that[r.integers(0, len(X_that), m)]
        hr = np.maximum(Xr @ D1 + d1, 0); sr = hr @ D2 + d2
        hf = np.maximum(Xg @ D1 + d1, 0); sf = hf @ D2 + d2
        pr, pf = 1 / (1 + np.exp(-sr)), 1 / (1 + np.exp(-sf))
        gsr, gsf = (pr - 1) / m, pf / m
        gD2 = hr.T @ gsr + hf.T @ gsf; gd2 = gsr.sum(0) + gsf.sum(0)
        ghr = (gsr @ D2.T) * (hr > 0); ghf = (gsf @ D2.T) * (hf > 0)
        gD1 = Xr.T @ ghr + Xg.T @ ghf; gd1 = ghr.sum(0) + ghf.sum(0)
        D1 -= lr * gD1; d1 -= lr * gd1; D2 -= lr * gD2; d2 -= lr * gd2

        hf = np.maximum(Xg @ D1 + d1, 0); sf = hf @ D2 + d2
        pf = 1 / (1 + np.exp(-sf))
        # d/ds log(1-sigma(s)) = -sigma(s);  d/ds log sigma(s) = 1 - sigma(s)
        gsf = (pf * (1 - pf) / m) if bao_hoa else (-(1 - pf) / m)
        ghf = (gsf @ D2.T) * (hf > 0)
        gXg = ghf @ D1.T
        gG2 = hg.T @ gXg; gg2 = gXg.sum(0)
        ghg = (gXg @ G2.T) * (hg > 0)
        gG1 = z.T @ ghg; gg1 = ghg.sum(0)
        G1 -= lr * gG1; g1 -= lr * gg1; G2 -= lr * gG2; g2 -= lr * gg2
    z = r.normal(0, 1, (3000, d_z))
    return np.maximum(z @ G1 + g1, 0) @ G2 + g2

K_MODE = 8
goc = np.stack([np.cos(2 * np.pi * np.arange(K_MODE) / K_MODE),
                np.sin(2 * np.pi * np.arange(K_MODE) / K_MODE)], 1) * 2.4
r = np.random.default_rng(3)
X_gan = np.concatenate([g + 0.13 * r.normal(0, 1, (700, 2)) for g in goc])

def do_phu(Xg):
    d2 = ((Xg[:, None, :] - goc[None]) ** 2).sum(-1)
    trong = d2.min(1) < 0.55 ** 2
    return len(np.unique(d2.argmin(1)[trong])), trong.mean() * 100

print(f"Du lieu that: {K_MODE} cum dat deu tren mot vong tron.")
print(f"\n{'hàm mất mát của bộ sinh':<34}{'chế độ phủ được (4 hạt giống)':>32}{'% điểm rơi vào cụm':>22}")
Grows = {}
for ten, bh in [("tối thiểu log(1 − D(G(z))) — gốc", True),
                ("tối đa log D(G(z)) — không bão hoà", False)]:
    kq = [do_phu(gan(X_gan, seed=s, bao_hoa=bh)) for s in range(4)]
    Grows[bh] = kq
    phu = [k[0] for k in kq]
    tl = np.mean([k[1] for k in kq])
    print(f"{ten:<34}{str(phu):>32}{tl:21.1f}%")

print("\nDang bai toan MINIMAX goc that bai hoan toan: bo sinh khong hoc duoc gi, 0/8 che")
print("do. Ly do nam o dao ham: d/ds log(1 - sigma(s)) = -sigma(s). Luc dau bo phan biet")
print("de dang nhan ra hang gia nen sigma(s) ~ 0, va gradient cua bo sinh cung ~ 0 theo.")
print("Bo sinh bi ket ngay tu dau — chinh la dieu bai bao GAN goc canh bao, va la ly do")
print("ho de xuat doi sang toi da log D(G(z)), thu cho gradient LON khi bo sinh dang te.")
print("\nMot luu y trung thuc: o thiet lap hai chieu nay, ban khong bao hoa phu DU CA 8")
print("che do o moi hat giong, tuc KHONG quan sat duoc sup che do. Sup che do la hien")
print("tuong that va rat hay gap, nhung no can thiet lap kho hon (nhieu chieu, du lieu")
print("that) moi lo ra; khang dinh no tu thi nghiem nay thi se la noi qua.")

fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
for c_, (bh, ten) in enumerate([(True, "minimax gốc"), (False, "không bão hoà")]):
    for k_, seed in enumerate((0, 1)):
        a_ = ax[c_ * 2 + k_]
        Xg = gan(X_gan, seed=seed, bao_hoa=bh)
        phu, tl = do_phu(Xg)
        a_.scatter(*X_gan[::7].T, s=4, color=C_DIM, alpha=.35)
        a_.scatter(*Xg[::3].T, s=5, color=C_BAD, alpha=.55)
        a_.set_title(f"{ten} · hạt {seed}\nphủ {phu}/{K_MODE}", fontsize=8)
        a_.set_xlim(-4, 4); a_.set_ylim(-4, 4)
        a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
fig.tight_layout(); fig.savefig(OUT + "bd09_gan.png"); plt.close(fig)


# =====================================================================
# (H) Hình 10 — Khuếch tán: dạng đóng của quá trình thuận, kiểm chứng
# =====================================================================
head("(H) Mo hinh khuech tan: dang dong cua qua trinh thuan")

T = 1000
beta = np.linspace(1e-4, 0.02, T)          # lich tuyen tinh cua DDPM
alpha = 1 - beta
alpha_ngang = np.cumprod(alpha)

print("Quy tac tung buoc:  q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I)")
print("Gop T buoc lai cho dang dong:")
print("    q(x_t | x_0) = N(sqrt(alpha_ngang_t) x_0, (1 - alpha_ngang_t) I)")
print("Day la ly do khong can chay 1000 buoc de lay mau x_t — nhay thang mot phat.")

r = np.random.default_rng(9)
x0 = np.array([2.0])
N_MP = 200_000
print(f"\nMo phong {N_MP:,} quy dao roi so voi dang dong:")
print(f"{'t':>6}{'TB mô phỏng':>16}{'TB dạng đóng':>16}{'PS mô phỏng':>16}{'PS dạng đóng':>16}")
for t in (1, 10, 100, 400, 999):
    x = np.full(N_MP, x0[0])
    for s in range(t + 1):
        x = np.sqrt(alpha[s]) * x + np.sqrt(beta[s]) * r.normal(0, 1, N_MP)
    tb_mp, ps_mp = x.mean(), x.var()
    tb_dd = np.sqrt(alpha_ngang[t]) * x0[0]
    ps_dd = 1 - alpha_ngang[t]
    print(f"{t:6d}{tb_mp:16.5f}{tb_dd:16.5f}{ps_mp:16.5f}{ps_dd:16.5f}")
print("Bon cot khop nhau trong sai so lay mau — dang dong dung.")

snr = alpha_ngang / (1 - alpha_ngang)
print(f"\n{'t':>6}{'alpha_ngang':>15}{'SNR':>14}{'còn lại bao nhiêu tín hiệu':>30}")
for t in (0, 50, 200, 500, 800, 999):
    print(f"{t:6d}{alpha_ngang[t]:15.6f}{snr[t]:14.4f}{np.sqrt(alpha_ngang[t]) * 100:29.2f}%")
print(f"Tai t = {T-1}, alpha_ngang = {alpha_ngang[-1]:.2e} nen x_T gan nhu la nhieu Gauss")
print("thuan tuy — do dung la dieu ta can de bat dau lay mau tu N(0, I).")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
ax[0].plot(alpha_ngang, color=C_MAIN, lw=1.8)
ax[0].set_xlabel("bước $t$"); ax[0].set_ylabel("$\\bar\\alpha_t$")
ax[0].set_title("Tín hiệu còn lại theo bước", fontsize=9)
ax[1].semilogy(snr, color=C_BAD, lw=1.8)
ax[1].axhline(1, ls=":", color=C_DIM)
ax[1].text(20, 1.4, "SNR = 1", fontsize=7.5, color=C_DIM)
ax[1].set_xlabel("bước $t$"); ax[1].set_ylabel("SNR $=\\bar\\alpha_t/(1-\\bar\\alpha_t)$")
ax[1].set_title("Tỉ số tín hiệu trên nhiễu (log)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "bd10_diffusion.png"); plt.close(fig)


# =====================================================================
# (I) Hình 12 — Khám phá: epsilon không phải cơ chế duy nhất
# =====================================================================
head("(I) Q-learning: kham pha den tu epsilon VA tu cach khoi tao Q")

H_, W_ = 7, 7
DICH = (0, 6)
BAY = [(1, 1), (2, 4), (4, 2), (5, 5)]
HANH_DONG = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def buoc(s, a, phat):
    r_, c_ = s
    dr, dc = HANH_DONG[a]
    ns = (min(max(r_ + dr, 0), H_ - 1), min(max(c_ + dc, 0), W_ - 1))
    if ns == DICH:
        return ns, 10.0, True
    if ns in BAY:
        return ns, -10.0, True
    return ns, phat, False

def q_learning(eps, phat, q0=0.0, tap=6000, alpha=0.2, gamma=0.97, seed=0):
    r = np.random.default_rng(seed)
    Q = np.full((H_, W_, 4), q0)
    for _ in range(tap):
        s = (H_ - 1, 0)
        for _t in range(80):
            a = r.integers(4) if r.random() < eps else int(np.argmax(Q[s]))
            ns, rw, xong = buoc(s, a, phat)
            Q[s][a] += alpha * ((rw if xong else rw + gamma * Q[ns].max()) - Q[s][a])
            s = ns
            if xong:
                break
    # Danh gia CHINH SACH THAM LAM da hoc duoc, khong con kham pha nua.
    tong, dat = 0.0, 0
    for _ in range(200):
        s = (H_ - 1, 0)
        G = 0.0
        for _t in range(80):
            s, rw, xong = buoc(s, int(np.argmax(Q[s])), phat)
            G += rw
            if xong:
                dat += (s == DICH)
                break
        tong += G
    return Q, tong / 200, dat / 200 * 100

print(f"Luoi {H_}x{W_}: xuat phat goc duoi trai, dich {DICH} (+10), {len(BAY)} bay (-10).")
print("Ba cau hinh chi khac nhau o PHAT MOI BUOC va GIA TRI KHOI TAO cua Q.")

cau_hinh = [
    ("phạt −0,1 · $Q_0 = 0$", -0.1, 0.0),
    ("phạt 0 · $Q_0 = 0$", 0.0, 0.0),
    ("phạt −0,1 · $Q_0 = -20$", -0.1, -20.0),
]
Irows = {}
for ten, phat, q0 in cau_hinh:
    print(f"\n{ten}")
    print(f"{'epsilon':>10}{'thưởng của chính sách đã học':>32}{'% tới đích':>14}")
    hang = []
    for eps in [0.0, 0.05, 0.1, 0.3, 1.0]:
        Q, g, d = q_learning(eps, phat, q0)
        hang.append((eps, g, d, Q))
        print(f"{eps:10.2f}{g:32.3f}{d:13.1f}%")
    Irows[ten] = hang

print("\nBa bang nay noi mot dieu ma cau 'phai co epsilon de kham pha' bo mat:")
print("  Cau hinh 1 — MOI epsilon deu thanh cong, ke ca eps = 0. Vi khoi tao Q = 0 trong")
print("    khi moi phan thuong deu am chinh la KHOI TAO LAC QUAN: hanh dong chua thu bao")
print("    gio cung trong hap dan hon hanh dong da thu, nen tham lam tu no da di kham pha.")
print("  Cau hinh 2 — bo phat buoc di thi mat luon tinh lac quan ay, va eps = 0 THAT BAI")
print("    hoan toan (0% toi dich): moi Q deu bang 0 nen argmax luon chon cung mot huong.")
print("  Cau hinh 3 — khoi tao BI QUAN (Q0 = -20) con doc hon: tham lam bam chat lay hanh")
print("    dong dau tien tinh co thu, va eps toi 0,3 van khong du de thoat trong ngan sach")
print("    nay. Phai eps = 1 moi tim ra dich.")
print("\nKet luan: kham pha khong chi den tu epsilon. Cach khoi tao ham gia tri cung la mot")
print("co che kham pha, va doi khi la co che manh hon.")

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.1))
Q_tot = Irows[cau_hinh[0][0]][2][3]
V = Q_tot.max(2)
im = ax[0].imshow(V, cmap="RdYlGn")
for i_ in range(H_):
    for j_ in range(W_):
        if (i_, j_) == DICH:
            ax[0].text(j_, i_, "Đ", ha="center", va="center", fontsize=8, weight="bold")
        elif (i_, j_) in BAY:
            ax[0].text(j_, i_, "✕", ha="center", va="center", fontsize=9, weight="bold")
        else:
            ax[0].text(j_, i_, ["↑", "↓", "←", "→"][int(np.argmax(Q_tot[i_, j_]))],
                       ha="center", va="center", fontsize=11)
ax[0].set_title("Chính sách đã học (Đ = đích, ✕ = bẫy)", fontsize=9)
ax[0].set_xticks([]); ax[0].set_yticks([])
fig.colorbar(im, ax=ax[0], fraction=.046)

x_ = np.arange(5)
w_ = 0.26
for k_, (ten, _, _) in enumerate(cau_hinh):
    ax[1].bar(x_ + (k_ - 1) * w_, [h[2] for h in Irows[ten]], width=w_, label=ten)
ax[1].set_xticks(x_)
ax[1].set_xticklabels(["0", "0,05", "0,1", "0,3", "1,0"])
ax[1].set_xlabel("$\\varepsilon$"); ax[1].set_ylabel("% số lần tới được đích")
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("Khởi tạo $Q$ quyết định nhiều như $\\varepsilon$", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "bd12_rl.png"); plt.close(fig)


# =====================================================================
# (J) Policy gradient: vì sao phải có đường nền
# =====================================================================
head("(J) Gradient chinh sach: duong nen cat phuong sai bao nhieu")

K_PG = 6
r = np.random.default_rng(41)
thuong_that = np.array([1.0, 1.2, 0.9, 1.1, 3.0, 1.05])
logits = np.zeros(K_PG)
pi = softmax(logits)

def uoc_luong_gradient(n_mau, dung_nen, seed):
    """REINFORCE: g = E[ (R - b) * grad log pi(a) ]."""
    rr = np.random.default_rng(seed)
    a = rr.choice(K_PG, size=n_mau, p=pi)
    R = thuong_that[a] + 0.25 * rr.normal(0, 1, n_mau)
    b = R.mean() if dung_nen else 0.0
    G = np.zeros((n_mau, K_PG))
    G[np.arange(n_mau), a] = 1.0
    G = G - pi[None, :]                       # grad log pi(a) theo logits
    return ((R - b)[:, None] * G).mean(0)

print(f"Bai toan mot buoc voi {K_PG} hanh dong, thuong that = {thuong_that}")
print(f"Chinh sach hien tai deu: pi = {np.round(pi, 4)}")
print(f"\n{'số mẫu':>9}{'độ lệch chuẩn — không nền':>28}{'có nền':>14}{'giảm được':>13}")
Jrows = []
for n_mau in [16, 64, 256, 1024]:
    g0 = np.array([uoc_luong_gradient(n_mau, False, s) for s in range(400)])
    g1 = np.array([uoc_luong_gradient(n_mau, True, s) for s in range(400)])
    s0 = g0.std(0).mean()
    s1 = g1.std(0).mean()
    Jrows.append((n_mau, s0, s1))
    print(f"{n_mau:9d}{s0:28.5f}{s1:14.5f}{s0 / s1:12.2f}x")
tb0 = np.array([uoc_luong_gradient(4096, False, s) for s in range(200)]).mean(0)
tb1 = np.array([uoc_luong_gradient(4096, True, s) for s in range(200)]).mean(0)
print(f"\nDuong nen KHONG lam lech uoc luong — no chi cat phuong sai:")
print(f"  gradient TB khong nen : {np.round(tb0, 5)}")
print(f"  gradient TB co nen    : {np.round(tb1, 5)}")
print(f"  lech lon nhat: {np.abs(tb0 - tb1).max():.2e}")
print("Ly do toan hoc: E[b * grad log pi] = b * grad E[1] = 0 voi moi hang so b, nen tru")
print("di mot duong nen bat ky khong doi ky vong. Chon b = thuong trung binh thi phuong")
print("sai nho nhat trong so cac hang so.")


# =====================================================================
# (K) Hình 13 — RLHF: nghiệm của bài toán có ràng buộc KL, dạng đóng
# =====================================================================
head("(K) RLHF: nghiem toi uu cua muc tieu co rang buoc KL")

from scipy.optimize import minimize

N_TL = 8                                     # 8 cau tra loi kha di
r = np.random.default_rng(77)
pi_ref = softmax(r.normal(0, 1, N_TL))
thuong = r.normal(0, 1.4, N_TL)

print("Muc tieu cua RLHF:   max_pi  E_pi[r(y)]  -  beta * KL(pi || pi_ref)")
print("Cho dao ham bang 0 tren don hinh xac suat se ra DANG DONG:")
print("    pi*(y) = pi_ref(y) * exp(r(y)/beta) / Z")
print(f"\n{'beta':>8}{'lệch so với tối ưu số':>26}{'KL(pi*||pi_ref)':>19}{'E[r] đạt được':>17}")
for beta in [0.05, 0.2, 1.0, 5.0]:
    # Dang dong
    log_pi = np.log(pi_ref) + thuong / beta
    pi_dd = softmax(log_pi)

    # Toi uu bang so: tham so hoa bang logit roi chay BFGS, khong dung cong thuc tren.
    def am_muc_tieu(z):
        p = softmax(z)
        return -(p @ thuong) + beta * np.sum(p * (np.log(p + 1e-300) - np.log(pi_ref)))
    res = minimize(am_muc_tieu, np.zeros(N_TL), method="BFGS",
                   options={"gtol": 1e-13, "maxiter": 50000})
    pi_so = softmax(res.x)

    kl = float(np.sum(pi_dd * (np.log(pi_dd) - np.log(pi_ref))))
    print(f"{beta:8.2f}{np.abs(pi_dd - pi_so).max():26.3e}{kl:19.4f}{pi_dd @ thuong:17.4f}")
print("Hai duong hoan toan khac nhau — mot ben thay vao cong thuc, mot ben chay BFGS tren")
print("don hinh — cho cung mot chinh sach toi sai so cua bo toi uu.")
print("\nDoc y nghia cua beta:")
print("  beta -> 0   : bo qua pi_ref, don het xac suat vao cau tra loi co thuong cao nhat.")
print("  beta -> vo cuc: giu nguyen pi_ref, khong hoc duoc gi tu thuong.")
print("Day la ly do RLHF luon co so hang KL: khong co no, mo hinh se pha nat phan phoi")
print("ngon ngu da hoc duoc de chay theo diem thuong — tuc 'lach diem thuong'.")

fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.1))
x_ = np.arange(N_TL)
ax[0].bar(x_ - .2, pi_ref, width=.4, color=C_DIM, label="$\\pi_{\\mathrm{ref}}$")
for i_, (beta, c_) in enumerate([(5.0, C_MAIN), (0.2, C_BAD)]):
    p_ = softmax(np.log(pi_ref) + thuong / beta)
    ax[0].bar(x_ + .2 + i_ * .0, p_, width=.4, color=c_, alpha=.75,
              label=f"$\\pi^*$, $\\beta={beta}$")
ax[0].set_xlabel("câu trả lời"); ax[0].set_ylabel("xác suất")
ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title("$\\beta$ nhỏ kéo chính sách xa $\\pi_{\\mathrm{ref}}$", fontsize=9)
bs = np.logspace(-1.4, 1.2, 40)
kls = [float(np.sum(softmax(np.log(pi_ref) + thuong / b) *
       (np.log(softmax(np.log(pi_ref) + thuong / b)) - np.log(pi_ref)))) for b in bs]
ers = [softmax(np.log(pi_ref) + thuong / b) @ thuong for b in bs]
ax[1].plot(kls, ers, "-o", ms=3, color=C_MAIN)
ax[1].set_xlabel("$\\mathrm{KL}(\\pi^*\\,\\|\\,\\pi_{\\mathrm{ref}})$")
ax[1].set_ylabel("$\\mathbb{E}_{\\pi^*}[r]$")
ax[1].set_title("Mặt đánh đổi: thưởng mua bằng độ lệch khỏi $\\pi_{\\mathrm{ref}}$", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "bd13_rlhf.png"); plt.close(fig)


# =====================================================================
# (L) DPO: bỏ hẳn bước mô hình thưởng mà vẫn ra cùng một chính sách
# =====================================================================
head("(L) DPO cho cung chinh sach voi RLHF hai buoc — kiem chung bang so")

r = np.random.default_rng(101)
thuong_that_dpo = r.normal(0, 1.2, N_TL)
BETA = 0.5

# Sinh du lieu SO SANH tu mo hinh Bradley-Terry voi thuong that.
n_cap = 60000
ia = r.integers(0, N_TL, n_cap)
ib = r.integers(0, N_TL, n_cap)
giu = ia != ib
ia, ib = ia[giu], ib[giu]
p_a_thang = 1 / (1 + np.exp(-(thuong_that_dpo[ia] - thuong_that_dpo[ib])))
a_thang = r.random(len(ia)) < p_a_thang
y_w = np.where(a_thang, ia, ib)
y_l = np.where(a_thang, ib, ia)
print(f"Du lieu so sanh: {len(y_w):,} cap (y_thang, y_thua), sinh tu Bradley-Terry.")

# ---- Duong 1: RLHF hai buoc -------------------------------------
def am_hop_ly_bt(rr_):
    d = rr_[y_w] - rr_[y_l]
    return -np.mean(np.log(1 / (1 + np.exp(-d)) + 1e-300))
res_bt = minimize(am_hop_ly_bt, np.zeros(N_TL), method="BFGS",
                  options={"gtol": 1e-12, "maxiter": 80000})
r_hat = res_bt.x
pi_rlhf = softmax(np.log(pi_ref) + r_hat / BETA)

# ---- Duong 2: DPO mot buoc --------------------------------------
def mat_mat_dpo(z):
    log_pi = np.log(softmax(z) + 1e-300)
    d = BETA * ((log_pi[y_w] - np.log(pi_ref[y_w])) - (log_pi[y_l] - np.log(pi_ref[y_l])))
    return -np.mean(np.log(1 / (1 + np.exp(-d)) + 1e-300))
res_dpo = minimize(mat_mat_dpo, np.zeros(N_TL), method="BFGS",
                   options={"gtol": 1e-12, "maxiter": 80000})
pi_dpo = softmax(res_dpo.x)

print(f"\n{'câu trả lời':>13}{'pi_ref':>10}{'RLHF 2 bước':>14}{'DPO 1 bước':>13}{'lệch':>12}")
for i_ in range(N_TL):
    print(f"{i_:13d}{pi_ref[i_]:10.4f}{pi_rlhf[i_]:14.4f}{pi_dpo[i_]:13.4f}"
          f"{abs(pi_rlhf[i_] - pi_dpo[i_]):12.2e}")
print(f"\nLech lon nhat giua hai chinh sach: {np.abs(pi_rlhf - pi_dpo).max():.3e}")
print(f"Tuong quan thuong hoc duoc voi thuong that: "
      f"{np.corrcoef(r_hat - r_hat.mean(), thuong_that_dpo - thuong_that_dpo.mean())[0, 1]:.6f}")

print("\nDuong 1 chay HAI buoc: khop mo hinh thuong Bradley-Terry, roi giai bai toan RL")
print("co rang buoc KL. Duong 2 chay MOT buoc, khong he co mo hinh thuong nao.")
print("Chung ra cung mot chinh sach, va do la toan bo y cua DPO: neu nghiem toi uu von")
print("da co dang pi* = pi_ref exp(r/beta)/Z thi dao nguoc lai duoc")
print("    r(y) = beta * log[ pi(y) / pi_ref(y) ] + beta * log Z,")
print("nen co the thay thang bieu thuc ay vao hop ly Bradley-Terry. Hang log Z triet tieu")
print("khi lay HIEU giua hai cau tra loi, nen no khong can tinh — do la chi tiet lam cho")
print("ca cach lam nay chay duoc.")

print("\n" + "=" * 74)
print("Da luu 9 hinh vao thu muc figs/: bd02, bd03, bd04, bd06, bd08, bd09, bd10, bd12, bd13")
print("=" * 74)
