"""Thí nghiệm cho giáo trình "Ứng dụng LLM".

Mọi thí nghiệm chạy trên CPU, không gọi API trả phí và không tải mô hình lớn:

  (A) Số token của cùng một nội dung tiếng Việt và tiếng Anh với ba tokenizer thật
      (GPT-2, Qwen2.5, XLM-RoBERTa) và với tokenizer BPE tự huấn luyện.
  (B) Bộ nhớ KV cache theo độ dài ngữ cảnh, và chi phí một yêu cầu RAG.
  (C) Truy xuất trên chính văn bản tiếng Việt của bộ giáo trình: BM25, TF-IDF,
      LSA và kết hợp RRF; truy vấn là câu hỏi trắc nghiệm, nhãn đúng là chương
      và mục mà lời giải trỏ tới.
  (D) Ảnh hưởng của kích thước đoạn (chunk) tới truy xuất.
  (E) Tìm kiếm gần đúng: IVF, PQ và tìm kiếm trên đồ thị láng giềng.
  (F) Độ tin cậy của agent nhiều bước.
  (G) Cần bao nhiêu mẫu để đánh giá: khoảng tin cậy và so sánh cặp.

    python code/ungdung/experiments.py

Cần: numpy, scipy, scikit-learn, matplotlib, transformers, tokenizers, nltk.
Lần chạy đầu tải ba tokenizer (vài MB) từ Hugging Face và hai kho văn bản của NLTK.
Trên Windows, đặt PYTHONIOENCODING=utf-8 khi ghi kết quả ra tệp.
"""

import os
import re
import math
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figs") + os.sep
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
C_MAIN, C_WARN, C_BAD, C_DIM, C_BLUE = "#1f6f68", "#b8860b", "#b0413e", "#8a857c", "#2f5f9f"


def head(t):
    print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)


def doc(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def nfc(s):
    return unicodedata.normalize("NFC", s)


# ------------------------------------------------------------------ dữ liệu dùng chung
# Văn bản các giáo trình được chụp một lần vào kho_van_ban.json, nên mọi số liệu dựa
# trên nó, kể cả số token ở (A), không đổi khi nội dung các giáo trình được sửa về sau.
tu = lambda s: len(s.split())

SACH = [("nentang", "nentang-trac-nghiem.md"), ("models", "models-trac-nghiem.md"),
        ("bieudien", "bieudien-trac-nghiem.md"), ("quantization", "trac-nghiem.md"),
        ("mlops", "mlops-trac-nghiem.md")]
BO_QUA = re.compile(r"^(Bài tập|Ôn phỏng vấn|Câu hỏi phỏng vấn|Tài liệu tham khảo|Kiến thức nền|Ký hiệu)", re.I)


def cat_muc(sach):
    """Cắt một giáo trình thành các mục (###). Bỏ chương 0, bài tập, phỏng vấn, tài liệu."""
    md = nfc(doc(f"content/{sach}.md" if sach != "quantization" else "content/quantization.md"))
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    muc = []
    for ch in re.split(r"\n(?=## )", md):
        m = re.match(r"## (\d+)\. (.+)", ch)
        if not m or m.group(1) == "0" or BO_QUA.match(m.group(2)):
            continue
        so = m.group(1)
        for sec in re.split(r"\n(?=### )", ch):
            ms = re.match(r"### (\d+)\.(\d+)\. (.+)", sec)
            if ms and ms.group(1) == so:
                muc.append({"sach": sach, "chuong": so, "muc": f"{so}.{ms.group(2)}", "text": sec})
    return muc


KHO = os.path.join(ROOT, "code", "ungdung", "kho_van_ban.json")


def cat_cau_hoi(sach, tep):
    q = []
    chuong = None
    for khoi in re.split(r"\n(?=## |### )", nfc(doc("content/" + tep))):
        mc = re.match(r"## Chương (\d+)", khoi)
        if mc:
            chuong = mc.group(1); continue
        mq = re.match(r"### (.+)", khoi)
        if mq and chuong:
            giai = " ".join(l[1:] for l in khoi.split("\n") if l.startswith(">"))
            refs = sorted(set(f"{a}.{b}" for a, b in re.findall(r"Mục (\d+)\.(\d+)", giai) if a == chuong))
            q.append({"sach": sach, "chuong": chuong, "muc": refs, "text": mq.group(1)})
    return q


# Kho văn bản được "chụp" lại một lần vào kho_van_ban.json. Các lần chạy sau đọc lại đúng
# bản chụp ấy, nên số liệu không đổi dù nội dung các giáo trình được sửa về sau.
import json, sys
if os.path.exists(KHO) and "--chup-lai" not in sys.argv:
    _k = json.load(open(KHO, encoding="utf-8"))
    MUC, CAU = _k["muc"], _k["cau"]
    print(f"Doc kho van ban da chup: {os.path.relpath(KHO, ROOT)}")
else:
    MUC = [m for s_, _ in SACH for m in cat_muc(s_)]
    CAU = [c for s_, t in SACH for c in cat_cau_hoi(s_, t)]
    with open(KHO, "w", encoding="utf-8") as f:
        json.dump({"muc": MUC, "cau": CAU}, f, ensure_ascii=False)
    print(f"Da chup kho van ban vao {os.path.relpath(KHO, ROOT)}")


# =====================================================================
# (A) Token: cùng một nội dung, tiếng Việt tốn bao nhiêu token so với tiếng Anh
# =====================================================================
head("(A) So token cua cung mot noi dung tieng Viet va tieng Anh")

import nltk
nltk.download("udhr", quiet=True)
nltk.download("gutenberg", quiet=True)
from nltk.corpus import udhr, gutenberg

# Bản trong NLTK bị cắt ở khoảng 10 000 byte, nên hai bản dừng ở hai chỗ khác nhau.
# Lấy đúng cùng một đoạn nội dung: từ lời nói đầu tới hết Điều 20.
_en = nfc(udhr.raw("English-Latin1")); _vi = nfc(udhr.raw("Vietnamese-UTF8"))
en_udhr = _en[_en.index("Preamble"):_en.index("Article 21")].strip()
vi_udhr = _vi[_vi.index("Lời nói đầu"):_vi.index("Điều 21")].strip()
print(f"Van ban song song: Tuyen ngon Nhan quyen (NLTK udhr), tu Loi noi dau toi het Dieu 20")
print(f"  tieng Anh : {tu(en_udhr):5d} tu, {len(en_udhr):6d} ky tu, {len(en_udhr.encode('utf-8')):6d} byte UTF-8")
print(f"  tieng Viet: {tu(vi_udhr):5d} tu, {len(vi_udhr):6d} ky tu, {len(vi_udhr.encode('utf-8')):6d} byte UTF-8")

from transformers import AutoTokenizer
TOK = [("GPT-2", "gpt2"), ("Qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct"), ("XLM-RoBERTa", "FacebookAI/xlm-roberta-base")]
tok_rows = []
print(f"\n{'tokenizer':<13}{'kích thước từ vựng':>19}{'token (Anh)':>13}{'token (Việt)':>14}"
      f"{'Việt / Anh':>12}{'token/từ (Việt)':>17}")
for ten, repo in TOK:
    tk = AutoTokenizer.from_pretrained(repo)
    n_en = len(tk(en_udhr, add_special_tokens=False)["input_ids"])
    n_vi = len(tk(vi_udhr, add_special_tokens=False)["input_ids"])
    tok_rows.append((ten, len(tk), n_en, n_vi))
    print(f"{ten:<13}{len(tk):19,d}{n_en:13,d}{n_vi:14,d}{n_vi / n_en:12.2f}{n_vi / tu(vi_udhr):17.2f}")

# Một câu cụ thể để thấy cách cắt
cau = "Mô hình ngôn ngữ lớn dự đoán token tiếp theo."
print(f"\nCau vi du: '{cau}' ({tu(cau)} tu)")
for ten, repo in TOK:
    tk = AutoTokenizer.from_pretrained(repo)
    ids = tk(cau, add_special_tokens=False)["input_ids"]
    manh = [tk.decode([i]) for i in ids]
    print(f"  {ten:<12} {len(ids):3d} token: " + " | ".join(m.replace(' ', '·') for m in manh))

# Tokenizer BPE tự huấn luyện: dữ liệu huấn luyện quyết định độ nén
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

vi_corpus = "\n".join(m["text"] for m in MUC)   # 332 mục của năm giáo trình, từ bản chụp
en_corpus = "\n".join(gutenberg.raw(f) for f in ["austen-emma.txt", "austen-persuasion.txt",
                                                   "austen-sense.txt", "chesterton-thursday.txt"])


def bpe(texts, vocab):
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    t.train_from_iterator(texts, trainer=tr)
    return t


print(f"\nTokenizer BPE tu huan luyen (8 000 token), kho: Viet {tu(vi_corpus):,} tu / Anh {tu(en_corpus):,} tu")
print(f"{'dữ liệu huấn luyện':<24}{'token (Anh)':>13}{'token (Việt)':>14}{'Việt / Anh':>12}")
bpe_rows = []
for ten, texts in [("chỉ tiếng Anh", [en_corpus]), ("chỉ tiếng Việt", [vi_corpus]),
                   ("trộn Anh + Việt", [en_corpus, vi_corpus])]:
    t = bpe(texts, 8000)
    n_en = len(t.encode(en_udhr).ids); n_vi = len(t.encode(vi_udhr).ids)
    bpe_rows.append((ten, n_en, n_vi))
    print(f"{ten:<24}{n_en:13,d}{n_vi:14,d}{n_vi / n_en:12.2f}")

# Cả bộ giáo trình tiếng Việt tốn bao nhiêu token
print(f"\nCa kho van ban tieng Viet ({len(MUC)} muc cua 5 giao trinh, {tu(vi_corpus):,} tu):")
for ten, repo in TOK:
    tk = AutoTokenizer.from_pretrained(repo)
    n = sum(len(tk(m["text"], add_special_tokens=False)["input_ids"]) for m in MUC)
    print(f"  {ten:<12} {n:9,d} token  ({n / tu(vi_corpus):.2f} token/tu)")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
x = np.arange(len(tok_rows)); w = .38
ax[0].bar(x - w / 2, [r[2] for r in tok_rows], w, color=C_BLUE, label="tiếng Anh")
ax[0].bar(x + w / 2, [r[3] for r in tok_rows], w, color=C_BAD, label="tiếng Việt")
ax[0].set_xticks(x); ax[0].set_xticklabels([r[0] for r in tok_rows])
ax[0].set_ylabel("số token"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("Cùng một văn bản, ba tokenizer", fontsize=9)
x2 = np.arange(len(bpe_rows))
ax[1].bar(x2 - w / 2, [r[1] for r in bpe_rows], w, color=C_BLUE, label="tiếng Anh")
ax[1].bar(x2 + w / 2, [r[2] for r in bpe_rows], w, color=C_BAD, label="tiếng Việt")
ax[1].set_xticks(x2); ax[1].set_xticklabels([r[0] for r in bpe_rows], fontsize=8)
ax[1].set_title("BPE 8 000 token theo dữ liệu huấn luyện", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud02_token.png"); plt.close(fig)


# =====================================================================
# (B) KV cache và chi phí một yêu cầu
# =====================================================================
head("(B) Bo nho KV cache va chi phi mot yeu cau")

# Cấu hình công bố của Llama 3 8B: 32 lớp, 8 đầu key/value (GQA), mỗi đầu 128 chiều.
L, H_kv, d_h, byte = 32, 8, 128, 2
kv_moi_token = 2 * L * H_kv * d_h * byte
print(f"Llama 3 8B: 2 (K va V) x {L} lop x {H_kv} dau KV x {d_h} chieu x {byte} byte "
      f"= {kv_moi_token:,d} byte = {kv_moi_token / 1024:.0f} KiB moi token")
for n in [2048, 8192, 32768, 131072]:
    print(f"  ngu canh {n:7,d} token: KV cache {kv_moi_token * n / 2**30:6.2f} GiB cho mot chuoi")
kv_mha = 2 * L * 32 * d_h * byte
print(f"Neu khong dung GQA (32 dau KV): {kv_mha / 1024:.0f} KiB moi token, gap {kv_mha / kv_moi_token:.0f} lan")

# Một yêu cầu RAG điển hình, tính bằng token (đơn giá là tham số, không phải giá thật)
phan = [("chỉ dẫn hệ thống", 600), ("5 đoạn truy xuất × 350 token", 1750),
        ("lịch sử hội thoại", 800), ("câu hỏi", 60)]
n_vao = sum(v for _, v in phan); n_ra = 350
print(f"\nYeu cau RAG: dau vao {n_vao:,d} token, dau ra {n_ra} token")
for ten, v in phan:
    print(f"  {ten:<32}{v:6d} token ({v / n_vao * 100:4.1f}%)")
gia_vao, gia_ra, ti_le_cache = 1.0, 4.0, 0.1       # đơn vị tiền / triệu token, và giá phần đọc từ cache
chi_phi = (n_vao * gia_vao + n_ra * gia_ra) / 1e6
cache_duoc = 600                                   # phần chỉ dẫn hệ thống dùng lại giữa các yêu cầu
chi_phi_cache = ((n_vao - cache_duoc) * gia_vao + cache_duoc * gia_vao * ti_le_cache + n_ra * gia_ra) / 1e6
print(f"Voi don gia gia dinh 1 (vao) va 4 (ra) moi trieu token: {chi_phi * 1e3:.3f} phan nghin don vi / yeu cau")
print(f"Neu 600 token chi dan duoc doc tu cache voi gia 10%: {chi_phi_cache * 1e3:.3f} (giam "
      f"{(1 - chi_phi_cache / chi_phi) * 100:.1f}%)")
print(f"Dau ra chiem {n_ra * gia_ra / (n_vao * gia_vao + n_ra * gia_ra) * 100:.1f}% chi phi du chi bang "
      f"{n_ra / (n_vao + n_ra) * 100:.1f}% so token")


# =====================================================================
# (C) Truy xuất trên văn bản tiếng Việt của bộ giáo trình
# =====================================================================
head("(C) Truy xuat tren van ban tieng Viet: BM25, TF-IDF, LSA va ket hop RRF")


with open(os.path.join(ROOT, "code", "ungdung", "cau_hoi_dien_dat_lai.json"), encoding="utf-8") as f:
    VIET_LAI = [{"sach": x["sach"], "chuong": x["muc"][0].split(".")[0], "muc": x["muc"], "text": x["q"]}
                for x in json.load(f)]
co_muc = [c for c in CAU if c["muc"]]
for c in VIET_LAI:
    for mm in c["muc"]:
        assert any(m["sach"] == c["sach"] and m["muc"] == mm for m in MUC), ("muc khong co", c)
print(f"Kho van ban: {len(MUC)} muc tu {len(SACH)} giao trinh, {sum(tu(m['text']) for m in MUC):,} tu")
print(f"Bo truy van 1: {len(CAU)} cau hoi trac nghiem (nhan: chuong; {len(co_muc)} cau co nhan muc)")
print(f"Bo truy van 2: {len(VIET_LAI)} cau hoi dien dat lai theo cach nguoi hoc hoi (nhan: muc)")

TU_RE = re.compile(r"\w+", re.U)


def am_tiet(s):
    return TU_RE.findall(s.lower())


def token_bm25(s, bigram=True):
    a = am_tiet(s)
    return a + ([a[i] + "_" + a[i + 1] for i in range(len(a) - 1)] if bigram else [])


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75, bigram=True):
        self.bigram = bigram
        self.k1, self.b = k1, b
        self.tf = [Counter(token_bm25(d, bigram)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], float)
        self.avg = self.len.mean()
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}

    def score(self, q):
        qs = token_bm25(q, self.bigram)
        s = np.zeros(len(self.tf))
        for i, t in enumerate(self.tf):
            den_norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            v = 0.0
            for w in qs:
                f = t.get(w)
                if f:
                    v += self.idf[w] * f * (self.k1 + 1) / (f + den_norm)
            s[i] = v
        return s


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

van = [m["text"] for m in MUC]
tfidf = TfidfVectorizer(tokenizer=lambda s: token_bm25(s, True), lowercase=False,
                        token_pattern=None, sublinear_tf=True, min_df=1)
Xt = tfidf.fit_transform(van)
svd = TruncatedSVD(n_components=256, random_state=0)
Xl = normalize(svd.fit_transform(Xt))
print(f"Tu vung TF-IDF (am tiet + cap am tiet): {Xt.shape[1]:,}; LSA giu 256 chieu, "
      f"{svd.explained_variance_ratio_.sum() * 100:.1f}% phuong sai")

bm25 = BM25(van, bigram=True)
bm25_uni = BM25(van, bigram=False)


def xep_hang(diem):
    return np.argsort(-diem, kind="stable")


def rrf(*hangs, k=60):
    s = np.zeros(len(MUC))
    for h in hangs:
        s[h] += 1.0 / (k + np.arange(1, len(h) + 1))
    return xep_hang(s)


def danh_gia(ten, ham):
    # Bộ 1: tìm đúng chương
    r_ch = {1: 0, 5: 0}; mrr = 0.0
    for c in CAU:
        h = ham(c["text"])
        dung = [i for i, j in enumerate(h) if MUC[j]["sach"] == c["sach"] and MUC[j]["chuong"] == c["chuong"]]
        vt = dung[0] + 1 if dung else 10 ** 9
        for k in r_ch:
            r_ch[k] += vt <= k
        mrr += 1 / vt
    # Bộ 2: tìm đúng mục
    r_m = {1: 0, 5: 0}; mrr2 = 0.0
    for c in VIET_LAI:
        h = ham(c["text"])
        dung = [i for i, j in enumerate(h) if MUC[j]["sach"] == c["sach"] and MUC[j]["muc"] in c["muc"]]
        vt = dung[0] + 1 if dung else 10 ** 9
        for k in r_m:
            r_m[k] += vt <= k
        mrr2 += 1 / vt
    n1, n2 = len(CAU), len(VIET_LAI)
    kq = (ten, r_ch[1] / n1, r_ch[5] / n1, mrr / n1, r_m[1] / n2, r_m[5] / n2, mrr2 / n2)
    print(f"{ten:<24}{kq[1]:8.3f}{kq[2]:8.3f}{kq[3]:8.3f}  |{kq[4]:8.3f}{kq[5]:8.3f}{kq[6]:8.3f}")
    return kq


print(f"\n{'':<24}{'bộ 1: trắc nghiệm → chương':^24}  |{'bộ 2: diễn đạt lại → mục':^24}")
print(f"{'phương pháp':<24}{'R@1':>8}{'R@5':>8}{'MRR':>8}  |{'R@1':>8}{'R@5':>8}{'MRR':>8}")
q_tfidf = lambda q: xep_hang((Xt @ tfidf.transform([q]).T).toarray().ravel())
q_lsa = lambda q: xep_hang(Xl @ normalize(svd.transform(tfidf.transform([q]))).ravel())
kq_truy = [
    danh_gia("BM25, âm tiết", lambda q: xep_hang(bm25_uni.score(q))),
    danh_gia("BM25, âm tiết + cặp", lambda q: xep_hang(bm25.score(q))),
    danh_gia("TF-IDF cosine", q_tfidf),
    danh_gia("LSA 256 chiều", q_lsa),
    danh_gia("RRF: BM25 + LSA", lambda q: rrf(xep_hang(bm25.score(q)), q_lsa(q))),
]

fig, ax = plt.subplots(figsize=(6.6, 3.0))
x = np.arange(len(kq_truy)); w = .26
for k, (cot, ten, col) in enumerate([(2, "bộ 1: R@5 tìm đúng chương", C_BLUE),
                                      (4, "bộ 2: R@1 tìm đúng mục", C_DIM),
                                      (5, "bộ 2: R@5 tìm đúng mục", C_BAD)]):
    ax.bar(x + (k - 1) * w, [r[cot] for r in kq_truy], w, color=col, label=ten)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in kq_truy], fontsize=7.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("tỉ lệ tìm đúng"); ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
ax.set_title("Truy xuất trên văn bản tiếng Việt của bộ giáo trình", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud05_truyxuat.png"); plt.close(fig)


# =====================================================================
# (D) Kích thước đoạn
# =====================================================================
head("(D) Kich thuoc doan (chunk) va chat luong truy xuat")


def chia_doan(text, w, chong=0.2):
    tu_ = text.split()
    buoc = max(1, int(w * (1 - chong)))
    return [" ".join(tu_[i:i + w]) for i in range(0, max(1, len(tu_) - int(w * chong)), buoc)] or [text]


print(f"Truy van: {len(co_muc)} cau trac nghiem co nhan muc + {len(VIET_LAI)} cau dien dat lai; do: tim dung muc trong 5 ket qua")
print(f"{'kích thước đoạn':<18}{'số đoạn':>9}{'mục R@5 BM25':>15}{'mục R@5 LSA':>14}{'mục R@5 RRF':>14}")
kq_doan = []
for w in [50, 100, 200, 400, None]:
    doan, nhan = [], []
    for j, m in enumerate(MUC):
        ds = [m["text"]] if w is None else chia_doan(m["text"], w)
        doan += ds; nhan += [j] * len(ds)
    nhan = np.array(nhan)
    b = BM25(doan, bigram=True)
    tv = TfidfVectorizer(tokenizer=lambda s: token_bm25(s, True), lowercase=False, token_pattern=None,
                         sublinear_tf=True)
    Xd = tv.fit_transform(doan)
    sv = TruncatedSVD(n_components=min(256, Xd.shape[0] - 1), random_state=0)
    Ld = normalize(sv.fit_transform(Xd))

    def muc_dau(h, k=5):
        seen = []
        for i in h:
            if nhan[i] not in seen:
                seen.append(nhan[i])
            if len(seen) == k:
                break
        return seen

    diem = {"bm25": 0, "lsa": 0, "rrf": 0}
    for c in co_muc + VIET_LAI:
        hb = xep_hang(b.score(c["text"]))
        hl = xep_hang(Ld @ normalize(sv.transform(tv.transform([c["text"]]))).ravel())
        s = np.zeros(len(doan))
        s[hb] += 1 / (60 + np.arange(1, len(hb) + 1)); s[hl] += 1 / (60 + np.arange(1, len(hl) + 1))
        for ten, h in [("bm25", hb), ("lsa", hl), ("rrf", xep_hang(s))]:
            diem[ten] += any(MUC[j]["sach"] == c["sach"] and MUC[j]["muc"] in c["muc"] for j in muc_dau(h))
    ten_w = "cả mục" if w is None else f"{w} từ"
    r = [diem[k] / len(co_muc + VIET_LAI) for k in ("bm25", "lsa", "rrf")]
    kq_doan.append((ten_w, len(doan), *r))
    print(f"{ten_w:<18}{len(doan):9d}{r[0]:15.3f}{r[1]:14.3f}{r[2]:14.3f}")

# (Kết quả của mục này được trình bày cùng thí nghiệm có e5-small, Hình 6 của experiments_llm.py.)


# =====================================================================
# (E) Tìm kiếm gần đúng: IVF, PQ, đồ thị láng giềng
# =====================================================================
head("(E) Tim kiem gan dung tren vector: IVF, PQ va do thi lang gieng")

cau_all = []
for m in MUC:
    for c in re.split(r"(?<=[.!?:])\s+|\n+", m["text"]):
        if len(c.split()) >= 6:
            cau_all.append(c)
tv_c = TfidfVectorizer(tokenizer=lambda s: token_bm25(s, True), lowercase=False, token_pattern=None,
                       sublinear_tf=True, min_df=2)
Xc = tv_c.fit_transform(cau_all)
D = 128
V = normalize(TruncatedSVD(n_components=D, random_state=0).fit_transform(Xc)).astype(np.float32)
rng = np.random.default_rng(0)
perm = rng.permutation(len(V))
Q = V[perm[:300]]; B = V[perm[300:]]
N = len(B)
print(f"{N:,} vector {D} chieu (LSA cua cac cau trong kho), 300 truy van, tim 10 lang gieng gan nhat")

S_exact = Q @ B.T
top_exact = np.argsort(-S_exact, axis=1)[:, :10]


def recall10(pred):
    return np.mean([len(set(p[:10]) & set(t)) / 10 for p, t in zip(pred, top_exact)])


def kmeans(X, k, it=20, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(it):
        lab = np.argmax(X @ C.T - 0.5 * (C * C).sum(1), axis=1)  # tối đa -||x-c||^2/2 + const
        for j in range(k):
            pts = X[lab == j]
            if len(pts):
                C[j] = pts.mean(0)
    lab = np.argmax(X @ C.T - 0.5 * (C * C).sum(1), axis=1)
    return C, lab


# IVF
nlist = 128
Cc, lab = kmeans(B, nlist)
lists = [np.where(lab == j)[0] for j in range(nlist)]
print(f"\nIVF voi {nlist} cum (k-means); kich thuoc cum: nho nhat {min(map(len, lists))}, "
      f"lon nhat {max(map(len, lists))}")
print(f"{'nprobe':>8}{'recall@10':>12}{'tỉ lệ vector phải so':>24}")
kq_ivf = []
for nprobe in [1, 2, 4, 8, 16, 32, 64]:
    preds, quet = [], 0
    for q in Q:
        gan = np.argsort(-(Cc @ q))[:nprobe]
        ung = np.concatenate([lists[j] for j in gan])
        quet += len(ung)
        preds.append(ung[np.argsort(-(B[ung] @ q))[:10]])
    r = recall10(preds); f = quet / len(Q) / N
    kq_ivf.append((nprobe, r, f))
    print(f"{nprobe:8d}{r:12.3f}{f * 100:23.1f}%")

# PQ
print(f"\nProduct quantization: chia {D} chieu thanh m doan, moi doan 256 tam (1 byte)")
print(f"{'m':>4}{'byte/vector':>13}{'nén':>8}{'recall@10':>12}{'recall@10 sau xếp lại 100':>28}")
kq_pq = []
for m in [8, 16, 32, 64]:
    ds = D // m
    code = np.zeros((N, m), np.uint8); books = []
    for i in range(m):
        sub = B[:, i * ds:(i + 1) * ds]
        Cm, lb = kmeans(sub, 256, it=12, seed=i)
        books.append(Cm); code[:, i] = lb
    preds, preds_rr = [], []
    for q in Q:
        bang = np.stack([books[i] @ q[i * ds:(i + 1) * ds] for i in range(m)])   # m x 256
        s = bang[np.arange(m)[:, None], code.T].sum(0)
        top = np.argsort(-s)[:100]
        preds.append(top[:10])
        preds_rr.append(top[np.argsort(-(B[top] @ q))][:10])
    r1, r2 = recall10(preds), recall10(preds_rr)
    kq_pq.append((m, m, r1, r2))
    print(f"{m:4d}{m:13d}{D * 4 / m:7.0f}x{r1:12.3f}{r2:28.3f}")

# Đồ thị láng giềng: tìm kiếm tham lam theo kiểu best-first trên đồ thị k-NN
K_G = 16
knn = np.zeros((N, K_G), np.int64)
for s0 in range(0, N, 2000):
    blk = B[s0:s0 + 2000] @ B.T
    for r_, row in enumerate(blk):
        row[s0 + r_] = -np.inf
    knn[s0:s0 + 2000] = np.argpartition(-blk, K_G, axis=1)[:, :K_G]


def tim_do_thi(q, ef, seed):
    r = np.random.default_rng(seed)
    start = r.choice(N, 8, replace=False)
    diem = {int(i): float(B[i] @ q) for i in start}
    ung = sorted(diem, key=lambda i: -diem[i])
    da_mo = set(); n_tinh = len(start)
    while True:
        mo = [i for i in ung[:ef] if i not in da_mo]
        if not mo:
            break
        i = mo[0]; da_mo.add(i)
        for j in knn[i]:
            j = int(j)
            if j not in diem:
                diem[j] = float(B[j] @ q); n_tinh += 1
        ung = sorted(diem, key=lambda i: -diem[i])
    return ung[:10], n_tinh


print(f"\nTim kiem tren do thi k-NN ({K_G} canh moi dinh), best-first voi hang doi ef:")
print(f"{'ef':>6}{'recall@10':>12}{'số phép tính khoảng cách':>27}{'so với vét cạn':>17}")
kq_g = []
for ef in [10, 20, 40, 80, 160]:
    preds, tinh = [], 0
    for qi, q in enumerate(Q):
        p, n_ = tim_do_thi(q, ef, qi)
        preds.append(p); tinh += n_
    r = recall10(preds); t_ = tinh / len(Q)
    kq_g.append((ef, r, t_ / N))
    print(f"{ef:6d}{r:12.3f}{t_:27.0f}{t_ / N * 100:16.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
ax[0].plot([k[2] * 100 for k in kq_ivf], [k[1] for k in kq_ivf], "-o", ms=4, color=C_BLUE, label="IVF (nprobe)")
ax[0].plot([k[2] * 100 for k in kq_g], [k[1] for k in kq_g], "-s", ms=4, color=C_BAD, label="đồ thị k-NN (ef)")
ax[0].set_xscale("log"); ax[0].set_xlabel("% vector phải tính khoảng cách (log)")
ax[0].set_ylabel("recall@10"); ax[0].set_ylim(0, 1.02); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("Độ chính xác theo khối lượng tính", fontsize=9)
ax[1].plot([k[1] for k in kq_pq], [k[2] for k in kq_pq], "-o", ms=4, color=C_WARN, label="chỉ dùng mã PQ")
ax[1].plot([k[1] for k in kq_pq], [k[3] for k in kq_pq], "-o", ms=4, color=C_MAIN, label="PQ + xếp lại 100")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("byte cho mỗi vector (float32: 512)")
ax[1].set_ylabel("recall@10"); ax[1].set_ylim(0, 1.02); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("Product quantization: bộ nhớ và độ chính xác", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud07_ann.png"); plt.close(fig)


# =====================================================================
# (F) Độ tin cậy của agent nhiều bước
# =====================================================================
head("(F) Do tin cay cua agent nhieu buoc")

print(f"{'số bước':>8}" + "".join(f"{'p = ' + str(p):>12}" for p in [0.99, 0.95, 0.90]))
for n in [1, 5, 10, 20, 50]:
    print(f"{n:8d}" + "".join(f"{p ** n:12.3f}" for p in [0.99, 0.95, 0.90]))

# Kiểm tra sau mỗi bước: phát hiện lỗi với xác suất c, làm lại tối đa r lần
def mo_phong(p, n, c, r, lan=200000, seed=0):
    g = np.random.default_rng(seed)
    thanh = 0; goi = 0
    for _ in range(lan // 1000):
        ok = np.ones(1000, bool); calls = np.zeros(1000)
        for _b in range(n):
            buoc_ok = np.zeros(1000, bool); con = np.ones(1000, bool)
            for lan_thu in range(r + 1):
                dung = g.random(1000) < p
                calls += con
                buoc_ok |= con & dung
                bat = g.random(1000) < c
                con = con & ~dung & bat          # sai và bị phát hiện thì thử lại
                if lan_thu == r:
                    break
            ok &= buoc_ok
        thanh += ok.sum(); goi += calls.sum()
    return thanh / lan, goi / lan


def cong_thuc(p, n, c, r):
    # một bước thành công nếu: đúng ở một lần thử nào đó trước khi hết lượt,
    # mỗi lần sai chỉ được thử lại khi bị phát hiện
    q = sum(p * ((1 - p) * c) ** k for k in range(r + 1))
    return q ** n


print(f"\nAgent 20 buoc, p = 0,95 moi buoc; kiem tra phat hien loi voi xac suat c, thu lai toi da r lan:")
print(f"{'c':>6}{'r':>4}{'công thức':>12}{'mô phỏng':>11}{'lời gọi / bước':>17}")
kq_ag = []
for c, r in [(0.0, 0), (0.5, 1), (0.8, 1), (0.8, 3), (0.95, 3)]:
    f = cong_thuc(0.95, 20, c, r); s, calls = mo_phong(0.95, 20, c, r)
    kq_ag.append((c, r, f, s, calls / 20))
    print(f"{c:6.2f}{r:4d}{f:12.3f}{s:11.3f}{calls / 20:17.3f}")

fig, ax = plt.subplots(figsize=(5.6, 2.9))
ns = np.arange(1, 51)
for p, col in [(0.99, C_MAIN), (0.95, C_BLUE), (0.90, C_BAD)]:
    ax.plot(ns, p ** ns, color=col, lw=1.6, label=f"p = {str(p).replace('.', ',')}")
qv = cong_thuc(0.95, 1, 0.8, 3)
ax.plot(ns, qv ** ns, "--", color=C_BLUE, lw=1.4, label="p = 0,95 có kiểm tra (c = 0,8; r = 3)")
ax.set_xlabel("số bước"); ax.set_ylabel("xác suất hoàn thành")
ax.legend(frameon=False, fontsize=7.5); ax.set_ylim(0, 1.02)
ax.set_title("Xác suất một agent hoàn thành cả chuỗi bước", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud11_agent.png"); plt.close(fig)


# =====================================================================
# (G) Cần bao nhiêu mẫu để đánh giá
# =====================================================================
head("(G) Can bao nhieu mau de danh gia mot ung dung LLM")

print("Khoang tin cay 95% cua do chinh xac uoc luong (p = 0,80):")
print(f"{'n':>6}{'nửa độ rộng (công thức)':>26}{'nửa độ rộng (mô phỏng)':>25}")
g = np.random.default_rng(1)
kq_ci = []
for n in [50, 100, 200, 500, 1000, 2000]:
    f = 1.96 * math.sqrt(0.8 * 0.2 / n)
    sim = g.binomial(n, 0.8, 20000) / n
    lo, hi = np.percentile(sim, [2.5, 97.5])
    kq_ci.append((n, f, (hi - lo) / 2))
    print(f"{n:6d}{f * 100:25.1f}%{(hi - lo) / 2 * 100:24.1f}%")

# Hai phiên bản prompt: A đúng 80%, B đúng 83%. Chấm trên cùng một bộ câu hỏi (ghép cặp)
# thì kết quả của hai phiên bản tương quan với nhau; chấm trên hai bộ khác nhau thì không.
from scipy.stats import norm, binomtest


def luc_kiem_dinh(n, rep=2000, seed=0):
    g2 = np.random.default_rng(seed)
    thang_cap = thang_rieng = 0
    for _ in range(rep):
        # A đúng 80% số câu. B giữ đúng 97% số câu A đúng và sửa được 27% số câu A sai,
        # nên B đúng 0,8 * 0,97 + 0,2 * 0,27 = 83% số câu.
        a = g2.random(n) < 0.80
        b = np.where(a, g2.random(n) < 0.97, g2.random(n) < 0.27)
        # McNemar chính xác trên các cặp bất đồng
        n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
        if n01 + n10 and binomtest(n01, n01 + n10, 0.5).pvalue < 0.05 and n01 > n10:
            thang_cap += 1
        # hai bộ câu hỏi độc lập: kiểm định hai tỉ lệ
        a2 = g2.random(n) < 0.80; b2 = g2.random(n) < 0.83
        p1, p2 = a2.mean(), b2.mean(); pp = (p1 + p2) / 2
        z = (p2 - p1) / math.sqrt(max(pp * (1 - pp) * 2 / n, 1e-12))
        if z > norm.ppf(0.975):
            thang_rieng += 1
    return thang_cap / rep, thang_rieng / rep


print("\nPhat hien cai thien tu 80% len khoang 83% (muc y nghia 5%), 2 000 lan mo phong moi n:")
print(f"{'n':>6}{'ghép cặp (McNemar)':>21}{'hai bộ độc lập':>17}")
kq_pow = []
for n in [100, 300, 1000, 3000]:
    pc, pr = luc_kiem_dinh(n)
    kq_pow.append((n, pc, pr))
    print(f"{n:6d}{pc:21.3f}{pr:17.3f}")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
ax[0].plot([k[0] for k in kq_ci], [k[1] * 100 for k in kq_ci], "-o", ms=4, color=C_MAIN)
ax[0].set_xscale("log"); ax[0].set_xlabel("số câu hỏi đánh giá (log)")
ax[0].set_ylabel("± điểm phần trăm"); ax[0].set_title("Khoảng tin cậy 95% của độ chính xác 80%", fontsize=9)
ax[1].plot([k[0] for k in kq_pow], [k[1] for k in kq_pow], "-o", ms=4, color=C_BAD, label="ghép cặp")
ax[1].plot([k[0] for k in kq_pow], [k[2] for k in kq_pow], "-s", ms=4, color=C_DIM, label="hai bộ độc lập")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("số câu hỏi mỗi phiên bản (log)")
ax[1].set_ylabel("xác suất phát hiện"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("Phát hiện cải thiện 80% → 83%", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ud12_danhgia.png"); plt.close(fig)

head("Da luu 5 hinh vao figs/: ud02, ud05, ud07, ud11, ud12")
