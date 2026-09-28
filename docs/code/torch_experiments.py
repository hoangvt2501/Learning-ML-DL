"""Thí nghiệm PyTorch THẬT trên bộ dữ liệu chữ số viết tay 8x8 (sklearn digits, 1797 ảnh, có sẵn offline).
Mọi con số trong bài phần thực nghiệm được in ra từ file này."""
import copy, os, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs") + os.sep
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ dữ liệu
d = load_digits()
X = (d.images / 16.0).astype(np.float32)[:, None]            # (1797, 1, 8, 8), giá trị trong [0, 1]
y = d.target.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = map(torch.from_numpy, (Xtr, Xte, ytr, yte))
Xcal = Xtr[:256]                                              # tập calibration: 256 ảnh từ tập train

def accuracy(model, X=Xte, y=yte):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

# ------------------------------------------------------------------ mô hình float (có BatchNorm, có depthwise)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16)
        self.dw, self.bn2 = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.BatchNorm2d(16)
        self.pw, self.bn3 = nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32)
        self.conv4, self.bn4 = nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dw(x)))
        x = F.max_pool2d(F.relu(self.bn3(self.pw(x))), 2)
        x = F.relu(self.bn4(self.conv4(x)))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))

def train(model, epochs=40, lr=3e-3, seed=0, wd=0.0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model.eval()

torch.manual_seed(0)
fp_model = train(Net())
acc_fp = accuracy(fp_model)
print(f"[FP32] test accuracy = {acc_fp:.4f}  (n_test = {len(yte)})")

# ------------------------------------------------------------------ (1) BN folding
def fold_bn(conv, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    b = bn.weight * (b0 - bn.running_mean) / std + bn.bias
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      groups=conv.groups, bias=True)
    fused.weight.data, fused.bias.data = w.detach().clone(), b.detach().clone()
    return fused

class FoldedNet(nn.Module):
    """Mạng sau khi gộp BN; có chỗ cắm fake-quant cho weight và activation."""
    def __init__(self, net):
        super().__init__()
        self.convs = nn.ModuleList([fold_bn(net.conv1, net.bn1), fold_bn(net.dw, net.bn2),
                                    fold_bn(net.pw, net.bn3), fold_bn(net.conv4, net.bn4)])
        self.fc = copy.deepcopy(net.fc)
        self.wq = None            # hàm fake-quant weight: (tên lớp, W) -> W_hat
        self.aq = None            # danh sách fake-quant activation: aq[i](x)
    def _w(self, i, w): return w if self.wq is None else self.wq(i, w)
    def _a(self, i, x): return x if self.aq is None else self.aq[i](x)
    def forward(self, x):
        x = self._a(0, x)                                     # lượng tử input
        for i, c in enumerate(self.convs):
            x = F.relu(F.conv2d(x, self._w(i, c.weight), c.bias, c.stride, c.padding, groups=c.groups))
            if i == 2: x = F.max_pool2d(x, 2)
            x = self._a(i + 1, x)
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return F.linear(x, self._w(4, self.fc.weight), self.fc.bias)

folded = FoldedNet(fp_model).eval()
with torch.no_grad():
    diff = (folded(Xte) - fp_model(Xte)).abs().max().item()
print(f"[BN folding] max |logit_folded - logit_goc| = {diff:.2e}")

# ------------------------------------------------------------------ (2) per-tensor vs per-channel
def fq_sym(w, bits, per_channel):
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        amax = w.abs().reshape(w.shape[0], -1).max(1).values.clamp_min(1e-8)
        amax = amax.reshape(-1, *([1] * (w.dim() - 1)))
    else:
        amax = w.abs().max().clamp_min(1e-8)
    s = amax / qmax
    q = torch.clamp(torch.round(w / s), -qmax, qmax) * s
    return w + (q - w).detach()                               # STE

dw_w = folded.convs[1].weight.detach()
dw_ranges = dw_w.abs().reshape(16, -1).max(1).values.numpy()
res_gran = {}
for bits in (8, 4):
    for pc in (False, True):
        folded.wq = lambda i, w, b=bits, p=pc: fq_sym(w, b, p)
        res_gran[(bits, pc)] = (accuracy(folded),
                                 torch.mean((fq_sym(dw_w, bits, pc) - dw_w) ** 2).item())
folded.wq = None
print("[Granularity] (bit, per_channel) -> (acc chỉ quantize weight, MSE lớp depthwise):")
for k, v in res_gran.items(): print("   ", k, f"acc={v[0]:.4f}  mse_dw={v[1]:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].bar(range(16), dw_ranges, color="#4a78c2")
axes[0].axhline(dw_ranges.max(), ls="--", color="#d9534f")
axes[0].text(15.5, dw_ranges.max() * 1.02, "dải per-tensor dùng chung", ha="right", color="#d9534f", fontsize=9)
axes[0].set_xlabel("Kênh của lớp depthwise (sau khi gộp BN)"); axes[0].set_ylabel("max|w| của kênh")
axes[0].set_title(f"Lớp depthwise: max|w| lớn nhất / nhỏ nhất ≈ {dw_ranges.max() / dw_ranges.min():.1f} lần", fontsize=10)
lab = ["INT8\nper-tensor", "INT8\nper-channel", "INT4\nper-tensor", "INT4\nper-channel"]
keys = [(8, False), (8, True), (4, False), (4, True)]
vals = [res_gran[k][0] * 100 for k in keys]
bars = axes[1].bar(lab, vals, color=["#9e9e9e", "#4a78c2", "#9e9e9e", "#4a78c2"])
axes[1].axhline(acc_fp * 100, ls="--", color="k"); axes[1].text(3.4, acc_fp * 100 + 0.5, "FP32", ha="right")
for bb, v in zip(bars, vals): axes[1].text(bb.get_x() + bb.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
axes[1].set_ylim(min(vals) - 8, 102); axes[1].set_ylabel("Độ chính xác test (%)")
axes[1].set_title("Chỉ lượng tử trọng số (activation giữ FP32)", fontsize=10)
fig.savefig(OUT + "fig08_per_channel.png"); plt.close(fig)

# ------------------------------------------------------------------ (3) calibration activation
def collect_acts(model, X):
    acts = [[] for _ in range(5)]
    def mk(i):
        def f(x): acts[i].append(x.detach().flatten()); return x
        return f
    model.aq = [mk(i) for i in range(5)]
    with torch.no_grad(): model(X)
    model.aq = None
    return [torch.cat(a) for a in acts]

def fq_asym(x, lo, hi, bits):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    qmax = 2 ** bits - 1
    s = max(hi - lo, 1e-8) / qmax
    z = float(np.clip(round(-lo / s), 0, qmax))
    q = torch.clamp(torch.round(x / s) + z, 0, qmax)
    xq = (q - z) * s
    return x + (xq - x).detach()

def calib_minmax(a, bits): return a.min().item(), a.max().item()
def calib_percentile(a, bits, p=99.99):
    return 0.0, torch.quantile(a[torch.randperm(len(a))[:1_000_000]], p / 100).item()
def calib_mse(a, bits):
    amax = a.max().item(); best = (None, float("inf"))
    for t in np.linspace(amax / 50, amax, 200):
        e = torch.mean((fq_asym(a, 0.0, t, bits) - a) ** 2).item()
        if e < best[1]: best = (t, e)
    return 0.0, best[0]
def calib_kl(a, bits, nbins=2048):
    """Thuật toán entropy calibration kiểu TensorRT (Migacz, 2017), cho activation không âm."""
    a = a.numpy(); a = a[a > 0]
    hist, edges = np.histogram(a, bins=nbins, range=(0, a.max()))
    nlev = 2 ** bits - 1
    best_kl, best_i = np.inf, nbins
    for i in range(nlev, nbins + 1):
        p = hist[:i].astype(np.float64).copy(); p[-1] += hist[i:].sum()        # dồn phần bị cắt vào bin cuối
        chunks = np.array_split(hist[:i].astype(np.float64), nlev)
        q = np.concatenate([np.where(c > 0, c.sum() / max((c > 0).sum(), 1), 0) for c in chunks])
        pm, qm = p / p.sum(), q / max(q.sum(), 1e-12)
        mask = pm > 0
        if np.any(qm[mask] == 0): continue
        kl = np.sum(pm[mask] * np.log(pm[mask] / qm[mask]))
        if kl < best_kl: best_kl, best_i = kl, i
    return 0.0, float(edges[best_i])

acts_cal = collect_acts(folded, Xcal)
methods = {"min–max": calib_minmax, "percentile 99.99%": calib_percentile, "MSE": calib_mse, "KL (entropy)": calib_kl}

def ptq_eval(model, bits_w, bits_a, method, keep_ends_8bit=True):
    ranges = []
    for i, a in enumerate(acts_cal):
        b = 8 if (i == 0 and keep_ends_8bit) else bits_a
        ranges.append(calib_minmax(a, b) if i == 0 else methods[method](a, b))
    model.aq = [lambda x, r=r, b=(8 if (i == 0 and keep_ends_8bit) else bits_a): fq_asym(x, r[0], r[1], b)
                for i, r in enumerate(ranges)]
    model.wq = lambda i, w: fq_sym(w, 8 if (keep_ends_8bit and i in (0, 4)) else bits_w, True)
    acc = accuracy(model); model.aq = model.wq = None
    return acc, ranges

calib_res = {}
for bits in (8, 4):
    for m in methods:
        calib_res[(bits, m)] = ptq_eval(folded, bits, bits, m)
print("[Calibration] W{b}A{b} PTQ, weight per-channel, lớp đầu/cuối 8 bit:")
for (b, m), (acc, rg) in calib_res.items():
    print(f"   {b} bit | {m:18s}: acc={acc:.4f} | ngưỡng lớp pw (act #3) = {rg[3][1]:.3f}")

a3 = acts_cal[3].numpy(); a3p = a3[a3 > 0]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.hist(a3p, bins=300, color="#ccc", log=True)
cols = {"min–max": "#d9534f", "percentile 99.99%": "#e0923a", "MSE": "#4a78c2", "KL (entropy)": "#4caf6e"}
for m, c in cols.items():
    t = calib_res[(4, m)][1][3][1]
    ax.axvline(t, color=c, lw=2, label=f"{m}: ngưỡng = {t:.2f}, acc W4A4 = {calib_res[(4, m)][0] * 100:.1f}%")
ax.set_xlabel("Giá trị activation dương sau lớp pointwise + ReLU (256 ảnh calibration)")
ax.set_ylabel("Số lượng (thang log)"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title(f"Ngưỡng cắt do các phương pháp calibration chọn (4 bit); {np.mean(a3 == 0):.0%} giá trị bằng 0 không vẽ")
fig.savefig(OUT + "fig11_calibration.png"); plt.close(fig)

# ------------------------------------------------------------------ (4) PTQ vs QAT theo số bit
class EMARange:
    def __init__(self, lo, hi, m=0.9): self.lo, self.hi, self.m = lo, hi, m
    def update(self, x):
        self.lo = self.m * self.lo + (1 - self.m) * min(x.min().item(), 0.0)
        self.hi = self.m * self.hi + (1 - self.m) * x.max().item()

def qat(fmodel, bits, epochs=15, lr=5e-4, seed=0):
    model = copy.deepcopy(fmodel)
    _, ranges = ptq_eval(model, bits, bits, "MSE")                 # khởi tạo dải bằng PTQ
    emas = [EMARange(lo, hi) for lo, hi in ranges]
    def mk(i):
        b = 8 if i == 0 else bits
        def f(x):
            if model.training: emas[i].update(x.detach())
            return fq_asym(x, emas[i].lo, emas[i].hi, b)
        return f
    model.aq = [mk(i) for i in range(5)]
    model.wq = lambda i, w: fq_sym(w, 8 if i in (0, 4) else bits, True)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    acc = accuracy(model); model.aq = model.wq = None
    return acc

bit_list = [8, 6, 4, 3, 2]
ptq_minmax = [ptq_eval(folded, b, b, "min–max")[0] for b in bit_list]
ptq_mse = [ptq_eval(folded, b, b, "MSE")[0] for b in bit_list]
qat_acc = [np.mean([qat(folded, b, seed=s) for s in range(3)]) for b in bit_list]
print("[PTQ vs QAT] bits:", bit_list)
print("   PTQ min-max:", [f"{a:.4f}" for a in ptq_minmax])
print("   PTQ MSE    :", [f"{a:.4f}" for a in ptq_mse])
print("   QAT (TB 3 seed):", [f"{a:.4f}" for a in qat_acc])

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.axhline(acc_fp * 100, ls="--", color="k", label=f"FP32 ({acc_fp * 100:.1f}%)")
ax.plot(bit_list, np.array(ptq_minmax) * 100, "s-", color="#d9534f", label="PTQ, calibration min–max")
ax.plot(bit_list, np.array(ptq_mse) * 100, "o-", color="#e0923a", label="PTQ, calibration MSE")
ax.plot(bit_list, np.array(qat_acc) * 100, "^-", color="#4a78c2", label="QAT (15 epoch, trung bình 3 lần chạy)")
ax.set_xticks(bit_list, [f"W{b}A{b}" for b in bit_list]); ax.invert_xaxis()
ax.set_ylabel("Độ chính xác test (%)"); ax.legend(frameon=False, fontsize=9)
ax.set_title("PTQ và QAT theo độ rộng bit (lớp đầu/cuối giữ 8 bit)")
fig.savefig(OUT + "fig13_ptq_qat.png"); plt.close(fig)

# ------------------------------------------------------------------ (5) Backend int8 thật của PyTorch
import torch.ao.quantization as tq

def size_kb(m):
    torch.save(m.state_dict(), "/tmp/_m.pt"); s = os.path.getsize("/tmp/_m.pt") / 1024; os.remove("/tmp/_m.pt"); return s

class QNet(nn.Module):
    """Cùng kiến trúc Net nhưng viết theo yêu cầu của eager-mode quantization."""
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU()
        self.dw, self.bn2, self.relu2 = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.BatchNorm2d(16), nn.ReLU()
        self.pw, self.bn3, self.relu3 = nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv4, self.bn4, self.relu4 = nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
        self.dequant = tq.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.dw(x)))
        x = self.pool(self.relu3(self.bn3(self.pw(x))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.fc(torch.flatten(self.gap(x), 1))
        return self.dequant(x)

qm = QNet(); qm.load_state_dict(fp_model.state_dict(), strict=False); qm.eval()
assert accuracy(qm) == acc_fp
torch.backends.quantized.engine = "x86"
qm.qconfig = tq.get_default_qconfig("x86")
tq.fuse_modules(qm, [["conv1", "bn1", "relu1"], ["dw", "bn2", "relu2"], ["pw", "bn3", "relu3"],
                     ["conv4", "bn4", "relu4"]], inplace=True)
tq.prepare(qm, inplace=True)
with torch.no_grad(): qm(Xcal)
tq.convert(qm, inplace=True)
acc_eager = accuracy(qm)
print(f"[torch.ao eager static int8] acc = {acc_eager:.4f}; size FP32 = {size_kb(fp_model):.1f} KB, INT8 = {size_kb(qm):.1f} KB")
print("   ", qm.conv1)

# PT2E với torchao
try:
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
    m = copy.deepcopy(fp_model).eval()
    ep = torch.export.export(m, (Xcal[:4],), dynamic_shapes={"x": {0: torch.export.Dim("batch")}}).module()
    quantizer = xiq.X86InductorQuantizer().set_global(xiq.get_default_x86_inductor_quantization_config())
    pm = prepare_pt2e(ep, quantizer)
    with torch.no_grad(): pm(Xcal)
    cm = convert_pt2e(pm)
    with torch.no_grad():
        acc_pt2e = (cm(Xte).argmax(1) == yte).float().mean().item()
    print(f"[torchao PT2E + X86InductorQuantizer] acc = {acc_pt2e:.4f}")
except Exception as e:
    print("[PT2E] lỗi:", repr(e)[:300])

# torchao quantize_ (weight-only / dynamic) trên một MLP lớn hơn + đo độ trễ
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
def mlp(): return nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1024)).eval()
torch.manual_seed(0); base = mlp(); xin = torch.randn(16, 1024)
def lat(m, n=30):
    with torch.no_grad():
        for _ in range(5): m(xin)
        t = time.perf_counter()
        for _ in range(n): m(xin)
    return (time.perf_counter() - t) / n * 1000
with torch.no_grad(): ref = base(xin)
rows = [("FP32", base)]
dyn_ao = tq.quantize_dynamic(copy.deepcopy(base), {nn.Linear}, dtype=torch.qint8); rows.append(("torch.ao quantize_dynamic (int8)", dyn_ao))
for name, cfg in [("torchao Int8WeightOnly", Int8WeightOnlyConfig()),
                  ("torchao Int8DynamicActivationInt8Weight", Int8DynamicActivationInt8WeightConfig())]:
    try:
        mm = copy.deepcopy(base); quantize_(mm, cfg); rows.append((name, mm))
    except Exception as e:
        print(name, "lỗi:", repr(e)[:200])
print("[MLP 1024-4096-4096-1024, batch 16, 1 luồng CPU]")
for name, m in rows:
    try:
        with torch.no_grad(): o = m(xin)
        err = ((o - ref).norm() / ref.norm()).item()
        print(f"   {name:42s} size={size_kb(m) / 1024:6.1f} MB  latency={lat(m):7.2f} ms  rel_err={err:.2e}")
    except Exception as e:
        print(f"   {name}: lỗi khi chạy {repr(e)[:200]}")
