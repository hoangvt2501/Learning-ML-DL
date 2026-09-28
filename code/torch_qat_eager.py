"""QAT eager mode của torch.ao: fine-tune từ trọng số FP32 đã huấn luyện rồi convert sang INT8.
Chạy: python code/torch_qat_eager.py (tự huấn luyện lại mô hình FP32 trước)."""
import os
import warnings; warnings.filterwarnings("ignore")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_experiments.py')).read()
exec(src.split('# ------------------------------------------------------------------ (1) BN folding')[0])
exec('import torch.ao.quantization as tq\n' + 'class QNet' + src.split('class QNet')[1].split('qm = QNet()')[0])
import torch.ao.nn.intrinsic.qat as nniqat
model = QNet(); model.load_state_dict(fp_model.state_dict(), strict=False)   # bắt đầu từ trọng số FP32 đã huấn luyện
model.train()
torch.backends.quantized.engine = "x86"
model.qconfig = tq.get_default_qat_qconfig("x86")
tq.fuse_modules_qat(model, [["conv1","bn1","relu1"],["dw","bn2","relu2"],["pw","bn3","relu3"],["conv4","bn4","relu4"]], inplace=True)
tq.prepare_qat(model, inplace=True)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
g = torch.Generator().manual_seed(0)
for ep in range(10):
    if ep == 6: model.apply(tq.disable_observer)          # cố định dải lượng tử
    if ep == 7: model.apply(nniqat.freeze_bn_stats)       # cố định thống kê BN
    model.train()
    perm = torch.randperm(len(Xtr), generator=g)
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i+64]
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
model.eval()
qmodel = tq.convert(model)
print("QAT eager int8 acc:", accuracy(qmodel))
