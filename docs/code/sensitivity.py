"""Phân tích độ nhạy theo lớp: lượng tử từng lớp một xuống 3 bit (trọng số + activation đầu ra), các lớp khác giữ FP32."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_experiments.py")).read()
exec(src.split("# ------------------------------------------------------------------ (4) PTQ vs QAT")[0])

names = ["conv1 (3x3)", "dw (depthwise 3x3)", "pw (pointwise 1x1)", "conv4 (3x3)", "fc (Linear)"]
_, ranges = ptq_eval(folded, 8, 8, "MSE")               # dải activation tham chiếu
acts_ranges3 = [calib_mse(a, 3) for a in acts_cal]       # ngưỡng MSE cho 3 bit
BITS = 3
print(f"FP32: {acc_fp:.4f}")
for li, name in enumerate(names):
    folded.wq = lambda i, w, li=li: fq_sym(w, BITS, True) if i == li else w
    # activation đầu ra của lớp li là điểm aq[li+1] (với fc: không có activation sau, chỉ lượng tử trọng số)
    def mk(i, li=li):
        if i == li + 1 and i < 5:
            lo, hi = acts_ranges3[i]
            return lambda x: fq_asym(x, lo, hi, BITS)
        return lambda x: x
    folded.aq = [mk(i) for i in range(5)]
    acc = accuracy(folded)
    folded.wq = folded.aq = None
    print(f"{name:22s}: acc = {acc:.4f}  (giảm {100 * (acc_fp - acc):.2f} điểm %)")
