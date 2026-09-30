Các script đặt hạt giống cố định nên độ chính xác lặp lại được trên cùng phiên bản thư viện.
Riêng độ trễ phụ thuộc phần cứng và dao động giữa các lần chạy.

```bash
pip install numpy scipy matplotlib scikit-learn torch torchao
python code/fig_basics.py
python code/fig_diagrams.py
python code/numpy_experiments.py
python code/torch_experiments.py        # vài phút trên CPU
python code/torch_qat_eager.py
python code/sensitivity.py
python code/sweep_alpha.py              # lời giải Bài 8, chỉ cần NumPy
```

Môi trường đã dùng để sinh mọi số liệu trong tài liệu: Python 3.12, NumPy 2.4, PyTorch 2.14 (CPU),
torchao 0.18, scikit-learn 1.8, CPU Intel Xeon 2,1 GHz (có AVX-512 VNNI và AMX-INT8), 1 luồng.
