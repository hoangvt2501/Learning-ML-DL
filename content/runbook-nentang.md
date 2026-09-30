Hai script dưới đây sinh ra toàn bộ 13 hình và mọi con số đo được của giáo trình.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả y hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/nentang/experiments.py     # Hình 2, 3, 4, 6–12 — vài phút
python code/nentang/fig_diagrams.py    # Hình 1, 5, 13
```

Môi trường đã dùng để sinh số liệu trong tài liệu: Python 3.13, NumPy 2.3, SciPy 1.16,
scikit-learn 1.7, matplotlib 3.10.

Số liệu ở đây chia làm hai loại rất khác nhau, và Phụ lục của giáo trình liệt kê đầy đủ.

Loại thứ nhất là **kiểm chứng một đẳng thức**: hồi quy logistic ≡ softmax với K=2, PCA qua
hiệp phương sai ≡ PCA qua SVD, sai số tái dựng ≡ tổng trị riêng bị bỏ, ridge ≡ MAP với tiên
nghiệm Gauss, và bốn khẳng định về đối ngẫu cùng KKT của SVM. Những kết quả này phải khớp tới
sai số máy — sáu trong mười đẳng thức khớp ở mức 10⁻¹⁵, bốn cái còn lại bị giới hạn bởi dung
sai của bộ giải chứ không phải bởi lý thuyết. Nếu chúng không khớp thì có lỗi trong mã.

Loại thứ hai là **mô phỏng trên dữ liệu sinh ra**, thiết kế để cô lập đúng một cơ chế mỗi lần.
Chúng chứng minh cơ chế tồn tại và có độ lớn đáng kể, không dùng để suy ra con số cho một bộ
dữ liệu cụ thể nào.
