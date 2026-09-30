Hai script dưới đây sinh ra toàn bộ 14 hình và mọi con số thực nghiệm của giáo trình.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả y hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/mlops/experiments.py     # Hình 6, 7, 9–14 — khoảng một phút
python code/mlops/fig_diagrams.py    # Hình 1–5 và Hình 8
```

Môi trường đã dùng để sinh số liệu trong tài liệu: Python 3.13, NumPy 2.3, SciPy 1.16,
scikit-learn 1.7, matplotlib 3.10.

Lưu ý về phạm vi: những con số đo được ở đây đến từ **dữ liệu mô phỏng**, được thiết kế để
cô lập đúng một cơ chế mỗi lần. Chúng dùng để chứng minh *cơ chế* tồn tại và có độ lớn đáng
kể, không dùng để suy ra con số cho một hệ thống cụ thể nào. Muốn có con số cho hệ thống của
mình thì chạy lại đúng thí nghiệm ấy trên dữ liệu của mình — cách làm được nêu trong từng bài
tập ở Chương 14.
