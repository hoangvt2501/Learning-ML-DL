Hai script dưới đây sinh ra toàn bộ 13 hình và mọi con số đo được của giáo trình.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả giống hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/nentang/experiments.py     # Hình 2, 3, 4, 6–12, vài phút
python code/nentang/fig_diagrams.py    # Hình 1, 5, 13
```

Trên Windows, nếu ghi kết quả ra tệp, đặt `PYTHONIOENCODING=utf-8` để các dòng chữ tiếng Việt không gây lỗi mã hoá.

Môi trường đã dùng để sinh số liệu: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

Số liệu của giáo trình gồm hai loại, liệt kê đầy đủ ở Phụ lục.

Loại thứ nhất là kiểm chứng đẳng thức: hồi quy logistic trùng hồi quy softmax với K = 2, PCA qua ma trận
hiệp phương sai trùng PCA qua SVD, sai số tái tạo bằng tổng các trị riêng bị bỏ, ridge trùng MAP với tiên
nghiệm Gauss, và các khẳng định về đối ngẫu và KKT của SVM. Những kết quả này phải khớp tới sai số của máy
tính: sáu trong mười đẳng thức khớp ở mức 10⁻¹⁵, bốn đẳng thức còn lại bị giới hạn bởi dung sai của bộ giải.
Nếu chúng không khớp thì mã có lỗi.

Loại thứ hai là mô phỏng trên dữ liệu sinh ngẫu nhiên, mỗi thí nghiệm cô lập một cơ chế. Chúng cho thấy cơ
chế tồn tại và có độ lớn đáng kể, nhưng không dùng để suy ra con số cho một tập dữ liệu thật cụ thể.
