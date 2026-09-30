Hai script dưới đây sinh ra toàn bộ 14 hình và mọi con số đo được của giáo trình.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả y hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/models/experiments.py     # Hình 2, 4, 5, 7, 9, 10, 13, 14 — vài phút
python code/models/fig_diagrams.py    # Hình 1, 3, 6, 8, 11, 12
```

Môi trường đã dùng để sinh số liệu trong tài liệu: Python 3.13, NumPy 2.3, SciPy 1.16,
scikit-learn 1.7, matplotlib 3.10.

Lưu ý về phạm vi, và ở giáo trình này nó có hai nửa khác nhau.

Phần lớn số liệu đến từ **dữ liệu mô phỏng**, được thiết kế để cô lập đúng một cơ chế mỗi
lần — ví dụ thí nghiệm gradient qua 40 lớp chỉ đổi một biến là cách dựng lớp. Chúng chứng
minh cơ chế tồn tại và có độ lớn đáng kể, không dùng để suy ra con số cho một mô hình cụ thể.

Nhưng bốn kết quả là **đối chiếu với con số đã công bố**, không phải mô phỏng: số tham số của
GPT-2 small, medium, large; số tham số của Llama-2 7B và 13B; ngân sách giờ-GPU của Llama-2 7B;
và phương sai tích vô hướng nêu trong chú thích 4 của bài báo Transformer. Bốn kết quả ấy là
phép kiểm tra rằng các công thức ở Chương 12 đúng chứ không chỉ hợp lý — chúng khớp tới từng
tham số, và với ngân sách giờ-GPU thì lệch 0,17%.
