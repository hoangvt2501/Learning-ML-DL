Ba script dưới đây sinh ra toàn bộ 15 hình và mọi con số đo được của giáo trình.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả y hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/models/experiments.py     # Hình 2, 4, 5, 7, 10, 11, 14, 15, vài phút
python code/models/thi_giac.py        # Hình 9 và số liệu về IoU, NMS, AP, vài giây
python code/models/fig_diagrams.py    # Hình 1, 3, 6, 8, 12, 13
```

Môi trường đã dùng để sinh số liệu: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

Về phạm vi: phần lớn số liệu đến từ dữ liệu mô phỏng, mỗi thí nghiệm được thiết kế để cô lập một cơ chế,
ví dụ thí nghiệm gradient qua 40 lớp chỉ đổi cách dựng lớp, và thí nghiệm phát hiện đối tượng chỉ làm nhiễu
hộp thật thay vì huấn luyện một mô hình. Chúng cho thấy cơ chế tồn tại và có độ lớn đáng kể, không dùng để
suy ra con số cho một mô hình cụ thể.

Một số kết quả là đối chiếu với con số đã công bố, không phải mô phỏng: số tham số của GPT-2 small, medium,
large; số tham số của Llama 2 7B và 13B; ngân sách giờ-GPU của Llama 2 7B; và phương sai của tích vô hướng
trong chú thích 4 của bài báo Transformer. Các phép đối chiếu này kiểm tra rằng công thức ở Chương 12 đúng,
không chỉ hợp lý.
