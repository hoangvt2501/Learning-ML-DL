Ba script dưới đây sinh ra toàn bộ 14 hình, mọi con số thực nghiệm trong bài và số liệu dùng trong lời giải bài tập.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả y hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/mlops/experiments.py     # Hình 6, 7, 9–14 và số liệu trong bài, khoảng một phút
python code/mlops/bai_tap.py         # số liệu cho lời giải bài tập, khoảng một phút rưỡi
python code/mlops/fig_diagrams.py    # Hình 1–5 và Hình 8
```

Môi trường đã dùng để sinh số liệu: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

Về phạm vi: số liệu thực nghiệm đến từ dữ liệu mô phỏng, mỗi thí nghiệm được thiết kế để cô lập một cơ chế.
Chúng cho thấy cơ chế đó tồn tại và có độ lớn đáng kể, không dùng để suy ra con số cho một hệ thống cụ thể.
Muốn có con số cho một hệ thống thật, cần chạy cùng thí nghiệm trên dữ liệu của hệ thống đó; các bài tập ở Chương 14
hướng dẫn cách làm.
