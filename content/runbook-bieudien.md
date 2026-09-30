Hai script dưới đây sinh ra toàn bộ 14 hình và mọi con số đo được của giáo trình.
Chúng đặt hạt giống cố định nên chạy lại cho kết quả y hệt.

```bash
pip install numpy scipy matplotlib scikit-learn
python code/bieudien/experiments.py     # Hình 2, 3, 4, 6, 8–10, 12, 13 — vài phút
python code/bieudien/fig_diagrams.py    # Hình 1, 5, 7, 11, 14
```

Môi trường đã dùng: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

Hai kết quả ở đây là **kiểm chứng một đẳng thức** chứ không phải mô phỏng: nghiệm của bài
toán RLHF có ràng buộc KL khớp dạng đóng tới 10⁻⁹, và chính sách DPO trùng chính sách RLHF
hai bước tới 4,16×10⁻⁸. Dạng đóng của quá trình khuếch tán cũng được đối chiếu với 200 000
quỹ đạo mô phỏng thật.

Ba khối cho số liệu **mâu thuẫn với điều tôi định viết**, và phần chữ đã được sửa theo số
chứ không ngược lại: thí nghiệm GAN không cho thấy sụp chế độ (nó cho thấy một thứ khác quan
trọng hơn), thí nghiệm Q-learning cho thấy ε = 0 vẫn thành công trong một cấu hình, và bảng
quét số chiều embedding không phải một đường cong đánh đổi đẹp. Phụ lục của giáo trình nói
rõ cả ba.
