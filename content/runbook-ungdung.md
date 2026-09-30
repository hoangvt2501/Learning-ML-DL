Ba script dưới đây sinh ra toàn bộ 15 hình và mọi con số đo được của giáo trình.

```bash
pip install numpy scipy scikit-learn matplotlib nltk transformers tokenizers torch
python code/ungdung/experiments.py        # vài phút; tải ba tokenizer (vài MB) và hai kho văn bản NLTK
python code/ungdung/experiments_llm.py    # 1 tới 2 giờ trên CPU; tải khoảng 1,5 GB mô hình
python code/ungdung/fig_diagrams.py
```

Trên Windows, đặt `PYTHONIOENCODING=utf-8` khi ghi kết quả ra tệp. Nếu ổ đĩa mặc định không đủ chỗ cho
mô hình tải về, đặt biến môi trường `HF_HOME` trỏ tới một thư mục trên ổ khác.

Kho văn bản dùng cho các thí nghiệm truy xuất và RAG là bản chụp nội dung các giáo trình của lộ trình tại
thời điểm viết, lưu trong `kho_van_ban.json` và `kho_trac_nghiem.json`. Các lần chạy sau đọc lại bản chụp nên
số liệu không đổi khi nội dung giáo trình thay đổi; chạy `experiments.py --chup-lai` để chụp lại.

Mô hình ngôn ngữ trong thí nghiệm là Qwen2.5-0.5B-Instruct, nhỏ hơn nhiều so với mô hình dùng trong sản
phẩm. Các thí nghiệm cho thấy cơ chế và hướng của hiệu ứng, không dùng để đánh giá mô hình lớn.
