# Quantization trong Deep Learning — giáo trình tự học

Trang web đọc giáo trình *“Quantization trong Deep Learning: từ nguyên lý đến thực hành”*,
kèm bài tập có lời giải, ngân hàng trắc nghiệm và năm công cụ tương tác.

Toàn bộ site là **HTML tĩnh**: mở `docs/index.html` bằng trình duyệt là đọc được ngay,
không cần chạy máy chủ, không cần mạng (trừ font chữ, có sẵn phương án dự phòng).

## Xem ngay

```bash
npm install     # chỉ cần cho việc dựng lại site
npm run dev     # dựng lại rồi phục vụ tại http://localhost:4173
```

Hoặc mở thẳng `docs/index.html`.

## Có gì bên trong

| Trang | Nội dung |
| --- | --- |
| **15 chương + phụ lục** | Nguyên văn giáo trình, công thức dựng bằng KaTeX, mọi cụm “Mục 6.2”, “Chương 11”, “Hình 7” đều thành liên kết bấm được. Cuối mỗi chương có phần tự kiểm tra. |
| **Bài tập** | 9 bài của giáo trình, mỗi bài kèm **lời giải chi tiết** ẩn sẵn, cộng phần luyện tính tay tự chấm và 48 câu trắc nghiệm theo chương. |
| **Phòng thí nghiệm** | 5 công cụ chạy trong trình duyệt: máy lượng tử affine, đánh đổi làm tròn – cắt, soi bit số thực, requantization dấu chấm tĩnh, tính dung lượng mô hình và KV cache. |
| **Ghi chú biên tập** | 5 chỗ trong giáo trình được phát biểu chặt lại, mỗi chỗ kèm kiểm chứng bằng mã. Ghi chú hiện ngay cuối mục tương ứng khi đọc chương. |
| **Thư viện hình** | Toàn bộ 17 hình ở một chỗ, bấm để phóng to, có liên kết về đúng mục đã dùng. |
| **Từ điển thuật ngữ** | Hơn 60 thuật ngữ đối chiếu Việt – Anh, lọc tại chỗ. |
| **Mã nguồn** | 7 script Python sinh ra mọi hình và mọi con số, kèm kết quả in ra. Tải về là chạy được. |
| **Toàn văn** | Cả giáo trình trên một trang, tiện Ctrl+F và in ra giấy. |

Ngoài ra: tìm kiếm toàn văn (bấm `/` hoặc `Ctrl`+`K`, **bỏ dấu vẫn tìm được**),
giao diện sáng/tối, mục lục hai bên có bám theo vị trí đọc, và bố cục dùng được trên điện thoại.

## Nguyên tắc với nội dung gốc

`content/quantization.md` là **nguyên văn của tác giả và không bị sửa**. Khi rà soát phát hiện một
phát biểu đúng về ý nhưng rộng hơn mức chứng minh được, repo này **không viết lại lời tác giả** mà
thêm một **ghi chú biên tập** có nhãn rõ ràng, hiện ngay cuối mục đó và gom lại ở trang
`ghi-chu.html`. Cách này giữ được bản gốc, để người đọc tự đối chiếu, và biến chính chỗ chưa chặt
thành một điểm dạy học.

Năm ghi chú hiện có, tất cả đều đã kiểm chứng bằng mã chạy thật:

| Mục | Nội dung ghi chú |
| --- | --- |
| 5.2 | Ràng buộc nằm ở **trục thu gọn**, không ở “activation”: chia nhóm vẫn đặt được scale trên trục ấy, và depthwise convolution là ngoại lệ do cấu trúc. |
| 6.4 | `quantize_multiplier` chỉ đúng với $0 < M \le 1$ (với $M > 1$ thì $M_0$ tràn int32); và hàm dịch phải làm tròn nửa về phía $+\infty$, khác gemmlowp vốn làm tròn nửa ra xa số 0. |
| 6.5 | “Trùng khớp từng bit” là phép **tự đối chiếu** giữa hai cài đặt trong cùng tài liệu, không phải bảo đảm khớp với một backend thật. |
| 11.3 | Quét $\alpha$ trên chính dữ liệu của thí nghiệm cho cực tiểu ở **0,60**, không phải 0,50 — kèm lý do đo được. |
| 10.1.1 | Mốc đối chiếu phiên bản (28-09-2026): PyTorch 2.14, torchao 0.18, NumPy 2.4 đều vẫn là bản mới nhất. |

## Cấu trúc thư mục

```text
content/
  quantization.md      # nguyên văn giáo trình — GIỮ NGUYÊN, không sửa
  ghi-chu.md           # ghi chú biên tập, gắn theo số mục
  chapters.json        # tên ngắn + mô tả từng chương, dùng cho điều hướng và trang chủ
  loi-giai.md          # lời giải 9 bài tập
  trac-nghiem.md       # 48 câu trắc nghiệm theo chương
  thuat-ngu.md         # từ điển thuật ngữ
figs/                  # 17 hình PNG
code/                  # 7 script Python + kết quả chạy
src/
  build.mjs            # dựng site: cắt chương, dựng sổ tra cứu, chèn ghi chú, sinh trang
  markdown.mjs         # markdown-it + KaTeX + tô màu cú pháp + liên kết chéo + gom <figure>
  layout.mjs           # khung HTML dùng chung
  pages/               # bộ dựng cho từng trang chuyên biệt
  assets/              # style.css và các script chạy phía trình duyệt
  test.mjs             # kiểm chứng phần lõi số học
  check-links.mjs      # dò liên kết nội bộ
  serve.mjs            # máy chủ tĩnh tối giản
docs/                  # KẾT QUẢ DỰNG — commit sẵn, cũng là thư mục cho GitHub Pages
```

## Sửa nội dung

Mọi nội dung nằm trong `content/`. Sửa xong chạy `npm run build`.

- **Thêm ghi chú biên tập** → `content/ghi-chu.md`, cú pháp `## <số mục> — <tiêu đề>` rồi phần thân.
  Build sẽ **báo lỗi** nếu ghi chú trỏ tới một mục không có thật.
- **Thêm câu trắc nghiệm** → `content/trac-nghiem.md`:

  ```markdown
  ## Chương 5
  ### Câu hỏi ở đây?
  - [ ] đáp án sai
  - [x] đáp án đúng
  > Giải thích hiện ra sau khi trả lời.
  ```

  Build sẽ báo lỗi nếu có câu nào thiếu đáp án đúng.
- **Thêm thuật ngữ** → `content/thuat-ngu.md`, cú pháp `### Tiếng Việt | English` rồi định nghĩa.
- **Sửa lời giải** → `content/loi-giai.md`, mỗi bài là một `## Bài N` kèm dòng
  `@meta chuong=… | dang=… | kho=…`.
- **Sửa bài giảng** → chỉ khi thật sự cần. Tiêu đề `## N. …` mở một chương mới và phải khớp với
  một mục trong `content/chapters.json`.

## Kiểm thử

```bash
npm run verify     # build + test + check, chạy một lượt
npm test           # 95 phép kiểm tra phần lõi số học
npm run check      # dò 1337 liên kết nội bộ và neo trong docs/
```

`src/test.mjs` chạy lại phần lõi số học của Phòng thí nghiệm bằng **chính các con số đã in trong
giáo trình** — $S = 4/7$ và $Z = 2$ của Hình 4, $Z = 42$ của Bài 1 (làm tròn nửa về số chẵn),
`np.float16(2049) = 2048`, $M_0 = 1111811840$ với $n = 8$ của Mục 6.5, bảng số bit thực tế ở
Mục 11.8, và 26,84 GB KV cache của Bài 9. Có thêm một nhóm kiểm tra riêng cho hai điểm nêu trong
ghi chú Mục 6.4: hành vi khi $M \ge 1$, và chỗ hai quy ước làm tròn lệch nhau ở giá trị âm.

`code/sweep_alpha.py` dựng lại đúng trạng thái ngẫu nhiên của thí nghiệm ở Mục 11.3, nên nó tái
lập chính xác hai mốc **7,7435%** và **1,4430%** đã in trong giáo trình — đó là bằng chứng rằng
phần quét $\alpha$ trong lời giải Bài 8 đo trên cùng dữ liệu chứ không phải một mô phỏng khác.

## Đăng lên GitHub Pages

Repo đã commit sẵn thư mục `docs/` và tệp `.nojekyll`. Vào **Settings → Pages**, chọn
nguồn là nhánh `main` và thư mục `/docs`.

## Ghi chú về nội dung

Giáo trình cùng 17 hình và 6 script gốc trong `content/`, `figs/`, `code/` là tài liệu của tác giả;
repo này dựng giao diện đọc quanh chúng. Phần **lời giải bài tập**, **câu trắc nghiệm**,
**từ điển thuật ngữ**, **ghi chú biên tập**, **`code/sweep_alpha.py`** và **các công cụ tương tác**
được viết thêm cho repo này.

Hai bài tập yêu cầu chạy lại thí nghiệm mà môi trường ở đây chưa có đủ thư viện (Bài 6 cần
scikit-learn + PyTorch, Bài 7 cần PyTorch) nên lời giải ghi rõ đâu là **phân tích dự đoán** và đâu
là cách kiểm chứng, chứ không đưa ra số liệu chưa đo. Bài 8 thì đã chạy thật.
