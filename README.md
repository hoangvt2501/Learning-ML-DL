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
| **Bài tập** | 9 bài của giáo trình, mỗi bài kèm **lời giải chi tiết** ẩn sẵn, cộng phần luyện tính tay tự chấm và 42 câu trắc nghiệm theo chương. |
| **Phòng thí nghiệm** | 5 công cụ chạy trong trình duyệt: máy lượng tử affine, đánh đổi làm tròn – cắt, soi bit số thực, requantization dấu chấm tĩnh, tính dung lượng mô hình và KV cache. |
| **Thư viện hình** | Toàn bộ 17 hình ở một chỗ, bấm để phóng to, có liên kết về đúng mục đã dùng. |
| **Từ điển thuật ngữ** | Hơn 60 thuật ngữ đối chiếu Việt – Anh, lọc tại chỗ. |
| **Mã nguồn** | 6 script Python sinh ra mọi hình và mọi con số, kèm kết quả in ra. Tải về là chạy được. |
| **Toàn văn** | Cả giáo trình trên một trang, tiện Ctrl+F và in ra giấy. |

Ngoài ra: tìm kiếm toàn văn (bấm `/` hoặc `Ctrl`+`K`, **bỏ dấu vẫn tìm được**),
giao diện sáng/tối, mục lục hai bên có bám theo vị trí đọc, và bố cục dùng được trên điện thoại.

## Cấu trúc thư mục

```text
content/
  quantization.md      # nguyên văn giáo trình — nguồn duy nhất, không sửa khi dựng site
  chapters.json        # tên ngắn + mô tả từng chương, dùng cho điều hướng và trang chủ
  loi-giai.md          # lời giải 9 bài tập
  trac-nghiem.md       # 42 câu trắc nghiệm theo chương
  thuat-ngu.md         # từ điển thuật ngữ
figs/                  # 17 hình PNG
code/                  # 6 script Python + kết quả chạy
src/
  build.mjs            # dựng site: cắt chương, dựng sổ tra cứu, sinh trang
  markdown.mjs         # markdown-it + KaTeX + tô màu cú pháp + liên kết chéo + gom <figure>
  layout.mjs           # khung HTML dùng chung
  pages/               # bộ dựng cho từng trang chuyên biệt
  assets/              # style.css và các script chạy phía trình duyệt
  test.mjs             # kiểm chứng phần lõi số học
  serve.mjs            # máy chủ tĩnh tối giản
docs/                  # KẾT QUẢ DỰNG — commit sẵn, cũng là thư mục cho GitHub Pages
```

## Sửa nội dung

Mọi nội dung nằm trong `content/`. Sửa xong chạy `npm run build`.

- **Sửa bài giảng** → `content/quantization.md`. Tiêu đề `## N. …` mở một chương mới và phải
  khớp với một mục trong `content/chapters.json`.
- **Thêm câu trắc nghiệm** → `content/trac-nghiem.md`, cú pháp:

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

## Kiểm thử

```bash
npm run verify     # build + test + check, chạy một lượt
npm test           # 77 phép kiểm tra phần lõi số học
npm run check      # dò 1194 liên kết nội bộ và neo trong docs/
```

`src/test.mjs` chạy lại phần lõi số học của Phòng thí nghiệm bằng **chính các con số đã in trong
giáo trình** — $S = 4/7$ và $Z = 2$ của Hình 4, $Z = 42$ của Bài 1 (làm tròn nửa về số chẵn),
`np.float16(2049) = 2048`, $M_0 = 1111811840$ với $n = 8$ của Mục 6.5, bảng số bit thực tế ở
Mục 11.8, và 26,84 GB KV cache của Bài 9. Nếu có ai sửa công thức trong `playground.js` mà làm
lệch khỏi giáo trình, bộ kiểm tra này sẽ báo ngay.

## Đăng lên GitHub Pages

Repo đã commit sẵn thư mục `docs/` và tệp `.nojekyll`. Vào **Settings → Pages**, chọn
nguồn là nhánh `main` và thư mục `/docs`.

## Ghi chú về nội dung

Giáo trình và toàn bộ hình, mã nguồn trong `content/`, `figs/`, `code/` là tài liệu gốc do tác giả
cung cấp; repo này chỉ dựng giao diện đọc quanh chúng. Phần **lời giải bài tập**, **câu trắc
nghiệm**, **từ điển thuật ngữ** và **các công cụ tương tác** được viết thêm cho repo này.

Ba bài tập yêu cầu chạy lại thí nghiệm (Bài 6, 7, 8) nên lời giải ghi rõ đâu là **phân tích dự
đoán** và đâu là cách kiểm chứng, chứ không đưa ra số liệu chưa đo.
