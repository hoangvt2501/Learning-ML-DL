# Hai giáo trình tự học: Quantization và MLOps

Một site tĩnh chứa hai giáo trình viết theo cùng một lối — *động cơ → định nghĩa → suy luận →
ví dụ số → thí nghiệm kiểm chứng* — kèm bài tập có lời giải, ngân hàng trắc nghiệm và công cụ
tương tác.

Mở `docs/index.html` bằng trình duyệt là đọc được ngay: không cần máy chủ, không cần mạng
(trừ font chữ, có sẵn phương án dự phòng).

## Xem ngay

```bash
npm install     # chỉ cần cho việc dựng lại site
npm run dev     # dựng lại rồi phục vụ tại http://localhost:4173
```

## Hai giáo trình

### 1 · Quantization trong Deep Learning

Từ công thức $S$ và $Z$ cho tới GPTQ, NF4 và KV cache. 17 chương, **17 hình sinh bằng mã**,
9 bài tập có lời giải, 48 câu trắc nghiệm, 5 công cụ tương tác.

Nguyên văn giáo trình do tác giả cung cấp và **được giữ nguyên**. Năm chỗ phát biểu rộng hơn
mức chứng minh được đã được bổ sung **ghi chú biên tập** có nhãn rõ ràng, chèn ngay cuối mục
tương ứng và gom ở trang `ghi-chu.html` — xem mục [Ghi chú biên tập](#ghi-chú-biên-tập).

### 2 · MLOps: đưa mô hình ra sản xuất và giữ cho nó sống

Vì sao mô hình tốt vẫn chết trong sản xuất, và phải dựng những gì quanh nó để nó sống.
18 chương, **14 hình sinh bằng mã**, 10 bài tập có lời giải, 48 câu trắc nghiệm, 4 công cụ
tương tác, và một chương riêng để **ôn phỏng vấn**.

Nội dung dựa trên nguồn gốc, và mỗi khẳng định đều ghi rõ nó đến từ đâu:

| Nguồn | Đóng góp vào tài liệu |
| --- | --- |
| Sculley và cộng sự, *Hidden Technical Debt in ML Systems* (NeurIPS 2015) | CACE, mười ba dạng nợ có tên, con số 5% / 95% mã keo, prediction bias / action limits / up-stream producers |
| Breck và cộng sự, *The ML Test Score* (2017) | 28 mục kiểm thử, cách tính điểm lấy **min** của bốn nhóm, bảng diễn giải |
| Google Cloud, *MLOps: CD and automation pipelines* | ba mức tự động hoá, tám bước pipeline, định nghĩa training–serving skew |
| Shankar và cộng sự, *Operationalizing ML: An Interview Study* (2022) | ba chữ V, phổ lỗi cứng → mềm → dịch chuyển, bốn nỗi đau và bốn anti-pattern |
| Huyen, *Data Distribution Shifts and Monitoring* | định nghĩa ba loại dịch chuyển, degenerate feedback loop |
| Yurdakul & Naranjo; du Pisanie & Visagie | phê phán ngưỡng PSI cố định |

Phần còn lại là **số liệu đo được trong chính repo này**, sinh bởi `code/mlops/`.

## Những con số đáng nhớ nhất, và chúng đến từ đâu

| Khẳng định | Nguồn |
| --- | --- |
| Concept drift làm độ chính xác rơi 75,6% → **26,9%** trong khi kiểm định KS trên đặc trưng cho $p = 0{,}48$ | `experiments.py` (D) |
| PSI kỳ vọng khi **không có dịch chuyển** là $2(k-1)/n$; ở $n=50$ quy tắc PSI > 0,25 báo động **73% số lần** | `experiments.py` (E), (F) |
| Với 100 nhánh song song, **63,4%** số yêu cầu chạm ít nhất một nhánh vượt p99 | `experiments.py` (A) |
| Bắt cải thiện tương đối 1% trên nền 5% cần **3 triệu mẫu mỗi nhánh** | `experiments.py` (C) |
| Thêm một đặc trưng làm chỉ số gộp **+19,8 điểm** nhưng nhóm 15% người dùng mới **−11,6 điểm** | `experiments.py` (B) |
| Không huấn luyện lại một năm mất **13,19 điểm** độ chính xác; từ hằng quý lên hằng tuần chỉ thêm **0,91 điểm** | `experiments.py` (G) |
| Không ngẫu nhiên hoá thì hệ gợi ý vĩnh viễn chỉ nhìn thấy **9,5%** danh mục | `experiments.py` (H) |

## Có gì trên site

| Trang | Nội dung |
| --- | --- |
| **Chương** | Nguyên văn, công thức dựng bằng KaTeX, mọi cụm “Mục 6.2”, “Chương 11”, “Hình 7” thành liên kết bấm được. Cuối mỗi chương có phần tự kiểm tra. |
| **Bài tập** | Mỗi bài kèm **lời giải chi tiết** ẩn sẵn, cộng ngân hàng trắc nghiệm theo chương. Sách 1 có thêm phần luyện tính tay tự chấm. |
| **Phòng thí nghiệm** | Sách 1: máy lượng tử affine, đánh đổi làm tròn – cắt, soi bit số thực, requantization, tính dung lượng. Sách 2: cỡ mẫu A/B có CUPED, hiệu chuẩn ngưỡng PSI, đuôi độ trễ khi toả nhánh, tự chấm ML Test Score. |
| **Thư viện hình** | Toàn bộ hình ở một chỗ, bấm để phóng to, có liên kết về đúng mục đã dùng. |
| **Từ điển thuật ngữ** | Hơn 60 thuật ngữ mỗi sách, đối chiếu Việt – Anh, lọc tại chỗ. |
| **Mã nguồn** | Mọi script sinh hình và số liệu, kèm kết quả in ra. Tải về là chạy được. |
| **Toàn văn** | Cả giáo trình trên một trang, tiện Ctrl+F và in ra giấy. |

Ngoài ra: tìm kiếm toàn văn qua cả hai giáo trình (bấm `/` hoặc `Ctrl`+`K`, **bỏ dấu vẫn tìm
được**), giao diện sáng/tối, mục lục hai bên bám theo vị trí đọc, và bố cục dùng được trên
điện thoại.

## Ghi chú biên tập

Với giáo trình 1, `content/quantization.md` là **nguyên văn của tác giả và không bị sửa**.
Khi rà soát phát hiện một phát biểu đúng về ý nhưng rộng hơn mức chứng minh được, repo này
không viết lại lời tác giả mà thêm một ghi chú có nhãn, hiện ngay cuối mục đó:

| Mục | Nội dung ghi chú |
| --- | --- |
| 5.2 | Ràng buộc nằm ở **trục thu gọn**, không ở “activation”: chia nhóm vẫn đặt được scale trên trục ấy, và depthwise convolution là ngoại lệ do cấu trúc. |
| 6.4 | `quantize_multiplier` chỉ đúng với $0 < M \le 1$; hàm dịch phải làm tròn nửa về $+\infty$, khác gemmlowp vốn làm tròn nửa ra xa số 0. |
| 6.5 | “Trùng khớp từng bit” là phép **tự đối chiếu** giữa hai cài đặt trong cùng tài liệu, không phải bảo đảm khớp với một backend thật. |
| 11.3 | Quét $\alpha$ trên chính dữ liệu của thí nghiệm cho cực tiểu ở **0,60**, không phải 0,50. |
| 10.1.1 | Mốc đối chiếu phiên bản (28-09-2026): PyTorch 2.14, torchao 0.18, NumPy 2.4 đều vẫn là bản mới nhất. |

## Cấu trúc thư mục

```text
content/
  books.json               # khai báo các giáo trình: tệp nguồn, tiền tố URL, công cụ đi kèm
  quantization.md          # nguyên văn giáo trình 1 — GIỮ NGUYÊN, không sửa
  mlops.md                 # giáo trình 2
  chapters.json            # metadata chương của sách 1
  mlops-chapters.json      # metadata chương của sách 2
  ghi-chu.md               # ghi chú biên tập cho sách 1
  {,mlops-}loi-giai.md     # lời giải bài tập
  {,mlops-}trac-nghiem.md  # ngân hàng trắc nghiệm
  {,mlops-}thuat-ngu.md    # từ điển thuật ngữ
  runbook-*.md             # hướng dẫn chạy lại thí nghiệm
  scripts-*.json           # mô tả từng script cho trang Mã nguồn
figs/                      # 31 hình PNG của cả hai giáo trình
code/                      # 7 script của sách 1
code/mlops/                # 2 script của sách 2
src/
  build.mjs                # dựng site cho mọi giáo trình khai báo trong books.json
  markdown.mjs             # markdown-it + KaTeX + tô màu cú pháp + liên kết chéo + gom <figure>
  layout.mjs               # khung HTML dùng chung, có bộ chuyển giáo trình
  pages/                   # bộ dựng cho từng loại trang
  assets/                  # style.css và các script chạy phía trình duyệt
  test.mjs                 # kiểm chứng phần lõi số học của cả hai phòng thí nghiệm
  check-links.mjs          # dò liên kết nội bộ
  serve.mjs                # máy chủ tĩnh tối giản
docs/                      # KẾT QUẢ DỰNG — commit sẵn, cũng là thư mục cho GitHub Pages
```

## Thêm một giáo trình nữa

Khai báo một mục trong `content/books.json` rồi đặt tệp nguồn vào `content/`. Build tự bỏ qua
giáo trình chưa có tệp nguồn, nên viết dở vẫn dựng được. Mỗi giáo trình có sổ tra cứu mục và
hình **riêng**, nên hai sách dùng trùng số mục (cả hai đều có “Mục 5.2”) vẫn không lẫn nhau.

## Sửa nội dung

- **Thêm câu trắc nghiệm** → `content/{,mlops-}trac-nghiem.md`:

  ```markdown
  ## Chương 5
  ### Câu hỏi ở đây?
  - [ ] đáp án sai
  - [x] đáp án đúng
  > Giải thích hiện ra sau khi trả lời.
  ```

  Build **báo lỗi** nếu có câu nào thiếu đáp án đúng.
- **Thêm thuật ngữ** → `content/{,mlops-}thuat-ngu.md`, cú pháp `### Tiếng Việt | English`.
- **Sửa lời giải** → `content/{,mlops-}loi-giai.md`, mỗi bài là `## Bài N` kèm dòng
  `@meta chuong=… | dang=… | kho=…`.
- **Thêm ghi chú biên tập** → `content/ghi-chu.md`, cú pháp `## <số mục> — <tiêu đề>`.
  Build báo lỗi nếu ghi chú trỏ tới một mục không có thật.

> **Lưu ý:** mọi bộ phân tích trong `src/` tách đoạn bằng `\n{2,}`, nên `read()` trong
> `build.mjs` chuẩn hoá xuống dòng về LF trước khi phân tích. Không có bước ấy thì một tệp
> lưu kiểu CRLF sẽ lặng lẽ không tách được đoạn nào.

## Kiểm thử

```bash
npm run verify     # build + test + check, chạy một lượt
npm test           # 126 phép kiểm tra phần lõi số học của cả hai sách
npm run check      # dò 2620 liên kết nội bộ và neo trên 47 trang
```

`src/test.mjs` đối chiếu phần lõi số học với **chính các con số đã in trong hai giáo trình**:

- Sách 1 — $S = 4/7$ và $Z = 2$ của Hình 4, $Z = 42$ của Bài 1 (làm tròn nửa về số chẵn),
  `np.float16(2049) = 2048`, $M_0 = 1111811840$ với $n = 8$, bảng số bit thực tế ở Mục 11.8,
  và 26,84 GB KV cache của Bài 9.
- Sách 2 — cỡ mẫu A/B (2 996 694 cho lift 1% trên nền 5%), $\mathbb{E}[\mathrm{PSI}] = 2(k-1)/n$,
  tỉ lệ báo động giả $P(\mathrm{PSI} > 0{,}10) = 0{,}00297$, p99 của max-of-$k$ (169,5 ms với
  $k=10$; 261,9 ms với $k=100$), và quy tắc lấy **min** của ML Test Score.

Nếu ai sửa công thức trong `playground.js` hoặc `lab-mlops.js` mà làm lệch khỏi giáo trình,
bộ kiểm tra này báo ngay.

## Đăng lên GitHub Pages

Repo đã commit sẵn `docs/` và `.nojekyll`. Vào **Settings → Pages**, chọn nguồn là nhánh `main`
và thư mục `/docs`.

## Ghi chú về nội dung

Giáo trình 1 cùng 17 hình và 6 script gốc là tài liệu của tác giả; repo này dựng giao diện đọc
quanh chúng. Mọi thứ còn lại — **giáo trình 2 (MLOps)**, lời giải bài tập, câu trắc nghiệm,
từ điển thuật ngữ, ghi chú biên tập, `code/sweep_alpha.py`, `code/mlops/` và các công cụ tương
tác — được viết thêm cho repo này.

Số liệu đo được trong `code/mlops/` đến từ **dữ liệu mô phỏng**, thiết kế để cô lập đúng một
cơ chế mỗi lần. Chúng chứng minh *cơ chế* tồn tại và có độ lớn đáng kể, không dùng để suy ra
con số cho một hệ thống cụ thể nào.
