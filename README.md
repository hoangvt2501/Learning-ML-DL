# Bốn giáo trình tự học, đọc như một mạch

Một site tĩnh chứa bốn giáo trình viết theo cùng một lối — *động cơ → định nghĩa → suy luận →
ví dụ số → thí nghiệm kiểm chứng* — kèm bài tập có lời giải, ngân hàng trắc nghiệm và công cụ
tương tác.

Mở `docs/index.html` bằng trình duyệt là đọc được ngay: không cần máy chủ, không cần mạng
(trừ font chữ, có sẵn phương án dự phòng).

## Xem ngay

```bash
npm install     # chỉ cần cho việc dựng lại site
npm run dev     # dựng lại rồi phục vụ tại http://localhost:4173
```

## Bốn giáo trình, một mạch

Bốn giáo trình được viết để **đọc nối nhau**, và trang `lo-trinh.html` xếp toàn bộ **58 bài**
của chúng thành một dãy đánh số liên tục:

| Phần | Giáo trình | Trả lời câu hỏi |
| --- | --- | --- |
| I | **Nền tảng** | học là gì, và tối ưu thế nào |
| II | **Mô hình & Kiến trúc** | thay mô hình tuyến tính bằng mạng sâu thì được gì |
| III | **Quantization** | làm mô hình chạy được trên phần cứng thật |
| IV | **MLOps** | giữ cho nó sống trong sản xuất |

Mỗi phần tự đứng vững được; thứ tự trên là thứ tự **ít phải quay lại nhất**. Tên tệp của từng
giáo trình giữ nguyên tiền tố riêng, nên thêm phần mới không làm hỏng liên kết cũ.


### 1 · Nền tảng Machine Learning: từ đại số tuyến tính tới SVM

Phần đứng trước mọi thứ còn lại. 21 chương, **13 hình sinh bằng mã**, 10 bài tập có lời giải,
54 câu trắc nghiệm, 111 mục tra cứu, và một chương riêng để **ôn phỏng vấn**.

Khác với ba giáo trình kia, tài liệu này không trích con số nào từ bài báo — mọi số đều sinh
tại chỗ. Nhưng chúng chia làm hai loại rạch ròi, và loại thứ nhất là **kiểm chứng một đẳng
thức**, phải khớp tới sai số máy:

| Đẳng thức | Sai số đo được |
| --- | --- |
| Hồi quy logistic ≡ softmax với $K=2$ | $1{,}6\times10^{-15}$ |
| PCA qua hiệp phương sai ≡ PCA qua SVD | $7{,}1\times10^{-15}$ |
| Eckart–Young: sai số tái dựng $=(n-1)\sum_{i>k}\lambda_i$ | $\sim10^{-13}$ |
| Ridge qua SVD ≡ ridge dạng đóng | $2{,}3\times10^{-15}$ |
| Hàm quyết định của LDA là tuyến tính | $4{,}4\times10^{-15}$ |
| SVM: $w=\sum_i\alpha_iy_ix_i$ dựng từ nghiệm đối ngẫu | $0$ |
| SVM: khe đối ngẫu bằng 0 | $4{,}0\times10^{-8}$ |
| Ridge dạng đóng ≡ MAP tối ưu bằng BFGS | $2{,}0\times10^{-8}$ |

Bốn dòng cuối không đạt sai số máy vì bị giới hạn bởi dung sai của bộ giải (libsvm, BFGS),
và tài liệu in ra con số thật thay vì làm tròn cho đẹp.

### 2 · Quantization trong Deep Learning

Từ công thức $S$ và $Z$ cho tới GPTQ, NF4 và KV cache. 17 chương, **17 hình sinh bằng mã**,
9 bài tập có lời giải, 48 câu trắc nghiệm, 5 công cụ tương tác.

Nguyên văn giáo trình do tác giả cung cấp và **được giữ nguyên**. Năm chỗ phát biểu rộng hơn
mức chứng minh được đã được bổ sung **ghi chú biên tập** có nhãn rõ ràng, chèn ngay cuối mục
tương ứng và gom ở trang `ghi-chu.html` — xem mục [Ghi chú biên tập](#ghi-chú-biên-tập).

### 3 · MLOps: đưa mô hình ra sản xuất và giữ cho nó sống

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

### 4 · Mô hình và kiến trúc: từ cây quyết định tới Transformer hiện đại

Từ đánh đổi thiên lệch–phương sai tới RoPE, GQA và SwiGLU. 17 chương, **14 hình sinh bằng mã**,
10 bài tập có lời giải, 43 câu trắc nghiệm, và một chương riêng để **ôn phỏng vấn**.

Mỗi cơ chế được trình bày theo cùng ba bước: vấn đề nó sinh ra để chữa, cơ chế kèm một con số
đo được, và cái giá phải trả. Bốn kết quả ở Chương 12 không phải mô phỏng mà là **đối chiếu với
số đã công bố**:

| Khẳng định | Đối chiếu |
| --- | --- |
| $N pprox 12Ld^2$ cho GPT-2 small, medium, large | khớp **chính xác tới từng tham số** với 124 439 808 / 354 823 168 / 774 030 080 |
| Công thức tham số kiểu Llama (SwiGLU ba ma trận, GQA, RMSNorm) | khớp chính xác với 6 738 415 616 (7B) và 13 015 864 320 (13B) |
| $C pprox 6ND$ trên 2 nghìn tỉ token | 184 011 giờ-GPU so với **184 320** Meta công bố — lệch **0,17%** |
| $\operatorname{Var}(q \cdot k) = d_k$, chú thích 4 bài báo Transformer | đo được 1021,97 khi $d_k = 1024$ |

Phần còn lại là **số liệu đo được trong chính repo này**, sinh bởi `code/models/`.

## Những con số đáng nhớ nhất, và chúng đến từ đâu

| Khẳng định | Nguồn |
| --- | --- |
| Concept drift làm độ chính xác rơi 75,6% → **26,9%** trong khi kiểm định KS trên đặc trưng cho $p = 0{,}48$ | `experiments.py` (D) |
| PSI kỳ vọng khi **không có dịch chuyển** là $2(k-1)/n$; ở $n=50$ quy tắc PSI > 0,25 báo động **73% số lần** | `experiments.py` (E), (F) |
| Với 100 nhánh song song, **63,4%** số yêu cầu chạm ít nhất một nhánh vượt p99 | `experiments.py` (A) |
| Bắt cải thiện tương đối 1% trên nền 5% cần **3 triệu mẫu mỗi nhánh** | `experiments.py` (C) |
| Thêm một đặc trưng làm chỉ số gộp **+19,8 điểm** nhưng nhóm 15% người dùng mới **−11,6 điểm** | `experiments.py` (B) |
| Không huấn luyện lại một năm mất **13,19 điểm** độ chính xác; từ hằng quý lên hằng tuần chỉ thêm **0,91 điểm** | `experiments.py` (G) |
| Không ngẫu nhiên hoá thì hệ gợi ý vĩnh viễn chỉ nhìn thấy **9,5%** danh mục | `mlops/experiments.py` (H) |
| Bộ phân loại **đoán tất cả là âm** đạt độ chính xác **0,9902**, cao hơn bộ dùng được (0,9900) | `nentang/experiments.py` (G) |
| Cùng mô hình ấy: ROC-AUC **0,9715** nhưng PR-AUC chỉ **0,4931** | `nentang/experiments.py` (G) |
| PCA giữ **96,1%** phương sai mà cho AUC **0,502** — vô dụng cho phân loại; LDA cho 0,9987 | `nentang/experiments.py` (K) |
| Cộng tuyến làm **hệ số** dao động gấp 25 lần trong khi **dự báo** không đổi | `nentang/experiments.py` (A) |
| Phân rã ma trận dưới ~20 đánh giá mỗi người còn **tệ hơn đoán bừa**; trên ngưỡng thì RMSE giảm 11 lần | `nentang/experiments.py` (M) |
| K-means khởi tạo ngẫu nhiên kẹt ở nghiệm tồi **71,5%** số lần; k-means++ hạ xuống 46,5% | `nentang/experiments.py` (L) |
| Kiểm tra đạo hàm: $\varepsilon$ tối ưu là $u^{1/3}$, và $\varepsilon=10^{-13}$ tệ hơn $10^{-6}$ bảy bậc | `nentang/experiments.py` (C) |
| Cùng mạng 40 lớp, đổi hệ số khởi tạo từ 0,5 sang 2,0 làm gradient đi từ $10^{-18}$ tới $10^{6}$ | `models/experiments.py` (D) |
| **Kết nối tắt một mình làm bùng nổ** ($2{,}4 	imes 10^{8}$); phải kèm chuẩn hoá mới ổn định (2,22) | `models/experiments.py` (D) |
| Không chia $\sqrt{d_k}$ thì entropy attention rơi còn **0,118 nat** trên tối đa 4,159 | `models/experiments.py` (E) |
| Mạng sâu $k$ lớp cần $6k$ tham số; mạng một lớp cần bề rộng **đúng bằng $2^k$** | `models/experiments.py` (C) |
| LSTM: độ lệch cổng quên 1 so với 4 làm gradient sau 100 bước chênh **12,8 bậc độ lớn** | `models/experiments.py` (F) |

## Có gì trên site

| Trang | Nội dung |
| --- | --- |
| **Chương** | Nguyên văn, công thức dựng bằng KaTeX, mọi cụm “Mục 6.2”, “Chương 11”, “Hình 7” thành liên kết bấm được. Cuối mỗi chương có phần tự kiểm tra. |
| **Bài tập** | Mỗi bài kèm **lời giải chi tiết** ẩn sẵn, cộng ngân hàng trắc nghiệm theo chương. Sách 1 có thêm phần luyện tính tay tự chấm. |
| **Phòng thí nghiệm** | Sách 1: máy lượng tử affine, đánh đổi làm tròn – cắt, soi bit số thực, requantization, tính dung lượng. Sách 2: cỡ mẫu A/B có CUPED, hiệu chuẩn ngưỡng PSI, đuôi độ trễ khi toả nhánh, tự chấm ML Test Score. Sách 3 không có công cụ tương tác — phần tính toán của nó nằm ở Chương 12 và các bài tập. |
| **Thư viện hình** | Toàn bộ hình ở một chỗ, bấm để phóng to, có liên kết về đúng mục đã dùng. |
| **Từ điển thuật ngữ** | Hơn 60 thuật ngữ mỗi sách, đối chiếu Việt – Anh, lọc tại chỗ. |
| **Ôn phỏng vấn** | Sách 1, 3 và 4 có chương riêng: khung trả lời, câu hỏi theo nhóm kèm con số để dẫn ra, và bảng những câu trả lời tự tố cáo. |
| **Lộ trình** | `lo-trinh.html` gom toàn bộ 58 bài của bốn giáo trình thành một dãy đánh số liên tục, kèm sơ đồ mối nối giữa các phần. |
| **Mã nguồn** | Mọi script sinh hình và số liệu, kèm kết quả in ra. Tải về là chạy được. |
| **Toàn văn** | Cả giáo trình trên một trang, tiện Ctrl+F và in ra giấy. |

Ngoài ra: tìm kiếm toàn văn qua cả bốn giáo trình (bấm `/` hoặc `Ctrl`+`K`, **bỏ dấu vẫn tìm
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
  nentang.md               # giáo trình Nền tảng (phần I)
  models.md                # giáo trình Mô hình & Kiến trúc (phần II)
  quantization.md          # nguyên văn giáo trình Quantization — GIỮ NGUYÊN, không sửa
  mlops.md                 # giáo trình MLOps (phần IV)
  chapters.json            # metadata chương của Quantization
  {nentang,models,mlops}-chapters.json   # metadata chương của ba sách còn lại
  ghi-chu.md               # ghi chú biên tập cho Quantization
  {,nentang-,models-,mlops-}loi-giai.md     # lời giải bài tập
  {,nentang-,models-,mlops-}trac-nghiem.md  # ngân hàng trắc nghiệm
  {,nentang-,models-,mlops-}thuat-ngu.md    # từ điển thuật ngữ
  runbook-*.md             # hướng dẫn chạy lại thí nghiệm
  scripts-*.json           # mô tả từng script cho trang Mã nguồn
figs/                      # 58 hình PNG của cả bốn giáo trình
code/                      # 7 script của Quantization
code/nentang/              # 2 script của Nền tảng
code/models/               # 2 script của Mô hình & Kiến trúc
code/mlops/                # 2 script của MLOps
src/
  build.mjs                # dựng site cho mọi giáo trình khai báo trong books.json
  markdown.mjs             # markdown-it + KaTeX + tô màu cú pháp + liên kết chéo + gom <figure>
  layout.mjs               # khung HTML dùng chung, có bộ chuyển giáo trình
  pages/                   # bộ dựng cho từng loại trang, kể cả curriculum.mjs (Lộ trình)
  assets/                  # style.css và các script chạy phía trình duyệt
  test.mjs                 # kiểm chứng phần lõi số học của cả bốn giáo trình
  check-links.mjs          # dò liên kết nội bộ
  serve.mjs                # máy chủ tĩnh tối giản
docs/                      # KẾT QUẢ DỰNG — commit sẵn, cũng là thư mục cho GitHub Pages
```

## Thêm một giáo trình nữa

Khai báo một mục trong `content/books.json` rồi đặt tệp nguồn vào `content/`. Build tự bỏ qua
giáo trình chưa có tệp nguồn, nên viết dở vẫn dựng được. Mỗi giáo trình có sổ tra cứu mục và
hình **riêng**, nên các sách dùng trùng số mục (cả ba đều có “Mục 5.2”) vẫn không lẫn nhau.

Tiêu đề trang chủ, câu mở đầu và phụ đề trên thanh tiêu đề đều **sinh ra từ `books.json`**,
nên thêm sách không phải sửa chỗ nào trong `src/`. Chương bài tập cần `"kind": "exercises"`
trong tệp `*-chapters.json`; thiếu nó thì trang bài tập không được dựng. Mục nào trong tệp
nguồn không có mục tương ứng trong `*-chapters.json` sẽ **bị bỏ qua không báo lỗi** — dễ mất
phần Phụ lục, nên nhớ khai báo nó với `"num": "PL"`.

Trang **Lộ trình** (`src/pages/curriculum.mjs`) tự gom mọi giáo trình theo thứ tự trong
`books.json` và đánh số bài liên tục, nên thêm một phần mới chỉ cần thêm một mục vào tệp ấy.
Các chương có nhãn *Bài tập*, *Ôn phỏng vấn*, *Tài liệu tham khảo* và mục `"num": "PL"` được
xếp riêng vì chúng là phần tra cứu chứ không phải bài học.

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
npm test           # 222 phép kiểm tra phần lõi số học của cả bốn sách
npm run check      # dò 5704 liên kết nội bộ và neo trên 94 trang
```

`src/test.mjs` đối chiếu phần lõi số học với **chính các con số đã in trong bốn giáo trình**:

- Sách 1 — $S = 4/7$ và $Z = 2$ của Hình 4, $Z = 42$ của Bài 1 (làm tròn nửa về số chẵn),
  `np.float16(2049) = 2048`, $M_0 = 1111811840$ với $n = 8$, bảng số bit thực tế ở Mục 11.8,
  và 26,84 GB KV cache của Bài 9.
- Sách 2 — cỡ mẫu A/B (2 996 694 cho lift 1% trên nền 5%), $\mathbb{E}[\mathrm{PSI}] = 2(k-1)/n$,
  tỉ lệ báo động giả $P(\mathrm{PSI} > 0{,}10) = 0{,}00297$, p99 của max-of-$k$ (169,5 ms với
  $k=10$; 261,9 ms với $k=100$), và quy tắc lấy **min** của ML Test Score.
- **Nền tảng** — nghiệm phương trình chuẩn tắc và ridge của Bài 1, điểm tối ưu $\varepsilon^{*}\sim u^{1/3}$
  khi kiểm tra đạo hàm, bậc $O(\kappa)$ so với $O(\sqrt{\kappa})$ của số vòng lặp, chặn Novikoff,
  các chỉ số từ ma trận nhầm lẫn của Bài 6 (kể cả việc bộ đoán tất cả là âm có độ chính xác
  **cao hơn**), nghiệm SVM hai điểm cùng khe đối ngẫu bằng 0, Eckart–Young, và quy mô của
  phân rã ma trận ở Bài 10.
- **Mô hình & Kiến trúc** — số tham số của GPT-2 (ba kích thước) và Llama-2 (7B, 13B) khớp **chính xác** với
  con số đã công bố, $d_{	ext{ff}} = 11008$ đúng bằng $	frac{8}{3}d$ làm tròn lên bội 256,
  ngân sách 184 011 giờ-GPU so với 184 320 đã công bố, KV cache 80 GiB so với 10 GiB của GQA,
  ngưỡng $T > 6d$, và các con số tính tay của Bài 4, 6 và 7.

Nếu ai sửa công thức trong `playground.js` hoặc `lab-mlops.js` mà làm lệch khỏi giáo trình,
bộ kiểm tra này báo ngay. Với hai sách viết thêm gần đây thì các phép kiểm tra chạy thẳng trên
công thức viết lại trong `test.mjs`, nên chúng bắt được cả lỗi trong chính công thức in trong bài.

## Đăng lên GitHub Pages

Repo đã commit sẵn `docs/` và `.nojekyll`. Vào **Settings → Pages**, chọn nguồn là nhánh `main`
và thư mục `/docs`.

## Ghi chú về nội dung

Giáo trình **Quantization** cùng 17 hình và 6 script gốc là tài liệu của tác giả; repo này dựng
giao diện đọc quanh chúng. Mọi thứ còn lại — **Nền tảng**, **Mô hình & Kiến trúc**, **MLOps**,
lời giải bài tập, câu trắc nghiệm, từ điển thuật ngữ, ghi chú biên tập, `code/sweep_alpha.py`,
`code/nentang/`, `code/models/`, `code/mlops/` và các công cụ tương tác — được viết thêm cho
repo này.

Hai tài liệu tiếng Việt được nêu ở Chương 19 của phần Nền tảng —
[Machine Learning cơ bản](https://machinelearningcoban.com/) của Vũ Hữu Tiệp và
[Deep AI KhanhBlog](https://phamdinhkhanh.github.io/) của Phạm Đình Khánh — là tài liệu đọc
thêm. Repo này **không sao chép** nội dung của chúng; cách tổ chức theo phần có tham khảo bản
ebook của tài liệu thứ nhất, còn chữ nghĩa và mọi con số đều tự viết và tự kiểm chứng. Riêng
ebook ấy ghi rõ yêu cầu xin phép tác giả trước khi sao chép hay in ấn.

Số liệu đo được trong `code/mlops/`, `code/nentang/` và phần lớn `code/models/` đến từ
**dữ liệu mô phỏng**, thiết kế để cô lập đúng một cơ chế mỗi lần. Chúng chứng minh *cơ chế* tồn tại và có độ lớn
đáng kể, không dùng để suy ra con số cho một hệ thống cụ thể nào.

Có hai loại ngoại lệ. Thứ nhất là bốn kết quả ở Chương 12 của **Mô hình & Kiến trúc** — số
tham số GPT-2, số tham số Llama-2, ngân sách giờ-GPU của Llama-2 7B, và phương sai tích vô
hướng — được **đối chiếu với con số đã công bố**. Thứ hai là mười đẳng thức ở phần **Nền tảng**
liệt kê trong bảng ở trên, vốn phải khớp tới sai số máy chứ không phải khớp xấp xỉ. Cả hai loại
kiểm tra rằng các công thức **đúng** chứ không chỉ hợp lý.
