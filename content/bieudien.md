# Biểu diễn, Sinh và Căn chỉnh: cách một trợ lý ngôn ngữ ra đời

> **Giáo trình tự học, viết theo lối bài giảng.** Khác ba giáo trình trước một chút về nhịp: ở đây tôi cố ý viết dài hơn, giải thích từng bước một, và dừng lại ở những chỗ người học hay vướng thay vì đi lướt qua. Chỗ nào quan trọng thì đào sâu, chỗ nào chỉ là chi tiết kỹ thuật thì nói rõ là chi tiết kỹ thuật.
>
> **Phần này lấp những gì.** Bốn giáo trình trước đi từ toán nền tới Transformer, rồi tới lượng tử hoá và vận hành. Nhưng chúng bỏ trống một mảng lớn: *làm sao biến dữ liệu thô thành vector*, *làm sao tái dùng một mô hình khổng lồ cho việc của mình*, *làm sao sinh ra nội dung mới*, và *làm sao bắt một mô hình biết nói trở thành một trợ lý biết nghe lời*. Bốn câu hỏi ấy là bốn phần của tài liệu này.
>
> **Về độ tin cậy của số liệu.** Mọi con số đều sinh từ `code/bieudien/`, hạt giống cố định. Hai kết quả là **kiểm chứng đẳng thức** chứ không phải mô phỏng: nghiệm tối ưu của bài toán RLHF có ràng buộc KL khớp với dạng đóng tới $10^{-9}$, và chính sách DPO trùng với chính sách RLHF hai bước tới $4{,}2 \times 10^{-8}$.
>
> **Và vài chỗ số liệu nói ngược lại điều tôi định viết.** Tôi giữ nguyên chúng và viết lại phần chữ. Thí nghiệm GAN ở Chương 10 **không** cho thấy sụp chế độ như tôi dự đoán — nó lại cho thấy một điều khác đáng giá hơn. Thí nghiệm học tăng cường ở Chương 12 cho thấy $\varepsilon = 0$ vẫn chạy tốt trong một cấu hình, điều mà câu "phải có $\varepsilon$ để khám phá" không giải thích nổi. Những chỗ như vậy được nói thẳng, vì đó mới là chỗ học được nhiều nhất.

---

## Mục lục

0. [Kiến thức nền và quy ước](#0-kiến-thức-nền-và-quy-ước)
1. [Bản đồ: bốn câu hỏi](#1-bản-đồ-bốn-câu-hỏi)
2. [Word2Vec và thứ nó thực sự phân rã](#2-word2vec-và-thứ-nó-thực-sự-phân-rã)
3. [Hình học của không gian embedding](#3-hình-học-của-không-gian-embedding)
4. [Embedding hiện đại và tìm kiếm vector](#4-embedding-hiện-đại-và-tìm-kiếm-vector)
5. [Vì sao tiền huấn luyện lại hiệu quả](#5-vì-sao-tiền-huấn-luyện-lại-hiệu-quả)
6. [Tinh chỉnh: đóng băng tới đâu](#6-tinh-chỉnh-đóng-băng-tới-đâu)
7. [LoRA và tinh chỉnh tiết kiệm tham số](#7-lora-và-tinh-chỉnh-tiết-kiệm-tham-số)
8. [Bài toán sinh, và ba cách trả lời](#8-bài-toán-sinh-và-ba-cách-trả-lời)
9. [Tự mã hoá biến phân](#9-tự-mã-hoá-biến-phân)
10. [Mạng đối sinh](#10-mạng-đối-sinh)
11. [Mô hình khuếch tán](#11-mô-hình-khuếch-tán)
12. [Nhập môn học tăng cường](#12-nhập-môn-học-tăng-cường)
13. [Gradient chính sách](#13-gradient-chính-sách)
14. [RLHF: học từ so sánh của con người](#14-rlhf-học-từ-so-sánh-của-con-người)
15. [DPO: bỏ hẳn mô hình thưởng](#15-dpo-bỏ-hẳn-mô-hình-thưởng)
16. [Bài tập](#16-bài-tập)
17. [Ôn phỏng vấn](#17-ôn-phỏng-vấn)
18. [Tài liệu tham khảo](#18-tài-liệu-tham-khảo)

---

## 0. Kiến thức nền và quy ước

Tài liệu này giả định người đọc đã qua hai giáo trình đầu của bộ, hoặc biết tương đương: hồi quy logistic và softmax, gradient descent, entropy chéo, hợp lý cực đại và MAP, cùng kiến trúc Transformer ở mức đọc hiểu. Chỗ nào cần dùng lại, tôi đều ghi rõ nó nằm ở chương nào để quay về xem.

| Ký hiệu | Ý nghĩa |
|---|---|
| $w$, $c$ | từ trung tâm và từ ngữ cảnh trong word2vec |
| $\#(w,c)$ | số lần $w$ và $c$ đồng hiện trong cửa sổ |
| $\mathrm{PMI}(w,c)$ | thông tin tương hỗ điểm: $\log \frac{p(w,c)}{p(w)p(c)}$ |
| $z$ | biến ẩn (VAE), hoặc nhiễu đầu vào (GAN) |
| $q_\phi(z\mid x)$ | bộ mã hoá — phân phối hậu nghiệm xấp xỉ |
| $p_\theta(x\mid z)$ | bộ giải mã |
| $\beta$ | hệ số cân giữa hai số hạng — trong VAE và trong RLHF, hai nghĩa khác nhau |
| $\bar\alpha_t$ | lượng tín hiệu còn lại sau $t$ bước khuếch tán |
| $r(x,y)$ | hàm thưởng: câu hỏi $x$, câu trả lời $y$ |
| $\pi_\theta$, $\pi_{\text{ref}}$ | chính sách đang huấn luyện và chính sách tham chiếu |
| $\mathrm{KL}(p\,\|\,q)$ | phân kỳ Kullback–Leibler |
| $A$, $B$ | hai ma trận hạng thấp của LoRA |
| $r$ | hạng của LoRA — **không** phải hàm thưởng; ngữ cảnh luôn phân biệt được |

**Một lưu ý về chữ $\beta$.** Nó xuất hiện ở hai nơi trong tài liệu này với hai nghĩa hoàn toàn khác nhau, và đây là chỗ dễ lẫn nhất:

- Trong **VAE** (Chương 9), $\beta$ nhân vào số hạng KL giữa hậu nghiệm và tiên nghiệm. $\beta$ lớn ép không gian ẩn gọn lại.
- Trong **RLHF** (Chương 14), $\beta$ nhân vào KL giữa chính sách mới và chính sách tham chiếu. $\beta$ lớn giữ mô hình gần bản gốc.

Điểm chung là cả hai đều là "giá của việc đi xa khỏi một phân phối tham chiếu". Điểm khác là phân phối tham chiếu ấy là gì: tiên nghiệm $N(0,I)$ trong VAE, mô hình đã tinh chỉnh có giám sát trong RLHF.

**Quy ước về chữ.** *Biểu diễn* cho representation, *nhúng* hoặc giữ nguyên *embedding* tuỳ câu, *chuyển giao* cho transfer, *tinh chỉnh* cho fine-tuning, *đóng băng* cho freezing, *khuếch tán* cho diffusion, *khử nhiễu* cho denoising, *chính sách* cho policy, *thưởng* cho reward, *căn chỉnh* cho alignment. Từ điển đầy đủ nằm ở trang **Từ điển thuật ngữ**.

---

## 1. Bản đồ: bốn câu hỏi

### 1.1. Vì sao chương này tồn tại

Bốn nhóm kỹ thuật trong tài liệu này — embedding, học chuyển giao, mô hình sinh, căn chỉnh — thoạt nhìn không liên quan gì tới nhau. Word2Vec ra đời năm 2013 để làm vector cho từ; mô hình khuếch tán ra đời để vẽ tranh; RLHF ra đời để dạy chatbot lễ phép. Ba mục đích, ba cộng đồng, ba dòng bài báo.

Nhưng nếu nhìn theo **đường đi của một mô hình ngôn ngữ hiện đại từ lúc chưa có gì tới lúc thành trợ lý**, chúng xếp thành một hàng rất gọn, và mỗi cái giải quyết đúng vấn đề mà cái trước để lại.

![Hình 1](figs/bd01_bando.png)

**Hình 1.** Bốn câu hỏi, và bốn nhóm kỹ thuật trả lời chúng. Đọc từ trái sang phải là đọc theo thứ tự một trợ lý ra đời.

### 1.2. Bốn câu hỏi, và vì sao chúng phải theo thứ tự đó

**Câu hỏi thứ nhất: làm sao biến dữ liệu thô thành số?**

Đây là câu hỏi đứng trước mọi thứ khác, và nó khó hơn vẻ ngoài. Máy tính chỉ làm việc với số, nên chữ "mèo" phải thành một vector nào đó. Cách ngây thơ nhất là gán cho mỗi từ một chỉ số: mèo = 7, chó = 8, tủ lạnh = 9. Nhưng làm thế là ngầm khẳng định rằng mèo gần chó hơn là gần tủ lạnh **về mặt số học**, mà thật ra ta chỉ tình cờ đánh số như vậy. Nếu đánh số theo thứ tự chữ cái thì "mèo" và "mền" lại cạnh nhau.

Cách thứ hai là one-hot: mỗi từ là một vector toàn số 0 với đúng một số 1. Cách này không bịa ra quan hệ giả nào cả — nhưng nó cũng không có quan hệ **thật** nào. Mọi cặp từ đều cách nhau đúng $\sqrt2$, nên mèo cách chó đúng bằng mèo cách tủ lạnh. Ta đã đi từ "bịa quan hệ" sang "không có quan hệ gì", và cả hai đều vô dụng.

Điều ta thực sự muốn là một không gian mà **khoảng cách có nghĩa**. Chương 2 và 3 nói về cách làm ra không gian ấy, và điều bất ngờ là nó nổi lên chỉ từ thống kê đồng hiện, không cần ai dạy.

**Câu hỏi thứ hai: đã có mô hình lớn rồi, làm sao dùng lại?**

Huấn luyện một mô hình ngôn ngữ từ đầu tốn hàng triệu đô la và hàng nghìn tỉ token. Gần như không ai làm việc đó. Cái mọi người làm là lấy một mô hình đã có rồi sửa nó cho việc của mình — và câu hỏi là sửa tới đâu, sửa thế nào, với bao nhiêu dữ liệu.

Chương 5, 6, 7 trả lời. Có một kết quả đo được ở Mục 6.3 mà tôi thấy đáng nhớ: với **rất ít nhãn**, đóng băng toàn bộ mô hình và chỉ học một lớp tuyến tính lại **thắng** việc tinh chỉnh toàn phần. Nhưng khi dữ liệu nhiều lên thì thứ tự đảo ngược hẳn. Biết ngưỡng đảo ngược ấy ở đâu là biết chọn đúng cách làm.

**Câu hỏi thứ ba: làm sao sinh ra cái mới?**

Phân loại thì dễ hiểu: cho ảnh, đoán nhãn. Sinh thì ngược lại và khó hơn nhiều: **cho nhiễu ngẫu nhiên, sinh ra một bức ảnh chưa từng tồn tại nhưng trông như thật**. Chương 8 tới 11 trình bày ba cách trả lời — VAE, GAN, khuếch tán — và điều đáng nói là cả ba đều biến nhiễu Gauss thành mẫu, chỉ khác nhau ở **huấn luyện bằng gì**.

**Câu hỏi thứ tư: làm sao bắt nó nói thứ ta muốn?**

Một mô hình đã tiền huấn luyện biết ngôn ngữ nhưng không biết **nghe lời**. Hỏi nó "thủ đô Pháp là gì?" thì nó có thể trả lời đúng, mà cũng có thể viết tiếp thành một bài kiểm tra địa lý gồm mười câu hỏi tương tự — vì trong dữ liệu huấn luyện, một câu hỏi thường đi kèm các câu hỏi khác chứ không phải câu trả lời.

Chương 12 tới 15 nói về cách sửa điều đó. Và ở đây có một ý sâu đáng dừng lại: giai đoạn căn chỉnh là chỗ duy nhất mô hình học từ **so sánh** thay vì từ ví dụ mẫu. Vì sao điều đó quan trọng? Vì **nhận ra câu nào hay hơn thì dễ hơn nhiều so với tự viết ra câu hay nhất**. Một người bình thường không viết nổi một bài thơ hay, nhưng vẫn chỉ ra được bài nào hay hơn bài nào. Học từ so sánh cho phép mô hình vượt qua chất lượng của chính người dán nhãn — điều mà học bắt chước không bao giờ làm được.

### 1.3. Mối nối với các giáo trình khác

| Chỗ này cần | Xem ở đâu |
|---|---|
| Softmax, entropy chéo, hợp lý cực đại | [Chương 6](nentang-ch06.html) và [Chương 10 của *Nền tảng*](nentang-ch10.html) |
| Phân rã SVD và xấp xỉ hạng thấp | [Chương 14 của *Nền tảng*](nentang-ch14.html) |
| Tối ưu có ràng buộc, nhân tử Lagrange | [Chương 12 của *Nền tảng*](nentang-ch12.html) |
| Kiến trúc Transformer, attention | [Chương 9 của *Học sâu*](models-ch09.html) |
| Đếm tham số, bộ nhớ huấn luyện | [Chương 12 của *Học sâu*](models-ch12.html) |
| Vận hành mô hình ngôn ngữ trong sản xuất | [Chương 13 của *MLOps*](mlops-ch13.html) |

Đặc biệt, Chương 7 về LoRA dùng thẳng kết quả đếm tham số ở [Chương 12 của *Học sâu*](models-ch12.html), và Chương 15 về DPO dùng lại cách suy luận về tối ưu có ràng buộc ở [Chương 12 của *Nền tảng*](nentang-ch12.html). Nếu hai chỗ đó còn mờ thì nên quay lại trước.

---

## 2. Word2Vec và thứ nó thực sự phân rã

### 2.1. Đặt lại bài toán cho đúng

Ta muốn mỗi từ thành một vector sao cho các từ **giống nhau về nghĩa** thì vector gần nhau. Nhưng "giống nhau về nghĩa" là một khái niệm của con người. Máy không có nó. Vậy lấy tín hiệu ở đâu?

Câu trả lời đã có từ những năm 1950, trong ngôn ngữ học, trước khi có học máy: **một từ được đặc trưng bởi những từ đi kèm nó**. Nếu trong hàng tỉ câu tiếng Việt, "mèo" và "chó" thường xuất hiện cạnh những từ giống nhau — "nuôi", "sủa/kêu", "thú cưng", "cho ăn" — thì chúng phải có gì đó chung. Ta không cần biết cái chung ấy là gì; ta chỉ cần đo nó.

Đây là chỗ đáng dừng lại một nhịp, vì nó là bước nhảy khái niệm lớn nhất của cả chương. Ta đang đổi một câu hỏi **không đo được** ("hai từ này có gần nghĩa không?") lấy một câu hỏi **đo được** ("hai từ này có hay xuất hiện trong ngữ cảnh giống nhau không?"). Bước đổi ấy không hoàn hảo — nó không phân biệt được "tốt" với "xấu" vì hai từ trái nghĩa vẫn xuất hiện trong ngữ cảnh y hệt nhau — nhưng nó đủ tốt để làm nền cho gần như mọi thứ về sau.

### 2.2. Một kho ngữ liệu nhỏ đủ để thấy mọi thứ

Để nhìn rõ cơ chế, ta cần một bộ dữ liệu nhỏ tới mức in ra được nhưng vẫn có cấu trúc thật. Thí nghiệm trong `code/bieudien/experiments.py` dựng một kho như vậy: 20 từ, mỗi từ là một cặp **(chủ đề, vai trò)** với 5 chủ đề và 4 vai trò, đặt tên kiểu `vua_nam`, `mua_nho`, `cay_lon`.

Kho gồm 40 000 câu, chia đôi:

- Một nửa số câu gom các từ **cùng chủ đề**: `[vua_nam, vua_nho, vua_lon]`.
- Nửa kia gom các từ **cùng vai trò**: `[vua_nam, mua_nam, cay_nam]`.

Chú ý điều này: **không câu nào nói rằng "chủ đề" và "vai trò" là hai trục độc lập.** Không có nhãn nào ghi "đây là chủ đề". Dữ liệu chỉ là những dãy từ. Nếu về sau embedding tách được hai trục ấy ra thành hai hướng cộng được với nhau, thì đó hoàn toàn là thứ mô hình tự rút ra.

### 2.3. Bước một: từ đếm sang PMI

Đếm xong ta có ma trận đồng hiện $\#(w,c)$. Nhưng đếm thô có một vấn đề rõ ràng: từ "của" đồng hiện với mọi thứ, không phải vì nó liên quan tới mọi thứ mà vì nó **phổ biến**. Ta cần một đại lượng trả lời câu hỏi đúng hơn: *hai từ này đồng hiện nhiều hơn mức ta trông đợi nếu chúng độc lập, bao nhiêu lần?*

Đại lượng ấy là **thông tin tương hỗ điểm**:

$$\mathrm{PMI}(w,c) = \log \frac{p(w,c)}{p(w)\,p(c)}.$$

Tử số là xác suất thật; mẫu số là xác suất nếu hai từ độc lập. Tỉ số bằng 1 (tức PMI bằng 0) nghĩa là đồng hiện đúng mức ngẫu nhiên — không có tin gì. PMI dương nghĩa là hút nhau, PMI âm nghĩa là đẩy nhau.

Trên kho ngữ liệu nhỏ ở trên, ba con số này là **toàn bộ tín hiệu** mà mô hình có:

| Cặp từ | PMI |
|---|---|
| Cùng chủ đề, khác vai trò (`vua_nam`, `vua_nu`) | $+1{,}1876$ |
| Cùng vai trò, khác chủ đề (`vua_nam`, `mua_nam`) | $+0{,}9367$ |
| Khác cả hai (`vua_nam`, `mua_nu`) | $0{,}0000$ |

Đọc bảng này kỹ một chút. Hai từ chia nhau **một** thuộc tính thì PMI dương; chia nhau **không** thuộc tính nào thì PMI bằng đúng 0, tức đồng hiện đúng mức ngẫu nhiên. Cấu trúc hai trục của dữ liệu đã nằm sẵn trong ma trận PMI rồi — chưa cần học gì cả.

### 2.4. Bước hai: nén ma trận PMI lại

Ma trận PMI có $V \times V$ ô. Với từ vựng thật cỡ 400 000 từ thì đó là 160 tỉ số — không lưu nổi, và phần lớn là 0. Ta muốn một biểu diễn **dày và ngắn**: mỗi từ một vector cỡ vài trăm chiều.

Cách làm là xấp xỉ hạng thấp, đúng như [Chương 14 của *Nền tảng*](nentang-ch14.html): tìm $E$ sao cho $E E^\top \approx \mathrm{PPMI}$, với PPMI là PMI đã cắt các giá trị âm về 0 (vì PMI âm ước lượng rất nhiễu — nó đòi ta đếm chính xác những cặp từ **hiếm khi** đi cùng nhau, mà dữ liệu hiếm thì ước lượng tệ).

Dùng SVD và giữ 8 chiều, rồi kiểm bằng phép loại suy — kiểu `a : b :: c : ?`:

```text
vua_nam : vua_nu :: mua_nho : ?  ->  mua_nu
Do chinh xac tren 240 phep loai suy: 1.0000   (doan ngau nhien: 0.0588)
```

**Đúng cả 240 phép.** Vector $\overrightarrow{vua\_nu} - \overrightarrow{vua\_nam}$ — tức "phép đổi vai trò từ nam sang nữ" — cộng vào `mua_nho` lại ra đúng `mua_nu`. Mô hình đã tách được hai trục thành hai hướng cộng được với nhau, dù không ai nói với nó rằng có hai trục.

![Hình 2](figs/bd02_word2vec.png)

**Hình 2.** Trái: ma trận PMI. Giữa: các từ chiếu xuống hai chiều đầu, màu là chủ đề và hình là vai trò — hai trục tách rõ. Phải: xem Mục 2.6.

> **Một điều phải nói thẳng về cái bảng quét số chiều trong kết quả in ra.** Tôi có thử giữ 2, 4, 6, 8, 12, 16, 20 chiều và định kể câu chuyện "ít quá thì thiếu, nhiều quá thì nhiễu". **Số liệu không ủng hộ câu chuyện đó**: giữ 20 chiều (hạng đầy đủ) lại đạt 100% trở lại, vì khi đó ta tái dựng PPMI chính xác. Với từ vựng chỉ 20 từ thì không đủ chỗ để một đánh đổi thật sự lộ ra. Muốn thấy đường cong thật phải chạy trên kho ngữ liệu thật. Tôi để cái bảng ấy lại trong kết quả để nói rõ điều này, chứ không phải để kết luận.

### 2.5. Bước ba: skip-gram, và vì sao nó tránh được softmax

Phần trên dùng SVD. Word2Vec thật thì không — nó chạy xuống dốc. Vì sao?

Vì ma trận PMI của từ vựng thật quá lớn để lập ra, chứ đừng nói tới phân rã. Word2Vec giải cùng bài toán ấy mà **không bao giờ lập ma trận**: nó lấy từng cặp $(w,c)$ quan sát được và đẩy $\langle w, c\rangle$ lên, rồi lấy vài cặp ngẫu nhiên và đẩy xuống.

Cụ thể, mô hình skip-gram muốn tối đa

$$p(c \mid w) = \frac{\exp(\langle w, c\rangle)}{\sum_{c' \in V}\exp(\langle w, c'\rangle)}.$$

Mẫu số là chỗ chết người: nó cộng trên **toàn bộ từ vựng**, cho **mỗi cặp huấn luyện**. Đếm phép tính:

| Từ vựng $V$ | Softmax đầy đủ | Lấy mẫu âm ($k=5$) | Rẻ hơn |
|---|---|---|---|
| 1 000 | 600 000 | 3 600 | 167× |
| 50 000 | 30 000 000 | 3 600 | 8 333× |
| 400 000 | 240 000 000 | 3 600 | 66 667× |
| 2 000 000 | 1 200 000 000 | 3 600 | **333 333×** |

![Hình 3](figs/bd03_negsampling.png)

**Hình 3.** Chi phí mỗi cặp huấn luyện. Cột giữa tăng tuyến tính theo $V$; cột phải là một đường nằm ngang.

**Lấy mẫu âm** thay bài toán "chọn đúng 1 trong $V$" bằng $k+1$ bài toán nhị phân độc lập: cặp thật này có phải cặp thật không? $k$ cặp ngẫu nhiên kia có phải cặp thật không?

$$\log \sigma(\langle w, c\rangle) + \sum_{i=1}^{k}\mathbb{E}_{c_i \sim P_n}\big[\log \sigma(-\langle w, c_i\rangle)\big].$$

Điểm mấu chốt: **chi phí không còn phụ thuộc $V$ nữa**. Đó là lý do word2vec huấn luyện được trên kho hàng tỉ từ vào năm 2013, khi phần cứng còn yếu hơn bây giờ rất nhiều.

### 2.6. Và đây là chỗ hay nhất: skip-gram đang phân rã cái gì

Năm 2014, Levy và Goldberg chứng minh một điều làm gọn cả lĩnh vực: **skip-gram với lấy mẫu âm ngầm phân rã ma trận PMI dịch đi một lượng $\log k$.**

Lập luận rất gọn, và đáng làm một lần bằng tay. Hàm mục tiêu kỳ vọng cho cặp $(i,j)$ là

$$\#(i,j)\,\log\sigma(s) + k\,\#(i)\,P_n(j)\,\log\sigma(-s), \qquad s = \langle w_i, c_j\rangle.$$

Lấy đạo hàm theo $s$ rồi cho bằng 0, dùng $\sigma' = \sigma(1-\sigma)$:

$$\#(i,j)\,(1-\sigma(s)) = k\,\#(i)\,P_n(j)\,\sigma(s) \;\Longrightarrow\; \frac{\sigma(s)}{1-\sigma(s)} = e^{s} = \frac{\#(i,j)}{k\,\#(i)P_n(j)}.$$

Thay $P_n(j) = \#(j)/N$:

$$\boxed{\;\langle w_i, c_j\rangle = \log\frac{\#(i,j)\,N}{\#(i)\,\#(j)} - \log k = \mathrm{PMI}(i,j) - \log k\;}$$

**Kiểm chứng bằng số.** Nếu điều trên đúng thì khi cho embedding đủ chiều để không bị ràng buộc gì (hạng đầy đủ), xuống dốc phải hội tụ về đúng ma trận PMI:

| Số chiều | Tương quan với PMI | \|lệch\| trung bình | \|lệch\| lớn nhất |
|---|---|---|---|
| 8 (hạng thấp) | 0,3727 | 0,2862 | 1,0566 |
| 20 (hạng đầy đủ) | **0,9995** | **0,0106** | 0,0251 |

Dòng dưới là kiểm chứng. Hai đường tính không liên quan gì tới nhau — một bên là công thức giải tích, một bên là chạy xuống dốc 4 000 vòng — mà ra cùng một ma trận.

Dòng trên cũng quan trọng không kém, và dễ bị bỏ qua: ở hạng thấp, mô hình **không thể** khớp chính xác. Nó buộc phải chọn giữ lại cái gì và vứt cái gì. **Chính sự ép buộc ấy tạo ra embedding có ích** — nếu khớp được hoàn hảo thì ta chỉ có một cách viết lại ma trận PMI chứ không có sự khái quát hoá nào.

> **Câu này đáng nhớ nguyên văn cho phỏng vấn:** *"Word2Vec không phải mạng nơ-ron sâu. Nó là một cách phân rã ma trận PMI mà không bao giờ phải lập ra ma trận ấy."* Nói được như vậy là cho thấy hiểu cả cơ chế lẫn động cơ, chứ không chỉ nhớ tên thuật toán.

### 2.7. Những chỗ hay nhầm

| Phát biểu | Thực tế |
|---|---|
| "Word2Vec là học sâu." | Nó là mô hình **một lớp**, không có phi tuyến ở giữa. Chỉ là phân rã ma trận có tên đẹp. |
| "Lấy mẫu âm là một phép xấp xỉ của softmax." | Không hẳn — nó là **một hàm mục tiêu khác**, và nghiệm của nó là PMI dịch $\log k$ chứ không phải nghiệm của softmax. |
| "Embedding hiểu nghĩa của từ." | Nó chỉ nắm được **thống kê đồng hiện**. Đó là lý do nó không phân biệt được "tốt" và "xấu" — hai từ trái nghĩa xuất hiện trong ngữ cảnh gần như y hệt nhau. |
| "Vector cộng trừ được nên mô hình biết suy luận." | Tính cộng được là hệ quả của cấu trúc ma trận PMI. Trên dữ liệu thật thì nó đúng ở một số quan hệ và sai ở nhiều quan hệ khác. |
| "Mũ 3/4 trong phân phối lấy mẫu âm là có cơ sở lý thuyết." | Bài báo gốc nói thẳng đó là con số **chọn bằng thực nghiệm**. Nó chạy tốt hơn mũ 1 và mũ 0; vì sao thì không ai chứng minh. |

---

## 3. Hình học của không gian embedding

### 3.1. Một câu hỏi tưởng đơn giản

Có embedding rồi, đo sự giống nhau thế nào? Câu trả lời mặc định của mọi người là cosine:

$$\cos(u,v) = \frac{\langle u, v\rangle}{\|u\|\,\|v\|}.$$

Vì sao cosine chứ không phải khoảng cách Euclid? Lý do thường được đưa ra là "vì độ dài vector phản ánh tần suất từ chứ không phản ánh nghĩa, nên nên bỏ nó đi". Lý do ấy đúng, nhưng nó che mất một vấn đề lớn hơn mà ít người nói tới.

### 3.2. Vấn đề: đám mây embedding không nằm quanh gốc

Cosine đo góc. Nhưng góc chỉ có nghĩa nếu đám mây điểm **trải đều quanh gốc toạ độ**. Nếu toàn bộ embedding lệch về một phía, thì mọi cặp vector đều tạo góc nhỏ với nhau — không phải vì chúng giống nhau, mà vì chúng cùng lệch về một hướng.

Đây gọi là **bất đẳng hướng** (anisotropy), và nó không phải trường hợp hiếm — nó là trường hợp **thường gặp** với embedding lấy từ các mô hình Transformer.

Thí nghiệm đo trực tiếp điều này trên ba đám mây điểm, mỗi đám 900 vector 64 chiều:

| Phân bố embedding | Cosine trung bình giữa hai vector bất kỳ | Sau khi trừ trung bình | Độ lệch tâm |
|---|---|---|---|
| Đều quanh gốc (lý tưởng) | $-0{,}0001$ | $-0{,}0011$ | 0,0318 |
| Lệch tâm nhẹ | $0{,}3899$ | $-0{,}0011$ | 0,6267 |
| Lệch tâm mạnh | **$0{,}8724$** | $-0{,}0011$ | 0,9342 |

![Hình 4](figs/bd04_anisotropy.png)

**Hình 4.** Phân bố cosine giữa mọi cặp vector. Vạch đỏ là trung bình. Ở cột phải, hai vector **ngẫu nhiên hoàn toàn** vẫn có cosine trung bình 0,87.

Hàng cuối là điều cần thấy. Đây là **900 vector ngẫu nhiên độc lập** — không có quan hệ ngữ nghĩa nào giữa chúng, vì chúng chỉ là nhiễu Gauss cộng một hằng số. Vậy mà cosine trung bình giữa hai vector bất kỳ là 0,87.

Nếu đem con số ấy đi báo cáo, ta sẽ nói "hai câu này giống nhau 87%". Thực tế thì nó chẳng nói gì về hai câu cả — nó chỉ nói rằng đám mây lệch tâm.

Cột thứ ba là cách chữa và nó hiệu quả đến mức gần như đáng ngờ: chỉ cần **trừ vector trung bình** của cả tập đi, cosine trung bình về $-0{,}0011$ ở cả ba trường hợp.

### 3.3. Ba cách chữa, theo thứ tự nên thử

1. **Trừ trung bình.** Tính vector trung bình trên một mẫu đại diện rồi trừ khỏi mọi embedding trước khi so sánh. Rẻ nhất, và như bảng trên cho thấy, thường đã đủ.
2. **Làm trắng.** Ngoài trừ trung bình còn chuẩn hoá phương sai theo từng hướng riêng, tức nhân với $\Sigma^{-1/2}$. Mạnh hơn nhưng cần ước lượng $\Sigma$, mà ước lượng ma trận hiệp phương sai cần nhiều dữ liệu (xem [Mục 7.4 của *Nền tảng*](nentang-ch07.html) về cái giá của việc ước lượng $\Sigma$ đầy đủ).
3. **Huấn luyện có mục tiêu tương phản.** Cách triệt để nhất: ngay từ đầu đã huấn luyện embedding sao cho cặp giống nhau gần nhau và cặp khác nhau xa nhau. Đây là cách các mô hình embedding câu hiện đại làm, và Mục 4.2 nói kỹ hơn.

> **Hệ quả thực tế.** Nếu bạn đang xây hệ tìm kiếm ngữ nghĩa và thấy **mọi** kết quả đều có điểm cosine cao (0,8 trở lên) kể cả những kết quả rõ ràng không liên quan, thì gần như chắc chắn đây là nguyên nhân. Đừng đi chỉnh ngưỡng; hãy trừ trung bình đi rồi đo lại.

### 3.4. Một cảnh báo về "chiều nào mang nghĩa gì"

Có một câu hỏi rất hay được hỏi: chiều thứ 37 của embedding mang nghĩa gì?

Câu trả lời là: **thường thì không mang nghĩa gì cả.** Lý do nằm ở toán chứ không ở cách huấn luyện. Nếu $E E^\top \approx M$ thì với mọi ma trận trực giao $Q$, ta có $(EQ)(EQ)^\top = E Q Q^\top E^\top = E E^\top$ — tức **xoay toàn bộ không gian đi thì không đổi gì cả**. Bài toán chỉ xác định $E$ tới một phép xoay, nên trục toạ độ hoàn toàn tuỳ tiện.

Thứ có nghĩa là **quan hệ giữa các vector** — khoảng cách, góc, hướng hiệu — chứ không phải toạ độ của từng vector. Đó cũng là lý do phép loại suy ở Mục 2.4 hoạt động: nó chỉ dùng hiệu vector, mà hiệu vector thì bất biến với phép xoay.

Ngoại lệ đáng biết: nếu cố tình ép tính thưa trong lúc huấn luyện (ví dụ bằng bộ tự mã hoá thưa), thì các chiều **có thể** trở nên diễn giải được. Đó là cả một nhánh nghiên cứu hiện nay về khả năng diễn giải của mô hình ngôn ngữ, và nó tồn tại chính vì mặc định thì các chiều không có nghĩa.

---

## 4. Embedding hiện đại và tìm kiếm vector

### 4.1. Vấn đề của embedding tĩnh

Word2Vec gán cho mỗi từ **đúng một** vector. Nghe hợp lý cho tới khi gặp từ đa nghĩa:

- "Con **đường** này dài quá."
- "Cho thêm **đường** vào cà phê."

Một vector duy nhất cho "đường" phải là một thoả hiệp giữa hai nghĩa — và thoả hiệp ấy không đúng ở cả hai câu. Với các từ có nhiều nghĩa, vector nằm đâu đó ở giữa, gần với nghĩa phổ biến hơn, và không phục vụ tốt nghĩa nào.

Đây không phải lỗi cài đặt. Nó là giới hạn **về mặt thiết kế** của việc gán một vector cho một chuỗi ký tự.

### 4.2. Embedding ngữ cảnh

Cách chữa là hiển nhiên khi đã nhìn ra vấn đề: đừng gán vector cho **từ**, hãy gán vector cho **một lần xuất hiện của từ trong một câu cụ thể**. Cùng chữ "đường" trong hai câu trên sẽ nhận hai vector khác nhau, vì ngữ cảnh khác nhau.

Đó chính xác là thứ mà Transformer làm — xem [Chương 9 của *Học sâu*](models-ch09.html). Mỗi lớp attention trộn thông tin giữa các vị trí, nên biểu diễn của một token ở lớp thứ 12 đã chứa thông tin từ cả câu.

Từ đó sinh ra một cách dùng rất phổ biến: lấy một mô hình đã tiền huấn luyện, cho câu chạy qua, rồi lấy trạng thái ẩn làm embedding của câu. Nhưng ở đây có một cái bẫy mà rất nhiều người vấp phải.

> **Cái bẫy:** trạng thái ẩn của một mô hình ngôn ngữ **không phải** embedding câu tốt, dù nghe rất hợp lý. Lý do là mô hình được huấn luyện để **đoán token tiếp theo**, chứ không phải để làm cho hai câu cùng nghĩa có biểu diễn gần nhau. Không có sức ép nào trong quá trình huấn luyện buộc điều đó xảy ra. Lấy trung bình các trạng thái ẩn rồi đem so cosine thường cho kết quả tệ hơn nhiều so với người ta trông đợi — và Chương 3 đã cho một lý do cụ thể: các biểu diễn ấy bất đẳng hướng nặng.

Cách làm đúng là huấn luyện thêm một bước với **mục tiêu tương phản**: cho các cặp câu được biết là cùng nghĩa (câu hỏi và câu trả lời, câu và bản dịch, hai câu diễn đạt lại của nhau), tối đa

$$\mathcal{L} = -\log \frac{\exp(\cos(u, v^{+})/\tau)}{\sum_{j}\exp(\cos(u, v_j)/\tau)},$$

trong đó $v^+$ là cặp đúng và các $v_j$ là những câu khác trong cùng lô. Hàm này có dạng y hệt entropy chéo của softmax — xem [Mục 6.4 của *Nền tảng*](nentang-ch06.html) — chỉ khác ở chỗ "lớp" ở đây là "câu nào trong lô là cặp đúng".

Chi tiết đáng chú ý là **các câu khác trong lô đóng vai trò mẫu âm**. Ta không phải đi tìm mẫu âm ở đâu cả; lô càng lớn thì càng nhiều mẫu âm, nên các mô hình embedding tốt thường được huấn luyện với lô rất lớn. Đây cũng là một dạng lấy mẫu âm, đúng tinh thần Mục 2.5.

### 4.3. Tìm kiếm vector, và vì sao nó khó

Có embedding rồi, việc tiếp theo thường là: cho một truy vấn, tìm $k$ mục gần nhất trong một kho hàng triệu mục.

Cách ngây thơ là tính khoảng cách tới **tất cả** rồi lấy $k$ nhỏ nhất. Với $N$ mục và $d$ chiều, chi phí là $O(Nd)$ cho mỗi truy vấn. Với $N = 10^7$ và $d = 768$, đó là khoảng 7,7 tỉ phép tính cho **một** truy vấn — không chấp nhận được nếu cần trả lời trong vài chục mili-giây.

Và [Mục 7.2 của *Nền tảng*](nentang-ch07.html) đã cho thấy vì sao không thể dùng các cấu trúc dữ liệu cổ điển như cây k-d: **lời nguyền số chiều** làm chúng thoái hoá về tìm kiếm tuyến tính khi số chiều vượt vài chục.

Cách thoát là chấp nhận một điều mà lúc đầu nghe hơi khó chịu: **bỏ yêu cầu chính xác**. Tìm kiếm láng giềng gần **xấp xỉ** chỉ hứa trả về các mục gần, không hứa trả về đúng $k$ mục gần nhất. Đổi lại, nó nhanh hơn hàng trăm lần.

| Họ phương pháp | Ý tưởng | Đánh đổi |
|---|---|---|
| IVF (phân cụm đảo) | K-means chia kho thành các cụm; truy vấn chỉ quét vài cụm gần nhất | Rất gọn bộ nhớ; hụt các mục nằm sát biên cụm |
| HNSW (đồ thị nhiều tầng) | Dựng đồ thị láng giềng nhiều tầng rồi đi tham lam từ tầng thô xuống tầng mịn | Nhanh và chính xác nhất hiện nay; **tốn bộ nhớ** vì phải lưu cả đồ thị |
| PQ (lượng tử hoá tích) | Chia vector thành nhiều đoạn, lượng tử hoá từng đoạn bằng một bảng mã nhỏ | Nén rất mạnh; mất chính xác do lượng tử hoá |

Dòng cuối nối thẳng sang [giáo trình *Quantization*](ch03.html): lượng tử hoá tích chính là uniform affine quantization áp cho từng đoạn của vector, với cùng một đánh đổi giữa sai số và dung lượng.

**Con số cần nhớ khi thiết kế hệ thống.** Một kho 10 triệu vector 768 chiều ở FP32 chiếm

$$10^7 \times 768 \times 4 \text{ byte} = 30{,}7 \text{ GB}.$$

Chỉ riêng con số này đã quyết định phần lớn kiến trúc: nó không vừa RAM của một máy thường, nên hoặc phải nén (PQ, hoặc hạ xuống FP16 để còn 15,4 GB), hoặc phải chia nhỏ ra nhiều máy. Tính con số này **trước** khi chọn thư viện là thói quen đáng có.

### 4.4. Đánh giá hệ tìm kiếm vector

Vì đã chấp nhận xấp xỉ, ta phải đo mình mất bao nhiêu. Chỉ số chuẩn là **recall@k**: trong $k$ kết quả trả về, có bao nhiêu phần trăm thuộc về $k$ kết quả đúng thật sự (tính bằng quét toàn bộ trên một mẫu nhỏ).

Đường cong cần vẽ là **recall theo độ trễ**, và cách đọc nó giống hệt cách đọc đường ROC ở [Mục 8.3 của *Nền tảng*](nentang-ch08.html): không có điểm nào "tốt nhất", chỉ có một đường đánh đổi, và chọn điểm nào trên đường ấy là quyết định nghiệp vụ.

Một lưu ý cuối, và nó quan trọng hơn mọi chỉ số ở trên: **recall@k cao không đảm bảo hệ thống hữu ích**. Nếu bản thân embedding không nắm được thứ người dùng coi là "liên quan", thì tìm chính xác 100% các láng giềng gần nhất vẫn cho kết quả vô dụng. Chất lượng embedding và chất lượng chỉ mục là hai thứ tách rời, và phải đo tách rời.

---

## 5. Vì sao tiền huấn luyện lại hiệu quả

### 5.1. Một sự thật kinh tế đứng sau mọi thứ

Trước khi vào cơ chế, hãy nhìn con số. Dữ liệu **có nhãn** thì đắt: thuê người đọc và gán nhãn 10 000 câu có thể tốn vài nghìn đô và vài tuần. Dữ liệu **không nhãn** thì gần như miễn phí: toàn bộ văn bản trên internet đã có sẵn.

Tỉ lệ giữa hai loại là khoảng một phần triệu. Nên bất kỳ cách nào biến dữ liệu không nhãn thành thứ có ích đều đáng giá hơn mọi cải tiến thuật toán trên dữ liệu có nhãn.

Tiền huấn luyện là cách làm đó. Nó dựng ra một nhiệm vụ mà **nhãn tự sinh từ chính dữ liệu**: che một từ đi rồi đoán nó, hoặc đoán từ tiếp theo. Không ai phải gán nhãn gì cả, mà mô hình vẫn có tín hiệu học.

### 5.2. Nhưng vì sao đoán từ tiếp theo lại dạy được nhiều đến thế?

Đây mới là câu hỏi thú vị, và câu trả lời hời hợt kiểu "vì nó thấy nhiều dữ liệu" không đủ.

Lý do thật sự là: **để đoán được từ tiếp theo cho đúng trong mọi ngữ cảnh, mô hình buộc phải học rất nhiều thứ khác.** Hãy thử vài ví dụ cụ thể:

- *"Thủ đô của nước Pháp là ___"* — muốn đoán đúng phải **biết sự thật về thế giới**.
- *"Cô ấy mở tủ lạnh và lấy ra một chai ___"* — phải **hiểu ngữ cảnh vật lý** rằng trong tủ lạnh có gì.
- *"Mặc dù trời mưa rất to, anh ấy vẫn ___"* — phải nắm được **quan hệ nhượng bộ** trong ngữ pháp.
- *"2 + 3 = ___"* — phải làm được **số học** ở mức tối thiểu.
- *"def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + ___"* — phải hiểu **cấu trúc mã**.

Không có bài toán nào trong số đó được dạy riêng. Chúng đều là **hệ quả bắt buộc** của việc giảm mất mát đoán từ tiếp theo trên một kho đủ lớn và đủ đa dạng. Mô hình không học chúng vì ta muốn; nó học vì không học thì không đoán đúng được.

Cách nói gọn: đoán từ tiếp theo là một nhiệm vụ **đại diện** cực kỳ rộng. Nó không phải mục tiêu cuối, nó là cái cớ để mô hình phải xây dựng một biểu diễn hữu ích về thế giới.

### 5.3. Bốn cách tái dùng, và chúng khác nhau ở đâu

![Hình 5](figs/bd05_taidung.png)

**Hình 5.** Bốn cách. Ô màu xám là đóng băng, ô màu xanh là có học, ô màu tím là thêm tham số mới.

| Cách | Số tham số phải học | Cần bao nhiêu nhãn | Khi nào dùng |
|---|---|---|---|
| Dùng thẳng, không chỉnh | 0 | không cần | nhiệm vụ đã nằm trong khả năng sẵn có của mô hình |
| Đóng băng + đầu tuyến tính | chỉ lớp cuối | rất ít | ít nhãn; hoặc cần chạy nhiều nhiệm vụ trên cùng một bộ đặc trưng |
| Tinh chỉnh toàn phần | toàn bộ | nhiều | có nhiều nhãn và đủ bộ nhớ |
| LoRA | dưới 1% | ít tới vừa | có ít nhãn nhưng cần sửa sâu hơn một lớp tuyến tính |

Bảng này chỉ là khung. Câu hỏi thật là: **ở bao nhiêu nhãn thì nên chuyển từ cách này sang cách kia?** Chương 6 trả lời bằng số đo.

### 5.4. Tiền huấn luyện hỏng khi nào

Không có gì miễn phí, nên cần biết giới hạn:

- **Khi miền đích quá xa miền tiền huấn luyện.** Một mô hình học trên văn bản web không giúp được mấy cho tín hiệu cảm biến công nghiệp. Đặc trưng nó học không có gì dùng lại được.
- **Khi nhiệm vụ đích cần thứ mà nhiệm vụ đại diện không đụng tới.** Mô hình ngôn ngữ giỏi ngữ pháp nhưng không giỏi số học nhiều chữ số, vì đoán từ tiếp theo hiếm khi đòi hỏi tính toán chính xác.
- **Khi mô hình tiền huấn luyện mang sẵn thiên lệch mà nhiệm vụ đích không chấp nhận được.** Nó học từ dữ liệu người viết, nên nó kế thừa cả thiên kiến trong dữ liệu ấy. Tinh chỉnh làm giảm nhưng không xoá được.

Điểm thứ ba là vấn đề thật trong sản xuất, không phải vấn đề lý thuyết — [Chương 6 của *MLOps*](mlops-ch06.html) bàn về đánh giá theo lát cắt, và đó là công cụ để phát hiện nó.

---

## 6. Tinh chỉnh: đóng băng tới đâu

### 6.1. Câu hỏi cụ thể

Có mô hình tiền huấn luyện, có $n$ mẫu có nhãn cho việc của mình. Nên đóng băng phần đặc trưng và chỉ học lớp cuối, hay nên cho cả mô hình học lại?

Trực giác nói: ít dữ liệu thì đóng băng, nhiều dữ liệu thì tinh chỉnh. Trực giác ấy **đúng**, nhưng nó không nói ngưỡng ở đâu, và ngưỡng mới là thứ ta cần.

### 6.2. Thiết kế thí nghiệm cho đàng hoàng

Đây là chỗ tôi phải cẩn thận, và lần đầu tôi làm sai nên nói luôn để người đọc tránh.

Lần đầu tôi dùng `MLPClassifier` của scikit-learn cho cột "tinh chỉnh toàn phần". Nhưng scikit-learn **không cho nạp trọng số ban đầu**. Nên cái gọi là "tinh chỉnh" ấy thực ra là **huấn luyện từ đầu**, và quả nhiên hai cột cho số gần như y hệt nhau (0,6975 so với 0,6987). So sánh ấy vô nghĩa.

Cách làm đúng là tự cài một mạng nhỏ bằng NumPy để kiểm soát được trọng số khởi tạo. Khi đó ba cấu hình dùng **cùng một cài đặt mạng**, chỉ khác hai điều:

| Cấu hình | Trọng số ban đầu của lớp ẩn | Lớp ẩn có được cập nhật không |
|---|---|---|
| Đóng băng + đầu tuyến tính | tiền huấn luyện | **không** |
| Tinh chỉnh toàn phần | tiền huấn luyện | có |
| Huấn luyện từ đầu | ngẫu nhiên | có |

Dữ liệu mô phỏng đúng tình huống thật: có một phép chiếu "thật" sinh ra dữ liệu, và mô hình "tiền huấn luyện" là phép chiếu ấy **cộng nhiễu** — tức gần đúng nhưng không trùng khớp. Đó chính là quan hệ giữa một mô hình tiền huấn luyện và nhiệm vụ đích trong thực tế.

### 6.3. Kết quả

![Hình 6](figs/bd06_transfer.png)

**Hình 6.** Độ chính xác theo số nhãn, trục hoành log.

| Số mẫu có nhãn | Đóng băng + đầu tuyến tính | Tinh chỉnh toàn phần | Huấn luyện từ đầu |
|---|---|---|---|
| 20 | 0,4027 | 0,3875 | 0,4118 |
| 50 | 0,4800 | 0,4955 | 0,4820 |
| 150 | 0,5330 | **0,5713** | 0,5520 |
| 500 | 0,6220 | **0,6713** | 0,6350 |
| 2 000 | 0,6375 | **0,7515** | 0,6773 |
| 8 000 | 0,6452 | **0,8183** | 0,7545 |

Đọc bảng này theo ba tầng.

**Tầng một — ở $n = 20$ và $n = 50$, ba cột nằm trong khoảng nhiễu của nhau.** Tôi không kết luận gì ở hai hàng ấy. Với 20 mẫu và 4 lớp, mỗi lớp chỉ có 5 mẫu; chênh lệch 0,02 giữa các cột không nói lên điều gì. Nếu ai đó đưa cho bạn một bảng như vậy và kết luận từ hàng đầu tiên, hãy nghi ngờ.

**Tầng hai — từ $n = 150$ trở đi, tinh chỉnh toàn phần tách hẳn lên và không bao giờ bị đuổi kịp.** Khoảng cách giãn dần: 0,04 ở $n=150$, 0,11 ở $n=2000$, **0,17 ở $n=8000$**. Cơ chế rất rõ: đặc trưng tiền huấn luyện chỉ *gần đúng* cho nhiệm vụ này, và khi có đủ dữ liệu thì việc sửa lại chúng đáng giá hơn nhiều so với việc chỉ khớp một lớp tuyến tính lên trên.

**Tầng ba — cột đóng băng gần như đứng yên từ $n = 500$ trở đi** (0,6220 → 0,6375 → 0,6452). Đây là hiện tượng đáng chú ý nhất trong bảng và nó có tên: **trần chặn**. Khi đặc trưng bị cố định, mô hình chỉ còn là hồi quy softmax trên những đặc trưng ấy. Thêm dữ liệu giúp ước lượng lớp tuyến tính chính xác hơn, nhưng không phá được giới hạn do bản thân đặc trưng đặt ra.

Nhận ra trần chặn này có giá trị thực tế ngay: nếu đang đóng băng và thấy độ chính xác không nhích lên dù thêm dữ liệu, thì **thêm dữ liệu nữa cũng vô ích**. Phải mở đóng băng ra. Đây chính là chẩn đoán thiên lệch cao ở [Mục 2.4 của *Học sâu*](models-ch02.html), xuất hiện lại ở một dạng khác.

**Và một điều nữa đáng để ý:** tinh chỉnh **luôn** tốt hơn huấn luyện từ đầu, ở mọi cỡ dữ liệu từ 150 trở lên. Trọng số tiền huấn luyện không chỉ giúp khi thiếu dữ liệu — nó là một điểm khởi đầu tốt hơn về mọi mặt.

### 6.4. Vài quy tắc thực hành

Những điều dưới đây là kinh nghiệm chung của ngành chứ không phải kết quả đo trong tài liệu này, nên tôi ghi rõ như vậy:

- **Tốc độ học khi tinh chỉnh phải nhỏ hơn nhiều so với khi huấn luyện từ đầu**, thường nhỏ hơn 10 đến 100 lần. Lý do: trọng số hiện tại đã tốt, và bước đi lớn sẽ phá mất thứ đã học được trước khi kịp học cái mới.
- **Mở đóng băng dần từ trên xuống** thường ổn định hơn mở hết một lúc. Các lớp gần đầu vào học những đặc trưng chung nhất (cạnh, kết cấu, cú pháp cơ bản) và ít cần sửa nhất.
- **Tinh chỉnh trên ít dữ liệu có thể gây quên tai hại** — mô hình mất khả năng ở những việc nó từng làm được. Nếu điều đó quan trọng, hãy trộn một ít dữ liệu của miền gốc vào, hoặc dùng LoRA ở Chương 7, vốn giữ nguyên trọng số gốc theo đúng nghĩa đen.

---

## 7. LoRA và tinh chỉnh tiết kiệm tham số

### 7.1. Vấn đề mà LoRA giải

Chương 6 kết luận rằng tinh chỉnh toàn phần thắng khi có đủ dữ liệu. Nhưng "đủ bộ nhớ" là một điều kiện nặng hơn người ta tưởng.

Hãy tính cho Llama-2 7B, dùng lại cách đếm ở [Chương 12 của *Học sâu*](models-ch12.html). Mô hình có 6 476 005 376 tham số phi-embedding. Tinh chỉnh toàn phần bằng Adam ở FP32 cần:

| Thành phần | Dung lượng |
|---|---|
| Trọng số | 24,1 GiB |
| Gradient | 24,1 GiB |
| Trạng thái Adam ($m$ và $v$) | **48,2 GiB** |
| **Tổng, chưa kể kích hoạt** | **96,4 GiB** |

Một GPU A100 80 GB **không đủ**. Và đó mới chỉ là mô hình 7B — loại nhỏ nhất trong họ.

Thêm một vấn đề nữa về vận hành: nếu phục vụ 50 khách hàng, mỗi người một bản tinh chỉnh riêng, thì phải lưu **50 bản sao đầy đủ** của mô hình 13 GB. Đó là 650 GB chỉ để lưu trọng số.

### 7.2. Ý tưởng, và giả thiết đứng sau nó

![Hình 7](figs/bd07_lora.png)

**Hình 7.** LoRA giữ nguyên $W_0$ và chỉ học phần thêm vào, ép phần ấy có hạng thấp.

Thay vì cập nhật $W$, LoRA đóng băng nó và học một **phần thêm vào**:

$$W = W_0 + \Delta W, \qquad \Delta W = BA,$$

với $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$ và $r \ll d$.

Giả thiết cốt lõi, và đáng nói rõ vì nó là thứ có thể sai: **phần cần sửa khi chuyển sang nhiệm vụ mới có hạng thấp.** Nói cách khác, mô hình gốc đã đúng gần hết rồi; cái cần đổi chỉ là một chỉnh sửa trong một không gian con ít chiều.

Giả thiết này nghe táo bạo nhưng có lý do để tin. Nhiệm vụ đích thường chỉ đòi mô hình **nhấn mạnh lại** những gì nó đã biết — ưu tiên một văn phong, một định dạng đầu ra, một miền kiến thức — chứ không đòi nó học lại ngôn ngữ từ đầu. Loại chỉnh sửa ấy đúng là ít chiều.

Chi tiết cài đặt quan trọng: khởi tạo $A$ ngẫu nhiên và $B = 0$. Khi đó $BA = 0$ lúc bắt đầu, nên mô hình **khởi đầu đúng bằng mô hình gốc** rồi đi dần ra. Nếu khởi tạo cả hai ngẫu nhiên thì ngay bước đầu đã phá mô hình.

### 7.3. Đếm tham số

Một lớp $d \times d$ có $d^2$ tham số. LoRA hạng $r$ có $2dr$. Tỉ lệ là

$$\frac{2dr}{d^2} = \frac{2r}{d}.$$

Điều đáng chú ý: tỉ lệ này **giảm khi $d$ tăng**. Mô hình càng lớn thì LoRA càng có lợi.

| Lớp | $d$ | Toàn phần $d^2$ | LoRA $r=8$ | LoRA $r=64$ | % ($r=8$) |
|---|---|---|---|---|---|
| GPT-2 small | 768 | 589 824 | 12 288 | 98 304 | 2,083% |
| GPT-2 large | 1 280 | 1 638 400 | 20 480 | 163 840 | 1,250% |
| Llama-2 7B | 4 096 | 16 777 216 | 65 536 | 524 288 | **0,391%** |
| Llama-2 70B | 8 192 | 67 108 864 | 131 072 | 1 048 576 | **0,195%** |

Áp cho cả mô hình. LoRA thường chỉ gắn vào $W_Q$ và $W_V$ — không phải vì lý thuyết mà vì bài báo gốc thử nhiều tổ hợp và thấy tổ hợp này cho tỉ lệ chất lượng trên tham số tốt nhất:

| Hạng $r$ | Tham số huấn luyện | % mô hình 7B |
|---|---|---|
| 4 | 2 097 152 | 0,0324% |
| **8** | **4 194 304** | **0,0648%** |
| 16 | 8 388 608 | 0,1295% |
| 64 | 33 554 432 | 0,5181% |

Và đây là con số đắt nhất của cả chương:

$$\text{Trạng thái Adam với LoRA } r=8:\; \textbf{32 MiB} \qquad\text{so với}\qquad \text{tinh chỉnh toàn phần: } \textbf{48,2 GiB}.$$

Chênh nhau **1 540 lần**. Đó là khác biệt giữa "chạy được trên một GPU tiêu dùng" và "cần một cụm máy chủ".

### 7.4. Ba điều hay bị nói sai

**"LoRA làm mô hình chạy nhanh hơn."** Sai, và sai theo hướng ngược. Khi suy luận, ta cộng $BA$ vào $W_0$ **một lần** rồi dùng ma trận gộp — nên LoRA không nhanh hơn mà cũng không chậm hơn mô hình gốc. Nếu **không** gộp (để đổi adapter nhanh) thì nó chậm hơn một chút vì phải làm thêm hai phép nhân nhỏ.

**"LoRA chất lượng bằng tinh chỉnh toàn phần."** Bài báo gốc báo cáo chất lượng **rất gần** trên các nhiệm vụ họ thử, nhưng "rất gần" không phải "bằng". Với nhiệm vụ đòi mô hình học kiến thức thật sự mới — không chỉ đổi văn phong — thì giả thiết hạng thấp yếu đi và khoảng cách rộng ra.

**"Hạng càng cao càng tốt."** Không. Hạng cao hơn nghĩa là nhiều tham số hơn, tức nhiều phương sai hơn với cùng lượng dữ liệu — đúng câu chuyện ở [Chương 2 của *Học sâu*](models-ch02.html). Trên ít dữ liệu, $r = 4$ hoặc $r = 8$ thường cho kết quả tốt hơn $r = 64$.

### 7.5. Món lợi về vận hành mà người ta hay quên

Ngoài bộ nhớ lúc huấn luyện, LoRA còn giải quyết bài toán phục vụ nhiều khách hàng mà Mục 7.1 nêu ra.

Với tinh chỉnh toàn phần: 50 khách hàng = 50 bản sao 13 GB = **650 GB**.

Với LoRA: 50 khách hàng = **một** bản gốc 13 GB + 50 adapter × 16 MB ≈ **13,8 GB**.

Hơn thế nữa, vì tất cả dùng chung trọng số gốc, ta có thể giữ mô hình gốc trên GPU và chỉ hoán đổi adapter theo từng yêu cầu — nên phục vụ được nhiều khách hàng cùng lúc trên một GPU. Đây là lý do LoRA được dùng rộng rãi trong sản phẩm thật, và nó là lý do độc lập với chuyện tiết kiệm bộ nhớ huấn luyện.

### 7.6. Họ hàng của LoRA

| Phương pháp | Cách làm | Ghi chú |
|---|---|---|
| Adapter | chèn hẳn các lớp nhỏ mới vào giữa | Có trước LoRA; **thêm độ trễ** vì không gộp được |
| Prefix / prompt tuning | học vài vector "token ảo" đặt trước đầu vào | Rất ít tham số; khó huấn luyện hơn và chiếm mất ngữ cảnh |
| BitFit | chỉ tinh chỉnh các tham số độ lệch | Cực gọn; đủ cho nhiệm vụ dễ |
| QLoRA | LoRA + mô hình gốc lượng tử hoá 4 bit | Kết hợp với [giáo trình *Quantization*](ch11.html); hạ thêm bộ nhớ của phần đóng băng |

Dòng cuối đáng chú ý: mô hình gốc **chỉ đọc chứ không cập nhật**, nên lượng tử hoá nó không ảnh hưởng tới việc huấn luyện adapter. Đó là lý do QLoRA chạy được — hai kỹ thuật ghép vào nhau vì chúng tác động lên hai phần khác nhau của bộ nhớ.

---

## 8. Bài toán sinh, và ba cách trả lời

### 8.1. Bài toán khó ở đâu

Phân loại hỏi: cho $x$, đoán $y$. Sinh hỏi ngược lại và khó hơn nhiều: **học phân phối $p(x)$ rồi lấy mẫu từ nó.**

Vì sao khó hơn? Hãy nghĩ về không gian. Một ảnh màu $256\times256$ có $256\times256\times3 = 196\,608$ số, mỗi số 256 giá trị. Số ảnh khả dĩ là $256^{196608}$ — lớn hơn số nguyên tử trong vũ trụ nhiều bậc không tưởng tượng nổi.

Trong cái không gian khổng lồ ấy, **ảnh trông như thật chiếm một phần cực kỳ nhỏ**. Lấy ngẫu nhiên một điểm thì gần như chắc chắn được nhiễu. Thứ ta muốn học là hình dạng của cái tập con bé tí ấy — thường gọi là **đa tạp dữ liệu** — và học nó đủ tốt để lấy mẫu ra những điểm mới trên đó.

### 8.2. Vì sao không dùng thẳng hợp lý cực đại

[Chương 10 của *Nền tảng*](nentang-ch10.html) đã dạy công cụ chuẩn: hợp lý cực đại. Sao không áp thẳng vào đây?

Vướng ở **hằng số chuẩn hoá**. Nếu mô hình hoá $p_\theta(x) = \frac{1}{Z(\theta)}\exp(f_\theta(x))$ thì

$$Z(\theta) = \int \exp(f_\theta(x))\,dx$$

là tích phân trên toàn bộ không gian $196\,608$ chiều. Không tính được, không xấp xỉ nổi, và nó phụ thuộc $\theta$ nên không bỏ qua được khi lấy đạo hàm.

Ba họ mô hình trong các chương sau là **ba cách né hằng số ấy**, và đó là cách đọc gọn nhất để phân biệt chúng:

- **VAE** tối đa một **chặn dưới** của log hợp lý — chặn dưới ấy tính được.
- **GAN** bỏ hẳn hợp lý, thay bằng một **trò chơi** giữa hai mạng.
- **Khuếch tán** biến bài toán thành một chuỗi bài **hồi quy nhỏ**, mỗi bài đều có mất mát tính được.

### 8.3. Ba họ, cùng một khung

![Hình 11](figs/bd11_mohinhsinh.png)

**Hình 11.** Ba cách trả lời cùng một câu hỏi. Cả ba đều biến nhiễu Gauss thành mẫu.

| | VAE | GAN | Khuếch tán |
|---|---|---|---|
| Huấn luyện bằng | chặn dưới ELBO | trò chơi hai bên | hồi quy đoán nhiễu |
| Ổn định | cao | **thấp** | cao |
| Chất lượng mẫu | thường **mờ** | sắc nét | sắc nét |
| Phủ hết các chế độ | có | **hay thiếu** | có |
| Tốc độ lấy mẫu | một lần chạy | một lần chạy | **nhiều bước** |
| Có không gian ẩn dùng được | **có** | có phần | không trực tiếp |
| Đánh giá được bằng hợp lý | có (chặn dưới) | **không** | có (chặn dưới) |

Bảng này giải thích luôn diễn biến lịch sử. GAN thống trị tạo ảnh khoảng 2015–2020 vì nó cho mẫu sắc nét nhất. Khuếch tán thay thế nó từ khoảng 2021 vì nó giữ được độ sắc nét **mà không phải trả giá bằng sự bất ổn** — đổi lại là lấy mẫu chậm, và chậm thì còn tối ưu được chứ bất ổn thì không.

---

## 9. Tự mã hoá biến phân

### 9.1. Bắt đầu từ bộ tự mã hoá thường, và xem nó thiếu gì

Bộ tự mã hoá là ý tưởng đơn giản: nén $x$ thành $z$ ngắn hơn rồi dựng lại, huấn luyện để bản dựng lại giống bản gốc.

$$x \;\xrightarrow{\text{mã hoá}}\; z \;\xrightarrow{\text{giải mã}}\; \hat x, \qquad \min \|x - \hat x\|^2.$$

Nó nén tốt. Nhưng nó **không sinh được**, và lý do đáng hiểu rõ.

Muốn sinh mẫu mới, ta phải lấy một $z$ ngẫu nhiên rồi giải mã. Nhưng lấy ngẫu nhiên **từ phân phối nào**? Bộ tự mã hoá không hứa gì về hình dạng của tập các $z$ mà nó dùng. Chúng có thể nằm rải rác thành những cụm rời rạc, hoặc trên một đường cong mảnh, hoặc bất kỳ hình gì. Lấy một điểm ngẫu nhiên trong không gian ẩn thì nhiều khả năng rơi vào **chỗ trống** mà bộ giải mã chưa từng thấy, và nó sẽ cho ra rác.

Vấn đề không phải nén kém. Vấn đề là **không gian ẩn không có cấu trúc để lấy mẫu**.

### 9.2. Cách sửa: ép không gian ẩn có hình dạng biết trước

VAE sửa đúng chỗ đó. Thay vì cho bộ mã hoá trả về **một điểm** $z$, nó trả về **một phân phối** $q_\phi(z\mid x) = \mathcal{N}(\mu_\phi(x),\, \sigma^2_\phi(x))$. Rồi nó thêm một số hạng phạt, ép phân phối ấy phải gần với một tiên nghiệm cố định $p(z) = \mathcal{N}(0, I)$.

Hàm mục tiêu là **chặn dưới bằng chứng** (ELBO):

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{tái dựng}} \;-\; \beta\,\underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{ép về tiên nghiệm}}.$$

Đọc hai số hạng bằng lời:

- Số hạng đầu nói: **giải mã lại phải ra đúng $x$.** Đây là phần "nén không mất thông tin".
- Số hạng sau nói: **phân phối của $z$ phải trông như $\mathcal{N}(0,I)$.** Đây là phần làm cho việc lấy mẫu trở nên hợp lệ — vì nếu mọi $q(z|x)$ đều gần $\mathcal{N}(0,I)$ thì lấy mẫu từ $\mathcal{N}(0,I)$ sẽ rơi vào vùng bộ giải mã đã quen.

Hai số hạng kéo ngược nhau, và $\beta$ là nút chỉnh giữa chúng. Với $\beta = 1$ ta có VAE gốc; $\beta$ khác 1 cho $\beta$-VAE.

> **Một chi tiết kỹ thuật cần biết nhưng đừng để nó che mất ý chính.** Lấy mẫu $z \sim q_\phi(z|x)$ là một phép ngẫu nhiên, mà gradient không đi qua phép ngẫu nhiên được. Mẹo **tái tham số hoá** giải quyết: viết $z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$ với $\varepsilon \sim \mathcal{N}(0,I)$. Bây giờ phần ngẫu nhiên nằm ở $\varepsilon$ — thứ không phụ thuộc tham số — nên gradient chảy được qua $\mu$ và $\sigma$ bình thường. Đó là một mẹo cài đặt chứ không phải ý tưởng của VAE, và không nên nhầm hai thứ.

### 9.3. $\beta$ thực sự điều khiển cái gì

Câu trả lời thường nghe là "$\beta$ cân giữa tái dựng và phạt chuẩn". Đúng nhưng mơ hồ. Thí nghiệm sau cho câu trả lời sắc hơn.

Thiết kế: dữ liệu 600 điểm trong $\mathbb{R}^4$, sinh từ **đúng 2** yếu tố ẩn cộng nhiễu nhỏ. Ta cố tình cho VAE **6 chiều ẩn** — nhiều hơn cần — rồi xem nó làm gì với phần thừa.

Cách đo: một chiều gọi là **còn sống** nếu KL của riêng chiều ấy lớn hơn 0 đáng kể, tức hậu nghiệm $q(z_i|x)$ còn khác tiên nghiệm $\mathcal{N}(0,1)$. Chiều nào có KL bằng 0 thì nó đúng bằng tiên nghiệm, nghĩa là **không mang tin gì về $x$**.

![Hình 8](figs/bd08_vae.png)

**Hình 8.** Trái: hai số hạng đánh đổi nhau theo $\beta$. Phải: số chiều ẩn còn sống.

| $\beta$ | Sai số tái dựng | Tổng KL | Số chiều còn sống | KL của từng chiều |
|---|---|---|---|---|
| 0 | 0,1219 | 34,79 | **6** | 14,66 · 15,03 · 0,27 · 2,70 · 0,41 · 1,72 |
| 0,05 | 0,1296 | 5,66 | 6 | 2,57 · 2,60 · 0,09 · 0,16 · 0,10 · 0,13 |
| 0,2 | 0,1476 | 3,50 | 4 | 1,73 · 1,74 · 0,006 · 0,017 · 0,006 · 0,011 |
| **1,0** | 0,3861 | 2,02 | **2** | 1,015 · 1,006 · 0 · 0 · 0 · 0 |
| 4,0 | 4,78 | 0,61 | 2 | 0,36 · 0,007 · 0 · 0,24 · 0 · 0 |
| 16,0 | 15,28 | 0,0000 | **0** | 0 · 0 · 0 · 0 · 0 · 0 |

Ba hàng đáng dừng lại:

**$\beta = 0$: cả 6 chiều đều sống.** Không có sức ép nào cả, nên mô hình dùng hết chỗ được cho. Đây đúng là bộ tự mã hoá thường ở Mục 9.1 — tái dựng tốt nhất bảng (0,1219) nhưng không gian ẩn vô tổ chức, không lấy mẫu được.

**$\beta = 1$: đúng 2 chiều sống lại.** Đây là kết quả đẹp nhất của chương. Dữ liệu được sinh từ đúng 2 yếu tố, và VAE **tự tìm ra con số 2** — không ai nói với nó. Bốn chiều thừa bị tắt hẳn (KL về đúng 0). Nói cách khác, số hạng KL hoạt động như một **phép chọn số chiều tự động**.

**$\beta = 16$: không chiều nào sống.** Đây là **sụp hậu nghiệm**: $z$ không còn liên quan gì tới $x$, và bộ giải mã chỉ còn biết đoán giá trị trung bình. Sai số tái dựng nhảy lên 15,28 — tệ hơn 125 lần so với $\beta = 0$. Mô hình đã "thắng" trọn vẹn số hạng KL bằng cách vứt bỏ toàn bộ thông tin.

### 9.4. Vì sao mẫu của VAE thường mờ

Đây là câu hỏi phỏng vấn hay gặp và câu trả lời thì gọn hơn người ta tưởng.

Số hạng tái dựng với giả thiết nhiễu Gauss chính là bình phương sai lệch — [Mục 10.2 của *Nền tảng*](nentang-ch10.html) đã chứng minh điều đó. Mà thứ tối thiểu hoá bình phương sai lệch là **kỳ vọng có điều kiện**:

$$\hat x = \mathbb{E}[x \mid z].$$

Bây giờ hãy nghĩ điều đó nghĩa là gì với ảnh. Nếu ứng với một $z$ có nhiều ảnh thật khả dĩ — cùng một khuôn mặt nhưng biên tóc hơi khác chỗ — thì **trung bình của chúng là một ảnh nhoè ở biên**. Trung bình của nhiều ảnh sắc nét lệch nhau không phải một ảnh sắc nét.

Vậy độ mờ không phải lỗi huấn luyện hay thiếu dung lượng mô hình. Nó là **hệ quả trực tiếp của việc chọn hàm mất mát bình phương**, tức của việc giả định nhiễu Gauss. Đổi giả định thì đổi hiện tượng — và đó chính là điều GAN làm ở chương sau.

---

## 10. Mạng đối sinh

### 10.1. Ý tưởng: bỏ hẳn hàm hợp lý

Mục 9.4 cho thấy độ mờ của VAE đến từ việc phải viết ra một hàm mất mát đo "giống thật". GAN đặt câu hỏi táo bạo: **nếu không viết hàm ấy mà đi học nó thì sao?**

Hai mạng, hai mục tiêu ngược nhau:

- **Bộ sinh** $G$ biến nhiễu $z$ thành mẫu giả $G(z)$.
- **Bộ phân biệt** $D$ nhận một mẫu và đoán nó là thật hay giả.

$D$ được huấn luyện để phân biệt cho giỏi; $G$ được huấn luyện để làm $D$ nhầm. Ví von quen thuộc là kẻ làm tiền giả và cảnh sát, mỗi bên giỏi lên thì ép bên kia phải giỏi hơn.

Bài toán viết ra là:

$$\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))].$$

Điểm hay là ở chỗ **$G$ không bao giờ nhìn thấy dữ liệu thật**. Nó chỉ nhận gradient qua $D$. Toàn bộ thông tin về "thế nào là thật" đi qua một kênh duy nhất là ý kiến của $D$.

### 10.2. Chỗ hỏng của bài toán minimax gốc — và tôi đã đo được nó

Đây là phần tôi định viết về sụp chế độ. Số liệu không cho phép, và cái nó cho lại hay hơn.

Thí nghiệm: 8 cụm Gauss đặt đều trên một vòng tròn, GAN nhỏ sinh điểm 2 chiều, chạy với hai hàm mất mát khác nhau cho bộ sinh, mỗi hàm 4 hạt giống.

| Hàm mất mát của bộ sinh | Chế độ phủ được (4 hạt giống) | % điểm rơi vào cụm |
|---|---|---|
| Tối thiểu $\log(1 - D(G(z)))$ — **đúng bài toán gốc** | $[0, 0, 0, 0]$ | **0,0%** |
| Tối đa $\log D(G(z))$ — bản "không bão hoà" | $[8, 8, 8, 8]$ | 15,3% |

![Hình 9](figs/bd09_gan.png)

**Hình 9.** Hai hàm mất mát, hai hạt giống mỗi loại. Xám là dữ liệu thật, đỏ là mẫu sinh ra.

**Dạng minimax gốc thất bại hoàn toàn** — không học được gì, 0 trên 8 chế độ, ở cả bốn hạt giống.

Lý do nằm ở một phép lấy đạo hàm mà ai cũng làm được:

$$\frac{\partial}{\partial s}\log(1 - \sigma(s)) = -\sigma(s).$$

Lúc bắt đầu huấn luyện, bộ sinh còn tệ nên bộ phân biệt nhận ra hàng giả rất dễ, tức $\sigma(s) \approx 0$ cho mẫu giả. Nhưng gradient của bộ sinh **chính bằng** $-\sigma(s)$, nên nó cũng $\approx 0$.

Nói cách khác: **bộ sinh càng tệ thì càng ít tín hiệu để sửa.** Đó là một vòng luẩn quẩn chết người, và nó xảy ra ngay ở bước đầu tiên khi bộ sinh luôn luôn tệ.

Bài báo GAN gốc nêu đúng vấn đề này và đề xuất cách chữa: thay vì cho $G$ **tối thiểu** $\log(1-D(G(z)))$, cho nó **tối đa** $\log D(G(z))$. Hai hàm có cùng điểm tối ưu nhưng đạo hàm khác hẳn:

$$\frac{\partial}{\partial s}\log \sigma(s) = 1 - \sigma(s),$$

và đại lượng này **lớn nhất đúng khi bộ sinh đang tệ nhất** ($\sigma(s)\approx 0$). Đúng cái ta cần.

> **Đây là một bài học vượt ra ngoài GAN.** Hai hàm mất mát có cùng nghiệm tối ưu vẫn có thể cho hành vi huấn luyện khác nhau một trời một vực, vì cái quyết định việc học là **gradient**, không phải vị trí của điểm tối ưu. Cùng ý ấy xuất hiện ở [Chương 6 của *Nền tảng*](nentang-ch06.html) khi so mất mát perceptron với hinge và logistic.

### 10.3. Còn sụp chế độ thì sao — nói thẳng

Tôi vào thí nghiệm này với dự định minh hoạ sụp chế độ: bộ sinh tìm được một vài chỗ làm bộ phân biệt chịu thua rồi ở lì tại đó, bỏ qua phần còn lại của phân phối.

**Số liệu không ủng hộ.** Với bản không bão hoà, cả bốn hạt giống đều phủ **đủ cả 8 chế độ**. Tôi cũng thử làm bộ phân biệt yếu đi, giảm cỡ lô, giảm số vòng — vẫn 8/8 ở mọi cấu hình.

Nên tôi không viết rằng thí nghiệm này cho thấy sụp chế độ, vì nó không cho thấy. Sụp chế độ là hiện tượng **có thật và rất hay gặp** trên dữ liệu nhiều chiều; nhưng khẳng định nó từ một thí nghiệm hai chiều mà bản thân thí nghiệm ấy bác bỏ thì là nói quá.

Điều thí nghiệm **có** cho thấy về chất lượng: chỉ **15,3%** số điểm sinh ra rơi vào phạm vi một cụm. Phần lớn mẫu nằm rải giữa các cụm — tức bộ sinh phủ đúng vị trí các chế độ nhưng **phân phối không khớp**. Đó là một cách hỏng khác, nhẹ hơn nhưng vẫn là hỏng, và nó nhắc rằng "phủ hết chế độ" chưa phải "học đúng phân phối".

### 10.4. Vì sao GAN khó huấn luyện, nói cho đúng

Ngoài chuyện gradient ở Mục 10.2, còn ba lý do có tính cấu trúc:

1. **Không có hàm mục tiêu nào giảm đơn điệu.** Khi huấn luyện một mô hình thường, mất mát giảm là dấu hiệu tiến bộ. Với GAN, mất mát của $G$ tăng có thể là vì $G$ tệ đi, mà cũng có thể vì $D$ vừa giỏi lên. **Không đọc được đường cong mất mát** là điều gây bực bội nhất khi làm việc với GAN.
2. **Đây là bài toán điểm yên ngựa, không phải cực tiểu.** Ta đi tìm một cân bằng Nash chứ không phải đáy của một thung lũng. Xuống dốc không có bảo đảm hội tụ tới điểm yên ngựa, và có thể đi vòng quanh mãi.
3. **Không có cách đánh giá khách quan.** VAE và khuếch tán đều cho một chặn dưới của log hợp lý để so sánh giữa các mô hình. GAN không có. Các chỉ số như FID đều là chỉ số **thay thế**, và chúng có điểm mù riêng.

Ba điều này cộng lại giải thích vì sao khuếch tán thay thế được GAN gần như hoàn toàn, dù chất lượng mẫu ban đầu không hơn: **huấn luyện khuếch tán là giải một bài hồi quy bình thường**, với mất mát đọc được và hội tụ dự đoán được.

---

## 11. Mô hình khuếch tán

### 11.1. Ý tưởng, và vì sao nó lại hợp lý

Ý tưởng nghe gần như đùa: **phá dần một bức ảnh bằng nhiễu cho tới khi nó thành nhiễu thuần tuý, rồi học cách đi ngược lại.**

Vì sao cách này lại dễ hơn học thẳng $p(x)$? Vì nó chia một bài toán rất khó thành **rất nhiều bài toán rất dễ**.

Học sinh một bức ảnh từ nhiễu trong một bước là bài toán khó. Nhưng học **gỡ một chút nhiễu** khỏi một bức ảnh đã hơi nhiễu — đó là bài toán khử nhiễu bình thường, và nó chỉ là hồi quy. Ghép 1 000 bước khử nhiễu nhỏ lại thì được một bước sinh lớn.

Đây là một mẫu tư duy đáng nhớ và không chỉ dùng cho khuếch tán: **khi một phép biến đổi quá khó để học trực tiếp, hãy tìm cách viết nó thành chuỗi nhiều phép dễ.**

### 11.2. Quá trình thuận, và dạng đóng của nó

Quá trình thuận thêm nhiễu Gauss từng chút một:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big),$$

với $\beta_t$ nhỏ, tăng dần theo lịch định sẵn. Hệ số $\sqrt{1-\beta_t}$ co tín hiệu lại một chút mỗi bước, đúng bằng lượng cần để phương sai tổng giữ nguyên.

Nếu phải chạy cả $t$ bước để lấy được $x_t$ thì huấn luyện sẽ chậm không chịu nổi. Nhưng vì mọi bước đều là Gauss tuyến tính nên gộp lại được thành **dạng đóng**. Đặt $\alpha_t = 1-\beta_t$ và $\bar\alpha_t = \prod_{s\le t}\alpha_s$:

$$\boxed{\;q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)\;}$$

Đây là chi tiết làm cho cả phương pháp chạy được. Muốn lấy $x_t$ với $t$ bất kỳ, ta **nhảy thẳng một phát**, không cần mô phỏng 999 bước trung gian.

**Kiểm chứng bằng số.** Mô phỏng 200 000 quỹ đạo thật sự đi từng bước, rồi so mô men với dạng đóng:

| $t$ | TB mô phỏng | TB dạng đóng | Phương sai mô phỏng | Phương sai dạng đóng |
|---|---|---|---|---|
| 1 | 1,99984 | 1,99978 | 0,00022 | 0,00022 |
| 10 | 1,99763 | 1,99781 | 0,00219 | 0,00219 |
| 100 | 1,89199 | 1,89224 | 0,10485 | 0,10486 |
| 400 | 0,88021 | 0,87994 | 0,80511 | 0,80643 |
| 999 | 0,00936 | 0,01271 | 1,00484 | 0,99996 |

Bốn cột khớp nhau trong sai số lấy mẫu. Dạng đóng đúng.

### 11.3. Lịch nhiễu, đọc qua tỉ số tín hiệu trên nhiễu

$\bar\alpha_t$ nói trực tiếp còn lại bao nhiêu tín hiệu. Đại lượng hay dùng hơn là **tỉ số tín hiệu trên nhiễu**:

$$\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}.$$

![Hình 10](figs/bd10_diffusion.png)

**Hình 10.** Trái: tín hiệu còn lại. Phải: SNR, trục dọc log.

| $t$ | $\bar\alpha_t$ | SNR | Biên độ tín hiệu còn lại |
|---|---|---|---|
| 0 | 0,999900 | 9 999,0 | 99,99% |
| 50 | 0,969951 | 32,28 | 98,49% |
| 200 | 0,656347 | 1,91 | 81,02% |
| 500 | 0,077797 | 0,0844 | 27,89% |
| 800 | 0,001508 | 0,0015 | 3,88% |
| 999 | $4{,}0\times10^{-5}$ | $4{,}0\times10^{-5}$ | **0,64%** |

Ở $t = 999$, chỉ còn 0,64% biên độ tín hiệu — tức $x_T$ gần như là nhiễu Gauss thuần tuý. Đó đúng là điều ta cần, vì lúc lấy mẫu ta bắt đầu từ $\mathcal{N}(0,I)$ và cần điểm xuất phát ấy khớp với điểm kết thúc của quá trình thuận.

SNR cũng cho biết **mô hình đang học gì ở mỗi bước**, và đây là cách đọc mà tôi thấy hữu ích nhất:

- $t$ **nhỏ** (SNR rất cao): ảnh gần như nguyên vẹn, mô hình học **chi tiết nhỏ và kết cấu**.
- $t$ **lớn** (SNR rất thấp): gần như chỉ còn nhiễu, mô hình học **bố cục thô, màu tổng thể**.

Nên khi lấy mẫu và đi ngược từ $t=T$ về $t=0$, mô hình quyết định bố cục trước rồi mới điền chi tiết sau — giống hệt cách người ta vẽ tranh. Điều đó không được lập trình vào; nó là hệ quả của lịch nhiễu.

### 11.4. Huấn luyện: chỉ là hồi quy

Đây là chỗ khuếch tán trở nên dễ chịu so với GAN. Từ dạng đóng ta viết được

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I).$$

Bây giờ **huấn luyện một mạng đoán lại $\varepsilon$** từ $x_t$ và $t$:

$$\mathcal{L} = \mathbb{E}_{x_0,\,\varepsilon,\,t}\Big[\big\|\varepsilon - \varepsilon_\theta(x_t, t)\big\|^2\Big].$$

Nhìn kỹ công thức này. Nó là **bình phương sai lệch** — một bài hồi quy hoàn toàn bình thường, thứ đã học từ [Chương 4 của *Nền tảng*](nentang-ch04.html). Không có trò chơi, không có điểm yên ngựa, không có hai mạng đánh nhau. Mất mát giảm nghĩa là mô hình tốt lên, và đọc được đường cong mất mát.

Thủ tục huấn luyện gọn tới mức viết ra vừa mấy dòng:

1. Lấy một ảnh thật $x_0$ từ dữ liệu.
2. Chọn ngẫu nhiên một bước $t \in \{1,\dots,T\}$.
3. Lấy ngẫu nhiên $\varepsilon \sim \mathcal{N}(0,I)$, dựng $x_t$ bằng dạng đóng.
4. Cho mạng đoán $\varepsilon$, tính bình phương sai lệch, cập nhật.

Vì sao lại đoán $\varepsilon$ chứ không đoán thẳng $x_0$? Về mặt toán thì hai cách tương đương — biết một cái thì suy ra cái kia từ dạng đóng. Nhưng đoán $\varepsilon$ cho mục tiêu có **phương sai đồng đều hơn giữa các $t$**, nên huấn luyện ổn định hơn. Đây là lựa chọn thực nghiệm, và bài báo DDPM nói rõ như vậy.

### 11.5. Lấy mẫu chậm, và người ta chữa thế nào

Nhược điểm duy nhất nhưng lớn: sinh một mẫu cần chạy mạng $T$ lần. Với $T = 1000$ thì chậm hơn GAN khoảng một nghìn lần.

| Cách chữa | Ý tưởng | Kết quả điển hình |
|---|---|---|
| DDIM | bỏ tính ngẫu nhiên ở bước ngược, cho phép nhảy cách bước | 1000 → 50 bước |
| Lịch nhiễu tốt hơn | phân bố các bước theo SNR thay vì đều theo $t$ | ít bước hơn với cùng chất lượng |
| Chưng cất | huấn luyện một mô hình học **gộp** nhiều bước thành một | tới mức 1–4 bước |
| Khuếch tán trong không gian ẩn | chạy toàn bộ quá trình trong không gian ẩn nén của một VAE | rẻ hơn rất nhiều theo chiều không gian |

Dòng cuối là điều làm cho các mô hình tạo ảnh hiện nay chạy được trên máy cá nhân, và nó là một ví dụ đẹp của việc **ghép hai họ mô hình**: dùng VAE để nén ảnh xuống không gian ẩn nhỏ, rồi chạy khuếch tán ở đó. VAE lo phần nén, khuếch tán lo phần sinh, mỗi bên làm đúng việc nó giỏi.

---

## 12. Nhập môn học tăng cường

### 12.1. Khác học có giám sát ở chỗ nào

Trong học có giám sát, mỗi mẫu đi kèm **đáp án đúng**. Mô hình đoán, so với đáp án, sửa.

Trong học tăng cường không có đáp án. Chỉ có **phần thưởng** — một con số nói việc vừa làm tốt hay tệ tới đâu. Ba khác biệt kéo theo, và cả ba đều làm bài toán khó hơn hẳn:

| | Học có giám sát | Học tăng cường |
|---|---|---|
| Tín hiệu | đáp án đúng cho mỗi mẫu | một con số, không nói nên làm gì thay thế |
| Thời điểm | ngay lập tức | có thể **trễ rất lâu** sau hành động gây ra nó |
| Dữ liệu | cho sẵn, cố định | **do chính chính sách sinh ra** |

Dòng cuối là dòng nặng nhất và hay bị bỏ qua. Trong học có giám sát, tập dữ liệu là cố định. Trong học tăng cường, **hành động của tác tử quyết định nó nhìn thấy gì tiếp theo**. Một chính sách tồi sẽ chỉ thu thập được dữ liệu tồi, rồi từ dữ liệu tồi ấy lại học ra chính sách tồi. Đây là một vòng phản hồi, và nó là gốc của mọi khó khăn về sau — cùng bản chất với vòng phản hồi thoái hoá ở [Mục 12.5 của *MLOps*](mlops-ch12.html).

### 12.2. Những khái niệm tối thiểu

- **Trạng thái** $s$: tình hình hiện tại.
- **Hành động** $a$: việc có thể làm.
- **Phần thưởng** $r$: con số nhận được sau khi làm.
- **Chính sách** $\pi(a \mid s)$: quy tắc chọn hành động.
- **Hàm giá trị hành động** $Q(s,a)$: tổng phần thưởng **kỳ vọng trong tương lai** nếu làm $a$ tại $s$ rồi chơi tốt về sau.

Chữ "trong tương lai" là chìa khoá. Một hành động có thể cho phần thưởng tức thì rất tệ mà vẫn là hành động đúng, vì nó mở ra phần thưởng lớn sau đó. $Q$ tồn tại chính để nắm bắt điều ấy.

Phương trình Bellman nối $Q$ với chính nó:

$$Q(s,a) = \mathbb{E}\big[r + \gamma \max_{a'} Q(s', a')\big].$$

Đọc bằng lời: *giá trị của việc làm $a$ tại $s$ bằng phần thưởng nhận ngay, cộng với giá trị của nước đi tốt nhất ở trạng thái kế tiếp, chiết khấu đi hệ số $\gamma$.* Hệ số $\gamma \in (0,1)$ nói ta coi trọng tương lai tới mức nào; $\gamma$ gần 1 là nhìn xa, gần 0 là chỉ nhìn trước mắt.

**Q-learning** biến phương trình ấy thành quy tắc cập nhật, và đó là toàn bộ thuật toán:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\Big[\underbrace{r + \gamma\max_{a'}Q(s',a')}_{\text{mục tiêu}} - Q(s,a)\Big].$$

Phần trong ngoặc là **sai số thời gian** — chênh lệch giữa điều ta vừa quan sát được và điều ta đang tin. Cập nhật kéo niềm tin về phía quan sát, đúng tinh thần gradient descent.

### 12.3. Khám phá và khai thác

Nếu luôn chọn hành động có $Q$ cao nhất, ta không bao giờ thử thứ mới, nên không bao giờ biết có gì tốt hơn. Nếu luôn chọn ngẫu nhiên, ta biết nhiều nhưng không tận dụng được gì.

Cách chữa quen thuộc là **$\varepsilon$-tham lam**: với xác suất $\varepsilon$ chọn ngẫu nhiên, còn lại chọn tham lam.

Câu chuyện thường dừng ở đó. Nhưng thí nghiệm sau cho thấy nó bỏ sót một nửa vấn đề.

### 12.4. Thí nghiệm nói gì — và nó nói ngược lại điều tôi định viết

Thiết kế: lưới $7\times7$, xuất phát góc dưới trái, đích ở góc trên phải ($+10$), bốn ô bẫy ($-10$). Ba cấu hình chỉ khác nhau ở **phạt mỗi bước** và **giá trị khởi tạo của $Q$**. Đánh giá là chạy chính sách tham lam đã học, không còn khám phá.

![Hình 12](figs/bd12_rl.png)

**Hình 12.** Trái: chính sách học được. Phải: tỉ lệ tới đích theo $\varepsilon$ ở ba cấu hình.

**Cấu hình 1 — phạt $-0{,}1$ mỗi bước, $Q_0 = 0$:**

| $\varepsilon$ | 0 | 0,05 | 0,1 | 0,3 | 1,0 |
|---|---|---|---|---|---|
| % tới đích | **100%** | 100% | 100% | 100% | 100% |

$\varepsilon = 0$ vẫn thành công. Câu "phải có $\varepsilon$ để khám phá" không giải thích được điều này.

**Cấu hình 2 — bỏ phạt mỗi bước, $Q_0 = 0$:**

| $\varepsilon$ | 0 | 0,05 | 0,1 | 0,3 | 1,0 |
|---|---|---|---|---|---|
| % tới đích | **0%** | 100% | 100% | 100% | 100% |

Bây giờ $\varepsilon = 0$ thất bại hoàn toàn.

**Cấu hình 3 — phạt $-0{,}1$, khởi tạo bi quan $Q_0 = -20$:**

| $\varepsilon$ | 0 | 0,05 | 0,1 | 0,3 | 1,0 |
|---|---|---|---|---|---|
| % tới đích | 0% | 0% | 0% | **0%** | 100% |

Còn tệ hơn: $\varepsilon$ tới 0,3 vẫn không đủ.

**Giải thích, và đây là điều đáng mang đi.**

Ở cấu hình 1, mọi phần thưởng đều âm nhưng $Q$ khởi tạo bằng 0. Nghĩa là **hành động chưa thử bao giờ cũng trông hấp dẫn hơn hành động đã thử** — vì hành động đã thử có $Q$ bị kéo xuống âm, còn hành động chưa thử vẫn ở 0. Chính sách tham lam do đó **tự đi khám phá** mà không cần $\varepsilon$ nào cả. Đây gọi là **khởi tạo lạc quan**, và nó là một cơ chế khám phá hoàn chỉnh.

Ở cấu hình 2, bỏ phạt bước đi thì mất luôn tính lạc quan ấy: mọi $Q$ đều bằng 0, `argmax` luôn trả về cùng một chỉ số, nên tác tử đi mãi một hướng và không bao giờ tìm ra đích.

Ở cấu hình 3, khởi tạo bi quan làm điều ngược lại: hành động đầu tiên tình cờ thử được kéo **lên** gần 0 trong khi các hành động khác vẫn ở $-20$, nên tác tử bám chặt lấy nó. Đây là bi quan tạo ra sự cố chấp.

> **Kết luận:** khám phá không chỉ đến từ $\varepsilon$. **Cách khởi tạo hàm giá trị cũng là một cơ chế khám phá**, và trong một số trường hợp nó mạnh hơn. Tôi đã định viết "$\varepsilon = 0$ thì kẹt", nhưng cấu hình 1 bác bỏ điều đó, nên phần chữ được viết lại theo số liệu.

---

## 13. Gradient chính sách

### 13.1. Vì sao Q-learning không đủ cho mô hình ngôn ngữ

Q-learning cần tính $\max_{a'} Q(s',a')$ ở mỗi bước. Với lưới $7\times7$ và 4 hành động thì dễ. Với mô hình ngôn ngữ, "hành động" là chọn token tiếp theo trong từ vựng 50 000 — vẫn tính được — nhưng "trạng thái" là **toàn bộ văn bản đã sinh**, và số trạng thái là vô hạn.

Quan trọng hơn: ta không muốn một bảng $Q$. Ta muốn **chỉnh trực tiếp chính sách**, vì chính sách chính là mô hình ngôn ngữ mà ta đã có.

Đó là chỗ gradient chính sách vào cuộc: thay vì học giá trị rồi suy ra chính sách, **tham số hoá chính sách và đi gradient thẳng trên phần thưởng kỳ vọng**.

### 13.2. Định lý gradient chính sách

Ta muốn tối đa $J(\theta) = \mathbb{E}_{a\sim\pi_\theta}[R(a)]$. Vấn đề là $a$ được lấy mẫu **từ chính** $\pi_\theta$, nên không lấy đạo hàm theo kiểu thường được.

Mẹo là đồng nhất thức sau, thường gọi là thủ thuật log:

$$\nabla_\theta \pi_\theta(a) = \pi_\theta(a)\,\nabla_\theta \log \pi_\theta(a).$$

Từ đó:

$$\nabla_\theta J = \sum_a R(a)\,\nabla_\theta\pi_\theta(a) = \sum_a \pi_\theta(a)\,R(a)\,\nabla_\theta\log\pi_\theta(a) = \mathbb{E}_{a\sim\pi_\theta}\big[R(a)\,\nabla_\theta\log\pi_\theta(a)\big].$$

Vế phải là một **kỳ vọng**, nên ước lượng được bằng cách lấy mẫu. Đó là thuật toán REINFORCE.

Đọc công thức bằng lời cho dễ nhớ: *lấy mẫu một hành động, xem được thưởng bao nhiêu, rồi đẩy log xác suất của hành động ấy lên với cường độ tỉ lệ phần thưởng.* Thưởng cao thì làm hành động ấy dễ xảy ra hơn; thưởng âm thì đẩy ngược lại.

### 13.3. Vấn đề: phương sai

Ước lượng ấy **không thiên lệch** nhưng **phương sai rất lớn**. Lý do nhìn ra ngay từ công thức: nếu mọi phần thưởng đều dương — chẳng hạn nằm trong khoảng $[1, 3]$ — thì $R(a) > 0$ với mọi $a$, nên ta **đẩy log xác suất của mọi hành động lên**, kể cả hành động tệ. Việc học chỉ diễn ra nhờ chênh lệch *tương đối* giữa các cường độ đẩy, mà chênh lệch ấy chìm trong nhiễu lấy mẫu.

Cách chữa gọn tới bất ngờ: **trừ đi một đường nền** $b$.

$$\nabla_\theta J = \mathbb{E}\big[(R(a) - b)\,\nabla_\theta\log\pi_\theta(a)\big].$$

Vì sao được phép trừ? Vì

$$\mathbb{E}\big[b\,\nabla_\theta\log\pi_\theta(a)\big] = b\sum_a \pi_\theta(a)\nabla_\theta\log\pi_\theta(a) = b\sum_a\nabla_\theta\pi_\theta(a) = b\,\nabla_\theta\!\left(\sum_a \pi_\theta(a)\right) = b\,\nabla_\theta 1 = 0.$$

Tổng xác suất luôn bằng 1, nên đạo hàm của nó bằng 0, nên số hạng thêm vào có kỳ vọng bằng 0 với **mọi** hằng số $b$. Ta được đổi phương sai mà không mất gì.

### 13.4. Đo xem nó cắt được bao nhiêu

Bài toán một bước, 6 hành động, phần thưởng thật $[1{,}0;\ 1{,}2;\ 0{,}9;\ 1{,}1;\ \mathbf{3{,}0};\ 1{,}05]$ — tức hành động thứ 5 tốt hơn hẳn, còn lại xấp xỉ nhau. Chính sách hiện tại là phân phối đều.

| Số mẫu | Độ lệch chuẩn — không nền | Có nền | Giảm được |
|---|---|---|---|
| 16 | 0,13674 | 0,05776 | **2,37×** |
| 64 | 0,06745 | 0,02944 | 2,29× |
| 256 | 0,03376 | 0,01449 | 2,33× |
| 1 024 | 0,01632 | 0,00718 | 2,27× |

Đường nền cắt độ lệch chuẩn đi khoảng **2,3 lần** ở mọi cỡ mẫu. Vì cần số mẫu tỉ lệ với **bình phương** độ lệch chuẩn để đạt cùng độ chính xác, điều này nghĩa là cần ít hơn khoảng **5,3 lần** số mẫu.

Và kiểm chứng rằng nó không làm lệch ước lượng:

```text
gradient TB khong nen : [-0.06274 -0.03002 -0.08005 -0.04598  0.27275 -0.05396]
gradient TB co nen    : [-0.06272 -0.02927 -0.07917 -0.04593  0.2716  -0.05451]
lech lon nhat: 1.15e-03
```

Hai vector trùng nhau trong sai số lấy mẫu, và cả hai đều chỉ đúng về hành động thứ 5 — hành động duy nhất có thành phần dương. Đường nền chỉ cắt nhiễu chứ không đổi hướng.

### 13.5. Từ đường nền tới lợi thế

Trong bài toán nhiều bước, đường nền tốt nhất phụ thuộc trạng thái: $b(s) = V(s)$, tức giá trị kỳ vọng của trạng thái ấy. Khi đó

$$R - b = Q(s,a) - V(s) =: A(s,a),$$

gọi là **hàm lợi thế**. Nó trả lời đúng câu hỏi cần hỏi: *hành động này tốt hơn mức trung bình ở trạng thái này bao nhiêu?*

Đây là nền của họ thuật toán actor–critic, và của **PPO** — thứ được dùng trong RLHF. PPO thêm một chi tiết nữa: nó **giới hạn mức thay đổi của chính sách trong mỗi lần cập nhật**, vì bước đi quá lớn trong không gian chính sách có thể phá hỏng mọi thứ và không có cách quay lại. Tinh thần ấy giống hệt lý do tốc độ học khi tinh chỉnh phải nhỏ (Mục 6.4).

---

## 14. RLHF: học từ so sánh của con người

### 14.1. Vì sao cần giai đoạn này

![Hình 14](figs/bd14_rlhf_quytrinh.png)

**Hình 14.** Ba giai đoạn. DPO ở Chương 15 gộp 3a và 3b làm một.

Sau tiền huấn luyện, mô hình biết ngôn ngữ nhưng **không biết nghe lời**. Hỏi "thủ đô Pháp là gì?" thì nó có thể trả lời đúng, mà cũng có thể viết tiếp thành một đề kiểm tra địa lý mười câu — vì trong dữ liệu web, một câu hỏi thường đi kèm các câu hỏi khác chứ không phải câu trả lời.

Tinh chỉnh có giám sát (bước 2) chữa được điều đó: cho mô hình xem hàng chục nghìn cặp (yêu cầu, câu trả lời tốt) và bắt chước. Nhưng nó có một trần chặn rõ ràng — **mô hình chỉ giỏi bằng người viết câu trả lời mẫu**. Học bắt chước không bao giờ vượt được người được bắt chước.

Bước 3 phá trần ấy, và cách nó phá là điều đáng hiểu kỹ.

### 14.2. Vì sao học từ so sánh lại mạnh hơn

Lý do nằm ở một bất đối xứng của nhận thức con người: **đánh giá dễ hơn sáng tạo.**

Một người bình thường không viết nổi một bài thơ hay. Nhưng đưa hai bài thơ, họ chỉ ra được bài nào hay hơn. Tương tự, viết một đoạn mã tối ưu thì khó, nhưng nhìn hai đoạn mã và nói đoạn nào rõ ràng hơn thì dễ.

Nên nếu ta thu thập **so sánh** thay vì **ví dụ mẫu**, ta lấy được tín hiệu chất lượng **cao hơn mức mà người dán nhãn tự tạo ra được**. Đó là lý do bước 3 vượt được trần của bước 2. Và đó cũng là lý do người ta chịu bỏ công xây cả một quy trình phức tạp chỉ để dùng một loại nhãn khác.

Thêm một lý do thực tế: so sánh **nhất quán giữa những người dán nhãn** hơn là chấm điểm tuyệt đối. Hỏi mười người "câu này mấy điểm trên thang 10" thì được mười câu trả lời khác nhau; hỏi "câu nào tốt hơn" thì họ đồng ý nhiều hơn hẳn.

### 14.3. Mô hình thưởng Bradley–Terry

Từ dữ liệu so sánh, ta muốn một hàm chấm điểm $r(x,y)$. Mô hình Bradley–Terry giả định xác suất người chọn $y_w$ hơn $y_l$ là

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big).$$

Đây chính là hồi quy logistic trên **hiệu** hai điểm thưởng — xem [Mục 6.3 của *Nền tảng*](nentang-ch06.html). Khớp nó bằng hợp lý cực đại.

Một hệ quả phải biết: $r$ chỉ xác định được **tới một hằng số cộng**. Cộng 100 vào mọi $r$ thì mọi hiệu không đổi, nên hợp lý không đổi. Điều này nghe như một phiền phức nhưng lại vô hại, vì bước sau chỉ dùng $r$ qua hàm mũ rồi chuẩn hoá, nên hằng số bị hằng số chuẩn hoá nuốt mất.

### 14.4. Tối ưu có ràng buộc KL, và dạng đóng của nó

Có $r$ rồi, ta muốn chính sách cho thưởng cao. Nhưng nếu chỉ tối đa thưởng thì mô hình sẽ **phá nát** khả năng ngôn ngữ để chạy theo bất kỳ điểm mù nào của $r$ — hiện tượng gọi là **lách điểm thưởng**. Mô hình thưởng chỉ là một xấp xỉ học từ dữ liệu hữu hạn; đẩy mạnh vào nó thì ta đang tối ưu cái xấp xỉ chứ không phải cái thật.

Cách chặn là buộc chính sách mới đừng đi quá xa chính sách sau bước 2:

$$\max_\pi\; \mathbb{E}_{y\sim\pi}\big[r(x,y)\big] \;-\; \beta\,\mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}}\big).$$

Bài toán này **có nghiệm dạng đóng**, và suy ra được bằng nhân tử Lagrange trên ràng buộc $\sum_y \pi(y) = 1$, đúng công cụ ở [Chương 12 của *Nền tảng*](nentang-ch12.html):

$$\boxed{\;\pi^{*}(y \mid x) = \frac{1}{Z(x)}\;\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big)\;}$$

Công thức này đáng nhìn kỹ. Nó nói: **chính sách tối ưu là chính sách tham chiếu, đánh trọng số lại theo hàm mũ của phần thưởng.** Câu trả lời thưởng cao được nhân lên, thưởng thấp bị nhân xuống, nhưng những gì $\pi_{\text{ref}}$ coi là không thể ($\pi_{\text{ref}} = 0$) thì vẫn không thể — vì nhân với 0 vẫn là 0. Đó chính là cơ chế giữ cho mô hình không nói năng lung tung.

**Kiểm chứng bằng số.** Giải bài toán theo hai đường độc lập: một bên thay vào công thức, một bên chạy BFGS trên đơn hình xác suất mà không biết gì về công thức.

| $\beta$ | Lệch giữa hai đường | $\mathrm{KL}(\pi^*\|\pi_{\text{ref}})$ | $\mathbb{E}[r]$ đạt được |
|---|---|---|---|
| 0,05 | $1{,}6\times10^{-7}$ | 4,741 | 2,438 |
| 0,20 | $3{,}7\times10^{-7}$ | 4,658 | 2,424 |
| 1,00 | $3{,}9\times10^{-9}$ | 0,417 | 0,303 |
| 5,00 | $4{,}6\times10^{-9}$ | 0,017 | $-0{,}350$ |

![Hình 13](figs/bd13_rlhf.png)

**Hình 13.** Trái: $\beta$ nhỏ kéo chính sách xa $\pi_{\text{ref}}$. Phải: mặt đánh đổi — thưởng mua được bằng độ lệch khỏi mô hình gốc.

Hình bên phải là cách đọc đúng về $\beta$: nó không phải một siêu tham số cần "chỉnh cho đúng", nó là **vị trí ta chọn trên một đường đánh đổi**. Muốn thưởng cao hơn thì phải chấp nhận lệch xa $\pi_{\text{ref}}$ hơn, và lệch xa hơn nghĩa là rủi ro lách điểm thưởng cao hơn.

### 14.5. Vì sao quy trình này phiền phức

Tóm lại RLHF cần: huấn luyện một mô hình thưởng riêng, rồi chạy PPO với **bốn** mô hình cùng lúc trong bộ nhớ — chính sách đang học, chính sách tham chiếu, mô hình thưởng, và mô hình giá trị của critic.

Điều đó vừa tốn bộ nhớ vừa khó ổn định, vì nó thừa hưởng toàn bộ khó khăn của học tăng cường đã nêu ở Chương 12 và 13. Trong thực tế, huấn luyện RLHF cần rất nhiều công chỉnh và dễ hỏng.

Câu hỏi tự nhiên: **có thể bỏ bớt không?** Chương sau trả lời là có.

---

## 15. DPO: bỏ hẳn mô hình thưởng

### 15.1. Quan sát then chốt

Nhìn lại công thức nghiệm tối ưu ở Mục 14.4:

$$\pi^{*}(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big).$$

Đây là một quan hệ giữa $\pi^*$ và $r$. Và quan hệ thì **đảo ngược được**. Lấy log rồi giải theo $r$:

$$r(x,y) = \beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).$$

Đọc câu này cho kỹ, vì nó là toàn bộ ý tưởng: **mọi chính sách đều ngầm định nghĩa một hàm thưởng.** Không cần huấn luyện một mô hình thưởng riêng — chính sách *đã là* một mô hình thưởng, viết ở dạng khác.

### 15.2. Và hằng số phiền phức tự biến mất

Vẫn còn $\log Z(x)$, thứ không tính được vì nó cộng trên mọi câu trả lời khả dĩ. Đây là chỗ mà mọi cách tiếp cận trước đó bế tắc.

Nhưng hãy thay biểu thức trên vào hợp lý Bradley–Terry ở Mục 14.3. Nó chỉ dùng **hiệu** hai phần thưởng:

$$r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)} + \underbrace{\beta\log Z(x) - \beta\log Z(x)}_{= \,0}.$$

$\log Z(x)$ phụ thuộc $x$ nhưng **không phụ thuộc $y$**, nên khi lấy hiệu hai câu trả lời cho **cùng một** câu hỏi, nó triệt tiêu sạch. Chi tiết nhỏ ấy là thứ làm cho cả phương pháp chạy được.

Kết quả là hàm mất mát DPO, tối ưu thẳng trên chính sách:

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

Không mô hình thưởng. Không PPO. Không critic. Chỉ một hàm mất mát tính được trực tiếp từ dữ liệu so sánh — và nó có dạng **entropy chéo nhị phân**, thứ quen thuộc từ [Chương 6 của *Nền tảng*](nentang-ch06.html).

### 15.3. Kiểm chứng: hai đường có thật sự gặp nhau không

Lập luận nghe chặt chẽ, nhưng chặt chẽ trên giấy và đúng khi chạy là hai chuyện. Thí nghiệm dựng một không gian nhỏ (8 câu trả lời) để giải được chính xác cả hai đường:

1. **Sinh dữ liệu so sánh** từ một hàm thưởng thật đã biết, qua mô hình Bradley–Terry — 52 588 cặp.
2. **Đường RLHF hai bước:** khớp $\hat r$ bằng hợp lý cực đại, rồi tính $\pi_{\text{RLHF}} \propto \pi_{\text{ref}}\exp(\hat r/\beta)$.
3. **Đường DPO một bước:** tối ưu thẳng $\mathcal{L}_{\text{DPO}}$ trên $\pi$, không hề có mô hình thưởng nào.

| Câu trả lời | $\pi_{\text{ref}}$ | RLHF 2 bước | DPO 1 bước | Lệch |
|---|---|---|---|---|
| 0 | 0,0669 | 0,0015 | 0,0015 | $2{,}0\times10^{-11}$ |
| 1 | 0,0246 | 0,0000 | 0,0000 | $3{,}0\times10^{-11}$ |
| 2 | 0,6201 | 0,4329 | 0,4329 | $4{,}2\times10^{-8}$ |
| 3 | 0,0087 | 0,0080 | 0,0080 | $4{,}7\times10^{-10}$ |
| 4 | 0,0845 | 0,0060 | 0,0060 | $2{,}8\times10^{-10}$ |
| 5 | 0,0378 | 0,0143 | 0,0143 | $2{,}2\times10^{-9}$ |
| 6 | 0,0306 | 0,2868 | 0,2868 | $2{,}1\times10^{-8}$ |
| 7 | 0,1267 | 0,2504 | 0,2504 | $1{,}8\times10^{-8}$ |

**Lệch lớn nhất: $4{,}16\times10^{-8}$.** Hai chính sách trùng nhau tới dung sai của bộ tối ưu.

Và mô hình thưởng học được có tương quan **0,999951** với thưởng thật — xác nhận rằng bước Bradley–Terry hoạt động đúng, nên phép so sánh là công bằng.

Đọc cột $\pi_{\text{ref}}$ và cột kết quả cạnh nhau cũng thú vị: câu trả lời số 6 từ 0,0306 lên 0,2868 (tăng 9,4 lần) còn câu số 0 từ 0,0669 xuống 0,0015 (giảm 45 lần). Đó là phép đánh trọng số lại theo hàm mũ ở Mục 14.4, nhìn thấy bằng mắt.

### 15.4. Cái giá của DPO

DPO đơn giản hơn nhiều, nhưng không miễn phí — và các bài so sánh thường bỏ qua phần này:

| | RLHF (PPO) | DPO |
|---|---|---|
| Số mô hình trong bộ nhớ | 4 | **2** |
| Cần mô hình thưởng riêng | có | **không** |
| Độ ổn định huấn luyện | thấp | **cao** |
| Dùng được dữ liệu ngoài phân phối | có, qua lấy mẫu mới | **kém hơn** |
| Học tiếp khi có dữ liệu mới | có, mô hình thưởng dùng lại được | phải huấn luyện lại |

Dòng thứ tư là điểm yếu thật sự. RLHF **lấy mẫu từ chính sách hiện tại** rồi chấm bằng mô hình thưởng, nên nó học được trên chính những câu mà mô hình đang thực sự sinh ra. DPO chỉ học trên tập so sánh **cố định** đã thu thập sẵn, nên khi chính sách đi xa khỏi phân phối của tập ấy, tín hiệu học yếu đi.

Dòng thứ năm cũng quan trọng về vận hành: mô hình thưởng là một **tài sản dùng lại được** — chấm được dữ liệu mới, đánh giá được mô hình khác, dò được suy giảm chất lượng. DPO không tạo ra tài sản đó.

> **Cách trả lời khi được hỏi "DPO hay RLHF".** Đừng nói cái nào tốt hơn. Nói: *"Chúng tối ưu cùng một mục tiêu và chứng minh được là cho cùng nghiệm tối ưu. DPO đơn giản và ổn định hơn nhiều nên là lựa chọn mặc định hợp lý. RLHF vẫn hơn khi cần học trên chính phân phối mô hình đang sinh ra, hoặc khi mô hình thưởng có giá trị riêng để đánh giá và giám sát."* Nói được rằng chúng **cùng nghiệm** là dấu hiệu hiểu, vì đó mới là nội dung chính của bài báo DPO.

### 15.5. Sau DPO

Hướng này vẫn đang động, nên tôi chỉ nêu tên và ý chính, không kết luận:

- **IPO** sửa một điểm yếu của DPO: khi dữ liệu so sánh gần như tất định (người dán nhãn luôn chọn cùng một bên), DPO có xu hướng đẩy tỉ lệ xác suất ra vô cùng — đúng hiện tượng dữ liệu tách được ở [Mục 6.5 của *Nền tảng*](nentang-ch06.html). IPO thêm phạt chuẩn để chặn.
- **KTO** bỏ luôn yêu cầu dữ liệu phải là **cặp** so sánh, chỉ cần nhãn "tốt" hoặc "tệ" cho từng câu riêng lẻ. Rẻ hơn nhiều khi thu thập.
- **Căn chỉnh không cần người** dùng chính mô hình ngôn ngữ để tạo dữ liệu so sánh thay cho người dán nhãn.

Điểm chung của cả ba: chúng chấp nhận cùng một khung — tối đa thưởng có ràng buộc KL — và chỉ đổi cách lấy tín hiệu hoặc cách chặn. Nắm được khung ở Mục 14.4 thì đọc bài báo mới chỉ còn là xem nó đổi chỗ nào.

---

## 16. Bài tập

**Bài 1 (suy luận).** Bài báo của Levy và Goldberg chứng minh skip-gram với lấy mẫu âm ngầm phân rã ma trận PMI dịch $\log k$.
(a) Viết hàm mục tiêu kỳ vọng cho một cặp $(i,j)$, rồi cho đạo hàm theo $s = \langle w_i, c_j\rangle$ bằng 0.
(b) Suy ra $\langle w_i, c_j\rangle = \mathrm{PMI}(i,j) - \log k$.
(c) Tăng $k$ từ 1 lên 15 thì ma trận đích đổi thế nào, và điều đó ảnh hưởng gì tới số ô bằng 0 sau khi cắt phần âm?
(d) Vì sao ở **hạng đầy đủ** thì xuống dốc hội tụ đúng về ma trận ấy, còn ở **hạng thấp** thì không? Kết quả đo ở Mục 2.6 nói gì?

**Bài 2 (chẩn đoán).** Bạn xây một hệ tìm kiếm ngữ nghĩa. Lấy trung bình trạng thái ẩn của một mô hình ngôn ngữ làm embedding câu, rồi xếp hạng bằng cosine. Kết quả: **mọi** cặp câu đều có điểm từ 0,82 tới 0,94, kể cả những cặp rõ ràng không liên quan.
(a) Chẩn đoán vấn đề và giải thích cơ chế.
(b) Nêu phép sửa rẻ nhất, và dùng số đo ở Mục 3.2 để nói nó hiệu quả tới đâu.
(c) Vì sao **không** nên chữa bằng cách nâng ngưỡng lên 0,93?
(d) Nêu cách chữa triệt để, và nói rõ nó đòi hỏi thêm gì.

**Bài 3 (tính tay).** Một mô hình có $d = 4096$, 32 lớp, tổng 6 476 005 376 tham số phi-embedding. Gắn LoRA hạng $r$ vào $W_Q$ và $W_V$ của mọi lớp.
(a) Viết công thức số tham số huấn luyện theo $r$, rồi tính cho $r = 4, 8, 16, 64$.
(b) Với $r = 8$, trạng thái Adam ở FP32 chiếm bao nhiêu? So với tinh chỉnh toàn phần.
(c) Tỉ lệ $2r/d$ đổi thế nào khi $d$ tăng từ 768 lên 8192 với $r$ cố định? Điều đó nói gì về mô hình càng lớn?
(d) Một công ty phục vụ 50 khách, mỗi khách một bản tinh chỉnh riêng của mô hình 13 GB. Tính dung lượng lưu trữ cho hai cách: tinh chỉnh toàn phần, và LoRA $r=8$ (giả sử mỗi adapter 16 MB).

**Bài 4 (suy luận).** Quá trình thuận của khuếch tán: $q(x_t\mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I)$.
(a) Chứng minh bằng quy nạp rằng $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$ với $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$.
(b) Vì sao dạng đóng này là điều kiện **bắt buộc** để huấn luyện khả thi?
(c) Với lịch tuyến tính $\beta$ từ $10^{-4}$ tới $0{,}02$ qua 1000 bước, $\bar\alpha_{999} = 4{,}0\times10^{-5}$. Còn lại bao nhiêu phần trăm biên độ tín hiệu, và vì sao con số ấy quan trọng?
(d) Giải thích vì sao $t$ nhỏ thì mô hình học chi tiết còn $t$ lớn thì học bố cục.

**Bài 5 (suy luận).** Hai hàm mất mát cho bộ sinh GAN: tối thiểu $\log(1-D(G(z)))$ và tối đa $\log D(G(z))$.
(a) Tính đạo hàm của mỗi hàm theo điểm số $s$ của bộ phân biệt.
(b) Lúc bắt đầu huấn luyện, $\sigma(s)\approx 0$ cho mẫu giả. Mỗi hàm cho gradient độ lớn bao nhiêu?
(c) Dùng kết quả đo ở Mục 10.2 để nói hậu quả.
(d) Hai hàm có cùng điểm tối ưu. Bài học tổng quát rút ra là gì, và nêu **một** ví dụ khác trong bộ giáo trình này minh hoạ cùng bài học.

**Bài 6 (tính tay).** Bài toán RLHF: $\max_\pi \mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ trên không gian rời rạc $N$ câu trả lời.
(a) Lập Lagrange với ràng buộc $\sum_y\pi(y)=1$, cho đạo hàm theo $\pi(y)$ bằng 0, và suy ra dạng đóng.
(b) $\beta\to 0$ và $\beta\to\infty$ cho chính sách gì? Giải thích bằng lời.
(c) Nếu $\pi_{\text{ref}}(y_0) = 0$ với một câu trả lời $y_0$ nào đó, thì $\pi^*(y_0)$ bằng bao nhiêu dù $r(y_0)$ lớn tới đâu? Điều đó có ý nghĩa gì về an toàn?
(d) Dùng số liệu ở Mục 14.4 để giải thích vì sao $\beta$ nên xem là "vị trí trên đường đánh đổi" chứ không phải "siêu tham số cần chỉnh đúng".

**Bài 7 (suy luận).** Suy ra hàm mất mát DPO.
(a) Từ dạng đóng ở Bài 6, giải ngược để viết $r$ theo $\pi$ và $\pi_{\text{ref}}$.
(b) Thay vào hợp lý Bradley–Terry và chỉ rõ **chính xác chỗ nào** $\log Z(x)$ triệt tiêu.
(c) Vì sao phép triệt tiêu ấy đòi hỏi hai câu trả lời phải cho **cùng một** câu hỏi $x$?
(d) Nêu **hai** tình huống mà RLHF vẫn hơn DPO, dù hai phương pháp có cùng nghiệm tối ưu.

**Bài 8 (chẩn đoán).** Với mỗi tình huống, nêu vấn đề và **hai** việc nên làm:
(a) Huấn luyện VAE, sai số tái dựng cao còn KL gần như bằng 0 ở mọi chiều ẩn.
(b) Tinh chỉnh một mô hình trên 80 mẫu có nhãn; độ chính xác kiểm định tệ hơn khi đóng băng đặc trưng.
(c) Đóng băng đặc trưng và thêm dữ liệu từ 500 lên 5 000 mẫu, độ chính xác chỉ nhích từ 0,62 lên 0,64.
(d) GAN huấn luyện 20 000 vòng, mất mát của bộ sinh **tăng** đều. Có nên dừng không?

**Bài 9 (thiết kế).** Bạn có một mô hình ngôn ngữ mã nguồn mở 7B và cần làm trợ lý hỗ trợ khách hàng cho một công ty bảo hiểm.
(a) Bạn có 3 000 đoạn hội thoại mẫu do nhân viên giỏi viết. Nên dùng tinh chỉnh toàn phần, LoRA, hay chỉ nhắc lệnh? Trả lời dựa trên Mục 6.3 và 7.3.
(b) Sau ba tháng, thu được 40 000 cặp so sánh từ chính khách hàng ("câu trả lời nào hữu ích hơn"). Thiết kế bước tiếp theo, chọn DPO hoặc RLHF và nêu lý do.
(c) Bộ phận pháp chế yêu cầu mô hình **không bao giờ** hứa chi trả. Giải thích vì sao ràng buộc KL trong Mục 14.4 **không đủ** để bảo đảm điều đó, và nêu cần thêm gì.
(d) Thiết kế cách giám sát trong sản xuất, nối với [Chương 10 của *MLOps*](mlops-ch10.html).

**Bài 10 (thí nghiệm).** Trong phần (G) của `code/bieudien/experiments.py`, GAN với hàm không bão hoà phủ đủ 8/8 chế độ ở mọi hạt giống, tức **không** quan sát được sụp chế độ.
(a) Nêu **ba** thay đổi có thể làm hiện tượng ấy lộ ra, và với mỗi cái nói rõ bạn kỳ vọng cơ chế nào gây ra nó.
(b) Cài một trong ba, chạy lại, và báo cáo kết quả **kể cả khi nó không ra như bạn đoán**.
(c) Nghĩ ra một chỉ số đo được sự "phủ" tốt hơn cách đếm số cụm chạm được. Gợi ý: xét phân phối số điểm rơi vào mỗi cụm.
(d) Vì sao đo được "phủ hết chế độ" vẫn chưa đủ để kết luận mô hình học đúng phân phối? Dùng con số 15,3% ở Mục 10.3.

---

## 17. Ôn phỏng vấn

### 17.1. Nhóm biểu diễn

**"Word2Vec hoạt động thế nào?"**

> Trả lời tốt không bắt đầu bằng kiến trúc mà bằng **giả thiết**: một từ được đặc trưng bởi những từ đi kèm nó. Skip-gram biến giả thiết ấy thành bài toán tối ưu — cho một từ, đoán các từ quanh nó.
>
> *Ghi điểm mạnh:* Levy và Goldberg chứng minh skip-gram với lấy mẫu âm **ngầm phân rã ma trận PMI dịch $\log k$**. Nên word2vec không phải học sâu — nó là một cách phân rã ma trận mà không bao giờ phải lập ra ma trận ấy.
>
> *Kể một con số:* ở hạng đầy đủ, xuống dốc hội tụ về đúng PMI với tương quan **0,9995**; ở hạng thấp thì không khớp được, và **chính sự ép buộc ấy** mới tạo ra embedding có ích.

**"Vì sao cần lấy mẫu âm?"**

> Vì mẫu số của softmax cộng trên toàn bộ từ vựng, cho mỗi cặp huấn luyện. Lấy mẫu âm đổi bài toán "chọn 1 trong $V$" thành $k+1$ bài nhị phân, nên **chi phí không còn phụ thuộc $V$**.
>
> *Kể một con số:* với từ vựng 2 triệu và $d = 300$, softmax đầy đủ tốn 1,2 tỉ phép tính mỗi cặp, lấy mẫu âm $k=5$ tốn 3 600 — rẻ hơn **333 000 lần**.
>
> *Trung thực về chi tiết:* mũ 3/4 trong phân phối lấy mẫu là con số **chọn bằng thực nghiệm**, bài báo gốc nói rõ vậy. Đừng bịa lý thuyết cho nó.

**"Vì sao cosine giữa hai embedding bất kỳ hay rất cao?"**

> Vì **bất đẳng hướng**: đám mây embedding lệch khỏi gốc toạ độ, nên mọi cặp đều có thành phần chung lớn.
>
> *Kể một con số:* với 900 vector **ngẫu nhiên độc lập** cộng một hằng số, cosine trung bình là **0,8724**. Không có quan hệ ngữ nghĩa nào ở đó cả. Chỉ cần trừ vector trung bình đi thì nó về $-0{,}0011$.
>
> *Hệ quả dùng được:* nếu hệ tìm kiếm cho mọi kết quả điểm 0,8+, đừng nâng ngưỡng — hãy trừ trung bình rồi đo lại.

**"Chiều thứ 37 của embedding mang nghĩa gì?"**

> Thường là không gì cả, và đây là kết quả toán chứ không phải nhận xét kinh nghiệm: nếu $EE^\top \approx M$ thì $(EQ)(EQ)^\top = EE^\top$ với mọi ma trận trực giao $Q$. Bài toán chỉ xác định $E$ **tới một phép xoay**, nên trục toạ độ tuỳ tiện. Thứ có nghĩa là quan hệ giữa các vector, không phải toạ độ.

### 17.2. Nhóm chuyển giao

**"Vì sao đoán từ tiếp theo lại dạy được nhiều thứ đến vậy?"**

> Vì để đoán đúng trong mọi ngữ cảnh, mô hình **buộc phải** học những thứ khác: sự thật về thế giới ("thủ đô Pháp là ___"), quan hệ ngữ pháp ("mặc dù… vẫn ___"), số học, cấu trúc mã. Không cái nào được dạy riêng; chúng đều là hệ quả bắt buộc của việc giảm mất mát trên kho đủ lớn.

**"Nên đóng băng hay tinh chỉnh?"**

> Phụ thuộc lượng nhãn, và có ngưỡng đo được.
>
> *Kể con số:* ở 150 mẫu trở lên, tinh chỉnh toàn phần tách hẳn lên và khoảng cách giãn dần — **0,8183 so với 0,6452 ở 8 000 mẫu**. Dưới 50 mẫu thì ba cách nằm trong khoảng nhiễu của nhau, không kết luận được.
>
> *Điều đáng nói thêm:* cột đóng băng **phẳng ra** từ 500 mẫu (0,622 → 0,638 → 0,645). Đó là **trần chặn** do đặc trưng cố định đặt ra, và nhận ra nó rất có ích: nếu đang đóng băng mà thêm dữ liệu không giúp gì, thì thêm nữa cũng vô ích — phải mở đóng băng ra.

**"LoRA hoạt động thế nào, và nó tiết kiệm cái gì?"**

> Đóng băng $W_0$, học phần thêm vào $\Delta W = BA$ với $A$ là $d\times r$ và $B$ là $r\times d$. Giả thiết cốt lõi: **phần cần sửa cho nhiệm vụ mới có hạng thấp**.
>
> *Nó tiết kiệm bộ nhớ lúc huấn luyện, không phải thời gian suy luận.* Kể con số: Llama-2 7B, LoRA $r=8$ trên $W_Q$ và $W_V$ cho **4 194 304 tham số** — 0,0648% mô hình. Trạng thái Adam: **32 MiB** so với **48,2 GiB** khi tinh chỉnh toàn phần, tức chênh 1 540 lần.
>
> *Chi tiết cài đặt đáng biết:* khởi tạo $B = 0$ để $BA = 0$ lúc đầu, nên mô hình khởi đầu đúng bằng bản gốc.
>
> *Món lợi hay bị quên:* phục vụ 50 khách hàng cần 13,8 GB với LoRA so với 650 GB với tinh chỉnh toàn phần, vì tất cả dùng chung một bản gốc.

**"LoRA có làm mô hình chạy nhanh hơn không?"** — Không. Khi suy luận thì gộp $BA$ vào $W_0$ một lần, nên tốc độ bằng đúng mô hình gốc. Nếu **không** gộp để đổi adapter nhanh thì còn chậm hơn một chút.

### 17.3. Nhóm mô hình sinh

**"VAE, GAN, khuếch tán khác nhau ở đâu?"**

> Cả ba đều biến nhiễu Gauss thành mẫu. Khác ở **huấn luyện bằng gì**: VAE tối đa một chặn dưới của log hợp lý; GAN chơi một trò chơi hai bên; khuếch tán giải một bài **hồi quy** đoán nhiễu.
>
> *Và đó là lý do lịch sử diễn ra như vậy:* khuếch tán thay GAN vì nó giữ được độ sắc nét mà không phải trả giá bằng sự bất ổn — huấn luyện nó là bình phương sai lệch bình thường, mất mát đọc được.

**"Vì sao mẫu của VAE mờ?"**

> Vì số hạng tái dựng với nhiễu Gauss chính là bình phương sai lệch, mà thứ tối thiểu hoá bình phương sai lệch là **kỳ vọng có điều kiện** $\mathbb{E}[x\mid z]$. Nếu nhiều ảnh thật ứng với cùng một $z$ thì **trung bình của chúng nhoè ở biên**. Độ mờ là hệ quả trực tiếp của việc chọn hàm mất mát, không phải lỗi huấn luyện.

**"$\beta$ trong VAE điều khiển gì?"**

> Không chỉ "cân giữa tái dựng và phạt chuẩn" — nó điều khiển **số chiều ẩn còn mang thông tin**.
>
> *Kể con số:* dữ liệu sinh từ đúng 2 yếu tố, cho VAE 6 chiều ẩn. Ở $\beta = 0$: cả 6 chiều sống. Ở $\beta = 1$: **đúng 2 chiều** sống lại — mô hình tự tìm ra con số đúng. Ở $\beta = 16$: **0 chiều**, tức sụp hậu nghiệm, và sai số tái dựng nhảy lên 15,28.

**"Vì sao bài báo GAN gốc phải đổi hàm mất mát của bộ sinh?"**

> Vì $\frac{\partial}{\partial s}\log(1-\sigma(s)) = -\sigma(s)$, mà lúc đầu bộ sinh tệ nên $\sigma(s)\approx 0$ cho mẫu giả — **gradient cũng gần 0**. Bộ sinh càng tệ càng ít tín hiệu để sửa.
>
> *Kể con số:* trên 8 cụm Gauss, dạng minimax gốc phủ **0/8** chế độ ở cả bốn hạt giống; bản không bão hoà phủ **8/8**. Đạo hàm của bản mới là $1-\sigma(s)$, **lớn nhất đúng khi bộ sinh đang tệ nhất**.
>
> *Bài học rộng hơn:* hai hàm mất mát cùng điểm tối ưu vẫn có thể cho hành vi huấn luyện khác hẳn, vì cái quyết định việc học là **gradient** chứ không phải vị trí điểm tối ưu.

**"Vì sao khuếch tán dễ huấn luyện hơn GAN?"**

> Vì nó **không phải trò chơi**. Mục tiêu là $\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$ — bình phương sai lệch, một bài hồi quy. Mất mát giảm nghĩa là mô hình tốt lên, không có điểm yên ngựa, không có hai mạng đánh nhau.
>
> *Chi tiết làm nó khả thi:* dạng đóng $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$ cho phép **nhảy thẳng** tới bước $t$ bất kỳ thay vì mô phỏng $t$ bước. Không có nó thì huấn luyện chậm gấp hàng trăm lần.

### 17.4. Nhóm căn chỉnh

**"Vì sao cần RLHF khi đã có tinh chỉnh có giám sát?"**

> Vì tinh chỉnh có giám sát là **bắt chước**, nên mô hình chỉ giỏi bằng người viết câu mẫu. RLHF học từ **so sánh**, và so sánh phá được trần ấy vì **đánh giá dễ hơn sáng tạo** — người ta chỉ ra được bài thơ nào hay hơn dù không viết nổi bài hay.
>
> *Lý do thực tế thứ hai:* so sánh nhất quán giữa những người dán nhãn hơn là chấm điểm tuyệt đối.

**"Nghiệm của bài toán RLHF có ràng buộc KL là gì?"**

> $$\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\big(r(x,y)/\beta\big).$$
>
> Đọc bằng lời: **chính sách tham chiếu, đánh trọng số lại theo hàm mũ của phần thưởng.**
>
> *Ghi điểm thêm:* nếu $\pi_{\text{ref}}(y) = 0$ thì $\pi^*(y) = 0$ **bất kể $r(y)$ lớn tới đâu** — nhân với 0 vẫn là 0. Đó chính là cơ chế giữ cho mô hình không đi lung tung.
>
> *Kể con số:* kiểm chứng bằng cách giải theo hai đường độc lập, lệch $10^{-9}$ tới $10^{-7}$.

**"Vì sao phải có số hạng KL?"**

> Để chặn **lách điểm thưởng**. Mô hình thưởng chỉ là một xấp xỉ học từ dữ liệu hữu hạn; đẩy mạnh vào nó thì ta đang tối ưu cái xấp xỉ chứ không phải cái thật, và mô hình sẽ phá nát khả năng ngôn ngữ để chạy theo điểm mù của $r$.
>
> *Cách đọc $\beta$ cho đúng:* nó không phải siêu tham số cần "chỉnh đúng" mà là **vị trí ta chọn trên một đường đánh đổi** giữa thưởng đạt được và độ lệch khỏi $\pi_{\text{ref}}$.

**"DPO hoạt động thế nào?"**

> Quan sát then chốt: dạng đóng trên **đảo ngược được** — mọi chính sách đều ngầm định nghĩa một hàm thưởng $r = \beta\log\frac{\pi}{\pi_{\text{ref}}} + \beta\log Z$.
>
> Thay vào hợp lý Bradley–Terry, mà hợp lý ấy chỉ dùng **hiệu** hai phần thưởng, nên $\log Z(x)$ — vốn không tính được — **triệt tiêu sạch** vì nó phụ thuộc $x$ chứ không phụ thuộc $y$. Kết quả là một hàm mất mát entropy chéo nhị phân tính thẳng từ dữ liệu so sánh.
>
> *Kể con số:* kiểm chứng trên không gian 8 câu trả lời với 52 588 cặp — chính sách DPO và chính sách RLHF hai bước lệch nhau **$4{,}2\times10^{-8}$**.

**"DPO hay RLHF?"**

> Đừng nói cái nào tốt hơn. Nói: chúng tối ưu **cùng một mục tiêu** và chứng minh được là cho **cùng nghiệm**. DPO đơn giản hơn (2 mô hình thay vì 4, không cần mô hình thưởng, ổn định hơn) nên là mặc định hợp lý.
>
> *RLHF vẫn hơn ở hai chỗ:* nó **lấy mẫu từ chính sách hiện tại** nên học được trên đúng phân phối mô hình đang sinh ra, trong khi DPO chỉ học trên tập so sánh cố định; và mô hình thưởng là **tài sản dùng lại được** để chấm dữ liệu mới, đánh giá mô hình khác, dò suy giảm chất lượng.

### 17.5. Những câu trả lời tự tố cáo

| Câu trả lời | Vì sao nó tố cáo |
|---|---|
| "Word2Vec là mạng nơ-ron sâu." | Nó có **một lớp**, không phi tuyến ở giữa. Là phân rã ma trận. |
| "Embedding hiểu nghĩa của từ." | Nó nắm thống kê đồng hiện. Đó là lý do nó không phân biệt được "tốt" và "xấu". |
| "Cosine 0,9 nghĩa là hai câu rất giống nhau." | Trên đám mây bất đẳng hướng, hai vector **ngẫu nhiên** cũng cho 0,87. |
| "LoRA làm mô hình chạy nhanh hơn." | Gộp $BA$ vào $W_0$ khi suy luận nên tốc độ **bằng** bản gốc. Nó tiết kiệm bộ nhớ **huấn luyện**. |
| "Hạng LoRA càng cao càng tốt." | Hạng cao = nhiều tham số = nhiều phương sai. Trên ít dữ liệu, $r=4$ thường hơn $r=64$. |
| "VAE mờ vì mô hình chưa đủ lớn." | Mờ vì bình phương sai lệch tối ưu ở **kỳ vọng có điều kiện**. Là hệ quả của hàm mất mát. |
| "GAN khó huấn luyện vì cần chỉnh siêu tham số." | Lý do sâu hơn: đây là **điểm yên ngựa** chứ không phải cực tiểu, và **không đọc được đường cong mất mát**. |
| "Khuếch tán sinh ảnh bằng cách khử nhiễu, nên nó chậm." | Đúng nhưng thiếu điểm hay: **huấn luyện** nó chỉ là hồi quy, và đó mới là lý do nó thắng GAN. |
| "RLHF dạy mô hình trả lời đúng." | Nó dạy mô hình trả lời theo **sở thích của người dán nhãn**. Hai thứ ấy không giống nhau. |
| "DPO là phiên bản đơn giản hoá của RLHF." | Nó **không phải xấp xỉ** — nó tối ưu cùng mục tiêu và cho cùng nghiệm, kiểm chứng được tới $10^{-8}$. |
| "Tăng $\beta$ trong RLHF thì mô hình an toàn hơn." | $\beta$ giữ mô hình gần $\pi_{\text{ref}}$. Nếu $\pi_{\text{ref}}$ đã không an toàn thì tăng $\beta$ chẳng giúp gì. |
| "$\varepsilon$-tham lam là cách duy nhất để khám phá." | Khởi tạo lạc quan cũng là một cơ chế khám phá, và đo được rằng $\varepsilon=0$ vẫn thành công 100% nhờ nó. |

---

## 18. Tài liệu tham khảo

**Biểu diễn và embedding**

1. T. Mikolov và cộng sự. *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop 2013. — bài báo word2vec gốc.
2. T. Mikolov và cộng sự. *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013. — nguồn của lấy mẫu âm và mũ 3/4; bài báo nói rõ mũ ấy chọn bằng thực nghiệm.
3. O. Levy, Y. Goldberg. *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS 2014. — chứng minh dùng ở Mục 2.6: SGNS ngầm phân rã PMI dịch $\log k$.
4. J. Pennington, R. Socher, C. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
5. J. Devlin và cộng sự. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019. — embedding theo ngữ cảnh.
6. N. Reimers, I. Gurevych. *Sentence-BERT.* EMNLP 2019. — nguồn của nhận định rằng trạng thái ẩn thô không phải embedding câu tốt.
7. J. Gao, D. He và cộng sự. *Representation Degeneration Problem in Training Neural Language Models.* ICLR 2019. — bất đẳng hướng.
8. T. Chen và cộng sự. *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* ICML 2020. — mục tiêu tương phản ở Mục 4.2.
9. Y. Malkov, D. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs.* TPAMI 2018.
10. H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor Search.* TPAMI 2011.

**Học chuyển giao**

11. J. Howard, S. Ruder. *Universal Language Model Fine-tuning for Text Classification.* ACL 2018. — mở đóng băng dần.
12. N. Houlsby và cộng sự. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019. — adapter.
13. E. Hu và cộng sự. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — nguồn của Chương 7, kể cả việc chỉ gắn vào $W_Q$ và $W_V$.
14. T. Dettmers và cộng sự. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
15. X. Lisa Li, P. Liang. *Prefix-Tuning.* ACL 2021.
16. E. Ben Zaken, S. Ravfogel, Y. Goldberg. *BitFit.* ACL 2022.

**Mô hình sinh**

17. D. Kingma, M. Welling. *Auto-Encoding Variational Bayes.* ICLR 2014. — VAE và mẹo tái tham số hoá.
18. I. Higgins và cộng sự. *beta-VAE.* ICLR 2017. — nguồn của $\beta$ ở Chương 9.
19. I. Goodfellow và cộng sự. *Generative Adversarial Nets.* NeurIPS 2014. — nguồn của cả bài toán minimax lẫn mẹo không bão hoà ở Mục 10.2.
20. M. Arjovsky, S. Chintala, L. Bottou. *Wasserstein GAN.* ICML 2017. — phân tích vì sao GAN khó huấn luyện.
21. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. — nguồn của dạng đóng và của lựa chọn đoán $\varepsilon$.
22. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. — DDIM.
23. R. Rombach và cộng sự. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. — khuếch tán trong không gian ẩn.
24. A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. — lịch nhiễu.

**Học tăng cường và căn chỉnh**

25. R. Sutton, A. Barto. *Reinforcement Learning: An Introduction.* MIT Press, 2018. — sách chuẩn; Chương 2 bàn về khởi tạo lạc quan, cơ chế đo được ở Mục 12.4.
26. C. Watkins, P. Dayan. *Q-learning.* Machine Learning, 1992.
27. R. Williams. *Simple Statistical Gradient-Following Algorithms.* Machine Learning, 1992. — REINFORCE và đường nền.
28. J. Schulman và cộng sự. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
29. J. Schulman và cộng sự. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. — hàm lợi thế.
30. P. Christiano và cộng sự. *Deep Reinforcement Learning from Human Preferences.* NeurIPS 2017. — bài báo đầu tiên của dòng RLHF.
31. L. Ouyang và cộng sự. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. — InstructGPT; nguồn của quy trình ba giai đoạn ở Hình 14.
32. Y. Bai và cộng sự. *Training a Helpful and Harmless Assistant with RLHF.* arXiv:2204.05862, 2022.
33. R. Rafailov và cộng sự. *Direct Preference Optimization.* NeurIPS 2023. — nguồn của toàn bộ Chương 15, kể cả phép suy ra ở Mục 15.2.
34. M. Azar và cộng sự. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. — IPO.
35. K. Ethayarajh và cộng sự. *KTO: Model Alignment as Prospect Theoretic Optimization.* ICML 2024.
36. R. Bradley, M. Terry. *Rank Analysis of Incomplete Block Designs.* Biometrika, 1952. — mô hình so sánh cặp dùng ở Mục 14.3.

---

## Phụ lục: chạy lại toàn bộ thí nghiệm

```text
code/bieudien/
├── experiments.py            # Hình 2, 3, 4, 6, 8, 9, 10, 12, 13 và mọi số liệu đo được
├── experiments_output.txt    # kết quả in ra của script trên
└── fig_diagrams.py           # Hình 1, 5, 7, 11, 14 (các sơ đồ khái niệm)
```

```bash
pip install numpy scipy matplotlib scikit-learn
python code/bieudien/experiments.py     # vài phút
python code/bieudien/fig_diagrams.py
```

Môi trường đã dùng: Python 3.13, NumPy 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10.

**Hai kết quả là kiểm chứng đẳng thức**, phải khớp tới dung sai của bộ giải chứ không phải khớp "đại khái":

| Đẳng thức | Sai số đo được |
|---|---|
| Nghiệm RLHF có ràng buộc KL khớp dạng đóng | $4 \times 10^{-9}$ tới $4\times10^{-7}$ |
| Chính sách DPO trùng chính sách RLHF hai bước | $4{,}16\times10^{-8}$ |
| Dạng đóng của quá trình thuận khuếch tán | khớp mô men trên 200 000 quỹ đạo |
| Skip-gram hạng đầy đủ hội tụ về PMI | tương quan 0,9995 |

**Và ba chỗ số liệu nói ngược lại điều tôi định viết.** Tôi giữ nguyên số và viết lại phần chữ:

1. **Thí nghiệm GAN (Mục 10.3)** không cho thấy sụp chế độ — bản không bão hoà phủ đủ 8/8 chế độ ở mọi hạt giống, và tôi đã thử nhiều cấu hình khác nhau. Thứ nó cho thấy lại quan trọng hơn: bài toán minimax gốc **thất bại hoàn toàn** vì gradient triệt tiêu.
2. **Thí nghiệm học tăng cường (Mục 12.4)** cho thấy $\varepsilon = 0$ vẫn đạt 100% trong một cấu hình, bác bỏ câu "phải có $\varepsilon$ để khám phá". Nguyên nhân là khởi tạo lạc quan, và đó mới là bài học.
3. **Bảng quét số chiều embedding (Mục 2.4)** không phải một đường cong đánh đổi đẹp — giữ hạng đầy đủ lại đạt 100% trở lại. Với từ vựng 20 từ thì không đủ chỗ để đánh đổi lộ ra, nên tôi nói rõ điều đó thay vì kết luận từ nó.

Ba chỗ ấy tôi nghĩ là phần đáng đọc nhất của tài liệu, vì chúng cho thấy sự khác nhau giữa một câu nghe hợp lý và một câu đã được kiểm.
