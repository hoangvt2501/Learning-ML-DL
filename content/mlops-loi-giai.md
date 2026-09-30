# Lời giải chi tiết — MLOps

Mỗi mục ứng với một bài trong Chương 14. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=9 | dang=Tính tay | kho=Cơ bản

**Bước 1 — PSI kỳ vọng khi không có dịch chuyển.** Dùng kết quả ở Mục 9.5:

$$\mathbb{E}[\mathrm{PSI}] \;\approx\; \frac{2(k-1)}{n} \;=\; \frac{2 \times 9}{500} \;=\; \boxed{0{,}036}.$$

Đối chiếu bảng đo được ở Mục 9.5: với $n = 500$, $k = 10$ thì PSI trung bình đo được là **0,0353**. Khớp.

**Bước 2 — có yên tâm với ngưỡng 0,10 không?** Giá trị trung bình 0,036 thì xa 0,10, nhưng **trung bình không phải là thứ gây báo động** — cái gây báo động là phần đuôi. Vì $\tfrac{n}{2}\mathrm{PSI} \sim \chi^2_{k-1}$, ta tính được phân bố của PSI:

| Phân vị của PSI khi không có dịch chuyển | Giá trị |
|---|---|
| 95% | 0,0677 |
| 99% | 0,0867 |
| 99,9% | 0,1115 |

Xác suất vượt ngưỡng 0,10 khi **không hề có dịch chuyển**:

$$P(\mathrm{PSI} > 0{,}10) = P\!\left(\chi^2_9 > \tfrac{0{,}10 \times 500}{2}\right) = P(\chi^2_9 > 25) = 0{,}00297.$$

Tức **0,3% mỗi lần đo, mỗi đặc trưng**. Nghe rất nhỏ. Nhưng bạn không đo một đặc trưng:

> Với **200 đặc trưng, đo mỗi ngày**, kỳ vọng là $200 \times 0{,}00297 \approx \mathbf{0{,}59}$ **báo động giả mỗi ngày** — tức khoảng **4 báo động giả mỗi tuần, trên một hệ thống hoàn toàn khoẻ mạnh.**

Đó chính là cơ chế sinh ra nỗi đau "báo động giả" ở Mục 3.2 và sự mất niềm tin ở Mục 10.5. Câu trả lời cho đề bài là: **không yên tâm** — không phải vì ngưỡng sai, mà vì **chưa hiệu chỉnh đa kiểm định**.

**Bước 3 — cần $n$ bao nhiêu để PSI kỳ vọng dưới 0,01?**

$$\frac{2(k-1)}{n} < 0{,}01 \;\Longleftrightarrow\; n > \frac{2 \times 9}{0{,}01} = \boxed{1800}.$$

**Điều rút ra.** Ngưỡng PSI không phải là một hằng số của ngành; nó là hàm của $(n, k, \text{số đặc trưng theo dõi})$. Ba cách xử lý, theo Mục 9.5: hiệu chuẩn ngưỡng bằng mô phỏng, hoặc chuyển sang kiểm định có p-value kèm hiệu chỉnh Benjamini–Hochberg, hoặc giữ $n$ và $k$ cố định rồi cảnh báo theo **độ dai** (ba kỳ liên tiếp vượt ngưỡng) thay vì theo một điểm.

## Bài 2
@meta chuong=8 | dang=Tính tay | kho=Cơ bản

**Bước 1 — xác định các đại lượng.** Cải thiện *tương đối* 5% trên nền 2%:

$$p_1 = 0{,}02, \qquad p_2 = 0{,}02 \times 1{,}05 = 0{,}021, \qquad \Delta = p_2 - p_1 = 0{,}001.$$

Chú ý: $\Delta$ tuyệt đối chỉ là **0,1 điểm phần trăm**. Đây là chỗ mọi người hay nhầm — "5%" nghe to, nhưng thứ đi vào mẫu số là 0,001.

**Bước 2 — thay vào công thức.** Với $z_{0{,}975} = 1{,}95996$ và $z_{0{,}80} = 0{,}84162$:

$$n = \frac{(1{,}95996 + 0{,}84162)^2 \left[0{,}02(0{,}98) + 0{,}021(0{,}979)\right]}{0{,}001^2}
     = \frac{7{,}8489 \times 0{,}040159}{10^{-6}} = \boxed{315\,203}$$

mẫu **mỗi nhánh**, tức hơn 630 000 mẫu cho cả thí nghiệm.

**Bước 3 — thời gian.** Với 40 000 lượt mỗi nhánh mỗi ngày:

$$\frac{315\,203}{40\,000} = \boxed{7{,}88 \text{ ngày}} \approx 8 \text{ ngày.}$$

(Trong thực tế nên làm tròn lên một tuần trọn vẹn để tránh hiệu ứng ngày trong tuần.)

**Bước 4 — rút xuống một nửa thời gian mà không giảm lực kiểm định.** Có bốn lựa chọn thật và hai lựa chọn giả.

| Lựa chọn | Cơ chế | Đánh giá |
|---|---|---|
| **Giảm phương sai bằng CUPED** | dùng dữ liệu trước thí nghiệm của chính người dùng làm hiệp biến để trừ bớt phương sai | **Tốt nhất.** Giảm phương sai 40% là mức thường thấy, cho $n = 189\,122$ tức **4,7 ngày**. Không đổi gì về mức ý nghĩa hay lực kiểm định. |
| **Tăng tỉ lệ lưu lượng vào thí nghiệm** | chia 50/50 thay vì 10/90 | Tốt, nếu chấp nhận phơi nhiễm nhiều người dùng hơn với bản chưa chắc chắn. |
| **Đổi sang chỉ số ít phương sai hơn** | ví dụ dùng chỉ số liên tục thay vì nhị phân | Tốt, nhưng phải kiểm tra chỉ số mới vẫn gắn với chỉ số sản phẩm (Mục 6.2). |
| **Chấp nhận bắt hiệu ứng lớn hơn** | bắt lift 7,1% thay vì 5% | Hợp lệ nhưng phải nói rõ: $n$ giảm còn 159 205, tức đúng một nửa — vì $n \propto 1/\Delta^2$ và $7{,}1\% \approx 5\% \times \sqrt{2}$. |
| ~~Dừng sớm khi thấy $p < 0{,}05$~~ | nhìn lén | **Sai.** Đây là *peeking* ở Mục 8.5, làm tỉ lệ dương tính giả vượt xa 5%. |
| ~~Giảm lực kiểm định xuống 60%~~ | | **Sai theo đề bài**, và nó chỉ chuyển rủi ro sang chỗ khác. |

**Điều rút ra.** Câu hỏi "chạy A/B test này bao lâu" luôn quy về **bốn** con số: tỉ lệ nền, mức cải thiện nhỏ nhất đáng quan tâm, lưu lượng, và phương sai. Đòn bẩy rẻ nhất trong bốn cái là **phương sai**, và đó là lý do CUPED đáng biết tên.

## Bài 3
@meta chuong=7 | dang=Tính tay | kho=Trung bình

**Phần a — tỉ lệ yêu cầu chạm ít nhất một dịch vụ chậm.** Mỗi dịch vụ vượt p99 với xác suất 0,01, độc lập:

$$P(\text{ít nhất một nhánh chậm}) = 1 - 0{,}99^{25} = \boxed{22{,}2\%}.$$

Hơn một phần năm số yêu cầu chạm phải một nhánh nằm ngoài p99, dù **mỗi** dịch vụ đều "nhanh ở mức p99".

**Phần b — mỗi dịch vụ phải nhanh tới phân vị nào.** Cần $P(\max \le t) = F(t)^{25} = 0{,}99$, nên

$$F(t) = 0{,}99^{1/25} = \boxed{0{,}999598} \quad \text{tức phân vị } \approx \mathbf{p99{,}96}.$$

Diễn giải cho dễ cảm: **chỉ 1 trong 2 488 yêu cầu tới mỗi dịch vụ được phép chậm.** Đây là yêu cầu khắc nghiệt hơn p99 khoảng 25 lần, và trong thực tế gần như không đạt được bằng cách tối ưu từng dịch vụ.

**Phần c — hai cách chữa không cần làm dịch vụ nhanh hơn.**

1. **Giảm $k$.** Gộp 25 lời gọi thành ít lời gọi hơn (gộp lô nhiều đặc trưng vào một yêu cầu), hoặc lưu đệm những đặc trưng ít đổi. Vì $1-0{,}99^k$ tăng gần tuyến tính theo $k$ khi $k$ nhỏ, giảm từ 25 xuống 5 đã kéo tỉ lệ chạm đuôi từ 22,2% xuống 4,9%.
2. **Đặt thời hạn chờ và trả lời một phần.** Cắt ở 80 ms, dùng giá trị mặc định hoặc giá trị đệm cho nhánh chưa kịp. Điều này đổi một bài toán về **độ trễ** thành một bài toán về **chất lượng**, và thường là đổi có lợi — chậm 300 ms thì người dùng bỏ đi, còn thiếu một đặc trưng thì mô hình chỉ kém đi một chút.

Cách thứ ba đáng nhắc: **yêu cầu dự phòng (hedged request)** — nếu sau 60 ms chưa có trả lời thì gửi thêm một bản sao tới máy chủ khác và lấy cái nào về trước. Tốn thêm vài phần trăm tài nguyên để cắt hẳn phần đuôi.

## Bài 4
@meta chuong=4 | dang=Suy luận | kho=Cơ bản

**Tập huấn luyện ĐÚNG.** Với mỗi dòng nhãn, lấy bản ghi đặc trưng có `valid_from` **lớn nhất nhưng không vượt quá** `event_time`:

| user_id | event_time | nhãn | tong_don_hang | lấy từ bản ghi |
|---|---|---|---|---|
| 7 | 2026-02-10 | 0 | **3** | `valid_from = 2026-01-01` |
| 7 | 2026-04-15 | 1 | **19** | `valid_from = 2026-03-01` |

Bản ghi `2026-06-01` (giá trị 52) **không được dùng cho dòng nào**, vì cả hai sự kiện đều xảy ra trước nó.

**Tập huấn luyện SAI mà một `JOIN` thông thường tạo ra.** Có hai biến thể, tuỳ bạn ghép vào đâu:

*Biến thể 1 — ghép vào bảng "giá trị hiện tại":* cả hai dòng đều nhận `tong_don_hang = 52`.

| user_id | event_time | nhãn | tong_don_hang |
|---|---|---|---|
| 7 | 2026-02-10 | 0 | 52 |
| 7 | 2026-04-15 | 1 | 52 |

*Biến thể 2 — `JOIN` thẳng vào bảng lịch sử mà quên điều kiện thời gian:* mỗi dòng nhãn nhân với cả ba bản ghi đặc trưng, ra **6 dòng** thay vì 2 — vừa rò rỉ tương lai vừa nhân bản dữ liệu, làm sai cả trọng số của mẫu.

**Mô hình học nhầm điều gì.** Ở biến thể 1, cả hai dòng có cùng đặc trưng 52 nhưng khác nhãn, nên đặc trưng ấy **mất hết sức phân biệt** trên tập này — và với dữ liệu thật, nơi người ở lại có số đơn tích luỹ cao hơn người rời bỏ, nó học được quan hệ ngược lại: *"số đơn hàng cao thì không rời bỏ"*. Quan hệ ấy có thật trong tập huấn luyện, nhưng nó là **hệ quả** của nhãn chứ không phải **nguyên nhân**: con số 52 tồn tại *vì* khách đã ở lại tới tháng 6.

Hậu quả đúng như Mục 4.2 mô tả: **chỉ số ngoại tuyến đẹp lên**, mô hình trông xuất sắc, rồi vô dụng khi triển khai — vì lúc dự đoán thật cho một khách hàng vào tháng 2, con số duy nhất bạn có là 3, không phải 52.

**Cách chặn từ gốc.** Hai lớp, dùng cả hai:

1. Ràng buộc trong hợp đồng dữ liệu ở Mục 3.3: `feature_valid_from <= event_time` phải đúng cho **mọi** dòng của tập huấn luyện. Đây là một phép kiểm tra chạy được bằng máy.
2. Dùng **as-of join** có sẵn của kho đặc trưng, thay vì tự viết `JOIN`.

## Bài 5
@meta chuong=9 | dang=Phân loại | kho=Cơ bản

| | Tình huống | Loại | Dò được chỉ bằng $X$? |
|---|---|---|---|
| (a) | Quảng cáo kéo về nhiều người dùng dưới 25 tuổi | **Covariate shift** | **Có** |
| (b) | Ngân hàng đổi định nghĩa "nợ xấu" từ 90 xuống 60 ngày | **Concept drift** | **Không** |
| (c) | Mùa lễ làm tỉ lệ gian lận tăng gấp ba, cách gian lận không đổi | **Label shift** | **Có** |
| (d) | Đối thủ ra tính năng mới, kỳ vọng của người dùng về "kết quả tốt" đổi | **Concept drift** | **Không** |

**(a) Covariate shift.** $P(X)$ đổi — phân phối tuổi dịch sang trái. Nhưng quan hệ "hồ sơ như thế này thì hành vi như thế kia" không đổi, tức $P(Y \mid X)$ giữ nguyên. Dò được dễ: histogram tuổi đổi rõ.

**(b) Concept drift.** Đây là trường hợp sạch nhất của concept drift vì **chính nhãn được định nghĩa lại**. Hồ sơ khách hàng đi vào hệ thống y như cũ, tức $P(X)$ không đổi một chút nào; nhưng với **cùng một hồ sơ**, xác suất bị gắn nhãn "nợ xấu" đã tăng. Kiểm chứng bằng mô phỏng: độ chính xác rơi xuống **25,9%** trong khi kiểm định KS trên đặc trưng cho $p = 0{,}127$ — hoàn toàn bình thường.

**(c) Label shift.** $P(Y)$ tăng gấp ba, còn $P(X \mid Y)$ — "giao dịch gian lận trông như thế nào" — giữ nguyên theo đúng đề bài. Dò được bằng $X$ **một cách gián tiếp**: vì $P(X) = \sum_y P(X\mid y)P(y)$ là hỗn hợp, đổi trọng số hỗn hợp thì phân phối biên của $X$ cũng đổi. Kiểm chứng: KS cho $p = 1{,}5 \times 10^{-15}$, còn độ chính xác gần như không đổi (**76,5%**) vì biên quyết định vẫn đúng chỗ.

> Lưu ý về (c): label shift **làm hỏng hiệu chuẩn và làm lệch precision/recall** dù độ chính xác trông ổn. Nếu ngưỡng quyết định được đặt cho tỉ lệ nền 1% mà nay nền là 3%, thì số cảnh báo sinh ra tăng vọt. Đây chính là **fixed threshold in dynamic system** ở Mục 10.6.

**(d) Concept drift.** Tinh tế hơn (b). Truy vấn người dùng gõ vào có thể không đổi, nhưng **cái được coi là câu trả lời đúng** đã đổi. Tức $P(Y\mid X)$ đổi. Đây là dạng khó nhất vì không có sự kiện nào trong hệ thống của bạn đánh dấu thời điểm nó xảy ra — thứ đổi nằm ở **bên ngoài**, đúng như mục *Changes in the External World* của Sculley và cộng sự.

**Điều rút ra.** Hai trong bốn trường hợp không dò được bằng cách giám sát $X$, và đó đúng là hai trường hợp **tàn phá nhất**. Vì vậy mọi thiết kế giám sát phải bắt đầu bằng câu hỏi ở Mục 3.5: *bao lâu thì tôi biết đáp án thật?*

## Bài 6
@meta chuong=6 | dang=Suy luận | kho=Trung bình

**Bước 1 — chấm từng nhóm.** Nhắc lại thang điểm ở Mục 6.5: 0,5 điểm nếu làm thủ công có ghi chép, 1 điểm nếu có hệ thống chạy tự động định kỳ.

| Nhóm | Mục đạt được | Điểm |
|---|---|---|
| **Dữ liệu** | Data 1 (schema, tự động) = 1; Data 7 (kiểm thử mã đặc trưng, tự động) = 1 | **2** |
| **Mô hình** | Đề bài không nêu mục nào | **0** |
| **Hạ tầng** | Infra 1 (huấn luyện lặp lại được) = 1; Infra 3 (kiểm thử tích hợp toàn pipeline) = 1; Infra 6 (canary) = 1; Infra 7 (quay lui) = 1 | **4** |
| **Giám sát** | Biểu đồ độ trễ và tỉ lệ lỗi là **chỉ số vận hành**, không nằm trong bốn nhóm của rubric | **0** |

**Bước 2 — điểm cuối.**

$$\text{ML Test Score} = \min(2,\ 0,\ 4,\ 0) = \boxed{0}.$$

**Bước 3 — vì sao quy tắc lấy nhỏ nhất cho ra con số ấy.** Bài báo chọn `min` vì cả bốn nhóm đều cần thiết: muốn nâng điểm thì **phải quan tâm tới cả bốn**. Ở đây đội có hạ tầng gần như hoàn hảo — 4/7 mục tự động, hơn phần lớn đội trong thực tế — nhưng vì hai nhóm bằng 0 nên điểm tổng bằng 0, và theo bảng diễn giải thì đó là *"giống một dự án nghiên cứu hơn là một hệ thống sản xuất"*.

Nghe khắc nghiệt, nhưng **đúng về bản chất**: một hệ thống huấn luyện lại hoàn hảo mà không ai biết nó đang hoạt động tốt hay không thì không an toàn hơn một hệ thống không có hạ tầng gì. Hạ tầng tốt chỉ làm cho việc triển khai **nhanh hơn** — kể cả triển khai một mô hình đã hỏng.

**Bước 4 — nên làm gì đầu tiên.** Điểm mấu chốt: vì điểm là `min` của bốn nhóm, **đầu tư vào nhóm mạnh nhất (Hạ tầng) không nâng được điểm một chút nào.** Phải nâng **cả hai** nhóm đang bằng 0 mới nhích được lên 1.

Thứ tự đề nghị, chọn theo tỉ lệ lợi ích trên công sức:

1. **Monitor 3 — đặc trưng lúc huấn luyện và lúc phục vụ tính ra cùng giá trị.** Rẻ nhất vì đội đã có canary; chỉ cần ghi log giá trị đặc trưng ở cả hai đường và so. Đây cũng là lỗi tốn kém nhất mà nó bắt được (Mục 4.1).
2. **Monitor 6 và 7 — phát hiện tụt chất lượng đột ngột và tụt do hồi quy.** Kể cả khi nhãn tới trễ, vẫn theo dõi được **độ lệch dự đoán** ở Mục 10.3.
3. **Model 6 — chất lượng đủ tốt trên mọi lát cắt quan trọng.** Rẻ, và bắt đúng loại sự cố ở Mục 6.3.
4. **Model 4 — biết ảnh hưởng của độ cũ.** Chạy một thí nghiệm kiểu Mục 11.1 là xong, và nó cho luôn câu trả lời về nhịp huấn luyện lại.

Bốn việc trên đưa điểm từ 0 lên khoảng 2 — tức **từ "dự án nghiên cứu" lên "đã có bước đầu đưa vào sản xuất"** — mà không cần đụng tới hạ tầng.

## Bài 7
@meta chuong=11 | dang=Thí nghiệm | kho=Trung bình

> **Đã chạy thật.** Kết quả dưới đây lấy từ phần (G) của `code/mlops/experiments.py` với ba giá trị `DRIFT`.

**Kết quả.** Độ chính xác trung bình qua 52 tuần:

| Tốc độ dịch chuyển | Không bao giờ | Mỗi quý | Mỗi tháng | Mỗi 2 tuần | Mỗi tuần | Nhịp nhỏ nhất còn trong 1 điểm |
|---|---|---|---|---|---|---|
| 0,010 rad/tuần (chậm) | 76,15% | 77,67% | 77,74% | 77,76% | 77,76% | **mỗi quý** |
| 0,035 rad/tuần (vừa) | 64,61% | 76,89% | 77,68% | 77,78% | 77,80% | **mỗi quý** |
| 0,080 rad/tuần (nhanh) | 45,08% | 73,04% | 76,96% | 77,50% | 77,66% | **mỗi tháng** |

**Quan hệ rút ra.** Ba quan sát, và cả ba đều dùng được trong công việc thật:

1. **Cái đổi mạnh nhất theo tốc độ dịch chuyển là cột "không bao giờ", không phải cột nhịp tối ưu.** Khi tốc độ tăng 8 lần (0,010 → 0,080), thiệt hại của việc không huấn luyện lại tăng từ 1,6 điểm lên **32,6 điểm**. Nhưng nhịp cần thiết chỉ đi từ "mỗi quý" sang "mỗi tháng" — tức tăng 3 lần.
2. **Nhịp hợp lý tỉ lệ nghịch với tốc độ dịch chuyển, nhưng rất thoải.** Lý do: một khi bạn huấn luyện lại đủ thường để mô hình luôn nằm trong vùng "gần đúng", phần thiệt hại còn lại là bậc hai theo lượng dịch chuyển tích luỹ giữa hai lần huấn luyện — nên giảm chu kỳ cho lợi ích giảm rất nhanh.
3. **Giá trị lớn nhất nằm ở lần huấn luyện lại đầu tiên.** Ở mọi tốc độ, bước từ "không bao giờ" sang "mỗi quý" chiếm phần lớn toàn bộ lợi ích; mọi bước sau đó đều nhỏ.

**Quy tắc thực hành rút ra.** Đừng chọn nhịp bằng cảm tính hay bằng thói quen ngành. Hãy chạy đúng thí nghiệm này trên bài toán của mình — nó chỉ cần dữ liệu lịch sử và vài giờ tính toán — rồi chọn **nhịp nhỏ nhất mà chênh lệch so với nhịp dày nhất còn dưới ngưỡng kinh doanh chấp nhận được**. Mục 11.1 cho thấy vì sao: chênh lệch giữa hằng tuần và hằng quý ở tốc độ vừa chỉ là 0,91 điểm, đổi lại chi phí vận hành gấp khoảng 13 lần.

## Bài 8
@meta chuong=12 | dang=Thí nghiệm | kho=Khó

> **Đã chạy thật.** Kết quả lấy từ phần (H) của `code/mlops/experiments.py`, trung bình 20 lần chạy, 30 vòng.

**Kết quả — đo theo cái hệ thống *biết*.** Chất lượng thật của top-10 mà hệ thống tin tưởng ở vòng cuối (trần có thể đạt là 0,537):

| $\varepsilon$ | 0,00 | 0,05 | 0,15 | 0,30 | 0,40 | 0,50 |
|---|---|---|---|---|---|---|
| Chất lượng top-10 | 0,348 | 0,377 | 0,393 | 0,401 | 0,409 | **0,422** |
| % của trần | 65% | 70% | 73% | 75% | 76% | **79%** |

**Có $\varepsilon$ tối ưu không? Theo thước đo này thì KHÔNG** — đường cong **đơn điệu tăng** trên cả khoảng $[0;\ 0{,}5]$. Càng khám phá nhiều thì hệ thống càng biết rõ danh mục.

**Nhưng đó là thước đo sai.** Nó chỉ đo *hệ thống biết gì*, mà **không tính giá của việc khám phá**: mỗi chỗ dành cho món ngẫu nhiên là một chỗ người dùng thật không nhận được món tốt nhất. Đo lại bằng **chất lượng thực sự giao tới người dùng** (trung bình chất lượng thật của những món đã được hiển thị):

| $\varepsilon$ | 0,00 | **0,05** | 0,10 | 0,20 | 0,30 | 0,40 | 0,50 |
|---|---|---|---|---|---|---|---|
| Chất lượng giao, trung bình cả kỳ | 0,3349 | **0,3349** | 0,3349 | 0,3255 | 0,3125 | 0,2944 | 0,2739 |
| Chất lượng giao, 5 vòng cuối | 0,3475 | **0,3588** | **0,3588** | 0,3530 | 0,3369 | 0,3179 | 0,3025 |

Bây giờ **có cực đại nội tại**, và nó nằm ở $\varepsilon \approx 0{,}05$–$0{,}10$:

- Ở 5 vòng cuối, $\varepsilon = 0{,}05$ cho **0,3588** so với **0,3475** của $\varepsilon = 0$ — tức **ngẫu nhiên hoá 5% đã trả đủ tiền cho chính nó** và còn lãi.
- $\varepsilon = 0{,}50$ làm chất lượng giao tụt xuống 0,3025, tức **mất 18%** so với mức tốt nhất.

**Vì sao đường cong có dạng như vậy.** Đây chính là đánh đổi khai thác – khám phá, và hai thước đo ở trên đo đúng hai vế của nó:

- **Vế khám phá** tăng đơn điệu theo $\varepsilon$: càng ngẫu nhiên thì càng mở rộng vùng nhìn thấy, nên *tri thức* của hệ thống càng tốt.
- **Vế khai thác** giảm tuyến tính theo $\varepsilon$: mỗi chỗ ngẫu nhiên là một chỗ mất đi cho món đã biết là tốt.
- Tổng của một hàm **tăng nhưng bão hoà** và một hàm **giảm tuyến tính** cho một cực đại nội tại, lệch về phía $\varepsilon$ nhỏ — vì lợi ích khám phá bão hoà rất nhanh (30% đầu tiên của thông tin rẻ hơn nhiều so với 30% cuối).

**Điều gì đổi nếu số vòng tăng từ 30 lên 300.** Cực đại **dịch về phía $\varepsilon$ nhỏ hơn nữa**, vì:

- Với chân trời dài, ngay cả $\varepsilon$ rất nhỏ cũng đủ thời gian để khám phá gần hết danh mục — lợi ích của $\varepsilon$ lớn bão hoà sớm.
- Trong khi cái giá của khai thác bị hy sinh thì **cộng dồn tuyến tính theo số vòng**, không bão hoà.

Đây đúng là trực giác đằng sau các chiến lược **giảm dần $\varepsilon$ theo thời gian** ($\varepsilon_t \propto 1/t$) và đằng sau các thuật toán bandit mà Sculley và cộng sự nhắc tới ở Mục 12.1 như lời giải đúng về mặt lý thuyết: khám phá nhiều lúc đầu, ít dần khi đã biết đủ.

**Điều rút ra cho công việc thật.** Một lượng ngẫu nhiên hoá **nhỏ** (5–10% số chỗ) gần như luôn là bữa trưa miễn phí — nó trả đủ tiền cho chính nó ngay trong vài chục vòng, đồng thời cho bạn nguồn dữ liệu không thiên lệch nói ở Mục 12.4. Một lượng **lớn** thì là thuế. Con số cụ thể phụ thuộc kích thước danh mục và chân trời thời gian, nên phải đo trên chính hệ thống của mình.

## Bài 9
@meta chuong=10 | dang=Thiết kế | kho=Khó

**Ràng buộc chi phối: nhãn trễ 30 ngày.** Mọi thiết kế phải xuất phát từ đây. Hệ quả trực tiếp: chỉ số chất lượng thật **luôn nói về quá khứ một tháng**, nên nó không bao giờ là hệ thống cảnh báo sớm — nó là hệ thống *xác nhận*.

**Bốn lớp giám sát.**

| Lớp | Giám sát gì | Nhịp | Dùng để làm gì |
|---|---|---|---|
| **1. Đầu vào thô** | lô tới đúng giờ không, số bản ghi trong khoảng $[0{,}7;\ 1{,}4]$ lần trung vị 7 ngày, schema khớp, tỉ lệ null từng cột | mỗi lô | chặn pipeline khi lỗi cứng |
| **2. Đặc trưng** | phân phối từng đặc trưng so với tham chiếu **cố định** và tham chiếu **trượt 30 ngày**; tỉ lệ danh mục lạ; **so giá trị đặc trưng giữa đường huấn luyện và đường phục vụ** | hằng ngày | bắt covariate shift và skew |
| **3. Dự đoán** | phân phối điểm số, **tỉ lệ gắn cờ**, độ lệch dự đoán theo lát cắt (thị trường, kênh, hạng khách, loại thẻ) | hằng giờ | **tuyến phòng thủ chính** — không cần nhãn |
| **4. Chất lượng** | precision/recall trên nhãn đã chín, AUC, tiền thiệt hại chặn được | hằng tuần, **trễ 30 ngày** | xác nhận và hiệu chuẩn lại |

**Chỉ số nào xứng đáng đánh thức người lúc 3 giờ sáng.** Chỉ ba, và cả ba đều là triệu chứng chứ không phải nguyên nhân (Mục 10.5):

1. **Pipeline dừng hoặc lô dữ liệu không tới** — không có dự đoán là sự cố tức thời.
2. **Tỉ lệ gắn cờ lệch khỏi dải lịch sử quá nhiều** — ví dụ nền 1,2% mà vọt lên 4,5% hoặc rơi xuống 0,2%. Cả hai chiều đều nguy: cao thì chặn nhầm khách thật, thấp thì đang để gian lận lọt.
3. **Chạm giới hạn hành động** — số tài khoản bị khoá trong một giờ vượt trần đã đặt (Mục 10.4). Đây là lưới an toàn cuối cùng và nó không cần biết mô hình đúng hay sai.

Mọi thứ khác — PSI của một đặc trưng, độ lệch phân phối điểm số — đi vào **vé** hoặc **bảng điều khiển**, không đánh thức ai.

**Phát hiện concept drift khi nhãn trễ một tháng.** Đây là phần khó nhất và là chỗ ghi điểm. Vì Chương 9 đã chứng minh **giám sát $X$ không bao giờ thấy concept drift**, cần bốn nguồn tín hiệu thay thế, xếp theo độ trễ:

| Nguồn | Độ trễ | Ghi chú |
|---|---|---|
| **Nhãn đại diện đến sớm** | vài giờ – vài ngày | khách gọi tổng đài khiếu nại giao dịch bị chặn; giao dịch bị hoàn thủ công; quyết định của đội điều tra nội bộ |
| **Phản hồi của nhân viên rà soát** | trong ngày | đội chống gian lận duyệt tay một mẫu các ca bị gắn cờ, cho nhãn "vàng" nhỏ nhưng nhanh |
| **Nhóm đối chứng nhỏ** | theo nhịp nhãn | một tỉ lệ nhỏ giao dịch không bị mô hình can thiệp, cho ước lượng **không thiên lệch** về tỉ lệ nền thật (Mục 12.4) |
| **Nhãn thật** | 30 ngày | dùng để xác nhận và hiệu chuẩn lại ba nguồn trên |

Thiết kế then chốt: **lấy mẫu rà soát tay có chủ đích**, không lấy ngẫu nhiên đều. Ưu tiên các ca ở gần ngưỡng quyết định và các ca trong lát cắt đang có tín hiệu lạ ở lớp 3 — đó là nơi thông tin trên mỗi nhãn là cao nhất.

Và một thực hành bắt buộc theo Mục 10.6: vì tỉ lệ nền gian lận **đổi theo mùa** (label shift, xem Bài 5c), **ngưỡng quyết định phải được hiệu chuẩn lại cùng mỗi lần huấn luyện lại** và phải là một phần của tạo tác mô hình, không phải một hằng số trong tệp cấu hình.

## Bài 10
@meta chuong=13 | dang=Thiết kế | kho=Khó

**Ràng buộc.** 50 000 lượt/ngày; ngân sách đánh giá bằng 3% chi phí suy luận. Nếu một lượt giám khảo tốn khoảng bằng một lượt phục vụ, thì 3% ngân sách cho phép chấm **khoảng 1 500 lượt mỗi ngày** — con số này quyết định mọi tỉ lệ lấy mẫu bên dưới.

**Ba tầng đánh giá.**

| Tầng | Tỉ lệ | Chi phí | Kiểm cái gì |
|---|---|---|---|
| **Kiểm tra quy tắc** | **100%** (50 000/ngày) | ≈ 0 | có trích dẫn không; trích dẫn có trỏ tới tài liệu thật trong ngữ cảnh không; độ dài trong khoảng cho phép; JSON hợp lệ; không lộ dữ liệu cá nhân; có từ chối khi ngoài phạm vi không |
| **Giám khảo LLM** | **3%** (1 500/ngày) | ~3% ngân sách | độ bám nguồn, độ liên quan của câu trả lời, độ liên quan của ngữ cảnh, giọng điệu |
| **Người chấm** | **~100 lượt/tuần** | công sức người | sự thật nền để **hiệu chuẩn lại giám khảo** |

Lấy mẫu cho tầng 2 **không nên ngẫu nhiên đều**. Phân bổ theo giá trị thông tin: khoảng một nửa lấy ngẫu nhiên để có ước lượng không thiên lệch, nửa còn lại **lấy có chủ đích** vào những lượt đã bị tầng 1 đánh dấu nghi ngờ, những lượt có điểm truy hồi thấp, và những chủ đề mới xuất hiện.

**Tách lỗi truy hồi khỏi lỗi sinh.** Đây là phép chẩn đoán bắt buộc (Mục 13.5). Với mỗi lượt được chấm, ghi lại đủ để trả lời **hai câu hỏi tách rời**:

```
1. Tài liệu đúng CÓ nằm trong ngữ cảnh không?
   → cần một tập câu hỏi có nhãn tài liệu đúng; đo recall@k, MRR
   → KHÔNG  ⇒ lỗi TRUY HỒI: sửa chia đoạn, embedding, truy hồi lai, rerank

2. Nếu CÓ, câu trả lời có được chống đỡ bởi ngữ cảnh không?
   → đo ĐỘ BÁM NGUỒN bằng giám khảo, có yêu cầu trích dẫn từng câu
   → KHÔNG  ⇒ lỗi SINH: sửa prompt, đổi mô hình, buộc trích dẫn
```

Phải ghi cả **danh sách id đoạn đã lấy về** và **điểm truy hồi** vào bản ghi vết, nếu không thì sau này không tách được — và đây là thứ phải quyết định ghi **trước**, giống như việc ghi xác suất hiển thị ở Mục 12.4.

**Hai vòng đánh giá.**

1. **Tập vàng làm cửa CI.** Khoảng 200–500 câu hỏi khó có nhãn tài liệu đúng và câu trả lời tham chiếu. Phải đạt trước **mọi** thay đổi của bất kỳ thứ nào trong sáu thành phần phiên bản ở Mục 13.8 — prompt, bản mô hình, cấu hình truy hồi, ảnh chụp corpus, định nghĩa công cụ, rubric.
2. **Chấm mẫu trực tuyến** như bảng trên, gắn điểm vào bản ghi vết để truy được về từng lượt.

**Giữ cho tập vàng sống.** Mỗi sự cố trong sản xuất phải sinh ra ít nhất một ca mới trong tập vàng. Đây là thực hành *tập kiểm định phải động* ở Mục 13.4; một tập vàng đóng băng chỉ đo những thứ bạn đã biết cách làm đúng.

**Phát hiện khi chính giám khảo đã trôi.** Bốn cơ chế, nên có cả bốn:

1. **Đo độ đồng thuận giám khảo – người** trên mẫu hằng tuần, và **vẽ nó theo thời gian**. Đây là chỉ số chính. Nó tụt là giám khảo đã trôi.
2. **Giữ một tập neo cố định** — khoảng 50 cặp câu hỏi/câu trả lời đã được người chấm và **đóng băng điểm**. Cho giám khảo chấm lại hằng tuần. Nếu điểm trên tập neo đổi mà nội dung không đổi, thì thứ đã đổi chính là giám khảo (hoặc bản mô hình của nhà cung cấp).
3. **Ghim phiên bản mô hình giám khảo** và coi mỗi lần nâng cấp là một lần phát hành đầy đủ: chạy song song bản cũ và bản mới trên tập neo trước khi chuyển.
4. **Kiểm tra thiên lệch định kỳ** (Mục 13.3): hỏi lại với thứ tự đảo để đo thiên lệch vị trí; đối chiếu điểm với độ dài câu trả lời để đo thiên lệch độ dài.

Cơ chế 2 là cơ chế rẻ nhất và mạnh nhất, vì nó tách bạch được hai nguyên nhân mà từ ngoài nhìn vào thì giống hệt nhau: *hệ thống của tôi kém đi* và *thước đo của tôi đổi*.
