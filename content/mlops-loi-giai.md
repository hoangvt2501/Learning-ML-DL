# Lời giải chi tiết — MLOps

Mỗi mục ứng với một bài trong Chương 14. Dòng `@meta` được script build đọc để gắn nhãn
chương, dạng bài và độ khó; nó không hiện ra trên trang. Số liệu của Bài 1, 2, 3, 4, 7 và 8
được in ra bởi `code/mlops/bai_tap.py`.

## Bài 1
@meta chuong=9 | dang=Tính tay | kho=Cơ bản

**(a) PSI kỳ vọng khi không có dịch chuyển.** Theo Mục 9.5, với hai mẫu cùng cỡ $n$ và $k$ bin,

$$\mathbb{E}[\mathrm{PSI}] \approx \frac{2(k-1)}{n} = \frac{2 \times 9}{500} = 0{,}036.$$

Mô phỏng 20 000 lần trong `bai_tap.py` cho trung bình 0,0364, khớp với công thức.

**(b) Số cảnh báo giả mỗi ngày.** Vì $\tfrac{n}{2}\,\mathrm{PSI}$ có phân phối xấp xỉ $\chi^2_{k-1}$, xác suất PSI vượt 0,10 khi không có dịch chuyển là

$$P(\mathrm{PSI} > 0{,}10) = P\!\left(\chi^2_9 > \frac{0{,}10 \times 500}{2}\right) = P(\chi^2_9 > 25) \approx 0{,}0030.$$

Mô phỏng cho 0,0036, lớn hơn công thức một chút, vì các bin được xác định từ chính mẫu tham chiếu và $n$ hữu hạn. Phân phối của PSI khi không có dịch chuyển:

| Phân vị | Theo công thức $\tfrac{2}{n}\chi^2_{9,\,q}$ | Mô phỏng |
|---|---|---|
| 95% | 0,0677 | 0,0684 |
| 99% | 0,0867 | 0,0874 |
| 99,9% | 0,1115 | 0,1131 |

Với một đặc trưng, xác suất 0,3% mỗi lần đo là nhỏ. Nhưng hệ thống theo dõi 200 đặc trưng mỗi ngày, nên số cảnh báo giả trung bình là $200 \times 0{,}0030 \approx 0{,}6$ mỗi ngày, tức khoảng 4 cảnh báo giả mỗi tuần trên một hệ thống không có vấn đề gì. Đây là cơ chế sinh ra báo động giả ở Mục 3.2 và Mục 10.5. Vấn đề không nằm ở ngưỡng 0,10 cho một đặc trưng, mà ở việc chưa hiệu chỉnh cho kiểm định nhiều lần: với 200 đặc trưng, ngưỡng cho mỗi đặc trưng phải chặt hơn nhiều, hoặc dùng hiệu chỉnh Bonferroni hay Benjamini–Hochberg.

**(c) Cỡ mẫu để PSI kỳ vọng nhỏ hơn 0,01.**

$$\frac{2(k-1)}{n} < 0{,}01 \iff n > \frac{2 \times 9}{0{,}01} = 1\,800.$$

**Nhận xét.** Ngưỡng PSI hợp lý phụ thuộc vào $n$, $k$ và số đặc trưng được theo dõi. Ba cách xử lý theo Mục 9.5: tính ngưỡng theo phân phối $\chi^2$ hoặc bằng mô phỏng; dùng kiểm định có p-value kèm hiệu chỉnh cho kiểm định nhiều lần; và yêu cầu tín hiệu kéo dài qua nhiều lần đo trước khi cảnh báo (Mục 10.5).

## Bài 2
@meta chuong=8 | dang=Tính tay | kho=Cơ bản

**(a) Cỡ mẫu mỗi nhánh.** Cải thiện tương đối 5% trên nền 2%:

$$p_1 = 0{,}02, \qquad p_2 = 0{,}02 \times 1{,}05 = 0{,}021, \qquad \Delta = p_2 - p_1 = 0{,}001.$$

Chênh lệch tuyệt đối chỉ là 0,1 điểm phần trăm; đây là đại lượng đi vào mẫu số của công thức. Với $z_{0{,}975} = 1{,}95996$ và $z_{0{,}80} = 0{,}84162$:

$$n = \frac{(1{,}95996 + 0{,}84162)^2\,\left[0{,}02 \times 0{,}98 + 0{,}021 \times 0{,}979\right]}{0{,}001^2} = \frac{7{,}8489 \times 0{,}040159}{10^{-6}} \approx 315\,203$$

mẫu mỗi nhánh, tức hơn 630 000 mẫu cho cả thí nghiệm.

**(b) Thời gian chạy.** Với 40 000 lượt mỗi nhánh mỗi ngày: $315\,203 / 40\,000 \approx 7{,}9$ ngày. Trong thực tế nên chạy tròn một hoặc hai tuần để mỗi ngày trong tuần xuất hiện như nhau, tránh ảnh hưởng của chu kỳ tuần.

**(c) Rút thời gian xuống một nửa mà không giảm lực kiểm định.**

| Cách | Cơ chế | Đánh giá |
|---|---|---|
| Giảm phương sai bằng CUPED | dùng giá trị của chỉ số trong giai đoạn trước thí nghiệm làm biến hiệp phương sai (Mục 8.4) | hợp lệ; giữ nguyên $\alpha$ và lực kiểm định. Nếu phương sai giảm 40%, cỡ mẫu còn 189 122, tức 4,7 ngày |
| Tăng tỉ lệ lưu lượng dành cho thí nghiệm | ví dụ từ 10% lên 20% tổng lưu lượng cho mỗi nhánh | hợp lệ, nếu chấp nhận nhiều người dùng hơn tiếp xúc với phiên bản chưa được kiểm chứng |
| Đổi sang chỉ số có phương sai nhỏ hơn | ví dụ chỉ số gần với thay đổi hơn, như tỉ lệ nhấp vào kết quả thay vì tỉ lệ mua hàng | hợp lệ, nhưng phải kiểm tra chỉ số mới vẫn gắn với chỉ số sản phẩm (Mục 6.2) |
| Chỉ phát hiện mức cải thiện lớn hơn | mức cải thiện tương đối 7,11% cho cỡ mẫu 157 602, đúng một nửa | hợp lệ về thống kê, nhưng thay đổi câu hỏi: cải thiện thật 5% sẽ thường không được phát hiện |
| Dừng sớm khi thấy $p < 0{,}05$ | xem kết quả giữa chừng | không hợp lệ: đây là lỗi peeking ở Mục 8.5, làm tỉ lệ dương tính giả vượt xa 5% |
| Giữ nguyên thiết kế, chỉ chạy 3,9 ngày | | không hợp lệ theo đề bài: lực kiểm định giảm từ 80% xuống 51% |

Mức 7,11% gần bằng $5\% \times \sqrt 2 \approx 7{,}07\%$ vì $n$ tỉ lệ nghịch với $\Delta^2$; chênh lệch nhỏ đến từ việc phương sai $p_2(1-p_2)$ cũng thay đổi theo $p_2$.

**Nhận xét.** Thời gian chạy một A/B test phụ thuộc bốn đại lượng: tỉ lệ nền, mức cải thiện nhỏ nhất cần phát hiện, lưu lượng, và phương sai của chỉ số. Giảm phương sai là cách duy nhất rút ngắn thời gian mà không đổi câu hỏi và không giảm độ tin cậy.

## Bài 3
@meta chuong=7 | dang=Tính tay | kho=Trung bình

**(a) Tỉ lệ yêu cầu gặp ít nhất một dịch vụ chậm.** Mỗi dịch vụ chậm hơn p99 của nó với xác suất 0,01, và các dịch vụ độc lập:

$$P(\text{ít nhất một nhánh chậm}) = 1 - 0{,}99^{25} \approx 0{,}222.$$

Hơn một phần năm số yêu cầu gặp một dịch vụ chậm hơn 80 ms, dù mỗi dịch vụ chỉ chậm như vậy ở 1% số lần gọi.

**(b) Phân vị mỗi dịch vụ phải đạt.** Cần $F(t)^{25} = 0{,}99$, nên

$$F(t) = 0{,}99^{1/25} \approx 0{,}999598,$$

tức phân vị khoảng 99,96: mỗi dịch vụ chỉ được chậm hơn 80 ms một lần trong khoảng 2 488 lần gọi. Yêu cầu này chặt hơn p99 khoảng 25 lần, và trong thực tế rất khó đạt bằng cách tối ưu từng dịch vụ.

**(c) Hai cách giảm phần đuôi mà không làm từng dịch vụ nhanh hơn.**

1. **Giảm số nhánh.** Gộp nhiều đặc trưng vào một lời gọi, hoặc lưu đệm những đặc trưng ít thay đổi. Khi $k$ nhỏ, $1 - 0{,}99^k$ tăng gần tuyến tính theo $k$: giảm từ 25 xuống 10 nhánh làm tỉ lệ gặp nhánh chậm giảm từ 22,2% xuống 9,6%; xuống 5 nhánh thì còn 4,9%.
2. **Đặt thời hạn chờ và trả lời với kết quả một phần.** Dừng chờ ở một mức, chẳng hạn 80 ms, và dùng giá trị mặc định hoặc giá trị lưu đệm cho những nhánh chưa trả lời. Cách này đổi vấn đề độ trễ thành vấn đề chất lượng: mô hình thiếu một đặc trưng thường chỉ kém đi một chút, trong khi một yêu cầu chậm vài trăm mili giây có thể làm người dùng rời đi. Mô hình nên được huấn luyện với cả những trường hợp thiếu đặc trưng như vậy.

Cách thứ ba là **yêu cầu dự phòng** (Dean và Barroso, 2013): nếu sau một khoảng, ví dụ bằng p95 của dịch vụ, chưa có kết quả thì gửi thêm một yêu cầu tới bản sao khác và dùng kết quả về trước. Cách này tốn thêm một phần nhỏ tài nguyên để cắt phần lớn phần đuôi.

## Bài 4
@meta chuong=4 | dang=Suy luận | kho=Cơ bản

**Tập huấn luyện đúng.** Với mỗi dòng nhãn, lấy bản ghi đặc trưng có `valid_from` lớn nhất nhưng không vượt quá `event_time`:

| user_id | event_time | nhan | tong_don_hang | lấy từ bản ghi |
|---|---|---|---|---|
| 7 | 2026-02-10 | 0 | 3 | `valid_from = 2026-01-01` |
| 7 | 2026-04-15 | 1 | 19 | `valid_from = 2026-03-01` |

Bản ghi ngày 2026-06-01 (giá trị 52) không được dùng cho dòng nào, vì cả hai sự kiện xảy ra trước ngày đó.

**Tập huấn luyện sai.** Có hai trường hợp, tuỳ cách ghép.

*Ghép với bảng giá trị hiện tại:* cả hai dòng đều nhận giá trị 52.

| user_id | event_time | nhan | tong_don_hang |
|---|---|---|---|
| 7 | 2026-02-10 | 0 | 52 |
| 7 | 2026-04-15 | 1 | 52 |

*Ghép với bảng lịch sử nhưng thiếu điều kiện thời gian:* mỗi dòng nhãn ghép với cả ba bản ghi, cho 6 dòng thay vì 2. Tập dữ liệu vừa chứa thông tin tương lai, vừa bị nhân bản, làm sai cả trọng số của các mẫu.

**Mô hình học sai điều gì.** Trong tập dữ liệu thật có nhiều khách hàng, những khách ở lại lâu tích luỹ nhiều đơn hàng hơn. Khi mọi dòng dùng giá trị hiện tại, mô hình học quan hệ "số đơn hàng cao thì không rời bỏ". Quan hệ này có thật trong tập huấn luyện, nhưng nó là hệ quả của nhãn chứ không phải nguyên nhân: con số 52 tồn tại vì khách đã ở lại tới tháng 6. Kết quả đánh giá ngoại tuyến tốt, nhưng khi dự đoán thật cho một khách vào tháng 2, giá trị có được là 3 chứ không phải 52, nên mô hình hoạt động kém hơn nhiều so với đánh giá (Mục 4.2).

**Cách phòng.** Dùng cả hai lớp:

1. Ràng buộc trong hợp đồng dữ liệu (Mục 3.3): `valid_from <= event_time` phải đúng với mọi dòng của tập huấn luyện. Đây là một kiểm tra chạy tự động được.
2. Dùng phép ghép theo thời điểm (as-of join) có sẵn trong kho đặc trưng hoặc thư viện dữ liệu, thay vì tự viết `JOIN`.

## Bài 5
@meta chuong=9 | dang=Phân loại | kho=Cơ bản

| | Tình huống | Loại | Phát hiện được chỉ bằng $X$? |
|---|---|---|---|
| (a) | Quảng cáo thu hút nhiều người dùng dưới 25 tuổi | covariate shift | có |
| (b) | Định nghĩa "nợ xấu" đổi từ quá hạn 90 ngày thành 60 ngày | concept drift | không |
| (c) | Tỉ lệ gian lận tăng gấp ba, cách gian lận không đổi | label shift | có, một cách gián tiếp |
| (d) | Kỳ vọng của người dùng về kết quả tìm kiếm tốt thay đổi | concept drift | không |

**(a)** $P(X)$ thay đổi: phân phối tuổi dịch về phía trẻ hơn. Quan hệ giữa hồ sơ và hành vi không đổi, tức $P(Y \mid X)$ giữ nguyên. Dễ phát hiện qua phân phối của đặc trưng tuổi.

**(b)** Đây là trường hợp rõ nhất của concept drift, vì chính định nghĩa của nhãn thay đổi. Hồ sơ khách hàng đi vào hệ thống như cũ, nên $P(X)$ không đổi; nhưng với cùng một hồ sơ, xác suất bị gắn nhãn "nợ xấu" tăng lên. Mọi phép kiểm tra trên $X$ đều không thấy gì, như kịch bản concept drift ở Mục 9.2.

**(c)** $P(Y)$ tăng gấp ba, còn $P(X \mid Y)$, tức đặc điểm của một giao dịch gian lận, giữ nguyên. Phát hiện được qua $X$ một cách gián tiếp: vì $P(X) = \sum_y P(X \mid y)\,P(y)$ là một hỗn hợp, thay đổi tỉ lệ hỗn hợp làm phân phối của $X$ thay đổi theo. Độ chính xác có thể gần như không đổi (như kịch bản label shift ở Mục 9.2), nhưng precision, recall và hiệu chuẩn xác suất thay đổi. Nếu ngưỡng quyết định được đặt cho tỉ lệ nền 1% mà nay tỉ lệ nền là 3%, số giao dịch bị đánh dấu thay đổi theo; đây là vấn đề ngưỡng cố định ở Mục 10.6.

**(d)** Truy vấn của người dùng có thể giữ nguyên, nhưng kết quả được coi là tốt đã thay đổi, tức $P(Y \mid X)$ thay đổi. Trường hợp này khó hơn (b), vì không có sự kiện nào trong hệ thống đánh dấu thời điểm thay đổi xảy ra: nguyên nhân nằm ở bên ngoài, đúng như phần *Dealing with Changes in the External World* của Sculley và cộng sự (2015).

**Nhận xét.** Hai trong bốn trường hợp không phát hiện được bằng cách giám sát $X$, và đó là hai trường hợp gây hại nhiều nhất cho độ chính xác. Vì vậy thiết kế giám sát phải bắt đầu bằng câu hỏi ở Mục 3.5: sau bao lâu thì biết nhãn thật?

## Bài 6
@meta chuong=6 | dang=Suy luận | kho=Trung bình

**(a) Chấm từng nhóm.** Nhắc lại thang điểm ở Mục 6.5: 0,5 điểm nếu làm thủ công và có ghi lại kết quả, 1 điểm nếu có hệ thống chạy tự động và định kỳ.

| Nhóm | Các mục đạt | Điểm |
|---|---|---|
| Dữ liệu | Data 1 (lược đồ, tự động) = 1; Data 7 (kiểm thử mã đặc trưng, tự động) = 1 | 2 |
| Phát triển mô hình | đề bài không nêu mục nào | 0 |
| Hạ tầng | Infra 1 (tái lập) = 1; Infra 3 (kiểm thử tích hợp) = 1; Infra 6 (canary) = 1; Infra 7 (quay lui) = 1 | 4 |
| Giám sát | một biểu đồ độ trễ để xem khi cần, không có quy trình định kỳ và không ghi lại kết quả, nên chưa đủ cho Monitor 6 | 0 |

$$\text{ML Test Score} = \min(2,\ 0,\ 4,\ 0) = 0.$$

Kể cả nếu biểu đồ độ trễ được xem định kỳ và có ghi lại kết quả, nhóm giám sát chỉ được 0,5 điểm cho Monitor 6, và điểm cuối cùng vẫn là 0 vì nhóm phát triển mô hình bằng 0.

Quy tắc lấy giá trị nhỏ nhất được chọn vì cả bốn nhóm đều cần thiết. Nhóm này có hạ tầng tốt hơn nhiều hệ thống thực tế (4 trên 7 mục, đều tự động), nhưng theo bảng diễn giải của bài báo, điểm 0 nghĩa là hệ thống "giống một dự án nghiên cứu hơn là một hệ thống sản xuất". Nhận định này có cơ sở: hạ tầng tốt làm việc triển khai nhanh hơn, kể cả triển khai một mô hình đã hỏng, và khi không có giám sát thì không ai biết mô hình đang hoạt động tốt hay không.

**(b) Nên làm gì đầu tiên.** Vì điểm là giá trị nhỏ nhất của bốn nhóm, đầu tư thêm vào nhóm hạ tầng không làm tăng điểm. Phải nâng cả hai nhóm đang bằng 0. Thứ tự đề xuất, theo tỉ lệ lợi ích trên công sức:

1. **Monitor 3**, đặc trưng lúc huấn luyện và lúc phục vụ tính ra cùng giá trị. Nhóm đã có canary, nên chỉ cần ghi lại giá trị đặc trưng ở cả hai đường và so sánh. Mục này phát hiện một trong những lỗi tốn kém nhất (Mục 4.1).
2. **Monitor 7**, chất lượng dự đoán không suy giảm. Khi nhãn đến trễ, bắt đầu bằng độ lệch dự đoán theo lát cắt (Mục 10.3).
3. **Monitor 6**, thêm cảnh báo tự động cho độ trễ và bộ nhớ dựa trên biểu đồ đã có; đây là việc rẻ nhất.
4. **Model 6**, chất lượng đủ tốt trên mọi lát cắt quan trọng: rẻ, và phát hiện đúng loại vấn đề ở Mục 6.3.
5. **Model 4**, đã biết ảnh hưởng của việc mô hình cũ đi: một thí nghiệm như Mục 11.1 trên dữ liệu lịch sử, đồng thời trả lời câu hỏi về nhịp huấn luyện lại.

Năm việc trên, nếu tự động hoá, đưa hai nhóm đang bằng 0 lên khoảng 2 hoặc 3 điểm, và điểm cuối cùng lên 2, tức mức "đã có bước đầu đưa vào sản xuất", mà không cần thay đổi hạ tầng.

## Bài 7
@meta chuong=11 | dang=Thí nghiệm | kho=Trung bình

Kết quả từ `bai_tap.py`, dùng cùng mô phỏng như phần (G) của `experiments.py`, với ba tốc độ dịch chuyển và năm nhịp huấn luyện lại. Độ chính xác trung bình qua 52 tuần:

| Tốc độ dịch chuyển | Không bao giờ | Mỗi quý | Mỗi tháng | Mỗi 2 tuần | Mỗi tuần | Nhịp thưa nhất trong 1 điểm so với hằng tuần |
|---|---|---|---|---|---|---|
| 0,010 rad/tuần (chậm) | 76,15% | 77,67% | 77,74% | 77,76% | 77,76% | mỗi quý |
| 0,035 rad/tuần (vừa) | 64,61% | 76,89% | 77,68% | 77,78% | 77,80% | mỗi quý |
| 0,080 rad/tuần (nhanh) | 45,08% | 73,04% | 76,96% | 77,50% | 77,66% | mỗi tháng |

Hàng giữa trùng với bảng ở Mục 11.1, vì dùng cùng mô phỏng và cùng hạt giống.

**Nhận xét.**

1. **Tốc độ dịch chuyển ảnh hưởng mạnh nhất tới cột "không bao giờ".** Khi tốc độ tăng 8 lần, thiệt hại của việc không huấn luyện lại tăng từ 1,6 lên 32,6 điểm phần trăm. Nhịp cần thiết chỉ thay đổi từ mỗi quý sang mỗi tháng.
2. **Nhịp hợp lý thưa dần khi dịch chuyển chậm lại, nhưng thay đổi ít.** Khi mô hình được huấn luyện lại đủ thường để luôn gần với thế giới hiện tại, thiệt hại còn lại tăng theo lượng dịch chuyển tích luỹ giữa hai lần huấn luyện; làm dày thêm nhịp chỉ giảm thêm một phần nhỏ.
3. **Phần lớn lợi ích đến từ việc có huấn luyện lại.** Ở cả ba tốc độ, bước từ "không bao giờ" lên "mỗi quý" chiếm phần lớn tổng lợi ích; các bước sau nhỏ.

**Quy tắc thực hành.** Chọn nhịp huấn luyện lại bằng cách chạy thí nghiệm này trên dữ liệu lịch sử của chính bài toán, rồi chọn nhịp thưa nhất mà chênh lệch so với nhịp dày nhất nằm trong mức chấp nhận được về mặt kinh doanh. Ở tốc độ vừa, chênh lệch giữa hằng tuần và hằng quý chỉ 0,91 điểm, trong khi số lần huấn luyện gấp 13 lần.

## Bài 8
@meta chuong=12 | dang=Thí nghiệm | kho=Khó

Kết quả từ `bai_tap.py`. Mô phỏng giống phần (H) của `experiments.py`, nhưng dùng một danh mục sinh với hạt giống riêng: chất lượng trung bình của 10 món tốt nhất là 0,610 (so với 0,555 ở Mục 12.3). Mỗi cấu hình là trung bình của 20 lần chạy. Hai đại lượng được đo:

- **Chất lượng của danh sách xếp hạng:** chất lượng thật trung bình của 10 món mà hệ thống xếp đầu ở vòng cuối, tức những gì hệ thống biết.
- **Chất lượng giao tới người dùng:** chất lượng thật trung bình của các món đã thực sự hiển thị, gồm cả các món ngẫu nhiên, tính trên cả kỳ và trên 5 vòng cuối.

**Với 30 vòng:**

| $\varepsilon$ | 0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|---|
| Chất lượng danh sách, vòng cuối | 0,360 | 0,390 | 0,404 | 0,425 | 0,431 | 0,453 |
| Tỉ lệ so với mức tối đa | 59% | 64% | 66% | 70% | 71% | 74% |
| Chất lượng giao, cả kỳ | 0,3480 | 0,3495 | 0,3389 | 0,3302 | 0,3108 | 0,2932 |
| Chất lượng giao, 5 vòng cuối | 0,3604 | 0,3744 | 0,3686 | 0,3626 | 0,3428 | 0,3246 |

**(a) Có giá trị $\varepsilon$ tốt nhất không?** Tuỳ đại lượng:

- **Theo chất lượng danh sách: không.** Đại lượng này tăng đơn điệu theo $\varepsilon$ trên cả khoảng: càng khám phá nhiều, hệ thống càng biết rõ danh mục.
- **Theo chất lượng giao tới người dùng: có**, ở $\varepsilon = 0{,}1$, tức một vị trí ngẫu nhiên trong mười. Trên 5 vòng cuối, $\varepsilon = 0{,}1$ cho 0,3744 so với 0,3604 khi không ngẫu nhiên hoá; trên cả kỳ hai giá trị gần bằng nhau (0,3495 so với 0,3480), nghĩa là trong 30 vòng, một vị trí ngẫu nhiên đã bù đủ cho chi phí của nó. Với $\varepsilon = 0{,}5$, chất lượng giao trên cả kỳ giảm còn 0,2932, thấp hơn 16% so với mức tốt nhất.

Hình dạng của các đường cong phản ánh đánh đổi giữa khai thác và khám phá. Lợi ích của khám phá (hệ thống biết thêm về danh mục) tăng theo $\varepsilon$ nhưng tăng chậm dần. Chi phí của khám phá (mỗi vị trí ngẫu nhiên hiển thị một món có chất lượng trung bình chỉ khoảng 0,154) tăng tuyến tính theo $\varepsilon$. Tổng của một hàm tăng chậm dần và một hàm giảm tuyến tính có cực đại ở bên trong khoảng, lệch về phía $\varepsilon$ nhỏ.

**(b) Khi số vòng tăng lên 300:**

| $\varepsilon$ | 0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|---|
| Chất lượng danh sách, vòng cuối | 0,360 | 0,488 | 0,518 | 0,541 | 0,559 | 0,570 |
| Tỉ lệ so với mức tối đa | 59% | 80% | 85% | 89% | 92% | 93% |
| Chất lượng giao, cả kỳ | 0,3590 | 0,4237 | 0,4187 | 0,4034 | 0,3864 | 0,3602 |
| Chất lượng giao, 5 vòng cuối | 0,3601 | 0,4615 | 0,4588 | 0,4424 | 0,4153 | 0,3817 |

Ba thay đổi:

1. **Không ngẫu nhiên hoá thì không có gì thay đổi.** Với $\varepsilon = 0$, chất lượng danh sách vẫn là 0,360 sau 300 vòng, như sau 30 vòng: hệ thống chỉ hiển thị các món đã gặp ở vòng khởi động, và thêm thời gian không giúp nó học thêm.
2. **Giá trị của khám phá tăng mạnh theo độ dài thời gian.** Với 30 vòng, $\varepsilon = 0{,}1$ chỉ hơn $\varepsilon = 0$ khoảng 0,4% về chất lượng giao trên cả kỳ; với 300 vòng, mức chênh là 18% (0,4237 so với 0,3590). Thông tin thu được từ khám phá được dùng trong nhiều vòng hơn.
3. **Giá trị tốt nhất vẫn là $\varepsilon = 0{,}1$**, giá trị khác 0 nhỏ nhất trong mô phỏng này (vì $\varepsilon \times 10$ phải là số nguyên vị trí). Khoảng cách giữa $\varepsilon = 0{,}1$ và $0{,}2$ thu hẹp lại, vì khi đã biết khá rõ danh mục, lợi ích của khám phá thêm giảm nhanh.

Nhận xét này phù hợp với các chiến lược giảm dần $\varepsilon$ theo thời gian, và với các thuật toán bandit mà Sculley và cộng sự nhắc tới ở Mục 12.1: khám phá nhiều khi còn biết ít, ít dần khi đã biết đủ. Các thuật toán như UCB hay Thompson sampling còn hiệu quả hơn, vì chúng hướng việc khám phá vào những món chưa chắc chắn thay vì chọn ngẫu nhiên đều.

**Kết luận cho thực tế.** Một mức ngẫu nhiên hoá nhỏ gần như luôn có lợi: nó bù đủ chi phí của mình sau một số vòng, và cung cấp nguồn dữ liệu không bị mô hình làm sai lệch (Mục 12.4). Mức lớn thì làm giảm chất lượng người dùng nhận được. Giá trị cụ thể phụ thuộc kích thước danh mục và độ dài thời gian, nên cần đo trên chính hệ thống.

## Bài 9
@meta chuong=10 | dang=Thiết kế | kho=Khó

**Ràng buộc chính: nhãn trễ 30 ngày.** Chỉ số chất lượng thật luôn phản ánh tình hình của một tháng trước, nên không thể dùng làm cảnh báo sớm; nó dùng để xác nhận và hiệu chỉnh.

**Bốn lớp giám sát.**

| Lớp | Giám sát gì | Tần suất | Mục đích |
|---|---|---|---|
| 1. Đầu vào thô | lô dữ liệu tới đúng giờ; số bản ghi trong khoảng 0,7 tới 1,4 lần trung vị 7 ngày; lược đồ khớp; tỉ lệ giá trị thiếu từng cột | mỗi lô | dừng pipeline khi có lỗi cứng |
| 2. Đặc trưng | phân phối từng đặc trưng so với tham chiếu cố định và tham chiếu trượt 30 ngày; tỉ lệ giá trị danh mục lạ; so sánh giá trị đặc trưng giữa đường huấn luyện và đường phục vụ | hằng ngày | phát hiện covariate shift và lệch huấn luyện–phục vụ |
| 3. Dự đoán | phân phối điểm số; tỉ lệ giao dịch bị đánh dấu; độ lệch dự đoán theo lát cắt (thị trường, kênh, hạng khách hàng, loại thẻ) | hằng giờ | lớp giám sát chính, không cần nhãn |
| 4. Chất lượng | precision và recall trên nhãn đã có, AUC, số tiền thiệt hại chặn được | hằng tuần, trễ 30 ngày | xác nhận và hiệu chỉnh |

**Chỉ số đủ quan trọng để gọi người trực lúc nửa đêm.** Chỉ ba, đều là triệu chứng (Mục 10.5):

1. **Pipeline dừng hoặc không có dữ liệu tới:** không có dự đoán là sự cố tức thời.
2. **Tỉ lệ giao dịch bị đánh dấu lệch xa khỏi khoảng lịch sử,** ví dụ tỉ lệ nền 1,2% mà tăng lên 4,5% hoặc giảm xuống 0,2%. Cả hai chiều đều nguy hiểm: tăng thì chặn nhầm khách hàng thật, giảm thì để lọt gian lận.
3. **Chạm giới hạn hành động,** ví dụ số tài khoản bị khoá trong một giờ vượt mức trần (Mục 10.4). Đây là lớp an toàn cuối cùng, không cần biết mô hình đúng hay sai.

Các tín hiệu khác, như PSI của một đặc trưng hay thay đổi của phân phối điểm số, được đưa vào phiếu xử lý hoặc bảng theo dõi.

**Phát hiện concept drift khi nhãn trễ một tháng.** Mục 9.2 cho thấy giám sát $X$ không phát hiện được concept drift, nên cần các nguồn tín hiệu về nhãn đến sớm hơn:

| Nguồn | Độ trễ | Ghi chú |
|---|---|---|
| Nhãn thay thế đến sớm | vài giờ tới vài ngày | khách hàng khiếu nại giao dịch bị chặn; giao dịch bị hoàn tiền thủ công; kết luận của nhóm điều tra nội bộ |
| Nhân viên rà soát | trong ngày | nhóm chống gian lận xem xét một mẫu các giao dịch, tạo ra một tập nhãn nhỏ nhưng nhanh |
| Nhóm đối chứng | theo nhịp nhãn | một tỉ lệ nhỏ giao dịch không chịu tác động của mô hình, cho ước lượng không bị sai lệch về tỉ lệ gian lận thật (Mục 12.4) |
| Nhãn thật | 30 ngày | xác nhận và hiệu chỉnh ba nguồn trên |

Điểm quan trọng trong thiết kế: mẫu cho nhân viên rà soát nên được chọn có chủ đích, không chọn đều. Ưu tiên các giao dịch gần ngưỡng quyết định và các giao dịch thuộc lát cắt đang có tín hiệu bất thường ở lớp 3, vì đó là nơi mỗi nhãn mang nhiều thông tin nhất.

Ngoài ra, theo Mục 10.6: tỉ lệ gian lận thay đổi theo mùa (label shift, Bài 5c), nên ngưỡng quyết định phải được xác định lại mỗi lần huấn luyện lại và lưu cùng mô hình, không phải là một hằng số trong tệp cấu hình.

## Bài 10
@meta chuong=13 | dang=Thiết kế | kho=Khó

**Ràng buộc.** 50 000 lượt mỗi ngày; ngân sách đánh giá bằng 3% chi phí suy luận. Nếu một lượt gọi giám khảo tốn xấp xỉ một lượt phục vụ, ngân sách này cho phép chấm khoảng 1 500 lượt mỗi ngày. Con số này quyết định các tỉ lệ lấy mẫu dưới đây; nếu giám khảo dùng mô hình rẻ hơn mô hình phục vụ, có thể chấm nhiều hơn.

**Ba tầng đánh giá.**

| Tầng | Tỉ lệ | Chi phí | Kiểm tra gì |
|---|---|---|---|
| Kiểm tra theo quy tắc | mọi lượt (50 000 mỗi ngày) | gần như bằng 0 | có trích dẫn không; trích dẫn có trỏ tới đoạn thật trong ngữ cảnh không; độ dài trong khoảng cho phép; định dạng hợp lệ; không lộ dữ liệu cá nhân; từ chối khi câu hỏi ngoài phạm vi |
| Giám khảo LLM | khoảng 3% (1 500 mỗi ngày) | khoảng 3% chi phí suy luận | độ trung thành với ngữ cảnh, mức liên quan của câu trả lời, mức liên quan của ngữ cảnh, giọng văn |
| Người chấm | khoảng 100 lượt mỗi tuần | thời gian của người | đánh giá chuẩn để hiệu chỉnh giám khảo |

Mẫu cho tầng 2 không nên chọn đều. Khoảng một nửa chọn ngẫu nhiên, để có ước lượng không bị sai lệch về chất lượng chung; nửa còn lại chọn có chủ đích: các lượt bị tầng 1 đánh dấu nghi ngờ, các lượt có điểm truy xuất thấp, các chủ đề mới xuất hiện.

**Tách lỗi truy xuất khỏi lỗi sinh** (Mục 13.5). Với mỗi lượt được chấm, cần trả lời hai câu hỏi riêng:

```
1. Tài liệu cần thiết có nằm trong ngữ cảnh không?
   → cần một tập câu hỏi có nhãn tài liệu đúng; đo recall@k, MRR
   → nếu không: lỗi truy xuất; sửa cách chia đoạn, mô hình embedding, tìm kiếm kết hợp, xếp hạng lại

2. Nếu có, câu trả lời có được ngữ cảnh hỗ trợ không?
   → đo độ trung thành bằng giám khảo, yêu cầu trích dẫn cho từng khẳng định
   → nếu không: lỗi sinh; sửa prompt, đổi mô hình, yêu cầu trích dẫn
```

Bản ghi vết phải lưu danh sách các đoạn đã truy xuất và điểm truy xuất của chúng. Nếu không lưu từ đầu thì sau này không tách được hai loại lỗi, tương tự việc phải ghi lại xác suất hiển thị ở Mục 12.4.

**Hai vòng đánh giá.**

1. **Tập đánh giá chuẩn làm bước kiểm định trước khi phát hành:** khoảng 200 tới 500 câu hỏi, gồm cả câu khó, có nhãn tài liệu đúng và câu trả lời tham chiếu. Phải đạt trước mọi thay đổi của bất kỳ thành phần nào trong sáu thành phần phiên bản ở Mục 13.8: prompt, phiên bản mô hình, cấu hình truy xuất, ảnh chụp kho tài liệu, định nghĩa công cụ, tiêu chí chấm. Mục 12.4 của giáo trình *Ứng dụng LLM* cho thấy vài trăm câu đủ để phát hiện khác biệt lớn nhưng không đủ cho khác biệt vài điểm phần trăm, nên các thay đổi nhỏ cần so sánh ghép cặp.
2. **Chấm mẫu trực tuyến** như bảng trên, gắn điểm vào bản ghi vết để truy lại từng lượt.

Tập đánh giá chuẩn phải được cập nhật: mỗi sự cố trong sản xuất nên tạo ra ít nhất một câu hỏi mới trong tập (Mục 13.4).

**Phát hiện khi giám khảo thay đổi hành vi.** Bốn cơ chế:

1. **Đo mức đồng thuận giữa giám khảo và người chấm** trên mẫu hằng tuần, và theo dõi theo thời gian. Mức đồng thuận giảm là dấu hiệu giám khảo đã thay đổi.
2. **Giữ một tập neo cố định:** khoảng 50 cặp câu hỏi và câu trả lời đã được người chấm, điểm được giữ cố định. Cho giám khảo chấm lại hằng tuần. Nếu điểm trên tập neo thay đổi trong khi nội dung không đổi, thì thứ thay đổi là giám khảo, hoặc phiên bản mô hình của nhà cung cấp.
3. **Dùng định danh phiên bản cố định cho mô hình giám khảo,** và coi mỗi lần nâng cấp là một lần phát hành: chạy song song phiên bản cũ và mới trên tập neo trước khi chuyển.
4. **Kiểm tra thiên lệch định kỳ** (Mục 13.3): hỏi lại với thứ tự đảo để đo thiên lệch vị trí; xem tương quan giữa điểm và độ dài câu trả lời để đo thiên lệch độ dài.

Cơ chế 2 rẻ nhất và hữu ích nhất, vì nó phân biệt được hai trường hợp mà nhìn từ bên ngoài giống hệt nhau: hệ thống kém đi, và thước đo thay đổi.
