# MLOps: vận hành hệ thống học máy

> **Giáo trình 6 của lộ trình.** Tài liệu trình bày những gì xảy ra sau khi đã có một mô hình tốt: quản lý dữ liệu và đặc trưng, thí nghiệm lặp lại được, đánh giá cho sản xuất, phục vụ và ra mắt mô hình, phát hiện dịch chuyển phân phối, giám sát, huấn luyện lại, vòng phản hồi, và những thay đổi khi hệ thống dùng mô hình ngôn ngữ lớn (LLMOps).
>
> **Kiến thức cần có.** Quy trình huấn luyện và đánh giá một mô hình có giám sát ([*Nền tảng*](nentang-ch01.html)), kiểm định giả thuyết ở mức cơ bản, và Python ở mức đọc hiểu. Không cần biết trước công cụ MLOps nào: tên công cụ thay đổi nhanh, còn các vấn đề mà công cụ giải quyết thì không.
>
> **Về số liệu.** Tài liệu có hai loại con số. Loại thứ nhất trích từ tài liệu gốc, mỗi lần dùng đều ghi nguồn, danh sách nguồn ở Chương 16. Loại thứ hai là số liệu đo trong chính repo này, sinh bởi `code/mlops/experiments.py` (số liệu trong bài) và `code/mlops/bai_tap.py` (số liệu trong lời giải bài tập) với hạt giống cố định. Những chỗ là quy tắc kinh nghiệm của ngành được ghi rõ là quy tắc kinh nghiệm.

---

## Mục lục

0. Ký hiệu và quy ước
1. Nợ kỹ thuật của hệ thống học máy
2. Vòng đời và mức độ tự động hoá
3. Dữ liệu: hợp đồng, kiểm định và các loại lỗi
4. Đặc trưng và tính đúng theo thời điểm
5. Thí nghiệm lặp lại được
6. Đánh giá mô hình cho sản xuất
7. Đóng gói và phục vụ mô hình
8. Chiến lược ra mắt
9. Dịch chuyển phân phối
10. Giám sát và cảnh báo
11. Huấn luyện lại
12. Vòng phản hồi
13. LLMOps
14. Bài tập
15. Câu hỏi phỏng vấn
16. Tài liệu tham khảo

---

## 0. Ký hiệu và quy ước

### 0.1. Bảng ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $X$ | đặc trưng đầu vào |
| $Y$ | nhãn, biến mục tiêu |
| $P(X)$ | phân phối của đầu vào |
| $P(Y \mid X)$ | quan hệ giữa đầu vào và nhãn, điều mô hình học |
| $\hat y$ | dự đoán của mô hình |
| $t$, $t_{\text{pred}}$ | thời điểm; thời điểm phải đưa ra dự đoán |
| $n$ | cỡ mẫu |
| $k$ | số bin (Chương 9) hoặc số dịch vụ gọi song song (Chương 7), tuỳ ngữ cảnh |
| p50, p95, p99 | phân vị 50, 95, 99 của độ trễ |
| SLI, SLO | chỉ số mức dịch vụ, mục tiêu mức dịch vụ |
| CI, CD, CT | tích hợp liên tục, bàn giao liên tục, huấn luyện liên tục |
| PSI | population stability index, một đại lượng đo dịch chuyển phân phối |

### 0.2. Quy ước thuật ngữ

| Tiếng Anh | Dùng trong giáo trình |
|---|---|
| production | sản xuất (môi trường vận hành thật) |
| deployment, rollout | triển khai, ra mắt |
| serving | phục vụ (mô hình) |
| latency, throughput | độ trễ, thông lượng |
| distribution shift, drift | dịch chuyển phân phối |
| covariate shift, label shift, concept drift | giữ nguyên tiếng Anh |
| training–serving skew | lệch giữa huấn luyện và phục vụ |
| data leakage | rò rỉ dữ liệu |
| point-in-time correctness | tính đúng theo thời điểm |
| feature store | kho đặc trưng |
| model registry | sổ đăng ký mô hình |
| shadow, canary, A/B test, blue–green | giữ nguyên tiếng Anh |
| monitoring, alerting | giám sát, cảnh báo |
| feedback loop | vòng phản hồi |
| technical debt | nợ kỹ thuật |
| guardrail | rào chắn |

Bảng đối chiếu đầy đủ nằm ở trang Từ điển thuật ngữ.

---

## 1. Nợ kỹ thuật của hệ thống học máy

Một mô hình đạt kết quả tốt trên tập kiểm tra vẫn có thể suy giảm dần trong sản xuất mà không gây ra lỗi nào. Chương này giải thích vì sao, dựa trên khung nợ kỹ thuật của Sculley và cộng sự (2015): nguyên lý CACE, các dạng nợ có tên, và cách đọc đúng con số "5% mã học máy" hay được trích dẫn. Chương kết thúc bằng danh sách những nguyên nhân thường gặp làm mô hình suy giảm, và mỗi nguyên nhân được trình bày ở một chương sau.

### 1.1. Hệ thống học máy hỏng một cách im lặng

Một mô hình đạt AUC 0,94 trên tập kiểm tra. Sáu tháng sau nó vẫn chạy, không có ngoại lệ nào trong log, không ai báo lỗi, nhưng doanh thu từ tính năng dùng mô hình đã giảm 20%. Tình huống này phổ biến, và là một trong những lý do chính khiến MLOps trở thành một lĩnh vực riêng.

Phần mềm thông thường thường hỏng một cách dễ thấy: một lời gọi API sai kiểu dữ liệu ném ra ngoại lệ, và hệ thống giám sát báo động. Hệ thống học máy thường hỏng một cách im lặng: một mô hình nhận đầu vào có phân phối đã thay đổi vẫn trả về một con số trong khoảng $[0, 1]$, đúng kiểu, đúng định dạng, đúng thời hạn, nhưng sai. Không cơ chế kiểm tra nào của phần mềm thông thường phát hiện được loại lỗi này.

### 1.2. Nợ kỹ thuật

Khung khái niệm chuẩn cho vấn đề này là bài báo *Hidden Technical Debt in Machine Learning Systems* (Sculley và cộng sự, 2015). Ý chính: xây dựng và triển khai hệ thống học máy tương đối nhanh và rẻ, nhưng duy trì chúng theo thời gian thì khó và tốn kém.

Bài báo dùng ẩn dụ **nợ kỹ thuật** (technical debt) của Ward Cunningham: vay nợ không phải lúc nào cũng sai, nhưng mọi khoản nợ đều phải trả lãi. Điểm riêng của học máy là phần lớn khoản nợ nằm ở mức hệ thống chứ không ở mức mã nguồn, nên đọc mã không thấy được, và lãi tích luỹ mà không ai nhận ra.

![Hình 1](figs/mlops01_debt.png)

**Hình 1.** Vẽ lại ý của Hình 1 trong Sculley và cộng sự (2015): phần mã học máy ở giữa chỉ là một phần nhỏ; các thành phần xung quanh (thu thập dữ liệu, kiểm định, trích đặc trưng, quản lý tài nguyên, phục vụ, giám sát) mới là phần phải xây dựng và vận hành.

### 1.3. Nguyên lý CACE

> **Định nghĩa 1.1 (CACE).** **CACE** (Changing Anything Changes Everything): trong một hệ thống học máy, thay đổi bất kỳ thành phần nào cũng có thể thay đổi hành vi của toàn bộ hệ thống.

Lý do: một mô hình học một hàm từ toàn bộ các đầu vào cùng lúc. Thêm một đặc trưng, bỏ một đặc trưng hay chỉ đổi cách lấy mẫu dữ liệu đều làm trọng số của mọi đặc trưng còn lại thay đổi theo. Bài báo nói rõ nguyên lý này áp dụng cho cả siêu tham số, thiết lập huấn luyện, cách lấy mẫu, ngưỡng hội tụ, tức mọi thứ có thể điều chỉnh.

Hệ quả thực tế:

- **Không có thay đổi nào là nhỏ và cô lập trong một mô hình.** Mọi thay đổi đều cần đánh giá lại toàn bộ; không thể lập luận cục bộ như khi sửa một hàm trong phần mềm thông thường. Mục 6.3 cho một ví dụ đo được: thêm một đặc trưng làm độ chính xác tổng thể tăng gần 20 điểm phần trăm nhưng làm một nhóm người dùng giảm gần 12 điểm.
- Bài báo đề xuất một cách giảm nhẹ: tách bài toán thành các bài toán con độc lập và dùng các mô hình riêng. Cách này chỉ hợp khi các bài toán con thật sự tách rời nhau.

### 1.4. Các dạng nợ kỹ thuật

Bài báo đặt tên cho nhiều dạng nợ. Dưới đây chúng được nhóm theo nơi phát sinh; riêng vòng phản hồi (feedback loop), một dạng nợ khác trong bài báo, được trình bày ở Chương 12.

**a) Mô hình làm mờ ranh giới trừu tượng**

| Dạng nợ | Nội dung |
|---|---|
| Entanglement (rối) | chính là CACE ở Mục 1.3 |
| Correction cascade (thác hiệu chỉnh) | có mô hình $m_a$ cho bài toán A; để giải bài toán A' gần giống, học thêm một mô hình nhận đầu ra của $m_a$ làm đầu vào. Nhanh, nhưng tạo ra phụ thuộc mới vào $m_a$, và từ đó cải thiện $m_a$ trở nên rất tốn kém vì có thể làm hỏng mô hình phía sau |
| Undeclared consumer (bên sử dụng không khai báo) | đầu ra của mô hình được ghi ra ở đâu đó không có kiểm soát truy cập, rồi một hệ thống khác dùng nó làm đầu vào mà không ai biết. Thay đổi mô hình khi đó làm hỏng hệ thống kia, và có thể tạo ra vòng phản hồi ẩn (Chương 12) |

**b) Phụ thuộc dữ liệu tốn kém hơn phụ thuộc mã**

Phụ thuộc giữa các đoạn mã có thể phát hiện bằng trình biên dịch và công cụ phân tích tĩnh; phụ thuộc vào dữ liệu thì không có công cụ tương đương, nên rất dễ tích tụ những chuỗi phụ thuộc không ai kiểm soát được.

- **Phụ thuộc dữ liệu không ổn định** (unstable data dependency): đặc trưng lấy từ một hệ thống khác, và hệ thống đó thay đổi hành vi theo thời gian, ví dụ vì bản thân nó cũng là một mô hình được huấn luyện lại. Cách xử lý là cố định phiên bản của tín hiệu đầu vào, với cái giá là phải duy trì nhiều phiên bản song song.
- **Phụ thuộc dữ liệu ít giá trị** (underutilized data dependency): đặc trưng đóng góp rất ít nhưng vẫn nằm trong mô hình, làm hệ thống dễ hỏng một cách không cần thiết. Bài báo nêu bốn nguồn: đặc trưng cũ đã bị các đặc trưng mới thay thế nhưng không ai gỡ; một nhóm đặc trưng được thêm cả gói vì áp lực thời hạn; đặc trưng cải thiện độ chính xác rất ít nhưng làm hệ thống phức tạp hơn nhiều; và đặc trưng tương quan mạnh với một đặc trưng khác có quan hệ nhân quả thật. Cách phát hiện được đề xuất là định kỳ đánh giá lại mô hình khi bỏ từng đặc trưng (leave-one-feature-out).

**c) Nợ trong kiến trúc, cấu hình và quy trình**

| Dạng nợ | Nội dung |
|---|---|
| Glue code (mã keo) | mã nối các thư viện đa dụng lại với nhau; đây là nơi xuất hiện con số 5% ở Mục 1.5 |
| Pipeline jungle | trường hợp riêng của mã keo ở khâu chuẩn bị dữ liệu: các bước lấy, ghép, lấy mẫu dữ liệu chồng chất qua thời gian, với nhiều tệp trung gian |
| Dead experimental codepaths | các nhánh điều kiện để thử nghiệm còn nằm lại trong mã sản xuất |
| Abstraction debt | thiếu những trừu tượng chuẩn cho dữ liệu, mô hình và dự đoán, tương tự vai trò của cơ sở dữ liệu quan hệ trong phần mềm thông thường |
| Configuration debt | hệ thống lớn có rất nhiều tuỳ chọn cấu hình, thường ít được kiểm tra kỹ như mã, dù một dòng cấu hình sai có thể gây hại như một dòng mã sai |
| Reproducibility debt | khó chạy lại một thí nghiệm và nhận được cùng kết quả (Chương 5) |
| Cultural debt | ranh giới cứng giữa nhóm nghiên cứu và nhóm kỹ thuật; cần ghi nhận việc gỡ bớt đặc trưng, giảm độ phức tạp, cải thiện giám sát ngang với việc tăng độ chính xác |

Bài báo còn nêu ba dấu hiệu đáng ngờ (smell) trong mã: dùng kiểu dữ liệu nguyên thuỷ (float, int) cho những đại lượng giàu ngữ nghĩa, nên một tham số không cho biết nó là log-odds hay là ngưỡng quyết định; dùng nhiều ngôn ngữ lập trình trong một hệ thống, làm việc kiểm thử và bàn giao khó hơn; và thường xuyên phải thử ý tưởng trên một bản nguyên mẫu (prototype) thay vì trên hệ thống thật, dấu hiệu cho thấy hệ thống thật khó thay đổi hoặc thiếu những giao diện tốt.

### 1.5. Con số 5% mã học máy

Con số "5% mã học máy, 95% mã keo" thường bị trích dẫn sai. Ba điểm cần nói đúng:

1. Đây là nhận định trong mục *Glue Code* của bài báo, về hệ thống tái sử dụng các thư viện đa dụng, không phải chú thích của Hình 1.
2. Bài báo diễn đạt có giới hạn: một hệ thống trưởng thành có thể chỉ gồm **nhiều nhất** 5% mã học máy và **ít nhất** 95% mã keo. Đây là một nhận định định tính kèm độ lớn, không phải kết quả đo trên một mẫu hệ thống.
3. Kết luận bài báo rút ra thường bị bỏ qua: vì tỉ lệ này, đôi khi tự viết một giải pháp gọn còn rẻ hơn tái sử dụng một thư viện đa dụng, và cách chống mã keo là bọc các thư viện sau một giao diện chung.

### 1.6. Những nguyên nhân thường gặp làm mô hình suy giảm

Trở lại mô hình AUC 0,94 ở Mục 1.1. Nó có thể suy giảm vì một trong các nguyên nhân sau, mỗi nguyên nhân đã có tên ở trên và được trình bày ở một chương sau:

- một nguồn dữ liệu phía trước đổi đơn vị tính từ đồng sang nghìn đồng mà không báo (phụ thuộc dữ liệu không ổn định; Chương 3 và Mục 10.4);
- một đặc trưng được tính khác nhau lúc huấn luyện và lúc phục vụ (lệch giữa huấn luyện và phục vụ; Chương 4);
- phân phối dữ liệu đã thay đổi mà không ai đo (dịch chuyển phân phối; Chương 9);
- một ngưỡng quyết định đặt tay từ năm trước vẫn được dùng sau khi mô hình đã huấn luyện lại (ngưỡng cố định trong hệ thống thay đổi; Mục 10.6);
- chính mô hình làm sai lệch dữ liệu huấn luyện của những lần sau (vòng phản hồi; Chương 12).

---

## 2. Vòng đời và mức độ tự động hoá

Chương này mô tả vòng đời của một hệ thống học máy như một vòng lặp, trình bày ba yếu tố quyết định thành công theo nghiên cứu phỏng vấn của Shankar và cộng sự (2022), ba mức tự động hoá theo kiến trúc tham chiếu của Google Cloud, và phân biệt huấn luyện liên tục với CI/CD.

### 2.1. Vòng đời là một vòng lặp

Cách mô tả quen thuộc "thu thập dữ liệu, huấn luyện, triển khai" là một đường thẳng và vì thế thiếu phần quan trọng nhất. Vòng đời thật có nhiều đường quay lại: kết quả đánh giá dẫn tới thu thập thêm dữ liệu, giám sát trong sản xuất dẫn tới huấn luyện lại, sự cố dẫn tới sửa đặc trưng.

![Hình 2](figs/mlops02_lifecycle.png)

**Hình 2.** Vòng đời của một hệ thống học máy với các đường quay lại. Hệ thống tốt là hệ thống đi vòng này nhanh và ít tốn kém.

Shankar và cộng sự (2022) phỏng vấn 18 kỹ sư học máy đang vận hành hệ thống thật và gom công việc của họ thành bốn nhiệm vụ:

1. **Thu thập và gán nhãn dữ liệu**: tìm nguồn, đưa về kho tập trung, làm sạch, gán nhãn.
2. **Tạo đặc trưng và thí nghiệm mô hình**: cải thiện chỉ số, thay đổi dữ liệu hoặc thay đổi mô hình.
3. **Đánh giá và triển khai mô hình.**
4. **Giám sát và xử lý khi chất lượng suy giảm trong sản xuất.**

### 2.2. Ba yếu tố: tốc độ, kiểm định, phiên bản

Cùng nghiên cứu đó rút ra ba yếu tố quyết định thành công của việc đưa mô hình vào sản xuất, gọi tắt là ba chữ V:

| Yếu tố | Nội dung | Lý do quan trọng |
|---|---|---|
| Velocity (tốc độ) | đi nhanh từ một ý tưởng tới một mô hình đã huấn luyện, lý tưởng trong vòng một ngày | học máy mang tính thực nghiệm, nên số ý tưởng thử được trong một khoảng thời gian quyết định năng suất |
| Validation (kiểm định) | kiểm tra, loại bỏ ý tưởng kém và phát hiện lỗi càng sớm càng tốt | lỗi được phát hiện càng muộn thì càng tốn kém |
| Versioning (phiên bản) | lưu và quản lý nhiều phiên bản mô hình và dữ liệu | không lường trước được mọi lỗi, nên phải quay lại được phiên bản trước |

Tốc độ và kiểm định có vẻ mâu thuẫn: thêm bước kiểm định thì chậm hơn. Nhưng chúng cũng hỗ trợ nhau: loại được ý tưởng kém ở giai đoạn sớm thì tổng thời gian giảm. Vấn đề là đặt bước kiểm định ở đâu, không phải nhiều hay ít.

### 2.3. Ba mức tự động hoá

Kiến trúc tham chiếu được trích dẫn nhiều nhất là tài liệu *MLOps: Continuous delivery and automation pipelines in machine learning* của Google Cloud, chia mức trưởng thành thành ba mức.

![Hình 3](figs/mlops03_levels.png)

**Hình 3.** Ba mức tự động hoá. Thứ được bàn giao lớn dần theo mức: một mô hình, một pipeline, rồi một hệ thống tự cập nhật.

**Mức 0, quy trình thủ công.** Mọi bước, từ phân tích và chuẩn bị dữ liệu tới huấn luyện và kiểm định, đều làm tay. Nhóm khoa học dữ liệu bàn giao mô hình đã huấn luyện cho nhóm kỹ thuật triển khai; hai bên làm việc tách rời. Mô hình ít khi được cập nhật và không được giám sát hiệu năng.

**Mức 1, tự động hoá pipeline huấn luyện.** Toàn bộ pipeline huấn luyện được tự động hoá, nên mô hình có thể được **huấn luyện liên tục** (continuous training, CT) trên dữ liệu mới. Pipeline có bước kiểm định dữ liệu và kiểm định mô hình tự động. Thứ được triển khai không còn là một mô hình mà là cả pipeline huấn luyện.

**Mức 2, tự động hoá CI/CD cho pipeline.** Có hệ thống CI/CD cho chính pipeline, để thử nghiệm ý tưởng mới và đưa vào sản xuất nhanh chóng. Tài liệu của Google chia thành sáu giai đoạn: phát triển và thí nghiệm, tích hợp liên tục pipeline, bàn giao liên tục pipeline, kích hoạt tự động, bàn giao liên tục mô hình, và giám sát.

Tài liệu cũng liệt kê tám bước của một pipeline học máy: trích xuất dữ liệu, phân tích dữ liệu, chuẩn bị dữ liệu, huấn luyện, đánh giá, **kiểm định mô hình**, phục vụ, giám sát. Bước kiểm định mô hình được tách riêng khỏi bước đánh giá, vì một lý do Mục 11.4 trình bày.

### 2.4. Huấn luyện liên tục và CI/CD

CI/CD xử lý thay đổi của **mã**; CT xử lý thay đổi của **dữ liệu**. Trong phần mềm thông thường, hệ thống chỉ thay đổi khi có người sửa mã. Trong học máy có thêm một nguồn thay đổi không đi qua kho mã: thế giới thay đổi, dữ liệu thay đổi theo, và mô hình cũ dần. Vì vậy học máy cần thêm huấn luyện liên tục: tự động huấn luyện lại và triển khai lại, kích hoạt bởi dữ liệu mới, bởi chất lượng suy giảm hoặc bởi dịch chuyển phân phối (Chương 11).

Trong CI/CD thông thường, tạo tác (artifact) là mã đã biên dịch. Trong học máy, tạo tác gồm mã, dữ liệu và mô hình, ba thứ thay đổi theo nhịp khác nhau. Quản lý đồng thời ba nhịp đó là lý do MLOps là một lĩnh vực riêng.

### 2.5. Các thói quen xấu thường gặp

Shankar và cộng sự quan sát được bốn thói quen xấu (anti-pattern):

| Thói quen | Mô tả | Tác hại |
|---|---|---|
| Industry–classroom mismatch | kiến thức học ở trường không khớp với những vấn đề gặp khi vận hành; người mới phải học lại qua sự cố | có thể giảm bằng tài liệu và quy trình |
| Keeping GPUs warm | chạy càng nhiều thí nghiệm càng tốt để không lãng phí tài nguyên | thí nghiệm không có giả thuyết tạo ra nhiều kết quả khó diễn giải và nhiều phiên bản phải quản lý |
| Retrofitting an explanation | thấy kết quả tốt, đưa vào sản xuất, rồi mới tìm lý do vì sao thay đổi có tác dụng | lời giải thích gắn vào sau có thể sai, và dễ làm các quyết định tiếp theo đi sai hướng |
| Undocumented tribal knowledge | hiểu biết về pipeline chỉ nằm trong đầu vài người | khi người đó rời đi, hiểu biết mất theo |

Thói quen thứ hai nghe như một ưu điểm, tận dụng tài nguyên, nhưng nhầm lẫn về thứ cần tối ưu: tài nguyên khan hiếm thường không phải GPU mà là số giả thuyết tốt và thời gian của con người để đọc và diễn giải kết quả.

---

## 3. Dữ liệu: hợp đồng, kiểm định và các loại lỗi

Lỗi dữ liệu là nguồn sự cố phổ biến và khó phát hiện nhất trong hệ thống học máy. Chương này phân loại lỗi dữ liệu thành ba loại với ba cách xử lý khác nhau, trình bày hợp đồng dữ liệu như một cách biến các giả định ngầm thành các kiểm tra tự động, rồi bàn về nhãn và phiên bản hoá dữ liệu.

### 3.1. Vai trò của dữ liệu

Một nhận xét lặp lại trong các nghiên cứu về thực hành: kỹ sư học máy giỏi thường cải thiện hệ thống bằng cách thay đổi dữ liệu nhiều hơn thay đổi mô hình. Theo một quy tắc kinh nghiệm được nhắc ở [Mục 1.3 của *Học sâu*](models-ch01.html), với cùng công sức, sửa dữ liệu và nhãn thường cho cải thiện lớn hơn đổi kiến trúc. Quan trọng hơn, lỗi dữ liệu thường không gây lỗi chương trình nào, nên chỉ được phát hiện nếu được kiểm tra có chủ đích.

### 3.2. Ba loại lỗi dữ liệu

Shankar và cộng sự (2022) phân biệt ba loại lỗi, mỗi loại cần một cách phản ứng khác nhau:

![Hình 4](figs/mlops04_errors.png)

**Hình 4.** Ba loại lỗi dữ liệu, từ dễ phát hiện và gây hại ngay tới khó phát hiện và gây hại chậm.

| Loại | Ví dụ | Đặc điểm | Cách xử lý |
|---|---|---|---|
| Lỗi cứng | hai cột bị đảo, tuổi âm, sai kiểu dữ liệu, tệp rỗng | vi phạm một ràng buộc kiểm tra được; dự đoán sai rõ ràng | dừng pipeline: không có dự đoán tốt hơn có dự đoán sai |
| Lỗi mềm | một số trường bị thiếu, đơn vị tính thay đổi, xuất hiện giá trị danh mục lạ | dự đoán vẫn trông hợp lý nên vượt qua các kiểm tra đơn giản; khó định lượng | cảnh báo và theo dõi tỉ lệ, không dừng pipeline |
| Dịch chuyển | phân phối thay đổi dần theo tuần, theo mùa | không có gì sai về kỹ thuật; thế giới đã khác | điều tra, có thể huấn luyện lại (Chương 9, 11) |

Nghiên cứu ghi nhận hai khó khăn quanh việc kiểm tra dữ liệu. Thứ nhất, để người tự đặt các ràng buộc chất lượng dữ liệu bằng tay không bền vững: người biết vì sao một ngưỡng là 0,3 có thể rời đi, còn ngưỡng thì ở lại. Thứ hai, khó khăn được nhắc nhiều nhất là **báo động giả**: cảnh báo kích hoạt trong khi chất lượng của mô hình vẫn ổn. Hậu quả không chỉ là phiền toái mà là nhóm vận hành mất niềm tin vào cảnh báo, và bỏ qua cả khi có sự cố thật (Mục 10.5). Dùng cùng một cơ chế cảnh báo cho cả ba loại lỗi là một nguyên nhân chính của báo động giả.

### 3.3. Hợp đồng dữ liệu

Cách xử lý gốc cho lỗi cứng là biến các giả định ngầm về dữ liệu thành một **hợp đồng dữ liệu** (data contract) kiểm tra được bằng máy. Một hợp đồng tối thiểu gồm bốn tầng:

```python
# Tầng 1: lược đồ (schema) - tên cột, kiểu, cột nào được phép thiếu
schema = {
    "user_id":    {"type": "int64",    "nullable": False},
    "amount_vnd": {"type": "float64",  "nullable": False},
    "country":    {"type": "category", "nullable": False},
    "event_time": {"type": "datetime", "nullable": False},
}

# Tầng 2: miền giá trị - ràng buộc trên từng giá trị
domain = {
    "amount_vnd": {"min": 0, "max": 5e9},
    "country":    {"allowed": {"VN", "SG", "TH", "ID"}},
}

# Tầng 3: thống kê - ràng buộc trên phân phối của cả lô
batch_stats = {
    "amount_vnd": {"null_rate_max": 0.001, "mean_shift_max_sigma": 3},
    "country":    {"new_category_rate_max": 0.005},
}

# Tầng 4: quan hệ giữa các cột và giữa các bảng
invariants = [
    "event_time <= ingest_time",       # không có sự kiện từ tương lai
    "refund_amount <= amount_vnd",
    "count(rows) between 0.7*d7_median and 1.4*d7_median",   # số dòng không hụt, không phình bất thường
]
```

Ba nhận xét về thiết kế này:

- **Tầng 1 và 2 phát hiện lỗi cứng, tầng 3 và 4 phát hiện lỗi mềm.** Hai nhóm cần hai cách phản ứng khác nhau (Mục 3.2): vi phạm tầng 1, 2 thì dừng pipeline; vi phạm tầng 3, 4 thì cảnh báo.
- **Hợp đồng nên được suy ra từ dữ liệu rồi người duyệt lại,** thay vì chỉ viết tay: suy ra lược đồ và khoảng giá trị từ một tập dữ liệu tham chiếu đã được kiểm tra, rồi để người xem lại. Hợp đồng khi đó không phụ thuộc vào trí nhớ của một người cụ thể. Các thư viện như TensorFlow Data Validation (Breck và cộng sự, 2019) và Great Expectations cung cấp chức năng này.
- **Ràng buộc "không có sự kiện từ tương lai"** ở tầng 4 là tuyến phòng thủ đầu tiên chống rò rỉ dữ liệu theo thời gian (Chương 4).

### 3.4. Kiểm định ở đầu vào và đầu ra của pipeline

Một thực hành được nhắc lại nhiều lần trong nghiên cứu phỏng vấn: kiểm định dữ liệu cả khi đi vào và khi đi ra khỏi mỗi pipeline. Lý do: trong một pipeline nhiều bước, bước thứ ba hỏng thì bước thứ năm vẫn chạy, và chỉ tới đầu ra cuối cùng mới thấy con số bất thường; khi đó phải mất nhiều thời gian lần ngược để tìm bước hỏng. Nguyên tắc chung, mượn từ hệ thống phân tán, là phát hiện lỗi sớm, báo lỗi rõ ràng, và kiểm tra tại ranh giới giữa các thành phần, nơi trách nhiệm chuyển từ nhóm này sang nhóm khác.

### 3.5. Nhãn

Ba loại vấn đề về nhãn, xếp theo mức độ hay bị bỏ qua:

1. **Độ trễ nhãn** (label lag): khoảng thời gian từ lúc dự đoán tới lúc biết đáp án thật. Với dự đoán lượt nhấp là vài phút; với dự đoán vỡ nợ tín dụng là hàng tháng tới hàng năm. Độ trễ nhãn quyết định những gì giám sát được (Chương 10) và huấn luyện lại được nhanh tới đâu (Chương 11), và Mục 9.2 cho thấy nó quyết định cả khả năng phát hiện loại dịch chuyển nguy hiểm nhất.
2. **Nhãn nhiễu và bất đồng giữa người gán nhãn.** Mức đồng thuận giữa những người gán nhãn là giới hạn trên thực tế của độ chính xác đo được. Một mô hình đạt 95% trong khi hai người gán nhãn chỉ đồng ý với nhau 90% là một kết quả đáng nghi, có thể do rò rỉ hoặc do cách đo.
3. **Nhãn bị chính hệ thống làm sai lệch.** Nếu nhãn lấy từ hành vi của người dùng trên những kết quả do mô hình chọn hiển thị, nhãn đó không phải quan sát khách quan về thế giới (Chương 12).

### 3.6. Phiên bản hoá dữ liệu

Yếu tố phiên bản ở Mục 2.2 áp dụng cho cả dữ liệu, không chỉ mô hình. Câu hỏi cần trả lời được bất cứ lúc nào: mô hình đang chạy trong sản xuất được huấn luyện từ chính xác tập dữ liệu nào? Có ba mức trả lời, chi phí tăng dần:

| Mức | Cách làm | Trả lời được |
|---|---|---|
| Yếu | ghi đường dẫn và thời điểm | "dữ liệu ở thư mục đó, ngày đó"; không dựng lại được nếu thư mục bị ghi đè |
| Trung bình | lưu mã băm (hash) nội dung của tập dữ liệu cùng với mô hình | biết chắc có phải đúng tập đó không; chưa dựng lại được |
| Mạnh | lưu trữ bất biến, chỉ thêm (append-only), kèm truy vết nguồn gốc (lineage) | dựng lại được đúng tập huấn luyện của bất kỳ mô hình nào trong quá khứ |

Mức mạnh tốn kém, nhưng là điều kiện để điều tra sự cố. Khi mô hình hoạt động bất thường, câu hỏi đầu tiên thường là dữ liệu huấn luyện có gì bất thường không; nếu không dựng lại được tập dữ liệu đó, việc điều tra dừng ngay ở câu hỏi đầu tiên.

---

## 4. Đặc trưng và tính đúng theo thời điểm

Kho đặc trưng thường được giới thiệu như một loại cơ sở dữ liệu. Chương này giải thích hai vấn đề mà nó được tạo ra để giải quyết: lệch giữa huấn luyện và phục vụ, và rò rỉ thông tin tương lai khi dựng tập huấn luyện. Khái niệm trung tâm của chương là tính đúng theo thời điểm, kèm ví dụ SQL cho cách ghép dữ liệu sai và đúng. Phần cuối nêu khi nào không cần kho đặc trưng.

### 4.1. Hai vấn đề: lệch huấn luyện–phục vụ và rò rỉ

**Lệch giữa huấn luyện và phục vụ** (training–serving skew). Khi huấn luyện, đặc trưng "tổng chi tiêu 30 ngày qua" được tính bằng một truy vấn SQL trên kho dữ liệu. Khi phục vụ, truy vấn đó quá chậm, nên đặc trưng được tính lại bằng Python trong dịch vụ phục vụ. Hai đoạn mã do hai người viết ở hai thời điểm, và hiếm khi khớp nhau hoàn toàn: một bên tính 30 ngày theo lịch, bên kia theo 720 giờ; một bên loại giao dịch hoàn tiền, bên kia không. Tài liệu của Google Cloud định nghĩa lệch huấn luyện–phục vụ là khi đặc trưng dùng để phục vụ khác với đặc trưng dùng lúc huấn luyện, làm chất lượng mô hình giảm sau khi triển khai.

Loại lỗi này không gây ra lỗi chương trình nào. Mô hình vẫn chạy, chỉ là nhận đầu vào hơi khác những gì nó đã học, và mất vài điểm phần trăm mà không ai biết vì sao.

**Rò rỉ thông tin tương lai khi dựng tập huấn luyện.** Đây là nội dung của Mục 4.2 và 4.3.

### 4.2. Tính đúng theo thời điểm

> **Định nghĩa 4.1 (Tính đúng theo thời điểm).** Một tập huấn luyện có **tính đúng theo thời điểm** (point-in-time correctness) nếu với mỗi dòng có thời điểm dự đoán $t_{\text{pred}}$, giá trị của mọi đặc trưng là giá trị quan sát được tại $t_{\text{pred}}$, không phải giá trị tại thời điểm dựng tập dữ liệu.

Định nghĩa nghe hiển nhiên nhưng rất dễ vi phạm.

![Hình 5](figs/mlops05_pit.png)

**Hình 5.** Đặc trưng "tổng số đơn hàng của khách" thay đổi theo thời gian: 12 vào ngày 1/3, 31 vào ngày 15/3, 480 vào hôm nay. Một dòng huấn luyện cần dự đoán cho ngày 1/4 phải dùng giá trị 31; dùng 480 là đưa thông tin của tương lai vào quá khứ.

Trong SQL, cách ghép sai trông hoàn toàn bình thường:

```sql
-- SAI: lấy giá trị đặc trưng HIỆN TẠI cho mọi dòng lịch sử
SELECT  l.user_id, l.event_time, l.label,
        f.total_orders                  -- giá trị của hôm nay
FROM    labels l
JOIN    user_features f ON f.user_id = l.user_id;
```

Cách ghép đúng phải ghép thêm theo trục thời gian: mỗi dòng lấy bản ghi đặc trưng mới nhất nhưng không muộn hơn thời điểm dự đoán.

```sql
-- ĐÚNG: ghép theo thời điểm (point-in-time join, as-of join)
SELECT  l.user_id, l.event_time, l.label, f.total_orders
FROM    labels l
LEFT JOIN LATERAL (
    SELECT total_orders
    FROM   user_features_history h
    WHERE  h.user_id    = l.user_id
      AND  h.valid_from <= l.event_time      -- chỉ nhìn về quá khứ
    ORDER BY h.valid_from DESC
    LIMIT 1
) f ON TRUE;
```

Vi phạm tính đúng theo thời điểm nguy hiểm hơn phần lớn các lỗi khác vì nó không làm kết quả đánh giá xấu đi mà làm chúng **tốt lên**. Mô hình học được rằng "khách có 480 đơn hàng thì không rời bỏ dịch vụ", trong khi con số 480 chỉ tồn tại vì khách đã ở lại. Kết quả ngoại tuyến tốt nên không ai nghi ngờ, cho tới khi mô hình được triển khai và hoạt động kém. Nghiên cứu của Shankar và cộng sự ghi nhận đúng điều này: rò rỉ dữ liệu, tức giả định lúc huấn luyện rằng có những dữ liệu mà lúc phục vụ không có, thường chỉ bị phát hiện sau khi mô hình đã được triển khai. [Mục 8.5 của *Nền tảng*](nentang-ch08.html) trình bày các dạng rò rỉ khi chia dữ liệu.

### 4.3. Ba đường rò rỉ thông tin tương lai

Xếp theo mức độ khó phát hiện:

1. **Rò rỉ theo thời gian**: như Mục 4.2. Phòng bằng ghép theo thời điểm và ràng buộc "không có sự kiện từ tương lai" trong hợp đồng dữ liệu (Mục 3.3).
2. **Rò rỉ theo thực thể**: cùng một người dùng có dữ liệu ở cả tập huấn luyện lẫn tập kiểm tra khi chia ngẫu nhiên, nên mô hình được kiểm tra trên chính những người nó đã thấy. Phòng bằng chia theo nhóm thực thể, và với dữ liệu có yếu tố thời gian thì chia theo thời gian.
3. **Rò rỉ qua đặc trưng là hệ quả của nhãn** (proxy leakage): ví dụ đặc trưng "số lần gọi tổng đài sau sự cố" để dự đoán "khách có gặp sự cố không". Về thời gian có thể hợp lệ, nhưng đặc trưng này là hậu quả của nhãn chứ không phải nguyên nhân. Không có kiểm tra tự động nào phát hiện được loại này; chỉ có cách hỏi với từng đặc trưng: tại thời điểm cần dự đoán, giá trị này đã tồn tại chưa, và nó có tồn tại vì nhãn hay không?

### 4.4. Kho đặc trưng

Kho đặc trưng (feature store) là một lớp hạ tầng gồm ba thành phần:

| Thành phần | Nội dung | Yêu cầu |
|---|---|---|
| Kho ngoại tuyến (offline store) | toàn bộ lịch sử giá trị đặc trưng, kèm thời điểm có hiệu lực | thông lượng cao, hỗ trợ ghép theo thời điểm trên dữ liệu lớn |
| Kho trực tuyến (online store) | giá trị mới nhất của mỗi thực thể | độ trễ thấp, thường dưới 10 ms, tra cứu theo khoá |
| Sổ đăng ký (registry) | một định nghĩa duy nhất cho mỗi đặc trưng | nguồn chuẩn cho cả hai kho |

Sổ đăng ký là thành phần quan trọng nhất: nó giải quyết lệch huấn luyện–phục vụ bằng cách buộc hai đường huấn luyện và phục vụ dùng chung một định nghĩa thay vì hai bản cài đặt. Kho ngoại tuyến giải quyết rò rỉ bằng cách giữ lịch sử để ghép được theo thời điểm.

> **Lưu ý.** Kho đặc trưng không tự động loại bỏ lệch huấn luyện–phục vụ, mà chỉ tạo điều kiện. Nếu nhóm vẫn viết một đường tính riêng cho phục vụ vì lý do độ trễ, sự lệch quay lại. Các công cụ khác nhau ở điểm này: có công cụ chỉ cung cấp hạ tầng và để nhóm tự giữ kỷ luật, có công cụ buộc dùng một định nghĩa duy nhất.

### 4.5. Khi nào không cần kho đặc trưng

Kho đặc trưng là hạ tầng phức tạp. Nó đáng đầu tư khi có đồng thời:

- nhiều mô hình dùng chung đặc trưng (với một mô hình, lợi ích tái sử dụng gần như bằng 0);
- phục vụ trực tuyến với độ trễ thấp (nếu chỉ chấm điểm theo lô mỗi đêm, dùng trực tiếp kho dữ liệu là đủ);
- đặc trưng phụ thuộc thời gian và cập nhật thường xuyên (nếu đặc trưng tĩnh, vấn đề ghép theo thời điểm không tồn tại).

Thiếu các điều kiện trên, một bảng trong kho dữ liệu cộng với kỷ luật dùng chung một hàm biến đổi đặc trưng cho cả huấn luyện lẫn phục vụ rẻ hơn nhiều và đạt cùng mục tiêu.

---

## 5. Thí nghiệm lặp lại được

Khi một mô hình hoạt động bất thường, việc đầu tiên là dựng lại được đúng mô hình đó: cùng mã, cùng dữ liệu, cùng cấu hình. Chương này phân biệt ba mức "lặp lại được", liệt kê những thứ phải cố định để huấn luyện cho cùng kết quả, rồi trình bày hai công cụ lưu vết: theo dõi thí nghiệm và sổ đăng ký mô hình.

### 5.1. Ba mức: tái lập, tái tạo, tính bền

| Mức | Định nghĩa | Khi nào cần |
|---|---|---|
| Tái lập (reproducibility) | chạy lại với cùng mã, cùng dữ liệu, cùng môi trường thì được cùng kết quả | bắt buộc, để điều tra sự cố |
| Tái tạo (replicability) | người khác, trong môi trường khác, làm lại được kết quả tương tự | cần cho hợp tác và kiểm toán |
| Tính bền (robustness) | đổi hạt giống ngẫu nhiên, đổi cách chia dữ liệu mà kết luận không đổi | cần trước khi tin vào một cải thiện |

Các tài liệu dùng hai từ reproducibility và replicability không thống nhất, có nơi dùng theo nghĩa ngược lại; giáo trình dùng theo nghĩa ở bảng trên.

Sculley và cộng sự (2015) gọi sự thiếu hụt ở mức thứ nhất là **reproducibility debt** và nêu bốn nguyên nhân: thuật toán có yếu tố ngẫu nhiên, tính bất định vốn có của huấn luyện song song, sự phụ thuộc vào điều kiện khởi tạo, và tương tác với thế giới bên ngoài.

Mức thứ ba thường bị bỏ qua nhất. Một cải thiện 0,3 điểm phần trăm có thể nhỏ hơn độ dao động giữa hai lần chạy chỉ khác hạt giống. Trước khi kết luận mô hình mới tốt hơn, cần chạy vài hạt giống cho cả hai mô hình và so sánh chênh lệch với độ dao động đó ([Mục 12.4 của *Ứng dụng LLM*](ungdung-ch12.html) trình bày cách so sánh ghép cặp hai hệ thống trên cùng tập kiểm tra bằng kiểm định McNemar).

### 5.2. Những thứ phải cố định

Để đạt mức tái lập, phải cố định đủ sáu nhóm yếu tố. Thiếu một nhóm là đủ để hai lần chạy cho kết quả khác nhau:

```python
import os, random
import numpy as np
import torch

S = 1207

# 1. Hạt giống cho mọi nguồn ngẫu nhiên, không chỉ một
random.seed(S); np.random.seed(S); torch.manual_seed(S)
torch.cuda.manual_seed_all(S)

# 2. Phép toán tất định (chậm hơn)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # cần cho một số phép cuBLAS
torch.use_deterministic_algorithms(True)            # báo lỗi nếu phép toán không có bản tất định
torch.backends.cudnn.benchmark = False              # không chọn thuật toán theo đo thời gian

# 3. Thứ tự dữ liệu và các tiến trình con của DataLoader
def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

g = torch.Generator(); g.manual_seed(S)
loader = torch.utils.data.DataLoader(ds, shuffle=True, num_workers=4,
                                     worker_init_fn=seed_worker, generator=g)

# 4. Phiên bản: mã (commit), dữ liệu (hash), thư viện (lockfile), ảnh container (digest)
# 5. Cấu hình: một tệp duy nhất, lưu cùng kết quả
# 6. Phần cứng: loại GPU và số GPU
```

Ba điểm hay bị bỏ sót:

- **Các tiến trình con của DataLoader.** Với `num_workers > 0`, mỗi worker là một tiến trình riêng. PyTorch tự đặt hạt giống cho bộ sinh số của torch trong mỗi worker, nhưng các thư viện khác mà hàm tăng cường dữ liệu dùng tới thì không chắc. Tài liệu PyTorch khuyến nghị đặt hạt giống cho chúng trong `worker_init_fn`, như hàm `seed_worker` ở trên.
- **Số GPU là một siêu tham số.** Đổi từ 4 sang 8 GPU mà giữ nguyên kích thước lô trên mỗi GPU là nhân đôi kích thước lô hiệu dụng, tức đổi bài toán tối ưu. Ngay cả khi giữ kích thước lô hiệu dụng, thứ tự cộng gradient giữa các GPU thay đổi, và vì phép cộng số thực dấu phẩy động không có tính kết hợp, kết quả khác đi ở những chữ số cuối.
- **`cudnn.benchmark = True` không tất định**: nó chọn thuật toán tích chập bằng cách đo thời gian lúc chạy, và kết quả đo phụ thuộc tải của máy.

Tính tất định hoàn toàn trên GPU có giá: nhiều phép toán tất định chậm hơn bản thường. Cách làm phổ biến là bật chế độ tất định khi điều tra sự cố hoặc khi chạy kiểm thử hồi quy, còn khi huấn luyện thường thì chấp nhận dao động nhỏ và đo tính bền bằng nhiều hạt giống (Mục 5.1).

### 5.3. Theo dõi thí nghiệm

Nguyên tắc: ghi đủ để dựng lại một thí nghiệm, không chỉ đủ để so sánh các thí nghiệm. Tối thiểu bốn nhóm thông tin:

1. **Đầu vào**: commit mã, hash dữ liệu, tệp cấu hình đầy đủ, ảnh container.
2. **Quá trình**: đường cong hàm mất mát, learning rate theo bước, tài nguyên sử dụng, thời gian chạy.
3. **Đầu ra**: chỉ số theo lát cắt (Mục 6.3), ma trận nhầm lẫn, tạo tác mô hình kèm hash.
4. **Ngữ cảnh**: ai chạy, khi nào, để kiểm tra giả thuyết gì.

Các công cụ như MLflow hay Weights & Biases lưu được cả bốn nhóm, nhưng nhóm thứ tư thường bị bỏ trống vì không công cụ nào điền thay được. Nhóm này lại là cách phòng thói quen *retrofitting an explanation* ở Mục 2.5: nếu giả thuyết được ghi lại trước khi chạy, lời giải thích không thể được gắn vào sau.

### 5.4. Sổ đăng ký mô hình

Sổ đăng ký mô hình (model registry) là nơi mỗi mô hình có một định danh và một vòng đời. Tối thiểu, nó phải trả lời được:

- mô hình này được huấn luyện từ commit nào, dữ liệu nào, cấu hình nào;
- nó đã qua những bước kiểm định nào, với kết quả ra sao;
- nó đang ở trạng thái nào (`staging`, `production`, `archived`) và ai duyệt;
- phiên bản nào đang thực sự chạy trong sản xuất, và nếu cần quay lui thì quay về phiên bản nào.

Câu hỏi cuối là lý do chính để có sổ đăng ký: yếu tố phiên bản ở Mục 2.2 tồn tại để quay lui được. Nghiên cứu của Shankar và cộng sự ghi nhận hai thực hành liên quan: giữ các phiên bản cũ làm mô hình dự phòng, và duy trì một lớp quy tắc đơn giản (heuristic) phía sau mô hình, để khi mô hình hỏng thì hệ thống chuyển sang một hành vi đơn giản nhưng đoán trước được.

Mỗi mô hình trong sổ đăng ký nên kèm một **thẻ mô hình** (model card; Mitchell và cộng sự, 2019): mục đích sử dụng, những trường hợp không nên dùng, dữ liệu huấn luyện, và kết quả đánh giá tách theo các nhóm liên quan. Thẻ mô hình biến những hiểu biết vốn chỉ nằm trong đầu người huấn luyện thành tài liệu đi cùng mô hình.

---

## 6. Đánh giá mô hình cho sản xuất

Một mô hình được tóm tắt bằng một con số, chẳng hạn "độ chính xác 92%", là một mô hình chưa được đánh giá đủ cho sản xuất. Chương này trình bày ba lý do và cách xử lý tương ứng: chỉ số ngoại tuyến phải gắn với chỉ số sản phẩm, chỉ số gộp che mất các nhóm nhỏ, và chỉ số trung bình không cho biết hành vi trên những trường hợp cụ thể. Phần cuối là bộ tiêu chí ML Test Score để đánh giá mức sẵn sàng của cả hệ thống.

### 6.1. Ba giới hạn của một con số

1. **Con số đo trên tập kiểm tra, không đo trên sản phẩm.** Mô hình tốt hơn theo AUC chưa chắc làm doanh thu hay mức hài lòng tăng (Mục 6.2).
2. **Con số gộp là trung bình theo lưu lượng,** nên nhóm chiếm đa số quyết định kết quả. Một nhóm nhỏ có thể tệ đi mà con số gộp vẫn tăng (Mục 6.3).
3. **Con số trung bình không nói gì về từng trường hợp.** Mô hình có thể tăng một điểm độ chính xác nhưng sai ở một trường hợp hiển nhiên mà phiên bản trước làm đúng (Mục 6.4).

### 6.2. Gắn chỉ số ngoại tuyến với chỉ số sản phẩm

Shankar và cộng sự (2022) ghi nhận thực hành gắn chỉ số đánh giá mô hình với chỉ số của sản phẩm. Tiêu chí Model 2 của ML Test Score (Mục 6.5) phát biểu cùng ý: chỉ số ngoại tuyến phải tương quan với chỉ số trực tuyến.

Quan hệ giữa hai loại chỉ số hiếm khi tuyến tính. Bảng dưới là các ví dụ minh hoạ cho những dạng quan hệ thường gặp, không phải số đo:

| Chỉ số của mô hình | Chỉ số sản phẩm tương ứng | Dạng quan hệ thường gặp |
|---|---|---|
| AUC của mô hình phát hiện gian lận | số tiền thiệt hại chặn được | bão hoà: quá một mức nào đó, AUC tăng nhưng số tiền không đổi vì nhóm điều tra đã hết năng lực xử lý |
| NDCG của hệ thống xếp hạng | thời gian người dùng ở lại | có thể ngược chiều: xếp hạng quá sát sở thích cũ làm giảm khám phá (Chương 12) |
| Perplexity của mô hình ngôn ngữ | tỉ lệ người dùng chấp nhận câu trả lời | tương quan yếu, và yếu dần khi mô hình đã đủ tốt |

Quy tắc thực hành: dùng một chỉ số ngoại tuyến rẻ để lặp nhanh, nhưng định kỳ kiểm tra rằng nó vẫn tương quan với chỉ số sản phẩm. Khi tương quan đó không còn, cải thiện chỉ số ngoại tuyến không còn mang lại giá trị cho sản phẩm.

### 6.3. Đánh giá theo lát cắt

Đánh giá theo lát cắt (slice-based evaluation) là tính chỉ số riêng cho từng nhóm con của dữ liệu thay vì chỉ tính một con số trên toàn bộ tập kiểm tra. Đây là một trong những kỹ thuật cho lợi ích lớn nhất so với công sức bỏ ra.

**Thí nghiệm.** Thí nghiệm trong `code/mlops/experiments.py` mô phỏng một tình huống thường gặp. Có 8 000 mẫu huấn luyện và 8 000 mẫu kiểm tra, trong đó 15% là khách hàng mới (nhóm B) và 85% là khách hàng cũ (nhóm A). Nhãn phụ thuộc vào đặc trưng $x_1$ mà mọi khách hàng đều có. Mô hình v2 thêm đặc trưng $x_2$, "lịch sử mua hàng": với khách hàng cũ, $x_2$ mang nhiều thông tin về nhãn; với khách hàng mới, chưa có lịch sử nên $x_2$ chỉ là nhiễu. Cả v1 (chỉ dùng $x_1$) và v2 (dùng $x_1$, $x_2$) đều là hồi quy logistic, không biết khách hàng thuộc nhóm nào.

![Hình 6](figs/mlops06_slice.png)

**Hình 6.** Độ chính xác của v1 và v2 trên toàn bộ tập kiểm tra và trên từng nhóm. Chỉ số gộp tăng mạnh trong khi nhóm khách hàng mới giảm.

| Lát cắt | v1 | v2 | Thay đổi |
|---|---|---|---|
| Tổng thể | 69,14% | 88,96% | +19,83 điểm |
| Nhóm A: khách hàng cũ, 85% lưu lượng | 68,99% | 94,48% | +25,49 điểm |
| Nhóm B: khách hàng mới, 15% lưu lượng | 69,97% | 58,35% | −11,62 điểm |

Cơ chế: vì $x_2$ rất tốt với 85% dữ liệu, mô hình học cho nó trọng số lớn (2,28, so với 1,07 của $x_1$). Với 15% còn lại, chính trọng số lớn đó đưa nhiễu vào dự đoán. Nhóm B không chỉ không được lợi mà còn bị hại, xuống dưới cả mức của mô hình cũ. Đây là một minh hoạ của nguyên lý CACE (Mục 1.3): thêm một đặc trưng không cộng thêm một thứ vào mô hình mà thay đổi toàn bộ hàm dự đoán.

Các cách sửa, vốn chỉ nghĩ tới được sau khi đã thấy vấn đề qua lát cắt, gồm: mã hoá "chưa có lịch sử" thành một giá trị thiếu tường minh kèm một đặc trưng chỉ báo, thay vì để nó thành một số ngẫu nhiên; thêm tương tác giữa $x_2$ và nhóm khách hàng; hoặc dùng mô hình riêng cho khách hàng mới.

**Nên theo dõi những lát cắt nào.** Tối thiểu bốn loại:

1. **Nghiệp vụ**: khách hàng mới hay cũ, thị trường, hạng khách hàng, kênh bán.
2. **Kỹ thuật**: thiết bị, phiên bản ứng dụng, có hay không có một đặc trưng nào đó.
3. **Công bằng**: các nhóm được pháp luật bảo vệ, nếu bài toán liên quan.
4. **Thời gian**: giờ trong ngày, ngày trong tuần, ngày lễ.

Tiêu chí Model 6 của ML Test Score phát biểu đúng yêu cầu này: chất lượng mô hình phải đủ tốt trên mọi lát cắt dữ liệu quan trọng.

### 6.4. Kiểm thử hành vi

Ngoài chỉ số trên tập kiểm tra, mô hình cần loại kiểm thử mà phần mềm thông thường vẫn dùng: khẳng định về hành vi trên những đầu vào cụ thể. Ribeiro và cộng sự (2020), với bộ công cụ CheckList, phân biệt ba loại kiểm thử:

```python
# 1. Chức năng tối thiểu (minimum functionality): những trường hợp đơn giản không được phép sai
assert sentiment("Sản phẩm này tuyệt vời") == "tích cực"

# 2. Bất biến (invariance): thay đổi không liên quan thì dự đoán phải giữ nguyên
assert sentiment("Chuyến bay tới Hà Nội bị huỷ") == sentiment("Chuyến bay tới Đà Nẵng bị huỷ")

# 3. Kỳ vọng có hướng (directional expectation): thay đổi có hướng rõ thì dự đoán phải đổi đúng hướng
assert risk(income=50_000_000) <= risk(income=5_000_000)   # các đặc trưng khác giữ nguyên
```

Giá trị của các kiểm thử này: một mô hình có thể tăng một điểm độ chính xác trung bình mà mất khả năng xử lý đúng một trường hợp hiển nhiên; chỉ số trung bình không phát hiện được điều đó, còn kiểm thử hành vi thì có. Các kiểm thử hành vi nên được chạy tự động mỗi khi có mô hình mới, như kiểm thử đơn vị trong CI.

### 6.5. ML Test Score

Bộ tiêu chí đầy đủ nhất để trả lời câu hỏi "hệ thống đã sẵn sàng cho sản xuất chưa" là *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction* (Breck và cộng sự, 2017). Bộ tiêu chí gồm 28 mục, chia thành bốn nhóm, mỗi nhóm 7 mục.

**Nhóm 1: Dữ liệu và đặc trưng**

| Mục | Nội dung |
|---|---|
| Data 1 | Kỳ vọng về đặc trưng được ghi lại thành một lược đồ |
| Data 2 | Mọi đặc trưng đều có ích |
| Data 3 | Không đặc trưng nào có chi phí quá lớn so với lợi ích |
| Data 4 | Đặc trưng tuân thủ các yêu cầu ở mức chính sách (quyền riêng tư, pháp lý) |
| Data 5 | Pipeline dữ liệu có kiểm soát quyền riêng tư phù hợp |
| Data 6 | Có thể thêm đặc trưng mới nhanh chóng |
| Data 7 | Toàn bộ mã tính đặc trưng đều được kiểm thử |

**Nhóm 2: Phát triển mô hình**

| Mục | Nội dung |
|---|---|
| Model 1 | Mọi đặc tả mô hình đều được rà soát và đưa vào kho mã |
| Model 2 | Chỉ số ngoại tuyến tương quan với chỉ số trực tuyến |
| Model 3 | Mọi siêu tham số đã được tinh chỉnh |
| Model 4 | Đã biết ảnh hưởng của việc mô hình cũ đi (staleness) |
| Model 5 | Một mô hình đơn giản hơn không tốt hơn (kiểm tra định kỳ) |
| Model 6 | Chất lượng đủ tốt trên mọi lát cắt dữ liệu quan trọng |
| Model 7 | Mô hình đã được kiểm tra về tính bao hàm (inclusion) giữa các nhóm người dùng |

**Nhóm 3: Hạ tầng học máy**

| Mục | Nội dung |
|---|---|
| Infra 1 | Huấn luyện tái lập được |
| Infra 2 | Mã đặc tả mô hình có kiểm thử đơn vị |
| Infra 3 | Toàn bộ pipeline có kiểm thử tích hợp |
| Infra 4 | Chất lượng mô hình được kiểm định trước khi đưa vào phục vụ |
| Infra 5 | Mô hình gỡ lỗi được bằng cách quan sát từng bước tính |
| Infra 6 | Mô hình được thử qua quy trình canary trước khi phục vụ toàn bộ |
| Infra 7 | Có thể quay lui mô hình đang phục vụ nhanh và an toàn |

**Nhóm 4: Giám sát**

| Mục | Nội dung |
|---|---|
| Monitor 1 | Thay đổi ở các hệ thống phụ thuộc phía trước sinh ra thông báo |
| Monitor 2 | Các bất biến của dữ liệu đúng ở cả đầu vào huấn luyện lẫn đầu vào phục vụ |
| Monitor 3 | Đặc trưng lúc huấn luyện và lúc phục vụ tính ra cùng giá trị |
| Monitor 4 | Mô hình không quá cũ |
| Monitor 5 | Mô hình ổn định về mặt số học (không có NaN, không có giá trị vô cùng) |
| Monitor 6 | Không có suy giảm, đột ngột hay từ từ, về tốc độ huấn luyện, độ trễ phục vụ, thông lượng hay bộ nhớ |
| Monitor 7 | Chất lượng dự đoán trên dữ liệu thật không suy giảm |

**Cách tính điểm:**

- 0,5 điểm cho một mục nếu mục đó được thực hiện thủ công, có ghi lại kết quả và gửi cho những người liên quan;
- 1 điểm nếu có hệ thống chạy mục đó tự động và định kỳ;
- cộng điểm riêng cho từng nhóm trong bốn nhóm;
- **điểm cuối cùng là giá trị nhỏ nhất trong bốn điểm nhóm.**

Bài báo giải thích việc lấy giá trị nhỏ nhất: cả bốn nhóm đều quan trọng, nên muốn tăng điểm phải cải thiện cả bốn. Hệ quả: một hệ thống có hạ tầng huấn luyện hoàn hảo nhưng không giám sát gì có điểm 0, dù ba nhóm kia đạt tối đa.

**Cách đọc điểm** (Bảng V của bài báo):

| Điểm | Ý nghĩa |
|---|---|
| 0 | giống một dự án nghiên cứu hơn là một hệ thống sản xuất |
| (0, 1] | đã có kiểm thử, nhưng có thể còn những lỗ hổng nghiêm trọng về độ tin cậy |
| (1, 2] | đã có bước đầu đưa vào sản xuất, cần đầu tư thêm |
| (2, 3] | kiểm thử tương đối đủ, nhiều mục còn có thể tự động hoá |
| (3, 5] | kiểm thử và giám sát tự động ở mức cao, phù hợp với hệ thống trọng yếu |
| > 5 | kiểm thử và giám sát tự động ở mức rất cao |

Bài báo cũng lưu ý rằng hệ thống ở các giai đoạn khác nhau nên nhắm tới các mức điểm khác nhau; một nguyên mẫu không cần đạt mức của một hệ thống trọng yếu.

---

## 7. Đóng gói và phục vụ mô hình

Chương này trình bày ba chế độ phục vụ mô hình, cách lập ngân sách độ trễ, lý do phải theo dõi phân vị cao của độ trễ thay vì trung bình, và cách đóng gói mô hình để môi trường huấn luyện và môi trường phục vụ khớp nhau.

### 7.1. Ba chế độ phục vụ

| Chế độ | Cách chạy | Phù hợp với | Khó khăn chính |
|---|---|---|---|
| Theo lô (batch) | chấm điểm định kỳ, ghi kết quả vào bảng | gợi ý hằng ngày, chấm điểm rủi ro | dự đoán cũ khi người dùng vừa thay đổi hành vi |
| Trực tuyến (online, request–response) | gọi đồng bộ, trả kết quả trong vài chục mili giây | xếp hạng tìm kiếm, chống gian lận khi thanh toán | ngân sách độ trễ, phần đuôi của phân phối độ trễ |
| Luồng (streaming) | xử lý theo sự kiện, trạng thái cập nhật liên tục | phát hiện bất thường, cá nhân hoá theo phiên | tính đúng của trạng thái, xử lý sự kiện đến muộn |

Một sai lầm hay gặp là chọn phục vụ trực tuyến khi yêu cầu nghiệp vụ không cần. Nguyên tắc: chọn chế độ rẻ nhất vẫn đáp ứng yêu cầu, vì chi phí vận hành tăng nhanh theo thứ tự trong bảng. Chế độ phục vụ cũng quyết định cách giám sát: với chấm điểm theo lô, có thể kiểm tra toàn bộ kết quả trước khi dùng; với phục vụ trực tuyến, dự đoán đã tới người dùng trước khi ai kịp kiểm tra.

### 7.2. Ngân sách độ trễ

Thiết kế phục vụ trực tuyến bắt đầu bằng một phép tính. Ví dụ giả định: ngân sách 150 ms cho một lượt xếp hạng kết quả tìm kiếm.

| Thành phần | Ngân sách |
|---|---|
| Mạng, chiều đi và chiều về | 20 ms |
| Tra cứu đặc trưng ở kho trực tuyến | 25 ms |
| Tiền xử lý đặc trưng | 10 ms |
| Suy luận mô hình | 60 ms |
| Hậu xử lý, ghi log | 15 ms |
| Dự phòng | 20 ms |

Trong ví dụ này, suy luận mô hình chiếm chưa tới một nửa ngân sách. Tối ưu mô hình từ 60 ms xuống 30 ms chỉ giảm tổng thời gian 20%, trong khi bỏ một lần tra cứu đặc trưng thừa có thể giảm nhiều hơn. Đây là phiên bản về độ trễ của Hình 1: phần mô hình chỉ là một phần của hệ thống. Các kỹ thuật giảm thời gian suy luận như lượng tử hoá được trình bày ở giáo trình [*Quantization*](ch01.html).

### 7.3. Phân vị cao của độ trễ và hiệu ứng toả nhánh

Độ trễ trung bình ít có giá trị để giám sát, vì phân phối độ trễ thường lệch phải mạnh: đa số yêu cầu nhanh, một số ít rất chậm, và người dùng cảm nhận những yêu cầu chậm. Vì vậy các mục tiêu mức dịch vụ (SLO) thường đặt trên phân vị cao: p95, p99.

Lý do quan trọng hơn là **hiệu ứng toả nhánh** (fan-out), được Dean và Barroso (2013) phân tích trong bài *The Tail at Scale*. Nếu một yêu cầu phải gọi $k$ dịch vụ song song và chỉ trả lời khi cả $k$ lời gọi đã xong, thời gian trả lời là **giá trị lớn nhất** của $k$ biến ngẫu nhiên. Với $k$ lời gọi độc lập có cùng hàm phân phối $F$:

$$P(\max \le t) = F(t)^k \quad\Longrightarrow\quad \text{phân vị } q \text{ của max} = F^{-1}\!\left(q^{1/k}\right).$$

Nói cách khác, để toàn bộ yêu cầu đạt p99 thì mỗi nhánh phải đạt phân vị $0{,}99^{1/k}$; với $k = 100$, đó là phân vị 99,99.

Thí nghiệm trong `code/mlops/experiments.py` dùng phân phối log-chuẩn với trung vị 20 ms và p99 bằng 100 ms cho một dịch vụ:

![Hình 7](figs/mlops07_tail.png)

**Hình 7.** Trái: p99 của thời gian trả lời theo số nhánh gọi song song. Phải: tỉ lệ yêu cầu gặp ít nhất một nhánh chậm hơn p99 của một nhánh, bằng $1-0{,}99^k$.

| Số nhánh $k$ | p99 của thời gian trả lời | Tỉ lệ yêu cầu gặp ít nhất một nhánh chậm |
|---|---|---|
| 1 | 100,0 ms | 1,0% |
| 10 | 169,5 ms | 9,6% |
| 50 | 231,4 ms | 39,5% |
| 100 | 261,9 ms | 63,4% |

Với 100 nhánh, dù mỗi dịch vụ chỉ chậm hơn 100 ms ở 1% số lần gọi, 63% số yêu cầu gặp ít nhất một lời gọi chậm như vậy. Giảm độ trễ trung bình của từng dịch vụ không giải quyết được vấn đề này. Dean và Barroso nêu các cách xử lý: giảm số nhánh; đặt thời hạn chờ (timeout) và trả lời với kết quả một phần; và gửi yêu cầu dự phòng (hedged request), tức gửi lại yêu cầu tới một bản sao khác nếu bản đầu chưa trả lời sau một khoảng thời gian, rồi dùng kết quả về trước.

### 7.4. Đóng gói và ranh giới môi trường

Nghiên cứu của Shankar và cộng sự ghi nhận khó khăn thường gặp là sự khác biệt giữa môi trường phát triển và môi trường sản xuất. Hai yêu cầu kéo về hai phía: môi trường phát triển cần linh hoạt để thử nghiệm nhanh, môi trường sản xuất cần chặt chẽ; nhưng hai môi trường càng khác nhau thì càng khó phát hiện lỗi sớm.

Ba thực hành thu hẹp khoảng cách:

1. **Dùng cùng một ảnh container** cho huấn luyện và phục vụ, chỉ khác điểm vào (entrypoint).
2. **Dùng cùng một hàm biến đổi đặc trưng**, đóng gói cùng mô hình thay vì cài đặt lại ở dịch vụ phục vụ (Mục 4.4).
3. **Kiểm thử tích hợp toàn pipeline** trong CI trên một tập dữ liệu nhỏ nhưng thật, đúng như mục Infra 3 của ML Test Score.

Về định dạng lưu mô hình, có một đánh đổi:

| Cách lưu | Ưu điểm | Nhược điểm |
|---|---|---|
| Pickle, `state_dict` | đơn giản, giữ nguyên hệ sinh thái | phụ thuộc chặt vào phiên bản thư viện; nạp tệp pickle không tin cậy có thể chạy mã tuỳ ý |
| Đồ thị đã xuất (ONNX, TorchScript, SavedModel) | không phụ thuộc ngôn ngữ, tối ưu được, ranh giới rõ | kém linh hoạt; phải kiểm tra đồ thị xuất ra cho cùng kết quả với bản gốc |

Với cách thứ hai, luôn cần một kiểm thử so sánh: chạy bản gốc và bản đã xuất trên cùng một lô đầu vào và so sánh đầu ra. Vì hai bản có thể dùng các kernel khác nhau, phép so sánh thường dùng một dung sai (ví dụ sai số tương đối $10^{-3}$) thay vì đòi trùng khớp từng bit. Mục 6.5 của giáo trình [*Quantization*](ch06.html) là một ví dụ của cùng kỷ luật này: kiểm tra bằng mã rằng phép tính số nguyên cho cùng kết quả với phép mô phỏng.

---

## 8. Chiến lược ra mắt

Một mô hình đã qua đánh giá ngoại tuyến vẫn chưa được đưa ngay tới mọi người dùng. Chương này trình bày bốn chiến lược ra mắt, mỗi chiến lược trả lời một câu hỏi khác nhau, rồi đi vào chi tiết của shadow, canary và A/B test, gồm cách tính cỡ mẫu và những lỗi thống kê thường gặp.

### 8.1. Bốn chiến lược và câu hỏi mỗi chiến lược trả lời

![Hình 8](figs/mlops08_rollout.png)

**Hình 8.** Bốn chiến lược ra mắt. Chọn chiến lược theo câu hỏi cần trả lời ở từng bước.

| Chiến lược | Trả lời được | Không trả lời được |
|---|---|---|
| Shadow (chạy song song ngầm) | mô hình mới có lỗi, có chậm không; dự đoán khác mô hình cũ nhiều không | người dùng có phản ứng tốt hơn không, vì họ không thấy kết quả của mô hình mới |
| Canary (ra mắt dần) | có chỉ số vận hành nào xấu đi ở quy mô nhỏ không | hiệu ứng dài hạn, hiệu ứng mạng lưới |
| Blue–green | chủ yếu là cơ chế triển khai: chuyển toàn bộ lưu lượng giữa hai môi trường và quay lui tức thì | không trả lời câu hỏi nào về chất lượng |
| A/B test | mô hình mới có làm chỉ số sản phẩm tốt lên hay không, theo nghĩa nhân quả | câu trả lời nhanh: A/B test tốn thời gian và lưu lượng |

Shankar và cộng sự (2022) ghi nhận thực hành chia một lần ra mắt thành nhiều giai đoạn và đánh giá ở từng giai đoạn. Trình tự điển hình: đánh giá ngoại tuyến, shadow, canary 1%, A/B test, rồi toàn bộ. Mỗi giai đoạn loại được một loại rủi ro, và loại càng sớm càng rẻ. Nghiên cứu cũng ghi nhận cái giá: nhiều người được phỏng vấn cho rằng quá trình từ lúc có ý tưởng tới lúc kiểm chứng xong kéo dài quá lâu. Một người cho biết ở công ty của họ, thử một ý tưởng đặc trưng mới mất hơn ba tháng; khoảng 40 tới 50% ý tưởng đi tới được lần ra mắt ban đầu, và khoảng một nửa trong số đó sau đó bị loại vì lý do pháp lý, quyền riêng tư hoặc độ phức tạp. Loại được ý tưởng kém ở giai đoạn sớm vì thế làm tăng tốc độ chung (Mục 2.2).

### 8.2. Shadow

Ở chế độ shadow, mô hình mới nhận cùng lưu lượng thật với mô hình cũ, dự đoán của nó được ghi lại nhưng không được dùng. Shadow trả lời được ba câu hỏi mà người dùng không phải chịu rủi ro nào:

1. **Mô hình mới có chịu được tải thật không**: độ trễ, bộ nhớ, tỉ lệ lỗi.
2. **Đặc trưng lúc phục vụ có khớp với lúc huấn luyện không**: đây là mục Monitor 3 của ML Test Score, và shadow là cách rẻ nhất để đo nó trên lưu lượng thật.
3. **Phân phối dự đoán có khác bất thường không**: so sánh phân phối của $\hat y$ giữa mô hình mới và mô hình cũ.

Điểm thứ hai đáng chú ý: nhiều lỗi lệch huấn luyện–phục vụ chỉ xuất hiện trên phân phối đầu vào thật, không xuất hiện trên tập kiểm tra. Shadow phát hiện được chúng trước khi có người dùng nào bị ảnh hưởng. Giới hạn của shadow: với các hệ thống mà dự đoán ảnh hưởng tới hành vi người dùng, như gợi ý hay xếp hạng, shadow không đo được phản ứng của người dùng với mô hình mới.

### 8.3. Canary

Canary đưa mô hình mới tới một phần nhỏ lưu lượng, ví dụ 1%, rồi tăng dần. Canary chỉ có ý nghĩa nếu ở mức lưu lượng nhỏ vẫn đo được điều gì đó; với 1% lưu lượng, chỉ số sản phẩm gần như chắc chắn không đủ mẫu để có ý nghĩa thống kê (Mục 8.4). Vì vậy cần phân chia chỉ số theo độ nhạy:

| Đo được ở canary 1% | Phải đợi A/B test |
|---|---|
| tỉ lệ lỗi, độ trễ p50, p95, p99 | tỉ lệ chuyển đổi, doanh thu mỗi phiên |
| phân phối của dự đoán $\hat y$ | tỉ lệ giữ chân người dùng |
| tỉ lệ đặc trưng bị thiếu | các chỉ số dài hạn |

Canary bảo vệ khỏi sự cố lớn; A/B test trả lời câu hỏi về giá trị. Dùng canary để kết luận về giá trị sản phẩm là một lỗi thiết kế quy trình thường gặp.

### 8.4. A/B test và cỡ mẫu

Với chỉ số là một tỉ lệ (tỉ lệ nhấp, tỉ lệ chuyển đổi), cỡ mẫu mỗi nhánh cần để phát hiện chênh lệch $\Delta = p_2 - p_1$ với mức ý nghĩa $\alpha$ (hai phía) và lực kiểm định $1-\beta$ là

$$n \;=\; \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \left[p_1(1-p_1) + p_2(1-p_2)\right]}{\Delta^2}.$$

$n$ tỉ lệ nghịch với bình phương mức chênh lệch: muốn phát hiện một mức cải thiện nhỏ bằng một nửa thì cần gấp bốn lần số mẫu.

![Hình 9](figs/mlops09_abtest.png)

**Hình 9.** Cỡ mẫu mỗi nhánh theo mức cải thiện tương đối cần phát hiện, với ba mức tỉ lệ nền. Trục tung dùng thang logarit.

Bảng dưới lấy tỉ lệ nền 5%, $\alpha = 0{,}05$ hai phía và lực kiểm định 80%, tính trong `code/mlops/experiments.py`:

| Cải thiện tương đối cần phát hiện | Cỡ mẫu mỗi nhánh | Số ngày nếu mỗi nhánh có 100 000 lượt mỗi ngày |
|---|---|---|
| 1% (từ 5% lên 5,05%) | 2 996 694 | 30,0 ngày |
| 2% | 752 700 | 7,5 ngày |
| 5% | 122 121 | 1,2 ngày |
| 10% | 31 231 | 0,3 ngày |
| 20% | 8 155 | 0,1 ngày |

Hàng đầu tiên giải thích vì sao không thể A/B test mọi thay đổi: để phát hiện cải thiện tương đối 1% trên tỉ lệ nền 5% cần khoảng ba triệu mẫu mỗi nhánh, tức một tháng lưu lượng cho một ý tưởng. Lưu lượng là tài nguyên khan hiếm, và quy trình ra mắt tốt được thiết kế để không dùng nó cho những ý tưởng có thể loại sớm hơn bằng cách rẻ hơn (đánh giá ngoại tuyến, shadow).

**Giảm phương sai.** Công thức cho thấy ba cách giảm cỡ mẫu: chấp nhận chỉ phát hiện mức cải thiện lớn hơn, nới $\alpha$ hoặc lực kiểm định, và giảm phương sai của chỉ số. Cách thứ ba không làm giảm độ tin cậy của kết luận. Kỹ thuật phổ biến là CUPED (Deng và cộng sự, 2013): dùng giá trị của chính chỉ số đó ở mỗi người dùng trong giai đoạn trước thí nghiệm làm biến hiệp phương sai, và trừ phần biến động giải thích được bởi nó. Nếu hệ số tương quan giữa giá trị trước và trong thí nghiệm là $\rho$, phương sai giảm theo hệ số $1 - \rho^2$, và cỡ mẫu cần thiết giảm cùng tỉ lệ. [Công cụ cỡ mẫu](mlops-thuc-hanh.html#co-mau) ở trang Phòng thí nghiệm cho phép nhập mức giảm phương sai này.

### 8.5. Những lỗi thống kê thường gặp

Tài liệu tham khảo chính cho mục này là Kohavi, Tang và Xu (2020), *Trustworthy Online Controlled Experiments*.

1. **Xem kết quả giữa chừng (peeking).** Kiểm tra p-value liên tục và dừng ngay khi thấy $p < 0{,}05$ làm tỉ lệ dương tính giả cao hơn 5% nhiều. Cách xử lý: cố định cỡ mẫu trước khi chạy, hoặc dùng các phương pháp thiết kế cho việc theo dõi liên tục (kiểm định tuần tự).
2. **So sánh nhiều chỉ số cùng lúc.** Với 20 chỉ số không thật sự thay đổi, kiểm định ở mức $\alpha = 0{,}05$ cho trung bình 1 chỉ số "có ý nghĩa" do ngẫu nhiên, và xác suất có ít nhất một chỉ số như vậy là $1 - 0{,}95^{20} \approx 64\%$ nếu các chỉ số độc lập. Cách xử lý: khai báo trước một chỉ số chính; các chỉ số còn lại là chỉ số bảo vệ (guardrail metric), chỉ dùng để phát hiện tác hại.
3. **Hai nhánh ảnh hưởng lẫn nhau.** Trong mạng xã hội hoặc sàn giao dịch hai phía, người dùng ở nhánh A ảnh hưởng tới người dùng ở nhánh B, nên giả định độc lập không còn đúng. Cách xử lý: chia ngẫu nhiên theo cụm (theo vùng, theo nhóm bạn bè) thay vì theo cá nhân.
4. **Tỉ lệ mẫu lệch (sample ratio mismatch).** Thiết kế chia 50/50 nhưng số người dùng thực tế ở hai nhánh chênh lệch nhiều hơn mức ngẫu nhiên cho phép, ví dụ vì một nhánh gây lỗi làm mất log. Khi đó kết quả không đáng tin, dù p-value trông thế nào. Kiểm tra bằng kiểm định khi bình phương trên số người dùng của hai nhánh trước khi đọc kết quả.
5. **Hiệu ứng mới lạ.** Người dùng có thể phản ứng mạnh với một thay đổi chỉ vì nó mới, và hiệu ứng giảm dần sau vài tuần. Cách xử lý: chạy đủ lâu và xem hiệu ứng theo thời gian, đặc biệt ở những người dùng đã gặp thay đổi nhiều lần.

---

## 9. Dịch chuyển phân phối

Dữ liệu trong sản xuất khác dần với dữ liệu huấn luyện. Chương này định nghĩa ba loại dịch chuyển phân phối, dùng một thí nghiệm để chỉ ra loại nào phát hiện được mà không cần nhãn, trình bày các phép kiểm tra thống kê thường dùng, rồi phân tích chỉ số PSI: vì sao bộ ngưỡng quen thuộc của nó cho kết quả sai khi cỡ mẫu thay đổi, và nên dùng gì thay thế.

### 9.1. Ba loại dịch chuyển

Mô hình học quan hệ giữa $X$ và $Y$ từ phân phối chung $P(X, Y)$. Phân phối chung có thể tách theo hai cách:

$$P(X, Y) \;=\; P(Y \mid X)\,P(X) \;=\; P(X \mid Y)\,P(Y).$$

Mỗi cách tách cho một cách mô tả sự thay đổi (Quiñonero-Candela và cộng sự, 2009):

| Loại | Thành phần thay đổi | Thành phần giữ nguyên | Ví dụ |
|---|---|---|---|
| Covariate shift | $P(X)$ | $P(Y \mid X)$ | một chiến dịch quảng cáo thu hút nhiều khách hàng trẻ hơn; quan hệ giữa tuổi và hành vi không đổi, chỉ có tỉ lệ người trẻ tăng |
| Label shift (prior probability shift) | $P(Y)$ | $P(X \mid Y)$ | tỉ lệ giao dịch gian lận tăng gấp đôi dịp lễ, nhưng một giao dịch gian lận vẫn có đặc điểm như trước |
| Concept drift | $P(Y \mid X)$ | $P(X)$ | cùng một hồ sơ khách hàng, nhưng định nghĩa "rủi ro" thay đổi vì chính sách tín dụng mới |

Covariate shift và concept drift là hai thành phần của cùng một cách tách; label shift thuộc cách tách còn lại. Vì vậy câu hỏi "đây là covariate shift hay label shift" không phải lúc nào cũng có câu trả lời duy nhất: nó phụ thuộc vào việc coi $X$ gây ra $Y$ (ví dụ đặc điểm khách hàng quyết định hành vi) hay $Y$ gây ra $X$ (ví dụ bệnh gây ra triệu chứng). Trong thực tế, các loại dịch chuyển thường xảy ra cùng lúc.

### 9.2. Thí nghiệm: loại dịch chuyển nào phát hiện được mà không cần nhãn

**Thiết lập.** Thí nghiệm trong `code/mlops/experiments.py` dùng dữ liệu hai chiều: $X \sim \mathcal N(0, I)$, nhãn sinh theo mô hình logistic với trọng số $w = (1{,}5;\ -1{,}0)$. Một hồi quy logistic được huấn luyện trên 4 000 mẫu, rồi được đánh giá trên ba tập dữ liệu mới, mỗi tập 4 000 mẫu:

- **Covariate shift**: $X \sim \mathcal N(1{,}2;\ 1)$ trên mỗi trục, nhãn sinh theo cùng quy tắc.
- **Label shift**: lấy mẫu lại sao cho 80% mẫu có nhãn dương (dữ liệu gốc khoảng 50%), giữ nguyên $P(X \mid Y)$.
- **Concept drift**: $X$ có cùng phân phối như lúc huấn luyện, nhưng nhãn sinh theo trọng số mới $w = (-1{,}0;\ 1{,}5)$.

Với mỗi tập, đo hai đại lượng: độ chính xác (cần nhãn) và p-value của kiểm định Kolmogorov–Smirnov (KS) hai mẫu trên từng đặc trưng, so với dữ liệu huấn luyện (không cần nhãn). Một dịch chuyển được coi là phát hiện được nếu p-value nhỏ nhất trong hai đặc trưng dưới $10^{-3}$.

![Hình 10](figs/mlops10_shifts.png)

**Hình 10.** Dữ liệu nguồn (xám), dữ liệu mới (màu) và biên quyết định của mô hình trong ba kịch bản. Ở hai kịch bản đầu, đám mây dữ liệu dịch chuyển và nhìn thấy được. Ở kịch bản concept drift, đám mây không đổi, chỉ có nhãn thay đổi, và độ chính xác giảm mạnh.

| Kịch bản | Độ chính xác | p-value KS trên $x_1$ | p-value KS trên $x_2$ | Phát hiện được chỉ từ $X$? |
|---|---|---|---|---|
| Không dịch chuyển | 75,6% | | | |
| Covariate shift | 76,4% | $\approx 0$ | $\approx 0$ | có |
| Label shift | 75,1% | $1{,}2\times10^{-29}$ | $2{,}2\times10^{-7}$ | có |
| Concept drift | 26,9% | $0{,}48$ | $0{,}043$ | không |

Bảng cho hai kết luận ngược chiều nhau:

1. **Hai kịch bản giữa phát hiện được nhưng gần như vô hại.** Kiểm định KS cho p-value rất nhỏ, trong khi độ chính xác gần như không đổi. Trong thí nghiệm này, mô hình logistic có dạng đúng với quy tắc sinh nhãn, nên việc dữ liệu dịch sang vùng khác không làm hỏng nó; covariate shift gây hại khi mô hình phải ngoại suy ra vùng ít dữ liệu huấn luyện mà nó mô tả sai. Label shift phát hiện được từ $X$ vì $X$ mang thông tin về $Y$: đổi tỉ lệ nhãn thì phân phối của $X$ cũng đổi theo. Nếu cảnh báo chỉ dựa trên thay đổi của $X$, cả hai trường hợp đều thành báo động giả, loại khó khăn được nhắc nhiều nhất ở Mục 3.2.
2. **Kịch bản cuối không phát hiện được nhưng gây hại nặng.** Độ chính xác giảm từ 75,6% xuống 26,9%, thấp hơn cả đoán ngẫu nhiên, trong khi các kiểm định trên $X$ không thấy gì bất thường. Giá trị $p = 0{,}043$ trên $x_2$ chỉ là dao động ngẫu nhiên, vì phân phối của $X$ không đổi; với ngưỡng thông thường 0,05, nó sẽ bị tính là "phát hiện", một ví dụ của vấn đề kiểm định nhiều lần ở Mục 9.3.

> **Nhận xét.** Giám sát $P(X)$ không thể phát hiện concept drift, vì theo định nghĩa concept drift giữ nguyên $P(X)$. Muốn phát hiện concept drift phải có tín hiệu về nhãn: nhãn thật, nhãn đến trễ, nhãn thay thế, hoặc phản hồi gián tiếp của người dùng.

Đây là lý do Mục 3.5 nói độ trễ nhãn quyết định khả năng giám sát: nó quyết định hệ thống có phát hiện được loại dịch chuyển gây hại nhiều nhất hay không.

### 9.3. Các phép kiểm tra dịch chuyển

| Phép kiểm tra | Dùng cho | Ưu điểm | Nhược điểm |
|---|---|---|---|
| Kolmogorov–Smirnov hai mẫu | biến liên tục, một chiều | không giả định dạng phân phối, có p-value | chỉ một chiều; với $n$ rất lớn, những khác biệt không đáng kể cũng có ý nghĩa thống kê |
| Khi bình phương | biến rời rạc hoặc đã chia bin | có p-value, lý thuyết rõ ràng | phải chọn cách chia bin; kém chính xác khi có bin ít mẫu |
| PSI | biến đã chia bin | một con số dễ báo cáo | không có p-value; ngưỡng là quy ước (Mục 9.5) |
| MMD (maximum mean discrepancy) | dữ liệu nhiều chiều | xử lý được vector, kiểm định bằng hoán vị | tốn tính toán, phải chọn hàm kernel |
| Bộ phân loại miền (domain classifier) | dữ liệu nhiều chiều | trực quan: nếu phân biệt được dữ liệu cũ và mới thì đã có dịch chuyển; AUC đo mức độ | phải huấn luyện thêm một mô hình |

Rabanser và cộng sự (2019) so sánh có hệ thống các phương pháp này và thấy một cách đơn giản cho kết quả tốt: áp dụng kiểm định KS lên **đầu ra của mô hình** (xác suất dự đoán cho từng lớp) thay vì lên từng đặc trưng đầu vào, kèm hiệu chỉnh Bonferroni. Đầu ra có ít chiều hơn đầu vào nhiều và tập trung đúng vào những thay đổi ảnh hưởng tới dự đoán. Mục 10.3 dùng cùng ý này.

Hai lưu ý thực hành:

- **Kiểm định trên nhiều đặc trưng phải hiệu chỉnh cho kiểm định nhiều lần.** Chạy KS trên 200 đặc trưng ở mức $\alpha = 0{,}01$ thì trung bình có 2 đặc trưng báo động ở mỗi lần kiểm tra dù không có gì thay đổi. Cần hiệu chỉnh (Bonferroni, Benjamini–Hochberg) hoặc dùng một phép kiểm tra nhiều chiều.
- **Nhiều dịch chuyển nằm ở quan hệ giữa các đặc trưng, không ở từng đặc trưng.** Mỗi đặc trưng vẫn giữ phân phối cũ nhưng tương quan giữa chúng thay đổi; chỉ phép kiểm tra nhiều chiều hoặc bộ phân loại miền phát hiện được.

### 9.4. PSI

PSI (population stability index) được định nghĩa trên hai phân phối đã chia thành $k$ bin. Gọi $B_j$ là tỉ lệ mẫu rơi vào bin $j$ trong tập tham chiếu và $T_j$ là tỉ lệ đó trong tập hiện tại:

$$\mathrm{PSI} \;=\; \sum_{j=1}^{k} (T_j - B_j)\,\ln\frac{T_j}{B_j}.$$

Tách tổng thành hai phần:

$$\mathrm{PSI} \;=\; \sum_j T_j \ln\frac{T_j}{B_j} \;+\; \sum_j B_j \ln\frac{B_j}{T_j} \;=\; D_{\mathrm{KL}}(T \,\|\, B) + D_{\mathrm{KL}}(B \,\|\, T).$$

Vậy PSI là tổng hai chiều của phân kỳ KL, còn gọi là phân kỳ Jeffreys. Khác với phân kỳ KL, PSI đối xứng giữa hai phân phối. Vì có $\ln(T_j/B_j)$, PSI không xác định khi một bin không có mẫu nào; các cài đặt thường thay tỉ lệ 0 bằng một số dương rất nhỏ.

PSI xuất phát từ lĩnh vực chấm điểm tín dụng và được dùng rộng rãi ở đó, kèm bộ ngưỡng quy ước sau:

| PSI | Diễn giải theo quy ước |
|---|---|
| < 0,10 | phân phối ổn định |
| 0,10 tới 0,25 | có dịch chuyển nhỏ, nên xem xét |
| > 0,25 | dịch chuyển đáng kể, cần hành động |

### 9.5. Ngưỡng PSI phụ thuộc cỡ mẫu và số bin

Ba ngưỡng trên là quy tắc kinh nghiệm, không phải kết quả thống kê. Yurdakul và Naranjo (2020) chỉ ra rằng khi hai mẫu cùng phân phối, PSI nhân với một hệ số phụ thuộc cỡ mẫu có phân phối tiệm cận khi bình phương, và đề xuất dùng giá trị tới hạn tính từ phân phối đó thay cho quy tắc kinh nghiệm, vì giá trị tới hạn phụ thuộc cả cỡ mẫu lẫn số bin.

Có thể thấy điều này bằng mô phỏng. Lấy hai mẫu từ **cùng một** phân phối chuẩn, tức không có dịch chuyển nào, chia bin theo phân vị của mẫu tham chiếu, rồi tính PSI; lặp lại 300 lần và lấy trung bình:

![Hình 11](figs/mlops11_psi_null.png)

**Hình 11.** PSI trung bình khi hai mẫu cùng phân phối, theo cỡ mẫu, với 5, 10 và 20 bin. Đường chấm là giá trị lý thuyết $2(k-1)/n$; hai đường ngang là các ngưỡng 0,10 và 0,25.

| $n$ mỗi mẫu | $k = 5$ | $k = 10$ | $k = 20$ | Lý thuyết $2(k-1)/n$, $k = 10$ |
|---|---|---|---|---|
| 200 | 0,0399 | 0,0943 | 0,2050 | 0,0900 |
| 500 | 0,0164 | 0,0353 | 0,0795 | 0,0360 |
| 1 000 | 0,0080 | 0,0179 | 0,0381 | 0,0180 |
| 5 000 | 0,0015 | 0,0037 | 0,0076 | 0,0036 |
| 20 000 | 0,0004 | 0,0009 | 0,0019 | 0,0009 |

**Vì sao PSI trung bình xấp xỉ $2(k-1)/n$.** Khi hai mẫu cùng phân phối, $T_j$ và $B_j$ đều gần xác suất thật $p_j$ của bin $j$, và chênh lệch giữa chúng có độ lớn cỡ $1/\sqrt n$. Dùng xấp xỉ $\ln(T_j/B_j) \approx (T_j - B_j)/p_j$:

$$\mathrm{PSI} \approx \sum_j \frac{(T_j - B_j)^2}{p_j}.$$

Với hai mẫu độc lập cùng cỡ $n$, $\operatorname{Var}(T_j - B_j) = 2p_j(1-p_j)/n$, nên $\tfrac{n}{2}\,\mathrm{PSI}$ có phân phối xấp xỉ khi bình phương với $k-1$ bậc tự do, đúng với kết quả tiệm cận nêu trên. Lấy kỳ vọng:

$$\mathbb{E}[\mathrm{PSI} \mid \text{không có dịch chuyển}] \;\approx\; \frac{2(k-1)}{n}.$$

Hai hệ quả:

- Với $n = 200$ và 20 bin, PSI trung bình là 0,205 **khi không có dịch chuyển nào**, rất gần ngưỡng 0,25.
- Với $n = 20\,000$, PSI trung bình chỉ là 0,0009; ngưỡng 0,25 khó đạt tới ngay cả khi có dịch chuyển thật.

Cùng một ngưỡng PSI vì thế vừa gây báo động giả ở mẫu nhỏ, vừa bỏ sót dịch chuyển ở mẫu lớn. Thí nghiệm tiếp theo đo trực tiếp điều này: với mỗi cỡ mẫu $n$ và mỗi độ dịch chuyển $\delta$ (mẫu mới có trung bình dịch đi $\delta$ lần độ lệch chuẩn), lặp lại 400 lần và đếm tỉ lệ số lần mỗi quy tắc báo động. Quy tắc thứ nhất là PSI lớn hơn 0,25 với 10 bin; quy tắc thứ hai là kiểm định KS với $p < 0{,}01$.

![Hình 12](figs/mlops12_detect.png)

**Hình 12.** Tỉ lệ báo động theo độ dịch chuyển thật $\delta$. Trái: quy tắc PSI > 0,25. Phải: kiểm định KS với $p < 0{,}01$.

Mỗi ô ghi tỉ lệ báo động của quy tắc PSI, rồi của kiểm định KS:

| $n$ | $\delta = 0$ (không dịch chuyển) | $\delta = 0{,}1$ | $\delta = 0{,}3$ |
|---|---|---|---|
| 50 | 73% / 0% | 78% / 1% | 82% / 5% |
| 100 | 24% / 1% | 24% / 2% | 54% / 24% |
| 200 | 0% / 0% | 0% / 5% | 17% / 49% |
| 1 000 | 0% / 2% | 0% / 22% | 0% / 100% |
| 10 000 | 0% / 0% | 0% / 100% | 0% / 100% |

Cột $\delta = 0$ là tỉ lệ báo động giả. Với $n = 50$, quy tắc PSI báo động 73% số lần dù không có gì thay đổi; kiểm định KS giữ tỉ lệ báo động giả quanh mức danh nghĩa 1% ở mọi cỡ mẫu. Hai cột còn lại là độ nhạy. Với $n = 1\,000$ và dịch chuyển thật $0{,}3$ độ lệch chuẩn, PSI không báo động lần nào, trong khi KS báo động ở cả 400 lần. Ở cỡ mẫu nhỏ, tỉ lệ báo động của PSI hầu như không phụ thuộc vào việc có dịch chuyển hay không: 73% khi không có dịch chuyển, 78% với $\delta = 0{,}1$.

**Nên làm gì thay thế.** Theo thứ tự ưu tiên:

1. **Dùng kiểm định có p-value** (KS, khi bình phương, MMD với kiểm định hoán vị) và hiệu chỉnh cho kiểm định nhiều lần. Ngưỡng khi đó tự điều chỉnh theo $n$.
2. **Nếu buộc phải dùng PSI** (vì quy định của ngành hoặc vì báo cáo đã quen dùng), tính ngưỡng theo đúng $n$ và $k$ của mình. Với hai mẫu cùng cỡ $n$, dùng kết quả ở trên: ngưỡng ở mức ý nghĩa $\alpha$ là $\tfrac{2}{n}\,\chi^2_{k-1,\,1-\alpha}$. Ví dụ với $k = 10$ và $\alpha = 0{,}01$, $\chi^2_{9;\,0{,}99} = 21{,}67$, nên ngưỡng là 0,217 khi $n = 200$, 0,043 khi $n = 1\,000$ và 0,0043 khi $n = 10\,000$. Cũng có thể xác định ngưỡng bằng mô phỏng: lấy nhiều cặp mẫu từ dữ liệu tham chiếu, tính PSI, và dùng phân vị 99% làm ngưỡng.
3. **Giữ cố định $n$ và $k$ giữa các lần đo.** Nếu cỡ lô thay đổi theo ngày, PSI của hai ngày không so sánh được với nhau, dù không có dịch chuyển nào.

### 9.6. Chọn cửa sổ thời gian

Mọi phép kiểm tra dịch chuyển đều so sánh một cửa sổ **hiện tại** với một tập **tham chiếu**, và cả hai đều phải được chọn:

| Lựa chọn | Đánh đổi |
|---|---|
| Cửa sổ hiện tại ngắn (một giờ) | phát hiện nhanh, nhưng nhiễu lớn và nhiều báo động giả |
| Cửa sổ hiện tại dài (một tuần) | ổn định, nhưng thay đổi đột ngột bị pha loãng và phát hiện muộn |
| Tham chiếu cố định (tập huấn luyện) | trả lời đúng câu hỏi "dữ liệu đã khác dữ liệu huấn luyện tới đâu", nhưng tiếp tục báo động cả khi dịch chuyển đã ổn định thành trạng thái bình thường mới |
| Tham chiếu trượt (30 ngày gần nhất) | thích ứng với trạng thái bình thường mới, nhưng không phát hiện được dịch chuyển chậm, vì tham chiếu trôi theo dữ liệu |

Hàng cuối là cái bẫy: tham chiếu trượt làm dịch chuyển chậm trở nên không thấy được, trong khi Chương 11 cho thấy dịch chuyển chậm có thể gây thiệt hại lớn. Cách làm hợp lý là dùng cả hai: tham chiếu cố định để biết dữ liệu đã cách xa dữ liệu huấn luyện bao nhiêu, tham chiếu trượt để phát hiện thay đổi đột ngột.

Một cái bẫy khác là **tính mùa vụ**. Lưu lượng thứ Bảy khác thứ Ba, tháng Chạp khác tháng Ba. So dữ liệu thứ Bảy với tham chiếu tính trên các ngày thường sẽ báo động đều đặn mỗi tuần mà không có ý nghĩa gì. Cách xử lý tối thiểu là so sánh cùng kỳ: thứ Bảy với các thứ Bảy trước.

---

## 10. Giám sát và cảnh báo

Chương 9 trình bày cách phát hiện dịch chuyển. Chương này đặt các phép kiểm tra đó vào một hệ thống giám sát hoàn chỉnh: phân biệt hai loại chỉ số, bốn lớp giám sát xếp theo khoảng cách tới nhãn, ba cơ chế giám sát do Sculley và cộng sự đề xuất, và cách thiết kế cảnh báo để tránh báo động giả.

### 10.1. Chỉ số vận hành và chỉ số riêng của học máy

| Loại | Ví dụ | Khi có sự cố, có biết ngay không |
|---|---|---|
| Chỉ số vận hành | độ trễ, lưu lượng, tỉ lệ lỗi, mức sử dụng CPU, GPU, bộ nhớ | có |
| Chỉ số riêng của học máy | độ chính xác, phân phối dự đoán, phân phối đặc trưng, chất lượng dữ liệu đầu vào | thường là không |

Chỉ số vận hành giống với mọi dịch vụ phần mềm khác. Sách *Site Reliability Engineering* của Google (Beyer và cộng sự, 2016) gọi độ trễ, lưu lượng, tỉ lệ lỗi và mức bão hoà tài nguyên là bốn tín hiệu cơ bản (four golden signals), và định nghĩa hai khái niệm dùng trong chương này: **chỉ số mức dịch vụ** (SLI), một đại lượng đo được như p99 độ trễ, và **mục tiêu mức dịch vụ** (SLO), giá trị mục tiêu cho SLI đó, như "p99 dưới 200 ms trong 99,9% số phút". Phần còn lại của chương tập trung vào loại chỉ số thứ hai, loại chỉ có ở hệ thống học máy.

### 10.2. Bốn lớp giám sát

Xếp theo khoảng cách tới nhãn: càng gần nhãn thì càng có ý nghĩa, nhưng càng khó có được.

| Lớp | Giám sát gì | Có ngay không | Mức ý nghĩa |
|---|---|---|---|
| 1. Đầu vào thô | dữ liệu tới có đủ số lượng, đúng giờ, đúng lược đồ không | có | thấp: chỉ phát hiện lỗi cứng |
| 2. Đặc trưng | phân phối từng đặc trưng, tỉ lệ thiếu, giá trị danh mục lạ | có | trung bình: phát hiện covariate shift và lệch huấn luyện–phục vụ |
| 3. Dự đoán | phân phối của $\hat y$, tỉ lệ từng lớp, độ tự tin trung bình | có | khá: phát hiện nhiều sự cố mà không cần nhãn |
| 4. Chất lượng | độ chính xác, AUC, chỉ số sản phẩm | chỉ khi có nhãn | cao nhất, nhưng thường đến muộn |

Nguyên tắc: cảnh báo dựa trên lớp 4 khi có thể; dùng lớp 1 tới 3 để điều tra, và để cảnh báo khi lớp 4 chưa có. Với hệ thống có nhãn đến chậm, lớp 3 là lớp giám sát chính.

### 10.3. Độ lệch dự đoán

Sculley và cộng sự (2015) đề xuất một phép kiểm tra đơn giản nhưng hữu ích:

> **Định nghĩa 10.1 (Độ lệch dự đoán).** Trong một hệ thống hoạt động đúng, phân phối của các nhãn được dự đoán thường phải bằng phân phối của các nhãn quan sát được. **Độ lệch dự đoán** (prediction bias) là chênh lệch giữa hai phân phối đó.

Bài báo nêu rõ giới hạn: đây không phải một phép kiểm tra đầy đủ, vì một mô hình chỉ luôn dự đoán tỉ lệ trung bình cũng có độ lệch dự đoán bằng 0. Nhưng một thay đổi đột ngột của đại lượng này thường là dấu hiệu của sự cố.

Ưu điểm của nó là không cần nhãn cho từng dự đoán, chỉ cần biết tỉ lệ nền theo thời gian. Nếu tỉ lệ gian lận trong lịch sử là 1,2% mà mô hình đột nhiên đánh dấu 4,5% số giao dịch, có vấn đề cần điều tra, dù chưa có nhãn nào cho ngày hôm nay.

Nên tính độ lệch dự đoán theo lát cắt: theo thị trường, theo kênh, theo phiên bản ứng dụng. Độ lệch tổng thể bằng 0 vẫn có thể che giấu hai lát cắt lệch theo hai chiều ngược nhau.

### 10.4. Giới hạn hành động và giám sát hệ thống phía trước

Bài báo của Sculley và cộng sự đề xuất thêm hai cơ chế.

**Giới hạn hành động** (action limits). Với hệ thống thực hiện hành động thật, như đặt giá thầu quảng cáo, chặn tin nhắn hay khoá tài khoản, cần đặt và thực thi giới hạn cho các hành động như một phép kiểm tra hợp lý. Giới hạn phải đủ rộng để không kích hoạt khi hệ thống hoạt động bình thường; khi chạm giới hạn thì tự động cảnh báo và cần người xem xét. Cơ chế này không cần biết mô hình đúng hay sai, chỉ cần biết rằng, chẳng hạn, việc khoá 50 000 tài khoản trong một giờ là điều chưa từng xảy ra và không nên xảy ra.

**Hệ thống phía trước** (up-stream producers). Dữ liệu đi vào hệ thống học máy từ nhiều hệ thống phía trước. Các hệ thống này cần được giám sát, kiểm thử và có SLO phù hợp với nhu cầu của hệ thống học máy phía sau. Cảnh báo ở phía trước phải được chuyển tới hệ thống học máy; ngược lại, khi hệ thống học máy không đạt SLO, mọi bên sử dụng kết quả của nó phải được thông báo. Cơ chế này cũng là cách xử lý dạng nợ *undeclared consumer* ở Mục 1.4: biến các phụ thuộc ngầm thành phụ thuộc có tên, có cảnh báo theo cả hai chiều.

### 10.5. Thiết kế cảnh báo

Mục 3.2 đã nhắc tới khó khăn được nêu nhiều nhất trong nghiên cứu của Shankar và cộng sự: báo động giả. Hậu quả của báo động giả không chỉ là phiền toái mà là mất niềm tin: khi đa số cảnh báo là giả, người nhận bỏ qua cả những cảnh báo thật, và hệ thống cảnh báo khi đó tạo ra cảm giác an toàn không có cơ sở.

Bốn nguyên tắc thiết kế, phần lớn lấy từ kinh nghiệm vận hành phần mềm (Beyer và cộng sự, 2016):

1. **Mỗi cảnh báo phải gắn với một hành động.** Nếu người nhận không biết phải làm gì, đó là một biểu đồ, không phải một cảnh báo; đưa nó vào bảng theo dõi (dashboard) thay vì gửi thông báo.
2. **Phân mức độ.** Gọi người trực ngay (page) cho lỗi cứng và dịch vụ ngừng hoạt động; tạo phiếu xử lý (ticket) cho dịch chuyển và suy giảm chậm; phần còn lại đưa vào bảng theo dõi.
3. **Cảnh báo theo triệu chứng, không theo nguyên nhân.** "Tỉ lệ chuyển đổi giảm 15%" là triệu chứng đáng gọi người trực. "PSI của đặc trưng thứ 37 vượt 0,2" là một manh mối để điều tra, không phải một sự cố.
4. **Yêu cầu tín hiệu kéo dài.** Một điểm dữ liệu vượt ngưỡng có thể là nhiễu; vượt ngưỡng trong ba cửa sổ liên tiếp mới là tín hiệu.

### 10.6. Ngưỡng cố định trong hệ thống thay đổi

Sculley và cộng sự đặt tên cho một dạng nợ rất hay gặp: **ngưỡng cố định trong hệ thống thay đổi** (fixed thresholds in dynamic systems). Nhiều mô hình cần một ngưỡng quyết định, ví dụ để xếp một email là thư rác hay không. Ngưỡng thường được đặt bằng tay để cân bằng precision và recall. Khi mô hình được huấn luyện lại trên dữ liệu mới, phân phối điểm số của nó thay đổi, và ngưỡng cũ có thể không còn phù hợp.

Cập nhật ngưỡng bằng tay cho nhiều mô hình vừa tốn công vừa dễ quên. Cách xử lý mà bài báo đề xuất là xác định ngưỡng tự động trên một tập kiểm định riêng mỗi lần huấn luyện, và coi ngưỡng là một phần của mô hình (lưu cùng mô hình trong sổ đăng ký) thay vì một hằng số trong cấu hình của dịch vụ.

Đây cũng là một ví dụ của CACE: huấn luyện lại làm thay đổi phân phối điểm số, nên mọi thứ phụ thuộc vào điểm số đó cũng thay đổi theo, kể cả một con số trong tệp cấu hình mà không ai coi là một phần của mô hình.

---

## 11. Huấn luyện lại

Chương này trả lời ba câu hỏi: mô hình cũ đi nhanh tới mức nào, khi nào nên huấn luyện lại, và huấn luyện lại từ đầu hay cập nhật tiếp mô hình cũ. Phần cuối trình bày nguyên tắc quan trọng nhất khi tự động hoá: pipeline huấn luyện lại phải qua đủ các bước kiểm định như một lần phát hành thủ công.

### 11.1. Mô hình cũ đi nhanh tới mức nào

Trước khi quyết định bao lâu huấn luyện lại một lần, cần biết cái giá của việc không huấn luyện lại. ML Test Score có hai mục cho vấn đề này: Model 4 (đã biết ảnh hưởng của việc mô hình cũ đi) và Monitor 4 (mô hình không quá cũ).

**Thí nghiệm.** Thí nghiệm trong `code/mlops/experiments.py` mô phỏng một thế giới có concept drift chậm: dữ liệu hai chiều, biên quyết định thật xoay 0,035 radian (khoảng $2^\circ$) mỗi tuần, trong 52 tuần. Mỗi tuần có 4 000 mẫu mới. Mô hình ban đầu là hồi quy logistic huấn luyện trên dữ liệu tuần 0; khi tới lịch huấn luyện lại, mô hình mới được huấn luyện trên dữ liệu của tuần vừa qua. So sánh bốn nhịp huấn luyện lại:

![Hình 13](figs/mlops13_staleness.png)

**Hình 13.** Độ chính xác theo tuần của cùng một mô hình trong cùng một thế giới, chỉ khác nhịp huấn luyện lại.

| Nhịp huấn luyện lại | Số lần huấn luyện lại trong năm | Độ chính xác trung bình | Thấp nhất | Tuần 52 |
|---|---|---|---|---|
| Không bao giờ | 0 | 64,61% | 45,42% | 45,48% |
| Mỗi quý (12 tuần) | 4 | 76,89% | 74,50% | 78,25% |
| Mỗi tháng (4 tuần) | 13 | 77,68% | 75,80% | 78,25% |
| Mỗi tuần | 52 | 77,80% | 75,72% | 78,72% |

Ba nhận xét:

1. **Không huấn luyện lại thì mất 13,19 điểm phần trăm độ chính xác trung bình** trong một năm. Tới tuần 52, biên quyết định đã xoay hơn $90^\circ$, và mô hình chỉ còn đúng 45,5%, thấp hơn đoán ngẫu nhiên trên bài toán hai lớp cân bằng này. Độ chính xác giảm dần từng chút, nên không có tuần nào có một thay đổi đột ngột đủ để kích hoạt cảnh báo theo ngưỡng.
2. **Lợi ích giảm dần nhanh.** Từ không huấn luyện lại lên mỗi quý tăng 12,3 điểm; từ mỗi quý lên mỗi tuần chỉ tăng thêm 0,91 điểm, trong khi số lần huấn luyện tăng 13 lần.
3. **Nhịp huấn luyện lại là một quyết định kinh tế.** Nó được xác định bằng cách đo đường cong như Hình 13 trên chính bài toán của mình, rồi so sánh giá trị của một điểm phần trăm độ chính xác với chi phí của một lần huấn luyện, kiểm định và triển khai.

Các con số trên thuộc về mô phỏng này: dịch chuyển đều và chậm. Với những thay đổi đột ngột, như một sự kiện làm hành vi người dùng thay đổi trong vài ngày, huấn luyện lại theo lịch cố định phản ứng chậm, và cần thêm các loại kích hoạt ở Mục 11.2.

### 11.2. Bốn loại kích hoạt

| Kích hoạt | Cơ chế | Ưu điểm | Nhược điểm |
|---|---|---|---|
| Theo lịch | định kỳ: mỗi tuần, mỗi tháng | đơn giản, dễ dự trù tài nguyên | huấn luyện lại không cần thiết khi mọi thứ ổn định; quá muộn khi có thay đổi đột ngột |
| Theo chất lượng | chỉ số chất lượng giảm quá ngưỡng | phản ứng đúng lúc cần | cần nhãn, nên bị giới hạn bởi độ trễ nhãn |
| Theo dịch chuyển | phép kiểm tra ở Chương 9 báo động | không cần nhãn | phát hiện được covariate shift, không phát hiện được concept drift (Mục 9.2) |
| Theo lượng dữ liệu | có đủ $N$ mẫu mới | phù hợp khi dữ liệu tới không đều | không liên quan tới việc mô hình còn tốt hay không |

Cấu hình thường gặp là huấn luyện lại theo lịch làm nền, cộng thêm kích hoạt theo chất lượng hoặc theo dịch chuyển để huấn luyện lại sớm khi cần. Shankar và cộng sự ghi nhận rằng nhiều nhóm chọn cách đơn giản: huấn luyện lại thường xuyên trên dữ liệu mới nhất, thay vì xây dựng một cơ chế kích hoạt phức tạp.

### 11.3. Huấn luyện lại từ đầu hay cập nhật mô hình cũ

Huyen (2022) phân biệt hai cách: **huấn luyện lại từ đầu** (stateless retraining), mỗi lần huấn luyện một mô hình mới trên một cửa sổ dữ liệu; và **huấn luyện có trạng thái** (stateful training), tiếp tục cập nhật mô hình hiện có chỉ bằng dữ liệu mới.

| | Huấn luyện lại từ đầu | Cập nhật mô hình cũ |
|---|---|---|
| Dữ liệu dùng | toàn bộ cửa sổ dữ liệu lịch sử | chỉ dữ liệu mới |
| Chi phí mỗi lần | cao | thấp |
| Khả năng tái lập | tốt: dựng lại được từ dữ liệu, mã và hạt giống | kém: trạng thái mô hình phụ thuộc toàn bộ chuỗi cập nhật trước đó |
| Quay lui | dễ: nạp lại mô hình cũ | khó: phải phát lại lịch sử cập nhật |
| Rủi ro riêng | | quên thảm khốc (catastrophic forgetting); một lô dữ liệu lỗi ảnh hưởng tới mọi phiên bản sau |

Nên mặc định huấn luyện lại từ đầu, và chỉ chuyển sang cập nhật mô hình cũ khi có ràng buộc thật: dữ liệu quá lớn để huấn luyện lại toàn bộ, hoặc cần thích ứng trong vài phút. Lý do nằm ở hai hàng giữa của bảng: khả năng tái lập và khả năng quay lui chỉ thể hiện giá trị khi có sự cố, nhưng khi đó thì rất cần.

### 11.4. Pipeline huấn luyện lại phải qua đủ các bước kiểm định

Sai lầm nguy hiểm nhất khi tự động hoá huấn luyện lại là để pipeline bỏ qua các bước kiểm định. Một pipeline huấn luyện lại tự động phải qua đúng những bước mà một lần phát hành thủ công phải qua:

```
dữ liệu mới
  → kiểm định dữ liệu              (Chương 3; nếu lỗi thì dừng, không huấn luyện)
  → huấn luyện
  → đánh giá trên tập kiểm tra cố định, theo lát cắt   (Chương 6)
  → so sánh với mô hình đang chạy  ← bước hay bị bỏ qua nhất
  → kiểm thử hành vi               (Mục 6.4)
  → đăng ký vào sổ đăng ký, trạng thái staging
  → shadow → canary → toàn bộ      (Chương 8)
```

Bước so sánh với mô hình đang chạy cần được nói riêng. Tài liệu của Google Cloud tách bước **kiểm định mô hình** (model validation) khỏi bước **đánh giá mô hình** (model evaluation) trong tám bước ở Mục 2.3 vì lý do này: đánh giá trả lời câu hỏi "mô hình này tốt tới đâu", còn kiểm định trả lời câu hỏi "có nên thay mô hình đang chạy bằng mô hình này không". Một mô hình huấn luyện lại tự động mà kém hơn mô hình đang chạy, trên tổng thể hoặc trên một lát cắt quan trọng, phải bị chặn tự động.

Vì mọi bước đều tự động, cơ chế quay lui cũng phải tự động: đó là mục Infra 7 của ML Test Score, có thể quay lui mô hình đang phục vụ nhanh và an toàn.

---

## 12. Vòng phản hồi

Trong nhiều hệ thống, dự đoán của mô hình ảnh hưởng tới dữ liệu mà mô hình sẽ được huấn luyện trong tương lai. Chương này phân loại các vòng phản hồi, trình bày cơ chế của vòng phản hồi thoái hoá trong hệ thống gợi ý, đo thiệt hại bằng mô phỏng, và nêu các cách phát hiện và khắc phục.

### 12.1. Hai loại vòng phản hồi

Sculley và cộng sự (2015) phân biệt:

- **Vòng phản hồi trực tiếp** (direct feedback loop): mô hình ảnh hưởng trực tiếp tới việc chọn dữ liệu huấn luyện tương lai của chính nó. Bài báo ghi nhận rằng về lý thuyết, cách giải đúng là dùng thuật toán bandit (ví dụ contextual bandit), nhưng các thuật toán này không phải lúc nào cũng mở rộng được tới kích thước không gian hành động của bài toán thực tế. Các cách giảm nhẹ khả thi là thêm một mức ngẫu nhiên hoá, hoặc tách riêng một phần dữ liệu không chịu ảnh hưởng của mô hình.
- **Vòng phản hồi ẩn** (hidden feedback loop): hai hệ thống ảnh hưởng lẫn nhau một cách gián tiếp, thông qua thế giới bên ngoài. Ví dụ trong bài báo: hai hệ thống cùng quyết định nội dung của một trang web, một hệ thống chọn sản phẩm để hiển thị, một hệ thống chọn đánh giá liên quan. Cải thiện hệ thống này làm người dùng nhấp nhiều hơn hoặc ít hơn vào phần của hệ thống kia, và thay đổi dữ liệu huấn luyện của nó. Loại này khó phát hiện hơn nhiều, vì không có đường dữ liệu nào trong sơ đồ hệ thống cho thấy mối liên hệ.

### 12.2. Vòng phản hồi thoái hoá

Trường hợp quan trọng nhất trong thực tế là **vòng phản hồi thoái hoá** (degenerate feedback loop; Huyen, 2022): dự đoán ảnh hưởng tới phản hồi của người dùng, phản hồi được dùng làm nhãn để huấn luyện lần sau, nên hệ thống tự củng cố những gì nó đã tin.

Cơ chế trong một hệ thống gợi ý:

1. Món A tình cờ được xếp cao hơn món B ở vòng đầu, dù chất lượng thật tương đương.
2. A được hiển thị nhiều hơn nên nhận nhiều lượt nhấp hơn, chỉ vì được nhìn thấy nhiều hơn.
3. Lần huấn luyện sau thấy A có nhiều lượt nhấp, nên xếp A cao hơn nữa.
4. B không được hiển thị nữa, nên không có dữ liệu nào cho thấy B cũng tốt.

Vấn đề là dữ liệu vẫn trông tốt: tỉ lệ nhấp trên những gì được hiển thị vẫn cao, vì hệ thống chỉ hiển thị những gì nó tin là tốt, và các chỉ số ngoại tuyến tính trên dữ liệu log không cho thấy vấn đề.

### 12.3. Thí nghiệm: đo thiệt hại của vòng phản hồi

**Thiết lập.** Mô phỏng trong `code/mlops/experiments.py` dùng một danh mục 2 000 món. Chất lượng thật của mỗi món, tức xác suất người dùng nhấp khi thấy nó, lấy từ phân phối Beta(1,6; 9): phần lớn món bình thường, ít món thật sự tốt; hệ thống không biết các giá trị này. Hệ thống khởi động bằng 200 lượt hiển thị ngẫu nhiên. Sau đó, ở mỗi vòng trong 30 vòng, hệ thống xếp các món theo tỉ lệ nhấp đã quan sát, chọn 10 món đầu, hiển thị cho 3 000 người dùng, rồi cập nhật số liệu từ log nhấp của chính nó. Chiến lược ε thay $\varepsilon \times 10$ vị trí trong danh sách bằng các món chọn ngẫu nhiên từ toàn bộ danh mục. Kết quả là trung bình của 20 lần chạy.

![Hình 14](figs/mlops14_feedback.png)

**Hình 14.** Qua 30 vòng huấn luyện lại: tỉ lệ danh mục từng được hiển thị (trái), chất lượng thật của 10 món xếp đầu (giữa), và tỉ lệ của nhóm 1% món tốt nhất đã từng được hiển thị (phải). Không ngẫu nhiên hoá, hệ thống chỉ hiển thị những món đã gặp ở vòng khởi động.

| Chiến lược | Tỉ lệ danh mục từng được hiển thị | Chất lượng thật của 10 món xếp đầu | Tỉ lệ nhóm 1% tốt nhất từng được hiển thị |
|---|---|---|---|
| Không ngẫu nhiên hoá | 9,5% | 0,351 | 8% |
| ε = 10% | 10,8% | 0,384 | 10% |
| ε = 30% | 13,4% | 0,402 | 12% |

Để so sánh: chất lượng trung bình của 10 món tốt nhất thật sự là 0,555, và chất lượng trung bình của toàn danh mục là 0,153.

Đọc bảng:

- Không ngẫu nhiên hoá, hệ thống đạt chất lượng 0,351, khoảng 63% mức tối đa, và chỉ bao giờ hiển thị 9,5% danh mục, đúng những món đã xuất hiện trong 200 lượt khởi động. Những món chưa từng được hiển thị có tỉ lệ nhấp quan sát bằng 0 nên không bao giờ vào được danh sách. Hệ thống không hỏng; nó chỉ không thể tốt hơn.
- Dành 30% vị trí cho các món ngẫu nhiên nâng tỉ lệ danh mục được hiển thị từ 9,5% lên 13,4% (tăng 41% theo tỉ lệ tương đối) và nâng chất lượng của danh sách từ 0,351 lên 0,402 (tăng khoảng 15%).
- Cái giá là 30% số lượt hiển thị dành cho các món ngẫu nhiên, có chất lượng trung bình chỉ 0,153: hy sinh kết quả ngắn hạn để có thông tin cho dài hạn. Đây là đánh đổi giữa khai thác và khám phá (exploitation–exploration), và chính là "thêm một mức ngẫu nhiên hoá" mà Sculley và cộng sự đề xuất.
- Mức cải thiện nhỏ so với khoảng cách tới mức tối đa. Với 30 vòng và chỉ 1 tới 3 vị trí ngẫu nhiên mỗi vòng, khám phá ngẫu nhiên đi rất chậm: sau 30 vòng, chiến lược ε = 30% mới gặp 12% nhóm món tốt nhất. Các thuật toán bandit như UCB hay Thompson sampling hướng việc khám phá vào những món chưa chắc chắn thay vì chọn đều, nên hiệu quả hơn nhiều.

### 12.4. Phát hiện và khắc phục

**Dấu hiệu.** Không dấu hiệu nào nằm trong các chỉ số chất lượng thông thường:

- **Độ đa dạng giảm dần** qua các vòng: entropy của phân phối hiển thị, số món khác nhau được hiển thị, tỉ lệ danh mục được hiển thị.
- **Phân phối hiển thị lệch dần** về một nhóm nhỏ: tỉ trọng của 1% món được hiển thị nhiều nhất tăng đều.
- **Món mới không bao giờ nổi lên**: món mới thêm vào danh mục không vào được danh sách đầu.

**Cách khắc phục**, theo thứ tự chi phí tăng dần:

1. **Ngẫu nhiên hoá một phần vị trí hiển thị** (ε-greedy), hoặc dùng thuật toán bandit. Cách rẻ nhất; thí nghiệm ở Mục 12.3 cho thấy tác dụng và giới hạn của nó.
2. **Giữ một nhóm đối chứng không chịu ảnh hưởng của mô hình.** Một tỉ lệ nhỏ lưu lượng nhận kết quả ngẫu nhiên hoặc theo quy tắc đơn giản. Đây là nguồn dữ liệu không bị mô hình làm sai lệch, và là cách để đo mô hình thật sự tốt tới đâu.
3. **Hiệu chỉnh theo xác suất hiển thị** (inverse propensity weighting): khi huấn luyện, gán cho mỗi mẫu trọng số bằng nghịch đảo xác suất nó được hiển thị, để bù cho việc một số món được hiển thị nhiều hơn (Joachims và cộng sự, 2017). Cách này đòi hỏi ghi lại xác suất hiển thị tại thời điểm hiển thị; nếu không ghi lại từ đầu thì không thể khôi phục về sau.
4. **Đưa vị trí hiển thị vào mô hình:** khi huấn luyện, thêm đặc trưng "vị trí hiển thị" để mô hình tách được phần lượt nhấp do vị trí gây ra; khi phục vụ, đặt vị trí bằng cùng một hằng số cho mọi món. Đây là cách thường dùng để giảm thiên lệch vị trí.

Cách thứ hai đáng được ưu tiên: một nhóm đối chứng nhỏ, duy trì thường xuyên, vừa cung cấp dữ liệu không bị sai lệch để huấn luyện, vừa cho một thước đo đáng tin cậy về giá trị thật của hệ thống.

---

## 13. LLMOps

Khi hệ thống dùng mô hình ngôn ngữ lớn, phần lớn các nguyên tắc ở mười hai chương trước vẫn áp dụng, nhưng một số ràng buộc thay đổi. Chương này nêu những thay đổi đó rồi đi qua các vấn đề vận hành riêng: đánh giá bằng giám khảo LLM và các thiên lệch của nó, đánh giá hệ thống RAG, chi phí và độ trễ, rào chắn, và phiên bản hoá. Giáo trình [*Ứng dụng LLM*](ungdung-ch01.html) trình bày chi tiết về phía xây dựng ứng dụng; chương này tập trung vào phía vận hành.

### 13.1. Những gì thay đổi

Vẫn cần hợp đồng dữ liệu, quản lý phiên bản, shadow và canary, giám sát và quay lui. Có năm ràng buộc thay đổi:

| Ràng buộc | MLOps cho mô hình tự huấn luyện | LLMOps |
|---|---|---|
| Huấn luyện | tự huấn luyện mô hình | thường dùng mô hình của bên thứ ba; thay cho huấn luyện lại là thay prompt, ngữ cảnh hoặc phiên bản mô hình |
| Đầu ra | một số hoặc một lớp | văn bản tự do, không có hàm đúng sai hiển nhiên |
| Tính tất định | cố định hạt giống thì tái lập được | có yếu tố ngẫu nhiên khi lấy mẫu, và mô hình của nhà cung cấp có thể thay đổi |
| Chi phí | chủ yếu là chi phí huấn luyện, tương đối cố định | chủ yếu là chi phí suy luận theo token, thay đổi theo từng yêu cầu |
| Rủi ro | dự đoán sai | dự đoán sai, cộng thêm rò rỉ dữ liệu, nội dung có hại, prompt injection |

Hàng thứ ba đáng chú ý nhất về mặt vận hành. Với mô hình tự huấn luyện, mô hình là một tạo tác bất biến nằm trong sổ đăng ký. Với mô hình gọi qua API, hành vi có thể thay đổi khi nhà cung cấp cập nhật mô hình sau cùng một tên gọi; Chen, Zaharia và Zou (2023) đo được những thay đổi đáng kể về hành vi của cùng một tên mô hình GPT-4 giữa hai phiên bản cách nhau ba tháng. Đây là dạng cực đoan của phụ thuộc dữ liệu không ổn định ở Mục 1.4, và cách xử lý cũng tương tự: dùng định danh phiên bản cố định của mô hình (các nhà cung cấp thường có tên kèm ngày phát hành), và coi mỗi lần đổi phiên bản là một lần phát hành đầy đủ, phải qua bộ đánh giá, shadow và canary.

### 13.2. Đánh giá theo ba tầng

Vì không có hàm đúng sai hiển nhiên, đánh giá là phần tốn kém nhất của LLMOps. Thực hành phổ biến là chia thành ba tầng, đánh đổi giữa chi phí và độ tin cậy. Tỉ lệ lấy mẫu trong bảng là mức thường gặp, không phải quy chuẩn:

| Tầng | Áp dụng cho | Chi phí | Phát hiện được |
|---|---|---|---|
| Kiểm tra theo quy tắc | mọi lượt gọi | gần như bằng 0 | sai định dạng, JSON lỗi, thiếu trích dẫn, độ dài bất thường, từ khoá bị cấm |
| Giám khảo LLM (LLM-as-a-judge) | một mẫu, vài phần trăm tới vài chục phần trăm số lượt | trung bình | chất lượng về nội dung: bám nguồn, hữu ích, đúng giọng văn, từ chối đúng lúc |
| Người chấm | một mẫu nhỏ, định kỳ | cao | đánh giá chuẩn, dùng để hiệu chỉnh giám khảo LLM |

Cấu trúc này tương ứng với bốn lớp giám sát ở Mục 10.2: càng gần đánh giá thật thì càng đắt, nên phải lấy mẫu.

Tầng thứ ba hay bị bỏ qua nhất nhưng giữ cho cả hệ thống đánh giá đáng tin. Giám khảo LLM cũng là một mô hình, có sai số và có thể thay đổi theo phiên bản. Không đối chiếu định kỳ với người chấm, hệ thống đánh giá có thể báo mọi thứ đều ổn trong khi chất lượng đã giảm. [Chương 12 của *Ứng dụng LLM*](ungdung-ch12.html) trình bày chi tiết các loại đánh giá và cách tính số mẫu cần thiết.

### 13.3. Các thiên lệch của giám khảo LLM

Zheng và cộng sự (2023), khi xây dựng bộ đánh giá MT-Bench, phân tích các thiên lệch của mô hình ngôn ngữ khi làm giám khảo:

- **Thiên lệch vị trí**: khi so sánh hai câu trả lời, giám khảo có xu hướng ưu tiên một vị trí, thường là câu trả lời đặt trước. Cách xử lý: hỏi hai lần với thứ tự đảo ngược, và chỉ tính là thắng khi thắng ở cả hai lần.
- **Thiên lệch độ dài**: câu trả lời dài hơn thường được chấm cao hơn dù không tốt hơn. Cách xử lý: đưa yêu cầu về độ dài vào tiêu chí chấm, hoặc so sánh các câu trả lời có độ dài tương đương.
- **Thiên lệch tự ưu tiên**: có dấu hiệu giám khảo chấm cao hơn cho câu trả lời do chính nó sinh ra, dù các tác giả lưu ý dữ liệu chưa đủ để kết luận chắc chắn. Cách xử lý thận trọng: dùng mô hình giám khảo khác họ với mô hình đang phục vụ.
- **Tiêu chí chấm mơ hồ**: câu hỏi "câu trả lời này có tốt không" cho kết quả không ổn định. Cách xử lý: tiêu chí nhiều chiều, mỗi chiều là một câu hỏi có hoặc không, hoặc một thang điểm ngắn, kèm ví dụ mẫu.

Shankar và cộng sự (2024) ghi nhận thêm một hiện tượng khi xây dựng tiêu chí chấm: người dùng cần có tiêu chí để chấm đầu ra, nhưng chính việc đọc và chấm đầu ra lại làm họ thay đổi tiêu chí, hiện tượng mà các tác giả gọi là *criteria drift*. Hệ quả thực tế: không thể viết xong tiêu chí chấm một lần từ đầu; cần đọc một lượng đầu ra thật trước, và xem lại tiêu chí định kỳ.

Nguyên tắc chung: đo mức đồng thuận giữa giám khảo LLM và người chấm trên cùng một tập mẫu, ví dụ bằng hệ số kappa của Cohen, và coi đó là giới hạn trên cho độ tin cậy của mọi kết luận rút ra từ giám khảo, tương tự lập luận về mức đồng thuận giữa những người gán nhãn ở Mục 3.5.

### 13.4. Đánh giá ngoại tuyến và trực tuyến

Cách tổ chức phổ biến gồm hai vòng:

1. **Vòng ngoại tuyến, trên một tập đánh giá chuẩn** (golden set): một tập câu hỏi được chọn lọc, gồm cả các trường hợp khó, mà hệ thống phải đạt trước mỗi lần phát hành. Đây là bước kiểm định trong CI, tương đương bước "so sánh với mô hình đang chạy" ở Mục 11.4 ([Mục 12.5 của *Ứng dụng LLM*](ungdung-ch12.html) trình bày kiểm thử hồi quy cho prompt).
2. **Vòng trực tuyến, chấm mẫu trên lưu lượng thật:** chấm bất đồng bộ một tỉ lệ nhỏ số lượt gọi thật và gắn điểm vào bản ghi vết (trace) tương ứng ([Chương 13 của *Ứng dụng LLM*](ungdung-ch13.html)).

Tập đánh giá chuẩn phải được cập nhật liên tục. Shankar và cộng sự (2022) ghi nhận thực hành này cho học máy nói chung: tập kiểm định nên thay đổi theo thời gian, và mỗi sự cố trong sản xuất nên trở thành một trường hợp mới trong tập kiểm định. Một tập đánh giá không được cập nhật dần dần chỉ còn đo những gì hệ thống đã làm đúng.

### 13.5. Đánh giá và vận hành hệ thống RAG

Với hệ thống RAG, một câu trả lời sai có hai nguồn gốc khác nhau và cần hai cách sửa khác nhau:

| Nguồn gốc | Biểu hiện | Đo bằng | Cách sửa |
|---|---|---|---|
| Truy xuất | tài liệu cần thiết không có trong ngữ cảnh | recall@$k$, MRR trên tập câu hỏi có nhãn tài liệu đúng | cải thiện cách chia đoạn, mô hình embedding, tìm kiếm kết hợp, xếp hạng lại |
| Sinh | tài liệu cần thiết có trong ngữ cảnh nhưng câu trả lời vẫn sai | độ trung thành (faithfulness): các khẳng định trong câu trả lời có được ngữ cảnh hỗ trợ không | sửa prompt, đổi mô hình, yêu cầu trích dẫn |

Phân biệt hai nguồn gốc này là bước chẩn đoán đầu tiên cho mọi lỗi RAG. Gộp chúng vào một chỉ số duy nhất thì không biết cần sửa ở đâu. Es và cộng sự (2023), với thư viện RAGAS, đề xuất ba chỉ số theo dõi: độ trung thành của câu trả lời với ngữ cảnh, mức liên quan của câu trả lời với câu hỏi, và mức liên quan của ngữ cảnh với câu hỏi. Thí nghiệm ở [Mục 8.5 của *Ứng dụng LLM*](ungdung-ch08.html) cho một ví dụ vì sao cần theo dõi riêng bước truy xuất: ở những câu mà truy xuất không đưa về đoạn nào thuộc đúng chương, độ chính xác giảm từ 0,586 (không có ngữ cảnh) xuống 0,276 (có ngữ cảnh sai).

Ba vấn đề vận hành riêng của RAG:

- **Độ mới của chỉ mục.** Kho tài liệu thay đổi (tài liệu mới, tài liệu bị sửa hoặc xoá), nên chỉ mục phải được cập nhật, và cần giám sát độ trễ giữa lúc tài liệu thay đổi và lúc chỉ mục phản ánh thay đổi đó.
- **Đổi mô hình embedding phải lập chỉ mục lại toàn bộ.** Vector của hai mô hình embedding khác nhau không so sánh được với nhau, nên không thể trộn vector cũ và mới trong cùng một chỉ mục. Việc đổi mô hình embedding cần được lên kế hoạch như một lần di chuyển dữ liệu: lập chỉ mục mới song song, đánh giá, rồi chuyển.
- **Quyền truy cập.** Truy xuất phải lọc theo quyền của người hỏi; nếu không, hệ thống có thể trả lời bằng nội dung từ tài liệu mà người hỏi không được phép xem.

### 13.6. Chi phí và độ trễ

Đây là phần khác nhiều nhất so với MLOps cho mô hình tự huấn luyện, vì chi phí thay đổi theo từng yêu cầu. Chi phí cần được ghi lại theo từng bản ghi vết và tách thành các thành phần:

```
chi phí một lượt = (số token vào × đơn giá token vào) + (số token ra × đơn giá token ra)
                 + chi phí các lượt gọi giám khảo LLM      ← ghi thành một dòng riêng
                 + chi phí truy xuất (embedding + tìm kiếm vector)
```

Ghi riêng chi phí giám khảo giúp biết chi phí tăng vì lưu lượng tăng hay vì tăng tỉ lệ lấy mẫu đánh giá.

Ba cách giảm chi phí, theo thứ tự nên thử:

1. **Bộ nhớ đệm** (cache): lưu kết quả cho câu hỏi lặp lại chính xác, cho câu hỏi gần giống (semantic cache), và lưu đệm phần đầu cố định của prompt (prompt caching; [Mục 4.6 của *Ứng dụng LLM*](ungdung-ch04.html)).
2. **Định tuyến** (routing): câu hỏi dễ gửi tới mô hình nhỏ và rẻ, câu hỏi khó mới gửi tới mô hình lớn. Cần một bộ phân loại độ khó, và bộ phân loại này cũng là một mô hình phải được giám sát.
3. **Giảm ngữ cảnh**: bớt số đoạn truy xuất, tóm tắt lịch sử hội thoại. Chi phí tỉ lệ với số token đầu vào, và ngữ cảnh thường chiếm phần lớn số token đó: trong thí nghiệm RAG ở Mục 8.5 của *Ứng dụng LLM*, hai đoạn ngữ cảnh làm số token đầu vào tăng từ 137 lên 813.

Về độ trễ, có hai chỉ số riêng: **thời gian tới token đầu tiên** (time to first token), quyết định cảm nhận về độ nhanh khi bắt đầu trả lời, và **thời gian giữa các token**, quyết định tốc độ hiển thị câu trả lời ([Mục 2.5 của *Ứng dụng LLM*](ungdung-ch02.html)). Phân tích phần đuôi ở Mục 7.3 vẫn áp dụng: một agent gọi mô hình và công cụ qua $k$ bước nối tiếp có thời gian trả lời bằng tổng thời gian các bước, và xác suất gặp ít nhất một bước chậm vẫn là $1 - 0{,}99^k$ nếu mỗi bước chậm với xác suất 1%.

### 13.7. Rào chắn

**Rào chắn** (guardrails) là các kiểm tra chạy quanh lượt gọi mô hình, ở cả đầu vào và đầu ra:

| Ở đầu vào | Ở đầu ra |
|---|---|
| phát hiện prompt injection | kiểm tra định dạng (JSON hợp lệ, đúng lược đồ) |
| phát hiện và che dữ liệu cá nhân | phát hiện nội dung có hại |
| chặn yêu cầu ngoài phạm vi | kiểm tra bám nguồn với RAG |
| giới hạn tần suất theo người dùng | giới hạn hành động (Mục 10.4) |

Hàng cuối cần nhấn mạnh. Với agent có quyền gọi các công cụ có tác động thật, như gửi email, hoàn tiền hay sửa dữ liệu, cơ chế giới hạn hành động của Sculley và cộng sự áp dụng nguyên vẹn: đặt giới hạn số hành động mỗi loại trong mỗi khoảng thời gian, và khi chạm giới hạn thì dừng lại và chuyển cho người xử lý. Một mô hình ngôn ngữ trả lời sai mà có quyền hoàn tiền gây hậu quả lớn hơn nhiều so với một mô hình chấm điểm sai. [Chương 11 của *Ứng dụng LLM*](ungdung-ch11.html) trình bày chi tiết về prompt injection, dữ liệu nhạy cảm và rào chắn nội dung.

### 13.8. Phiên bản hoá

Yếu tố phiên bản ở Mục 2.2 áp dụng cho hệ thống LLM với một danh sách dài hơn. Một phiên bản của hệ thống LLM gồm ít nhất:

- **prompt**, gồm cả prompt hệ thống và các ví dụ few-shot ([Mục 4.7 của *Ứng dụng LLM*](ungdung-ch04.html));
- **phiên bản mô hình và tham số sinh**: nhiệt độ, top-p, số token tối đa;
- **cấu hình truy xuất**: mô hình embedding, cách chia đoạn, $k$, bộ xếp hạng lại;
- **ảnh chụp của kho tài liệu**: chỉ mục được lập tại thời điểm nào;
- **định nghĩa các công cụ** mà agent được phép gọi;
- **tiêu chí chấm và tập đánh giá chuẩn** đã dùng để duyệt phiên bản.

Thay đổi bất kỳ thành phần nào trong sáu thành phần trên là một phiên bản mới và cần được đánh giá lại. Đây là nguyên lý CACE (Mục 1.3) áp dụng cho hệ thống LLM: prompt, mô hình, kho tài liệu và công cụ phụ thuộc lẫn nhau, nên thay đổi một thành phần có thể thay đổi hành vi của cả hệ thống.

---

## 14. Bài tập

Lời giải chi tiết nằm ở trang Lời giải. Các số liệu dùng trong lời giải của Bài 1, 2, 3, 4, 7 và 8 được sinh bởi `code/mlops/bai_tap.py`.

**Bài 1 (tính tay).** Một hệ thống giám sát dịch chuyển bằng PSI với $k = 10$ bin; mỗi ngày lấy $n = 500$ mẫu cho cả tập tham chiếu và tập hiện tại.
(a) Tính PSI kỳ vọng khi không có dịch chuyển nào.
(b) Hệ thống cảnh báo khi PSI vượt 0,10 và theo dõi 200 đặc trưng. Khi không có dịch chuyển, trung bình mỗi ngày có bao nhiêu cảnh báo giả?
(c) Cần $n$ tối thiểu bằng bao nhiêu để PSI kỳ vọng khi không có dịch chuyển nhỏ hơn 0,01?
*Gợi ý: dùng kết quả $\tfrac{n}{2}\,\mathrm{PSI} \approx \chi^2_{k-1}$ ở Mục 9.5.*

**Bài 2 (tính tay).** Tỉ lệ chuyển đổi nền là 2%. Cần phát hiện một cải thiện tương đối 5% với $\alpha = 0{,}05$ hai phía và lực kiểm định 80%.
(a) Tính cỡ mẫu mỗi nhánh.
(b) Mỗi nhánh nhận 40 000 lượt mỗi ngày. Thí nghiệm phải chạy bao lâu?
(c) Cần rút thời gian xuống một nửa mà không giảm lực kiểm định. Nêu các cách hợp lệ, và chỉ ra những cách nghe hợp lý nhưng không hợp lệ.
*Gợi ý: công thức ở Mục 8.4, với $z_{0{,}975} = 1{,}96$ và $z_{0{,}80} = 0{,}84$.*

**Bài 3 (tính tay).** Một yêu cầu xếp hạng gọi song song 25 dịch vụ đặc trưng và phải đợi đủ 25 kết quả mới trả lời. Mỗi dịch vụ có p99 bằng 80 ms, và các dịch vụ độc lập với nhau.
(a) Tính tỉ lệ yêu cầu gặp ít nhất một dịch vụ chậm hơn 80 ms.
(b) Để p99 của cả yêu cầu bằng 80 ms, mỗi dịch vụ phải đạt phân vị nào?
(c) Nêu hai cách giảm phần đuôi mà không cần làm từng dịch vụ nhanh hơn.

**Bài 4 (suy luận).** Cho bảng lịch sử đặc trưng và bảng nhãn dưới đây. Viết tập huấn luyện đúng theo thời điểm; chỉ ra tập huấn luyện sai mà một phép `JOIN` thông thường tạo ra, và cho biết mô hình sẽ học sai điều gì.

| user_id | valid_from | tong_don_hang |
|---|---|---|
| 7 | 2026-01-01 | 3 |
| 7 | 2026-03-01 | 19 |
| 7 | 2026-06-01 | 52 |

| user_id | event_time | nhan |
|---|---|---|
| 7 | 2026-02-10 | 0 |
| 7 | 2026-04-15 | 1 |

**Bài 5 (phân loại).** Xếp mỗi tình huống vào covariate shift, label shift hay concept drift, và cho biết có phát hiện được chỉ bằng cách giám sát $X$ hay không:
(a) Một chiến dịch quảng cáo thu hút nhiều người dùng dưới 25 tuổi.
(b) Ngân hàng đổi định nghĩa "nợ xấu" từ quá hạn 90 ngày thành quá hạn 60 ngày.
(c) Mùa mua sắm cuối năm làm tỉ lệ giao dịch gian lận tăng gấp ba, nhưng cách gian lận không đổi.
(d) Một đối thủ ra mắt tính năng mới, làm thay đổi kỳ vọng của người dùng về một kết quả tìm kiếm tốt.
*Gợi ý: xem lại bảng ở Mục 9.2.*

**Bài 6 (ML Test Score).** Một nhóm có: lược đồ đặc trưng đầy đủ và kiểm thử đơn vị cho toàn bộ mã tính đặc trưng, chạy tự động; huấn luyện tái lập được và kiểm thử tích hợp toàn pipeline, chạy tự động; canary và quay lui tự động. Về giám sát, nhóm chỉ có một biểu đồ độ trễ để xem khi cần, không có cảnh báo tự động và không có mục nào khác thuộc nhóm giám sát. Đề bài không nói gì về các mục của nhóm phát triển mô hình.
(a) Tính điểm ML Test Score của nhóm và giải thích vì sao quy tắc lấy giá trị nhỏ nhất cho ra con số đó.
(b) Nhóm nên làm gì đầu tiên để tăng điểm?

**Bài 7 (thí nghiệm).** Sửa phần (G) của `code/mlops/experiments.py` để chạy với ba tốc độ dịch chuyển 0,010, 0,035 và 0,080 radian mỗi tuần, và thêm nhịp huấn luyện lại 2 tuần một lần. Với mỗi tốc độ, tìm nhịp thưa nhất mà độ chính xác trung bình không kém nhịp hằng tuần quá 1 điểm phần trăm. Rút ra quan hệ giữa tốc độ dịch chuyển và nhịp huấn luyện lại hợp lý.

**Bài 8 (thí nghiệm).** Trong phần (H) của `code/mlops/experiments.py`, cho $\varepsilon$ chạy từ 0 tới 0,5 với bước 0,1 (tức từ 0 tới 5 vị trí ngẫu nhiên trong danh sách 10 món). Với mỗi $\varepsilon$, đo hai đại lượng: chất lượng thật của 10 món mà hệ thống xếp đầu ở vòng cuối, và chất lượng thật trung bình của các món đã thực sự hiển thị cho người dùng.
(a) Theo mỗi đại lượng, có giá trị $\varepsilon$ tốt nhất không? Giải thích hình dạng của các đường cong.
(b) Điều gì thay đổi khi số vòng tăng từ 30 lên 300?

**Bài 9 (thiết kế).** Thiết kế hệ thống giám sát cho một mô hình phát hiện gian lận có độ trễ nhãn 30 ngày (tranh chấp giao dịch mất một tháng mới được xác nhận). Nêu rõ: giám sát gì ở mỗi lớp trong bốn lớp ở Mục 10.2; chỉ số nào đủ quan trọng để gọi người trực lúc nửa đêm; và phát hiện concept drift bằng cách nào khi nhãn đến trễ một tháng.

**Bài 10 (thiết kế, LLM).** Thiết kế quy trình đánh giá cho một trợ lý hỏi đáp dùng RAG trên tài liệu nội bộ của công ty, với 50 000 lượt hỏi mỗi ngày và ngân sách đánh giá bằng 3% chi phí suy luận. Nêu rõ: ba tầng đánh giá và tỉ lệ lấy mẫu của từng tầng; cách tách lỗi truy xuất khỏi lỗi sinh; và cách phát hiện khi chính giám khảo LLM thay đổi hành vi.

---

## 15. Câu hỏi phỏng vấn

Chương này không thêm kiến thức mới mà sắp xếp lại nội dung các chương trước theo dạng câu hỏi và câu trả lời thường gặp trong phỏng vấn cho vị trí kỹ sư học máy hoặc kỹ sư MLOps.

### 15.1. Cách trình bày câu trả lời

Phần lớn câu hỏi về MLOps là câu hỏi mở, và người phỏng vấn đánh giá cách lập luận hơn là danh sách công cụ. Một cách trình bày dùng được cho đa số câu hỏi gồm bốn bước:

1. **Nêu ràng buộc.** Ví dụ: "phụ thuộc vào độ trễ nhãn", "phụ thuộc vào số mô hình dùng chung đặc trưng". Nêu đúng ràng buộc cho thấy đã hiểu bài toán.
2. **Nêu đánh đổi** giữa các lựa chọn.
3. **Chọn một phương án và giải thích lý do**, thay vì dừng ở "còn tuỳ trường hợp".
4. **Nêu cách đo để biết lựa chọn có đúng không.**

Một con số cụ thể, như các số liệu đo được trong giáo trình, làm câu trả lời thuyết phục hơn nhiều so với một nhận định chung.

### 15.2. Nền tảng

**Câu hỏi: MLOps là gì, khác DevOps ở điểm nào?**

> **Trả lời.** DevOps quản lý một loại tạo tác, mã nguồn, thay đổi khi có người sửa mã. MLOps quản lý ba loại tạo tác: mã, dữ liệu và mô hình, thay đổi theo ba nhịp khác nhau, trong đó dữ liệu thay đổi mà không ai sửa kho mã. Vì vậy ngoài CI/CD, học máy cần thêm huấn luyện liên tục (CT). Hệ quả về vận hành: hệ thống học máy thường hỏng một cách im lặng; một mô hình nhận đầu vào có phân phối khác vẫn trả về kết quả đúng kiểu, đúng thời hạn, nhưng sai.

**Câu hỏi: Vì sao một mô hình tốt khi đánh giá lại suy giảm trong sản xuất?**

> **Trả lời.** Có bốn nguyên nhân chính, mỗi nguyên nhân có tên: lệch giữa huấn luyện và phục vụ (đặc trưng được tính khác nhau ở hai nơi), dịch chuyển phân phối (dữ liệu thay đổi theo thời gian), phụ thuộc dữ liệu không ổn định (một hệ thống phía trước thay đổi mà không thông báo), và vòng phản hồi (mô hình làm sai lệch dữ liệu huấn luyện của chính nó). Nguyên lý CACE giải thích vì sao khó phòng: trong một mô hình, thay đổi bất kỳ thành phần nào cũng có thể thay đổi toàn bộ hành vi, nên không thể lập luận cục bộ như khi sửa một hàm trong phần mềm thông thường.

**Câu hỏi: Con số "5% mã học máy" nghĩa là gì?**

> **Trả lời.** Nhận định này nằm trong mục về mã keo của Sculley và cộng sự (2015), không phải chú thích của hình vẽ nổi tiếng trong bài báo. Nó được diễn đạt có giới hạn: một hệ thống trưởng thành có thể chỉ gồm nhiều nhất 5% mã học máy và ít nhất 95% mã keo; đây là nhận định định tính, không phải phép đo. Kết luận bài báo rút ra là đôi khi tự viết một giải pháp gọn còn rẻ hơn dùng lại một thư viện đa dụng, và nên bọc các thư viện sau một giao diện chung.

**Câu hỏi: Các mức trưởng thành của MLOps là gì?**

> **Trả lời.** Theo kiến trúc tham chiếu của Google Cloud: mức 0, mọi bước làm thủ công, thứ được bàn giao là một mô hình; mức 1, pipeline huấn luyện được tự động hoá và có huấn luyện liên tục, thứ được bàn giao là cả pipeline; mức 2, có CI/CD cho chính pipeline, thứ được bàn giao là một hệ thống tự cập nhật. Tự động hoá không thay thế giám sát: một hệ thống ở mức 1 mà không có giám sát chỉ triển khai nhanh hơn, kể cả khi triển khai một mô hình đã hỏng.

### 15.3. Dữ liệu và đặc trưng

**Câu hỏi: Lệch giữa huấn luyện và phục vụ là gì và phòng tránh thế nào?**

> **Trả lời.** Là khi đặc trưng lúc phục vụ được tính khác với lúc huấn luyện, thường vì có hai bản cài đặt riêng: một truy vấn SQL trên kho dữ liệu khi huấn luyện, một đoạn mã viết lại trong dịch vụ phục vụ cho đủ nhanh. Cách phòng: một định nghĩa đặc trưng duy nhất cho cả hai đường, đóng gói hàm biến đổi cùng mô hình, và đo độ lệch bằng cách so giá trị đặc trưng ở hai nơi trên cùng lưu lượng, đúng mục Monitor 3 của ML Test Score. Chạy shadow là cách rẻ nhất để đo điều này trên lưu lượng thật.

**Câu hỏi: Tính đúng theo thời điểm là gì?**

> **Trả lời.** Với mỗi dòng huấn luyện có thời điểm dự đoán $t$, giá trị mọi đặc trưng phải là giá trị quan sát được tại $t$, không phải giá trị tại lúc dựng tập dữ liệu. Ví dụ: một dòng của tháng 3 phải dùng tổng số đơn hàng của khách tại tháng 3. Nếu dùng con số của hôm nay, mô hình học rằng "khách có 480 đơn hàng thì không rời bỏ", trong khi con số 480 chỉ tồn tại vì khách đã ở lại. Lỗi này nguy hiểm vì nó làm kết quả đánh giá ngoại tuyến tốt lên, nên không ai nghi ngờ cho tới khi mô hình được triển khai. Cách phòng là ghép theo thời điểm (as-of join) và thêm ràng buộc "không có sự kiện từ tương lai" vào hợp đồng dữ liệu.

**Câu hỏi: Khi nào cần kho đặc trưng?**

> **Trả lời.** Khi có đồng thời ba điều kiện: nhiều mô hình dùng chung đặc trưng, phục vụ trực tuyến với độ trễ thấp, và đặc trưng phụ thuộc thời gian, cập nhật thường xuyên. Thiếu các điều kiện đó, một bảng trong kho dữ liệu cộng với kỷ luật dùng chung một hàm biến đổi rẻ hơn nhiều. Kho đặc trưng không tự động loại bỏ lệch huấn luyện–phục vụ; nếu nhóm vẫn viết một đường tính riêng cho phục vụ thì sự lệch quay lại.

**Câu hỏi: Làm sao phát hiện dữ liệu có vấn đề?**

> **Trả lời.** Phân loại lỗi thành ba loại với ba cách phản ứng: lỗi cứng (vi phạm lược đồ, giá trị ngoài miền) thì dừng pipeline; lỗi mềm (thiếu dữ liệu một phần, đổi đơn vị, giá trị danh mục lạ) thì cảnh báo và theo dõi tỉ lệ; dịch chuyển phân phối thì điều tra và có thể huấn luyện lại. Các kiểm tra được viết thành hợp đồng dữ liệu, suy ra từ dữ liệu tham chiếu rồi người duyệt lại, và chạy ở cả đầu vào lẫn đầu ra của mỗi pipeline. Dùng chung một cơ chế cảnh báo cho cả ba loại là nguyên nhân phổ biến của báo động giả.

### 15.4. Đánh giá và ra mắt

**Câu hỏi: Vì sao một chỉ số tổng thể không đủ để đánh giá mô hình?**

> **Trả lời.** Chỉ số tổng thể là trung bình theo lưu lượng, nên nhóm đa số quyết định kết quả. Trong thí nghiệm ở Mục 6.3, thêm một đặc trưng làm độ chính xác tổng thể tăng 19,8 điểm phần trăm, nhóm khách hàng cũ tăng 25,5 điểm, nhưng nhóm 15% khách hàng mới giảm 11,6 điểm, xuống dưới cả mô hình cũ. Nguyên nhân: đặc trưng mới chỉ có thông tin với khách hàng cũ, nên với khách hàng mới, trọng số lớn của nó đưa nhiễu vào dự đoán. Vì vậy cần đánh giá theo lát cắt, cùng với kiểm thử hành vi cho những trường hợp cụ thể.

**Câu hỏi: Shadow, canary và A/B test khác nhau thế nào?**

> **Trả lời.** Mỗi cách trả lời một câu hỏi khác nhau. Shadow cho biết mô hình mới có chịu được tải thật không, đặc trưng có khớp với lúc huấn luyện không, mà người dùng không chịu rủi ro nào; nhưng không cho biết người dùng phản ứng ra sao. Canary cho biết có chỉ số vận hành nào xấu đi ở quy mô nhỏ không, và giới hạn phạm vi ảnh hưởng nếu có sự cố. A/B test cho biết mô hình mới có làm chỉ số sản phẩm tốt lên theo nghĩa nhân quả không. Canary không đo được chỉ số sản phẩm, vì ở 1% lưu lượng không đủ mẫu; dùng canary để kết luận về giá trị là một lỗi quy trình thường gặp.

**Câu hỏi: Vì sao không A/B test mọi thay đổi?**

> **Trả lời.** Vì lưu lượng có hạn, và cỡ mẫu tỉ lệ nghịch với bình phương mức cải thiện cần phát hiện. Với tỉ lệ nền 5%, phát hiện một cải thiện tương đối 1% với lực kiểm định 80% cần khoảng 3 triệu mẫu mỗi nhánh, tức khoảng một tháng lưu lượng nếu mỗi nhánh có 100 000 lượt mỗi ngày. Vì vậy quy trình tốt loại các ý tưởng kém bằng những cách rẻ hơn (đánh giá ngoại tuyến, shadow) trước khi dùng lưu lượng cho A/B test.

**Câu hỏi: Những lỗi thường gặp khi đọc kết quả A/B test là gì?**

> **Trả lời.** Xem kết quả giữa chừng và dừng khi thấy $p < 0{,}05$, làm tăng tỉ lệ dương tính giả (cố định cỡ mẫu trước, hoặc dùng kiểm định tuần tự); so sánh nhiều chỉ số cùng lúc (khai báo trước một chỉ số chính); hai nhánh ảnh hưởng lẫn nhau trong mạng xã hội hoặc sàn giao dịch (chia ngẫu nhiên theo cụm); tỉ lệ mẫu giữa hai nhánh lệch so với thiết kế (kiểm tra bằng kiểm định khi bình phương trước khi đọc kết quả); và hiệu ứng mới lạ (chạy đủ lâu, xem hiệu ứng theo thời gian).

**Câu hỏi: ML Test Score được tính thế nào?**

> **Trả lời.** Gồm 28 mục chia đều bốn nhóm: dữ liệu, phát triển mô hình, hạ tầng, giám sát. Mỗi mục được 0,5 điểm nếu làm thủ công có ghi lại kết quả, 1 điểm nếu có hệ thống chạy tự động và định kỳ. Cộng điểm riêng từng nhóm; điểm cuối cùng là giá trị nhỏ nhất trong bốn nhóm. Lý do lấy giá trị nhỏ nhất: cả bốn nhóm đều cần thiết, nên một hệ thống có hạ tầng hoàn hảo mà không có giám sát vẫn có điểm 0.

### 15.5. Giám sát và dịch chuyển

**Câu hỏi: Có những loại dịch chuyển phân phối nào?**

> **Trả lời.** Covariate shift: $P(X)$ thay đổi, $P(Y \mid X)$ giữ nguyên. Label shift: $P(Y)$ thay đổi, $P(X \mid Y)$ giữ nguyên. Concept drift: $P(Y \mid X)$ thay đổi. Hai loại đầu thường làm $P(X)$ thay đổi; concept drift thì không nhất thiết.

**Câu hỏi: Loại dịch chuyển nào phát hiện được mà không cần nhãn?**

> **Trả lời.** Covariate shift và label shift, vì cả hai làm phân phối của $X$ thay đổi. Concept drift không phát hiện được bằng cách giám sát $X$, vì theo định nghĩa nó có thể giữ nguyên $P(X)$. Trong thí nghiệm ở Mục 9.2, concept drift làm độ chính xác giảm từ 75,6% xuống 26,9%, trong khi kiểm định KS trên các đặc trưng không thấy gì bất thường ($p = 0{,}48$). Ngược lại, covariate shift cho p-value gần bằng 0 trong khi độ chính xác gần như không đổi, tức cảnh báo dựa trên $X$ sẽ là báo động giả. Vì vậy độ trễ nhãn quyết định khả năng phát hiện loại dịch chuyển gây hại nhiều nhất, và thiết kế giám sát phải bắt đầu bằng câu hỏi: sau bao lâu thì biết nhãn thật?

**Câu hỏi: Nên dùng ngưỡng PSI nào?**

> **Trả lời.** PSI là tổng hai chiều của phân kỳ KL (phân kỳ Jeffreys). Các ngưỡng 0,10 và 0,25 là quy tắc kinh nghiệm từ lĩnh vực chấm điểm tín dụng, không tính tới cỡ mẫu và số bin. Khi hai mẫu cùng phân phối, PSI có kỳ vọng xấp xỉ $2(k-1)/n$: với $n = 200$ và 20 bin, PSI trung bình là 0,205 dù không có dịch chuyển nào. Mô phỏng ở Mục 9.5 cho thấy với $n = 50$, quy tắc PSI > 0,25 báo động 73% số lần khi không có dịch chuyển; với $n = 1\,000$ và dịch chuyển thật 0,3 độ lệch chuẩn, quy tắc này không báo động lần nào, trong khi kiểm định KS báo động ở mọi lần. Nên dùng kiểm định có p-value kèm hiệu chỉnh cho kiểm định nhiều lần; nếu buộc phải dùng PSI, tính ngưỡng theo đúng $n$ và $k$, ví dụ $\tfrac{2}{n}\chi^2_{k-1,\,0{,}99}$, và giữ $n$, $k$ cố định giữa các lần đo.

**Câu hỏi: Khi không có nhãn thì giám sát gì?**

> **Trả lời.** Có bốn lớp giám sát, càng gần nhãn càng có ý nghĩa nhưng càng khó có: đầu vào thô, đặc trưng, dự đoán, chất lượng. Không có nhãn thì lớp dự đoán là lớp chính: phân phối điểm số, tỉ lệ từng lớp, và độ lệch dự đoán (phân phối nhãn dự đoán so với tỉ lệ nền lịch sử), tính theo lát cắt. Độ lệch dự đoán có giới hạn: một mô hình luôn dự đoán tỉ lệ trung bình cũng có độ lệch bằng 0; nhưng thay đổi đột ngột của nó thường là dấu hiệu sự cố. Với hệ thống có hành động thật, thêm giới hạn hành động: đặt trần cho số hành động trong mỗi khoảng thời gian, chạm trần thì cảnh báo.

**Câu hỏi: Vì sao cảnh báo hay bị bỏ qua, và thiết kế cảnh báo thế nào?**

> **Trả lời.** Vì báo động giả: khi phần lớn cảnh báo là giả, người nhận mất niềm tin và bỏ qua cả cảnh báo thật. Bốn nguyên tắc: mỗi cảnh báo phải gắn với một hành động cụ thể; phân mức (gọi người trực, tạo phiếu xử lý, hoặc chỉ hiển thị trên bảng theo dõi); cảnh báo theo triệu chứng (tỉ lệ chuyển đổi giảm) chứ không theo nguyên nhân (PSI của một đặc trưng); và yêu cầu tín hiệu kéo dài qua nhiều cửa sổ liên tiếp.

### 15.6. Thiết kế hệ thống

Với câu hỏi dạng "thiết kế hệ thống học máy cho bài toán X", đi theo thứ tự sau để không bỏ sót:

1. **Làm rõ bài toán:** chỉ số sản phẩm là gì, chỉ số nào của mô hình đại diện cho nó, và hai chỉ số tương quan tới đâu.
2. **Hỏi về nhãn:** lấy từ đâu, trễ bao lâu, có bị chính hệ thống làm sai lệch không. Câu trả lời quyết định phần lớn thiết kế còn lại.
3. **Chế độ phục vụ:** theo lô, trực tuyến hay luồng; chọn chế độ rẻ nhất đáp ứng được yêu cầu; nếu trực tuyến thì lập ngân sách độ trễ.
4. **Dữ liệu và đặc trưng:** nguồn, hợp đồng dữ liệu, có cần ghép theo thời điểm không, có cần kho đặc trưng không.
5. **Huấn luyện:** nhịp huấn luyện lại, loại kích hoạt, những gì cần cố định để tái lập được.
6. **Đánh giá:** tập kiểm tra cố định, các lát cắt, kiểm thử hành vi, bước so sánh với mô hình đang chạy.
7. **Ra mắt:** shadow, canary, A/B test, và cơ chế quay lui.
8. **Giám sát:** bốn lớp, chỉ số nào gọi người trực, quy trình xử lý khi có cảnh báo.
9. **Vòng phản hồi:** hệ thống có ảnh hưởng tới dữ liệu tương lai của nó không, có cần nhóm đối chứng không.

Bước 2 và bước 9 là hai bước hay bị bỏ qua nhất.

### 15.7. LLMOps

**Câu hỏi: LLMOps khác MLOps ở điểm nào?**

> **Trả lời.** Phần lớn nguyên tắc giữ nguyên: hợp đồng dữ liệu, quản lý phiên bản, shadow và canary, giám sát, quay lui. Có năm ràng buộc thay đổi: thường không tự huấn luyện mô hình; đầu ra là văn bản tự do, không có hàm đúng sai hiển nhiên; kết quả không tất định, và nhà cung cấp có thể cập nhật mô hình sau cùng một tên gọi; chi phí chủ yếu là chi phí suy luận theo token, thay đổi theo từng yêu cầu; và có thêm các rủi ro như nội dung có hại, rò rỉ dữ liệu, prompt injection. Việc nhà cung cấp cập nhật mô hình là dạng cực đoan của phụ thuộc dữ liệu không ổn định; cách xử lý là dùng định danh phiên bản cố định và coi mỗi lần đổi phiên bản là một lần phát hành đầy đủ.

**Câu hỏi: Đánh giá một hệ thống dùng LLM thế nào?**

> **Trả lời.** Ba tầng theo chi phí: kiểm tra theo quy tắc trên mọi lượt gọi (định dạng, JSON, trích dẫn, độ dài); giám khảo LLM trên một mẫu, cho chất lượng nội dung; người chấm trên một mẫu nhỏ, định kỳ, để hiệu chỉnh giám khảo. Cộng thêm hai vòng: tập đánh giá chuẩn làm bước kiểm định trước mỗi lần phát hành, và chấm mẫu trên lưu lượng thật. Cần biết các thiên lệch của giám khảo LLM: thiên lệch vị trí (xử lý bằng cách hỏi hai lần với thứ tự đảo), thiên lệch độ dài, và khả năng tự ưu tiên câu trả lời của chính mình. Giám khảo cũng là một mô hình có thể thay đổi, nên cần đo định kỳ mức đồng thuận giữa giám khảo và người chấm.

**Câu hỏi: Một câu trả lời của hệ thống RAG sai thì điều tra thế nào?**

> **Trả lời.** Tách thành hai câu hỏi. Tài liệu cần thiết có nằm trong ngữ cảnh không? Nếu không, đó là lỗi truy xuất: đo bằng recall@$k$, sửa bằng cách chia đoạn, mô hình embedding, tìm kiếm kết hợp, xếp hạng lại. Nếu có mà câu trả lời vẫn sai, đó là lỗi sinh: đo bằng độ trung thành với ngữ cảnh, sửa bằng prompt, đổi mô hình, yêu cầu trích dẫn. Để tách được hai loại lỗi, bản ghi vết phải lưu danh sách các đoạn đã truy xuất và điểm của chúng; gộp hai loại lỗi vào một chỉ số thì không biết cần sửa ở đâu.

**Câu hỏi: Làm sao giảm chi phí của một ứng dụng LLM?**

> **Trả lời.** Theo thứ tự nên thử: bộ nhớ đệm (cho câu hỏi lặp lại, câu hỏi gần giống, và phần đầu cố định của prompt); định tuyến câu hỏi dễ tới mô hình nhỏ; giảm ngữ cảnh (bớt số đoạn truy xuất, tóm tắt lịch sử hội thoại). Chi phí của giám khảo LLM nên được ghi thành một dòng riêng, để biết chi phí tăng vì lưu lượng hay vì tăng tỉ lệ lấy mẫu đánh giá.

### 15.8. Các câu trả lời chưa đạt

| Câu trả lời | Vì sao chưa đạt |
|---|---|
| "Tôi giám sát độ chính xác trong sản xuất." | Chỉ làm được khi có nhãn ngay; không hỏi về độ trễ nhãn là bỏ qua ràng buộc quan trọng nhất. |
| "Ngưỡng PSI là 0,25." | Ngưỡng cố định không tính tới cỡ mẫu và số bin (Mục 9.5). |
| "Dùng kho đặc trưng là hết lệch huấn luyện–phục vụ." | Kho đặc trưng tạo điều kiện, không bảo đảm. |
| "A/B test mọi thay đổi." | Cỡ mẫu tỉ lệ với $1/\Delta^2$; lưu lượng không đủ cho mọi ý tưởng. |
| "Huấn luyện lại hằng ngày cho chắc." | Lợi ích giảm dần nhanh (Mục 11.1), và không tính tới chi phí. |
| "Mô hình đạt 99% nên rất tốt." | Không hỏi về mất cân bằng lớp, về lát cắt, và về mức đồng thuận giữa người gán nhãn. |
| "Chúng tôi dùng công cụ X nên vấn đề đó đã được giải quyết." | Công cụ không thay được việc chọn đánh đổi; nêu tên công cụ trước khi nêu ràng buộc là dấu hiệu học thuộc. |
| "CI/CD là đủ cho học máy." | Thiếu huấn luyện liên tục, và bỏ qua việc dữ liệu thay đổi mà không ai sửa mã. |
| "Giám khảo LLM chấm 4,5/5 nên chất lượng tốt." | Không đối chiếu với người chấm thì không biết giám khảo có đáng tin không (Mục 13.3). |

---

## 16. Tài liệu tham khảo

**Nợ kỹ thuật và kiểm thử hệ thống học máy**

1. D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison. *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 2015. Nguồn của nguyên lý CACE, các dạng nợ kỹ thuật, nhận định về 5% mã học máy, vòng phản hồi, và ba cơ chế giám sát: độ lệch dự đoán, giới hạn hành động, hệ thống phía trước.
2. D. Sculley và cộng sự. *Machine Learning: The High-Interest Credit Card of Technical Debt.* SE4ML Workshop, NeurIPS 2014. Phiên bản đầu, ngắn hơn, của bài báo trên.
3. E. Breck, S. Cai, E. Nielsen, M. Salib, D. Sculley. *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE BigData 2017. Nguồn của 28 mục kiểm tra, cách tính điểm và bảng diễn giải ở Mục 6.5.
4. E. Breck, N. Polyzotis, S. Roy, S. E. Whang, M. Zinkevich. *Data Validation for Machine Learning.* SysML 2019. Kiểm định dữ liệu bằng lược đồ suy ra từ dữ liệu (Mục 3.3).
5. M. Zinkevich. *Rules of Machine Learning: Best Practices for ML Engineering.* Google. Tập hợp các quy tắc thực hành, bổ sung cho Chương 2 và Chương 6.

**Quy trình, kiến trúc và vận hành**

6. Google Cloud. *MLOps: Continuous delivery and automation pipelines in machine learning.* Nguồn của ba mức tự động hoá, tám bước của pipeline, sáu giai đoạn CI/CD ở mức 2, và định nghĩa lệch huấn luyện–phục vụ.
7. D. Sato, A. Wider, C. Windheuser. *Continuous Delivery for Machine Learning.* martinfowler.com, 2019.
8. D. Kreuzberger, N. Kühl, S. Hirschl. *Machine Learning Operations (MLOps): Overview, Definition, and Architecture.* IEEE Access, 2023.
9. B. Beyer, C. Jones, J. Petoff, N. R. Murphy (chủ biên). *Site Reliability Engineering: How Google Runs Production Systems.* O'Reilly, 2016. Nguồn của SLI, SLO, bốn tín hiệu cơ bản và các nguyên tắc cảnh báo ở Chương 10.
10. J. Dean, L. A. Barroso. *The Tail at Scale.* Communications of the ACM, 2013. Phân tích phần đuôi độ trễ khi toả nhánh và yêu cầu dự phòng (Mục 7.3).

**Thực hành trong công nghiệp**

11. S. Shankar, R. Garcia, J. M. Hellerstein, A. G. Parameswaran. *Operationalizing Machine Learning: An Interview Study.* arXiv:2209.09125, 2022. Nguồn của ba yếu tố tốc độ, kiểm định, phiên bản; bốn nhiệm vụ trong vòng đời; ba loại lỗi dữ liệu; và bốn thói quen xấu.
12. C. Huyen. *Designing Machine Learning Systems.* O'Reilly, 2022. Nguồn của khái niệm vòng phản hồi thoái hoá và cách phân biệt huấn luyện lại từ đầu với huấn luyện có trạng thái.
13. A. Paleyes, R.-G. Urma, N. D. Lawrence. *Challenges in Deploying Machine Learning: A Survey of Case Studies.* ACM Computing Surveys, 2022.
14. M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, T. Gebru. *Model Cards for Model Reporting.* FAT* 2019. Thẻ mô hình (Mục 5.4).

**Đánh giá và thí nghiệm có đối chứng**

15. M. T. Ribeiro, T. Wu, C. Guestrin, S. Singh. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL 2020. Ba loại kiểm thử hành vi ở Mục 6.4.
16. R. Kohavi, D. Tang, Y. Xu. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing.* Cambridge University Press, 2020. Tài liệu chuẩn về A/B test, gồm các lỗi ở Mục 8.5.
17. A. Deng, Y. Xu, R. Kohavi, T. Walker. *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.* WSDM 2013. Kỹ thuật CUPED ở Mục 8.4.

**Dịch chuyển phân phối**

18. J. Quiñonero-Candela, M. Sugiyama, A. Schwaighofer, N. D. Lawrence (chủ biên). *Dataset Shift in Machine Learning.* MIT Press, 2009. Nguồn học thuật của cách phân loại covariate shift, prior probability shift và concept shift.
19. J. G. Moreno-Torres, T. Raeder, R. Alaiz-Rodríguez, N. V. Chawla, F. Herrera. *A unifying view on dataset shift in classification.* Pattern Recognition, 2012.
20. S. Rabanser, S. Günnemann, Z. C. Lipton. *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 2019. So sánh các phương pháp phát hiện dịch chuyển (Mục 9.3).
21. B. Yurdakul, J. Naranjo. *Statistical properties of the population stability index.* Journal of Risk Model Validation, 2020. Phân phối tiệm cận của PSI và giá trị tới hạn phụ thuộc cỡ mẫu và số bin (Mục 9.5).
22. A. du Pisanie và cộng sự. *A critical review of existing and new population stability testing procedures in credit risk scoring.* arXiv:2303.01227, 2023.

**Vòng phản hồi**

23. T. Joachims, A. Swaminathan, T. Schnabel. *Unbiased Learning-to-Rank with Biased Feedback.* WSDM 2017. Hiệu chỉnh theo xác suất hiển thị (Mục 12.4).

**Kho đặc trưng**

24. Tài liệu của Feast (feast.dev) về ghép theo thời điểm, kho ngoại tuyến, kho trực tuyến và sổ đăng ký đặc trưng.

**LLMOps**

25. L. Zheng, W.-L. Chiang, Y. Sheng và cộng sự. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023, Datasets and Benchmarks Track. Các thiên lệch của giám khảo LLM (Mục 13.3).
26. S. Es, J. James, L. Espinosa-Anke, S. Schockaert. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217, 2023. Ba chỉ số đánh giá RAG (Mục 13.5).
27. S. Shankar, J. D. Zamfirescu-Pereira, B. Hartmann, A. G. Parameswaran, I. Arawjo. *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences.* UIST 2024. Hiện tượng criteria drift (Mục 13.3).
28. L. Chen, M. Zaharia, J. Zou. *How is ChatGPT's Behavior Changing over Time?* arXiv:2307.09009, 2023. Thay đổi hành vi của cùng một tên mô hình theo thời gian (Mục 13.1).
29. C. Huyen. *AI Engineering: Building Applications with Foundation Models.* O'Reilly, 2025. Đánh giá, vận hành và kiến trúc của ứng dụng dùng mô hình nền tảng.

---

## Phụ lục: chạy lại toàn bộ thí nghiệm

```
code/mlops/
├── experiments.py            # Hình 6, 7, 9–14 và mọi số liệu đo được trong bài
├── experiments_output.txt    # kết quả in ra của experiments.py
├── bai_tap.py                # số liệu cho lời giải bài tập Chương 14
├── bai_tap_output.txt        # kết quả in ra của bai_tap.py
└── fig_diagrams.py           # Hình 1–5 và Hình 8 (sơ đồ khái niệm)
```

```bash
pip install numpy scipy matplotlib scikit-learn
python code/mlops/experiments.py     # khoảng một phút
python code/mlops/bai_tap.py         # khoảng một phút rưỡi
python code/mlops/fig_diagrams.py
```

Các script đặt hạt giống cố định, nên mọi con số trong tài liệu lặp lại được y hệt với cùng phiên bản thư viện. Số liệu thực nghiệm đến từ dữ liệu mô phỏng, mỗi thí nghiệm được thiết kế để cô lập một cơ chế. Chúng cho thấy cơ chế đó tồn tại và có độ lớn đáng kể, không dùng để suy ra con số cho một hệ thống cụ thể.
