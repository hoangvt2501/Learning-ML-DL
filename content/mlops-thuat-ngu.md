# Từ điển thuật ngữ — MLOps

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Các định nghĩa theo đúng cách dùng trong giáo trình; số mục trong ngoặc chỉ nơi thuật ngữ được trình bày.

## Nợ kỹ thuật

### Nợ kỹ thuật | technical debt
Chi phí dài hạn phát sinh từ những lựa chọn giúp đi nhanh trong ngắn hạn. Không phải khoản nợ nào cũng sai, nhưng khoản nợ nào cũng phải trả lãi. Trong hệ thống học máy, phần lớn nợ nằm ở mức hệ thống chứ không ở mức mã nguồn, nên đọc mã không thấy được (Mục 1.2).

### Nguyên lý CACE | Changing Anything Changes Everything
Trong một hệ thống học máy, thay đổi bất kỳ thành phần nào cũng có thể thay đổi hành vi của toàn bộ hệ thống: thêm hay bớt một đặc trưng làm trọng số của mọi đặc trưng khác thay đổi; nguyên lý áp dụng cho cả siêu tham số, cách lấy mẫu và ngưỡng hội tụ. Vì vậy không có thay đổi nào là nhỏ và cô lập trong một mô hình (Mục 1.3).

### Entanglement | entanglement
Tên của dạng nợ mô tả bởi nguyên lý CACE: các thành phần của mô hình phụ thuộc lẫn nhau tới mức không thể lập luận cục bộ.

### Thác hiệu chỉnh | correction cascade
Học một mô hình mới nhận đầu ra của một mô hình có sẵn làm đầu vào, để giải một bài toán gần giống. Nhanh trước mắt, nhưng tạo ra phụ thuộc mới, và cải thiện mô hình gốc trở nên tốn kém vì có thể làm hỏng mô hình phía sau.

### Bên sử dụng không khai báo | undeclared consumer
Một hệ thống khác dùng đầu ra của mô hình làm đầu vào mà không ai biết, thường vì đầu ra được ghi ra nơi không có kiểm soát truy cập. Thay đổi mô hình khi đó làm hỏng hệ thống kia, và có thể tạo ra vòng phản hồi ẩn.

### Phụ thuộc dữ liệu không ổn định | unstable data dependency
Đặc trưng lấy từ một hệ thống thay đổi hành vi theo thời gian, ví dụ một mô hình khác được huấn luyện lại định kỳ. Cách xử lý là cố định phiên bản của tín hiệu đầu vào, đổi lại phải duy trì nhiều phiên bản song song. Mô hình ngôn ngữ gọi qua API là dạng cực đoan của loại phụ thuộc này (Mục 13.1).

### Phụ thuộc dữ liệu ít giá trị | underutilized data dependency
Đặc trưng đóng góp rất ít nhưng vẫn nằm trong mô hình, làm hệ thống dễ hỏng một cách không cần thiết. Bốn nguồn: đặc trưng cũ đã bị thay thế, đặc trưng thêm cả gói, đặc trưng cải thiện rất ít, và đặc trưng tương quan với một đặc trưng khác có quan hệ nhân quả thật. Phát hiện bằng cách định kỳ đánh giá lại mô hình khi bỏ từng đặc trưng.

### Mã keo | glue code
Mã nối các thư viện đa dụng lại với nhau. Theo Sculley và cộng sự (2015), một hệ thống trưởng thành có thể chỉ gồm nhiều nhất 5% mã học máy và ít nhất 95% mã keo; đây là nhận định định tính, không phải phép đo (Mục 1.5).

### Pipeline jungle | pipeline jungle
Trường hợp riêng của mã keo ở khâu chuẩn bị dữ liệu: các bước lấy, ghép, lấy mẫu dữ liệu chồng chất theo thời gian, với nhiều tệp trung gian.

### Nợ cấu hình | configuration debt
Nợ tích tụ trong các tuỳ chọn cấu hình của hệ thống. Cấu hình thường được kiểm tra ít kỹ hơn mã, dù một dòng cấu hình sai có thể gây hại như một dòng mã sai.

### Ngưỡng cố định trong hệ thống thay đổi | fixed thresholds in dynamic systems
Ngưỡng quyết định đặt bằng tay không còn phù hợp sau khi mô hình được huấn luyện lại, vì phân phối điểm số đã thay đổi. Cách xử lý: xác định ngưỡng tự động trên tập kiểm định mỗi lần huấn luyện và lưu ngưỡng cùng mô hình (Mục 10.6).

## Vòng đời và quy trình

### Huấn luyện liên tục | continuous training (CT)
Tự động huấn luyện lại và triển khai lại mô hình khi có dữ liệu mới, khi chất lượng giảm hoặc khi có dịch chuyển phân phối. CI/CD xử lý thay đổi của mã; CT xử lý thay đổi của dữ liệu, loại thay đổi không đi qua kho mã (Mục 2.4).

### Tốc độ, kiểm định, phiên bản | velocity, validation, versioning
Ba yếu tố quyết định thành công của việc đưa mô hình vào sản xuất theo nghiên cứu phỏng vấn của Shankar và cộng sự (2022): lặp thí nghiệm nhanh, phát hiện ý tưởng kém và lỗi càng sớm càng tốt, và quản lý nhiều phiên bản để quay lui được (Mục 2.2).

### Mức tự động hoá | MLOps maturity levels
Ba mức theo kiến trúc tham chiếu của Google Cloud: mức 0, mọi bước làm thủ công, bàn giao một mô hình; mức 1, tự động hoá pipeline huấn luyện, bàn giao cả pipeline; mức 2, có CI/CD cho chính pipeline (Mục 2.3).

### Kiểm định mô hình | model validation
Bước quyết định có nên thay mô hình đang chạy bằng mô hình mới hay không, tách khỏi bước đánh giá mô hình (mô hình mới tốt tới đâu). Là bước hay bị bỏ qua nhất trong pipeline huấn luyện lại tự động (Mục 11.4).

### Tái lập | reproducibility
Chạy lại với cùng mã, cùng dữ liệu, cùng môi trường thì được cùng kết quả. Cần để điều tra sự cố. Phân biệt với tái tạo (người khác làm lại được kết quả tương tự) và tính bền (kết luận không đổi khi đổi hạt giống hoặc cách chia dữ liệu) (Mục 5.1).

### Sổ đăng ký mô hình | model registry
Nơi mỗi mô hình có định danh và vòng đời: huấn luyện từ mã, dữ liệu và cấu hình nào, đã qua những bước kiểm định nào, đang ở trạng thái nào, phiên bản nào đang chạy trong sản xuất và có thể quay lui về phiên bản nào (Mục 5.4).

### Thẻ mô hình | model card
Tài liệu đi kèm một mô hình: mục đích sử dụng, những trường hợp không nên dùng, dữ liệu huấn luyện, kết quả đánh giá tách theo các nhóm liên quan (Mitchell và cộng sự, 2019).

## Dữ liệu và đặc trưng

### Hợp đồng dữ liệu | data contract
Các giả định về dữ liệu được viết thành ràng buộc kiểm tra được bằng máy, gồm bốn tầng: lược đồ, miền giá trị, thống kê của cả lô, và quan hệ giữa các cột hoặc các bảng. Nên được suy ra từ dữ liệu tham chiếu rồi người duyệt lại (Mục 3.3).

### Lỗi cứng, lỗi mềm, dịch chuyển | hard errors, soft errors, drift
Ba loại lỗi dữ liệu cần ba cách phản ứng: lỗi cứng (vi phạm ràng buộc kiểm tra được) thì dừng pipeline; lỗi mềm (dự đoán vẫn trông hợp lý) thì cảnh báo và theo dõi tỉ lệ; dịch chuyển thì điều tra và có thể huấn luyện lại. Dùng chung một cơ chế cảnh báo cho cả ba là nguyên nhân phổ biến của báo động giả (Mục 3.2).

### Độ trễ nhãn | label lag
Khoảng thời gian từ lúc dự đoán tới lúc biết nhãn thật: vài phút với dự đoán lượt nhấp, hàng tháng với dự đoán vỡ nợ. Độ trễ nhãn quyết định những gì giám sát được, huấn luyện lại được nhanh tới đâu, và có phát hiện được concept drift hay không (Mục 3.5).

### Lệch giữa huấn luyện và phục vụ | training–serving skew
Đặc trưng lúc phục vụ được tính khác lúc huấn luyện, thường vì có hai bản cài đặt riêng. Không gây ra lỗi chương trình nào; mô hình vẫn chạy nhưng kém đi (Mục 4.1).

### Tính đúng theo thời điểm | point-in-time correctness
Với mỗi dòng huấn luyện có thời điểm dự đoán $t$, giá trị mọi đặc trưng phải là giá trị quan sát được tại $t$, không phải giá trị tại lúc dựng tập dữ liệu. Vi phạm tính chất này là đưa thông tin tương lai vào tập huấn luyện (Mục 4.2).

### Ghép theo thời điểm | as-of join, point-in-time join
Phép ghép lấy, cho mỗi dòng nhãn, bản ghi đặc trưng mới nhất nhưng không muộn hơn thời điểm dự đoán của dòng đó (Mục 4.2).

### Rò rỉ dữ liệu | data leakage
Mô hình được huấn luyện với thông tin mà lúc dự đoán thật không có. Nguy hiểm vì nó làm kết quả đánh giá ngoại tuyến tốt lên, nên thường chỉ bị phát hiện sau khi triển khai. Ba đường rò rỉ: theo thời gian, theo thực thể, và qua đặc trưng là hệ quả của nhãn (Mục 4.3).

### Rò rỉ qua đặc trưng là hệ quả của nhãn | proxy leakage
Đặc trưng hợp lệ về thời gian nhưng tồn tại vì nhãn, ví dụ số lần gọi tổng đài sau sự cố khi dự đoán có sự cố hay không. Không kiểm tra tự động nào phát hiện được; phải xem xét từng đặc trưng.

### Kho đặc trưng | feature store
Hạ tầng gồm kho ngoại tuyến (lịch sử giá trị, hỗ trợ ghép theo thời điểm), kho trực tuyến (giá trị mới nhất, tra cứu với độ trễ thấp) và sổ đăng ký giữ một định nghĩa duy nhất cho mỗi đặc trưng. Tạo điều kiện, nhưng không tự động loại bỏ lệch huấn luyện–phục vụ (Mục 4.4).

## Đánh giá

### Đánh giá theo lát cắt | slice-based evaluation
Tính chỉ số riêng cho từng nhóm con của dữ liệu. Cần thiết vì chỉ số tổng thể là trung bình theo lưu lượng, nên nhóm đa số quyết định kết quả; trong thí nghiệm ở Mục 6.3, độ chính xác tổng thể tăng 19,8 điểm phần trăm trong khi một nhóm giảm 11,6 điểm.

### Kiểm thử hành vi | behavioral testing
Kiểm thử hành vi của mô hình trên những đầu vào cụ thể thay vì độ chính xác trung bình. Ba loại theo CheckList (Ribeiro và cộng sự, 2020): chức năng tối thiểu, bất biến, và kỳ vọng có hướng (Mục 6.4).

### ML Test Score | ML Test Score
Bộ 28 mục kiểm tra chia thành bốn nhóm: dữ liệu, phát triển mô hình, hạ tầng, giám sát (Breck và cộng sự, 2017). Mỗi mục được 0,5 điểm nếu làm thủ công có ghi lại kết quả, 1 điểm nếu tự động và định kỳ; điểm cuối cùng là giá trị nhỏ nhất trong bốn điểm nhóm (Mục 6.5).

### Chỉ số ngoại tuyến | offline metric
Chỉ số đo trên tập kiểm tra, rẻ và nhanh, dùng để lặp khi phát triển mô hình. Phải được kiểm tra định kỳ rằng nó vẫn tương quan với chỉ số sản phẩm đo trực tuyến (Mục 6.2).

## Phục vụ và ra mắt

### Ngân sách độ trễ | latency budget
Phân bổ tổng thời gian cho phép của một yêu cầu cho từng thành phần: mạng, tra cứu đặc trưng, tiền xử lý, suy luận, hậu xử lý, dự phòng. Suy luận mô hình thường chỉ chiếm một phần của ngân sách (Mục 7.2).

### Phân vị độ trễ | latency percentile
p50, p95, p99: giá trị mà 50%, 95%, 99% số yêu cầu có độ trễ không vượt quá. Mục tiêu mức dịch vụ thường đặt trên phân vị cao vì phân phối độ trễ lệch phải (Mục 7.3).

### Hiệu ứng toả nhánh | fan-out, tail at scale
Khi một yêu cầu gọi $k$ dịch vụ song song và đợi đủ $k$ kết quả, thời gian trả lời là giá trị lớn nhất của $k$ biến ngẫu nhiên; phân vị $q$ của nó ứng với phân vị $q^{1/k}$ của mỗi dịch vụ. Với $k = 100$, 63% số yêu cầu gặp ít nhất một dịch vụ chậm hơn p99 của nó (Mục 7.3).

### Yêu cầu dự phòng | hedged request
Nếu sau một khoảng thời gian chưa có kết quả, gửi thêm yêu cầu tới một bản sao khác và dùng kết quả về trước. Tốn thêm một phần nhỏ tài nguyên để giảm phần đuôi độ trễ (Dean và Barroso, 2013).

### Shadow | shadow deployment
Mô hình mới nhận cùng lưu lượng thật với mô hình cũ, dự đoán được ghi lại nhưng không được dùng. Người dùng không chịu rủi ro; là cách rẻ nhất để đo lệch huấn luyện–phục vụ trên lưu lượng thật (Mục 8.2).

### Canary | canary deployment
Đưa mô hình mới tới một phần nhỏ lưu lượng rồi tăng dần, dừng lại nếu chỉ số vận hành xấu đi. Bảo vệ khỏi sự cố lớn, nhưng không đo được chỉ số sản phẩm vì ở vài phần trăm lưu lượng không đủ mẫu (Mục 8.3).

### Blue–green | blue–green deployment
Duy trì hai môi trường giống nhau và chuyển toàn bộ lưu lượng từ môi trường này sang môi trường kia, cho phép quay lui tức thì. Là cơ chế triển khai, không cho biết gì về chất lượng mô hình.

### A/B test | A/B test, online controlled experiment
Chia ngẫu nhiên người dùng thành các nhánh nhận các phiên bản khác nhau, để đo tác động nhân quả của thay đổi lên chỉ số sản phẩm. Cỡ mẫu tỉ lệ nghịch với bình phương mức chênh lệch cần phát hiện (Mục 8.4).

### Lực kiểm định | statistical power
Xác suất phát hiện được một hiệu ứng có thật với kích thước cho trước, thường chọn 80%. Muốn phát hiện một hiệu ứng nhỏ bằng một nửa thì cần gấp bốn lần số mẫu.

### CUPED | controlled experiment using pre-experiment data
Kỹ thuật giảm phương sai trong A/B test: dùng giá trị của cùng chỉ số trong giai đoạn trước thí nghiệm làm biến hiệp phương sai. Nếu tương quan giữa hai giá trị là $\rho$, phương sai và cỡ mẫu cần thiết giảm theo hệ số $1 - \rho^2$ (Deng và cộng sự, 2013; Mục 8.4).

### Xem kết quả giữa chừng | peeking
Kiểm tra p-value liên tục và dừng khi thấy $p < 0{,}05$, làm tỉ lệ dương tính giả cao hơn nhiều so với 5%. Cách xử lý: cố định cỡ mẫu trước, hoặc dùng kiểm định tuần tự (Mục 8.5).

### Chỉ số bảo vệ | guardrail metric
Chỉ số theo dõi để phát hiện tác hại của một thay đổi, khác với chỉ số chính được khai báo trước khi chạy thí nghiệm (Mục 8.5).

### Tỉ lệ mẫu lệch | sample ratio mismatch
Số người dùng thực tế ở các nhánh chênh lệch so với tỉ lệ thiết kế nhiều hơn mức ngẫu nhiên cho phép. Là dấu hiệu thí nghiệm có lỗi, và kết quả khi đó không đáng tin (Mục 8.5).

## Dịch chuyển phân phối

### Dịch chuyển phân phối | distribution shift, dataset shift
Phân phối dữ liệu lúc sử dụng khác phân phối dữ liệu lúc huấn luyện. Ba loại chính: covariate shift, label shift, concept drift (Mục 9.1).

### Covariate shift | covariate shift
$P(X)$ thay đổi, $P(Y \mid X)$ giữ nguyên. Phát hiện được bằng cách giám sát $X$. Gây hại chủ yếu khi mô hình phải ngoại suy ra vùng ít dữ liệu huấn luyện.

### Label shift | label shift, prior probability shift
$P(Y)$ thay đổi, $P(X \mid Y)$ giữ nguyên. Phát hiện được gián tiếp qua $X$, vì phân phối của $X$ là hỗn hợp theo $Y$. Làm thay đổi precision, recall và hiệu chuẩn xác suất dù độ chính xác có thể gần như không đổi.

### Concept drift | concept drift
$P(Y \mid X)$ thay đổi. Không phát hiện được bằng cách chỉ giám sát $X$, vì $P(X)$ có thể giữ nguyên; cần tín hiệu về nhãn. Trong thí nghiệm ở Mục 9.2, concept drift làm độ chính xác giảm từ 75,6% xuống 26,9% trong khi kiểm định trên $X$ không thấy gì bất thường.

### PSI | population stability index
$\sum_j (T_j - B_j)\ln(T_j/B_j)$ trên hai phân phối đã chia bin; bằng tổng hai chiều của phân kỳ KL. Các ngưỡng 0,10 và 0,25 là quy tắc kinh nghiệm từ lĩnh vực chấm điểm tín dụng, không tính tới cỡ mẫu và số bin (Mục 9.4, 9.5).

### Phân kỳ Jeffreys | Jeffreys divergence
Tổng đối xứng $D_{\mathrm{KL}}(P \,\|\, Q) + D_{\mathrm{KL}}(Q \,\|\, P)$ của hai chiều phân kỳ KL. PSI chính là phân kỳ Jeffreys giữa hai phân phối đã chia bin.

### Kiểm định Kolmogorov–Smirnov hai mẫu | two-sample Kolmogorov–Smirnov test
Kiểm định so sánh hai phân phối một chiều mà không giả định dạng phân phối, dựa trên khoảng cách lớn nhất giữa hai hàm phân phối thực nghiệm. Có p-value, nên ngưỡng tự điều chỉnh theo cỡ mẫu (Mục 9.3).

### MMD | maximum mean discrepancy
Khoảng cách giữa hai phân phối nhiều chiều, tính qua một hàm kernel, kiểm định bằng hoán vị. Dùng khi dịch chuyển nằm ở quan hệ giữa các đặc trưng (Mục 9.3).

### Bộ phân loại miền | domain classifier
Mô hình được huấn luyện để phân biệt dữ liệu cũ với dữ liệu mới; nếu phân biệt được tốt hơn đoán ngẫu nhiên thì đã có dịch chuyển, và AUC của nó đo mức độ dịch chuyển (Mục 9.3).

### Hiệu chỉnh cho kiểm định nhiều lần | multiple testing correction
Điều chỉnh ngưỡng khi thực hiện nhiều kiểm định cùng lúc. Không hiệu chỉnh thì với 200 đặc trưng ở $\alpha = 0{,}01$, trung bình có 2 báo động giả mỗi lần kiểm tra. Các phương pháp phổ biến: Bonferroni, Benjamini–Hochberg (Mục 9.3).

## Giám sát

### Chỉ số vận hành | operational metrics
Độ trễ, lưu lượng, tỉ lệ lỗi, mức sử dụng tài nguyên. Khi có sự cố thì biết ngay, giống mọi dịch vụ phần mềm khác (Mục 10.1).

### Chỉ số riêng của học máy | ML-specific metrics
Độ chính xác, phân phối dự đoán, phân phối đặc trưng, chất lượng dữ liệu đầu vào. Khi có sự cố thường không biết ngay, nên hệ thống học máy cần giám sát riêng (Mục 10.1).

### SLI, SLO | service level indicator, service level objective
SLI là một đại lượng đo mức dịch vụ, ví dụ p99 độ trễ; SLO là giá trị mục tiêu cho SLI đó, ví dụ p99 dưới 200 ms trong 99,9% số phút (Beyer và cộng sự, 2016).

### Độ lệch dự đoán | prediction bias
Chênh lệch giữa phân phối nhãn được dự đoán và phân phối nhãn quan sát được. Không phải phép kiểm tra đầy đủ, vì một mô hình luôn dự đoán tỉ lệ trung bình cũng có độ lệch bằng 0, nhưng thay đổi đột ngột của nó thường là dấu hiệu sự cố. Không cần nhãn cho từng dự đoán, chỉ cần tỉ lệ nền theo thời gian (Mục 10.3).

### Giới hạn hành động | action limits
Mức trần cho số hành động thật mỗi loại trong mỗi khoảng thời gian, ví dụ số tài khoản bị khoá mỗi giờ; chạm trần thì cảnh báo và cần người xem xét. Không cần biết mô hình đúng hay sai (Mục 10.4).

### Hệ thống phía trước | up-stream producers
Các hệ thống cung cấp dữ liệu cho hệ thống học máy. Cần được giám sát và có SLO phù hợp với nhu cầu của hệ thống học máy; cảnh báo phải được truyền theo cả hai chiều (Mục 10.4).

### Báo động giả | false alarm
Cảnh báo kích hoạt khi hệ thống không có vấn đề. Khi báo động giả chiếm đa số, người nhận mất niềm tin và bỏ qua cả cảnh báo thật; đây là khó khăn được nhắc nhiều nhất trong nghiên cứu của Shankar và cộng sự (Mục 3.2, 10.5).

## Huấn luyện lại

### Độ cũ của mô hình | model staleness
Mức suy giảm chất lượng của mô hình theo thời gian kể từ lần huấn luyện cuối. Trong thí nghiệm ở Mục 11.1, không huấn luyện lại làm mất 13,19 điểm phần trăm độ chính xác trung bình sau một năm.

### Kích hoạt huấn luyện lại | retraining trigger
Điều kiện bắt đầu một lần huấn luyện lại: theo lịch, theo chất lượng, theo dịch chuyển, hoặc theo lượng dữ liệu mới. Kích hoạt theo dịch chuyển không phát hiện được concept drift (Mục 11.2).

### Huấn luyện lại từ đầu, huấn luyện có trạng thái | stateless retraining, stateful training
Huấn luyện lại từ đầu tạo mô hình mới trên một cửa sổ dữ liệu; huấn luyện có trạng thái tiếp tục cập nhật mô hình hiện có bằng dữ liệu mới. Cách thứ nhất tốn hơn nhưng dễ tái lập và dễ quay lui hơn (Huyen, 2022; Mục 11.3).

### Quên thảm khốc | catastrophic forgetting
Hiện tượng mô hình được cập nhật liên tục trên dữ liệu mới mất khả năng xử lý những gì đã học từ dữ liệu cũ.

## Vòng phản hồi

### Vòng phản hồi trực tiếp | direct feedback loop
Mô hình ảnh hưởng trực tiếp tới việc chọn dữ liệu huấn luyện tương lai của chính nó. Về lý thuyết nên dùng thuật toán bandit; cách giảm nhẹ thực tế là thêm ngẫu nhiên hoá hoặc tách riêng một phần dữ liệu không chịu ảnh hưởng của mô hình (Mục 12.1).

### Vòng phản hồi ẩn | hidden feedback loop
Hai hệ thống ảnh hưởng lẫn nhau một cách gián tiếp qua thế giới bên ngoài. Khó phát hiện vì không có đường dữ liệu nào trong sơ đồ hệ thống thể hiện mối liên hệ (Mục 12.1).

### Vòng phản hồi thoái hoá | degenerate feedback loop
Dự đoán ảnh hưởng tới phản hồi của người dùng, và phản hồi được dùng làm nhãn cho lần huấn luyện sau, nên hệ thống tự củng cố những gì nó đã tin. Dữ liệu vẫn trông tốt trong khi hệ thống chỉ còn hiển thị một phần nhỏ của danh mục (Huyen, 2022; Mục 12.2).

### Khai thác và khám phá | exploitation and exploration
Đánh đổi giữa dùng những lựa chọn đã biết là tốt và thử những lựa chọn chưa biết. Trong mô phỏng ở Bài 8, một vị trí ngẫu nhiên trong mười bù đủ chi phí của nó sau vài chục vòng; mức ngẫu nhiên hoá lớn làm giảm chất lượng người dùng nhận được.

### Hiệu chỉnh theo xác suất hiển thị | inverse propensity weighting
Gán cho mỗi mẫu huấn luyện trọng số bằng nghịch đảo xác suất nó được hiển thị, để bù cho việc một số lựa chọn được hiển thị nhiều hơn. Đòi hỏi ghi lại xác suất hiển thị tại thời điểm hiển thị (Mục 12.4).

### Nhóm đối chứng | holdout group
Một tỉ lệ nhỏ lưu lượng không chịu tác động của mô hình, nhận kết quả ngẫu nhiên hoặc theo quy tắc đơn giản. Cung cấp dữ liệu không bị mô hình làm sai lệch và thước đo đáng tin cậy về giá trị thật của hệ thống (Mục 12.4).

### Thiên lệch vị trí | position bias
Món được xếp ở vị trí cao nhận nhiều lượt nhấp hơn chỉ vì được nhìn thấy nhiều hơn. Cách xử lý thường dùng: đưa vị trí hiển thị vào mô hình khi huấn luyện, rồi đặt nó bằng cùng một hằng số khi phục vụ.

## LLMOps

### Giám khảo LLM | LLM-as-a-judge
Dùng một mô hình ngôn ngữ để chấm đầu ra theo tiêu chí cho trước. Rẻ hơn người chấm nhiều lần, nhưng có các thiên lệch đã biết (vị trí, độ dài, tự ưu tiên) và có thể thay đổi theo phiên bản, nên cần đối chiếu định kỳ với người chấm (Mục 13.2, 13.3).

### Criteria drift | criteria drift
Hiện tượng người xây dựng tiêu chí chấm thay đổi tiêu chí trong khi đọc và chấm đầu ra của mô hình (Shankar và cộng sự, 2024). Vì vậy tiêu chí chấm cần được xây dựng sau khi đọc một lượng đầu ra thật, và xem lại định kỳ (Mục 13.3).

### Tập đánh giá chuẩn | golden set
Tập câu hỏi được chọn lọc, gồm cả trường hợp khó, mà hệ thống phải đạt trước mỗi lần phát hành. Cần được cập nhật: mỗi sự cố trong sản xuất nên trở thành một trường hợp mới (Mục 13.4).

### Độ trung thành | faithfulness, groundedness
Mức độ các khẳng định trong câu trả lời được ngữ cảnh đã truy xuất hỗ trợ. Dùng để phát hiện lỗi ở bước sinh của hệ thống RAG, tách biệt với recall@$k$ dùng cho lỗi ở bước truy xuất (Mục 13.5).

### Rào chắn | guardrails
Các kiểm tra chạy quanh lượt gọi mô hình: phát hiện prompt injection, che dữ liệu cá nhân, chặn yêu cầu ngoài phạm vi ở đầu vào; kiểm tra định dạng, nội dung có hại, độ bám nguồn và giới hạn hành động ở đầu ra (Mục 13.7).

### Prompt injection | prompt injection
Nội dung do người dùng hoặc tài liệu bên ngoài đưa vào làm mô hình thực hiện chỉ dẫn không mong muốn thay cho chỉ dẫn gốc. Chi tiết ở Chương 11 của giáo trình *Ứng dụng LLM*.

### Thời gian tới token đầu tiên | time to first token (TTFT)
Thời gian từ lúc gửi yêu cầu tới lúc nhận token đầu tiên của câu trả lời, quyết định cảm nhận về độ nhanh khi bắt đầu trả lời. Phân biệt với thời gian giữa các token, quyết định tốc độ hiển thị câu trả lời (Mục 13.6).

### Bản ghi vết | trace
Bản ghi đầy đủ của một lượt xử lý: prompt, các đoạn đã truy xuất kèm định danh và điểm, đầu ra, điểm đánh giá, độ trễ, và chi phí tách theo từng thành phần. Thiếu định danh các đoạn truy xuất thì không tách được lỗi truy xuất khỏi lỗi sinh (Mục 13.5, 13.6).
