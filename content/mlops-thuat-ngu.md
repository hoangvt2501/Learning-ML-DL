# Từ điển thuật ngữ — MLOps

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mọi định nghĩa bám sát đúng cách dùng trong giáo trình.

## Nợ kỹ thuật

### Nợ kỹ thuật | technical debt
Chi phí dài hạn phải trả cho việc đi nhanh trong ngắn hạn. Không phải mọi khoản nợ đều xấu, nhưng **mọi khoản nợ đều phải trả lãi**. Nợ của hệ thống ML đặc biệt nguy hiểm vì nó **ẩn**: nó nằm ở mức hệ thống chứ không ở mức mã nguồn, nên đọc code không thấy.

### Nguyên lý CACE | Changing Anything Changes Everything
Đổi bất cứ thứ gì là đổi tất cả. Thêm hay bớt một đặc trưng làm đổi trọng số của mọi đặc trưng còn lại; nguyên lý này áp cho cả siêu tham số, thiết lập học, cách lấy mẫu và ngưỡng hội tụ. Hệ quả: **không tồn tại thay đổi nhỏ và cô lập trong một mô hình**.

### Rối | entanglement
Tên gọi khác của hệ quả CACE: các thành phần của hệ thống ML dính chặt vào nhau tới mức không lý luận cục bộ được.

### Thác hiệu chỉnh | correction cascade
Học một mô hình nhỏ nhận đầu ra của mô hình có sẵn làm đầu vào để sửa cho một bài toán hơi khác. Nhanh trước mắt, nhưng tạo phụ thuộc hệ thống mới và làm việc cải thiện mô hình gốc trở nên rất đắt.

### Người dùng không khai báo | undeclared consumer
Một hệ thống khác lặng lẽ dùng đầu ra mô hình của bạn làm đầu vào mà không ai biết. Trong công nghệ phần mềm gọi là **visibility debt**. Nguy hiểm vì nó tạo ràng buộc chặt mà không nhìn thấy được.

### Phụ thuộc dữ liệu không ổn định | unstable data dependency
Tín hiệu đầu vào lấy từ một hệ thống thay đổi hành vi theo thời gian — ví dụ nó cũng là một mô hình tự cập nhật. Cách chữa là đóng băng phiên bản, nhưng đóng băng cũng có giá: giá trị cũ dần và phải nuôi nhiều phiên bản.

### Phụ thuộc dữ liệu ít dùng | underutilized data dependency
Đặc trưng gần như không đóng góp gì nhưng vẫn ở trong mô hình, khiến hệ thống dễ vỡ không cần thiết. Bốn đường vào: *legacy feature*, *bundled feature*, *ε-feature*, *correlated feature*. Phát hiện bằng **đánh giá bỏ-từng-đặc-trưng định kỳ**.

### Mã keo | glue code
Mã để nối các gói đa dụng lại với nhau. Một hệ thống trưởng thành có thể chỉ gồm *nhiều nhất* 5% mã học máy và *ít nhất* 95% mã keo. Chiến lược chống lại: bọc các gói hộp đen sau một API chung.

### Rừng rậm pipeline | pipeline jungle
Trường hợp riêng của mã keo, mọc ở khâu chuẩn bị dữ liệu: cào, ghép, lấy mẫu chồng chất theo năm tháng, đầy tệp trung gian.

### Nợ cấu hình | configuration debt
Nợ tích tụ trong các tuỳ chọn cấu hình của hệ thống. Cả nhà nghiên cứu lẫn kỹ sư đều coi cấu hình là chuyện phụ, trong khi một dòng cấu hình sai có sức phá hoại ngang một dòng mã sai.

### Ngưỡng cố định trong hệ thống động | fixed threshold in dynamic system
Ngưỡng quyết định đặt bằng tay không còn hợp lệ sau khi mô hình được huấn luyện lại, vì phân phối điểm số đầu ra đã đổi. Cách chữa: **học ngưỡng cùng mô hình và coi nó là một phần của tạo tác mô hình**.

## Vòng đời và quy trình

### Huấn luyện liên tục | CT — continuous training
Chữ C thứ ba bên cạnh CI/CD. CI/CD lo việc **mã** thay đổi; CT lo việc **dữ liệu** thay đổi mà không ai chạm vào repo.

### Ba chữ V | Velocity, Validation, Versioning
Ba biến quyết định thành bại của một lần triển khai: tốc độ vòng lặp thí nghiệm, kiểm định càng sớm càng tốt, và giữ nhiều phiên bản để quay lui được.

### Ba mức tự động hoá | MLOps maturity levels 0 / 1 / 2
Mức 0 bàn giao **một mô hình**; mức 1 bàn giao **cả pipeline huấn luyện**; mức 2 bàn giao **một hệ thống tự cập nhật** có CI/CD cho chính pipeline.

### Kiểm định mô hình | model validation
Khác với **đánh giá mô hình**: đánh giá trả lời "mô hình này tốt đến đâu", kiểm định trả lời "**có nên thay thế mô hình đang chạy bằng nó không**". Là cửa hay bị quên nhất trong pipeline huấn luyện lại tự động.

### Sổ đăng ký mô hình | model registry
Nơi một tạo tác mô hình có danh tính và vòng đời: huấn luyện từ commit nào, dữ liệu nào, đã qua cửa nào, đang ở trạng thái nào, và **phiên bản nào đang thực sự chạy**.

## Dữ liệu và đặc trưng

### Hợp đồng dữ liệu | data contract
Biến giả định ngầm về dữ liệu thành ràng buộc kiểm tra được bằng máy. Bốn tầng: schema, miền giá trị, thống kê của lô, và bất biến giữa các cột hoặc bảng.

### Phổ lỗi dữ liệu | hard → soft → drift errors
Ba loại lỗi đòi hỏi **ba cách phản ứng khác nhau**: lỗi cứng thì chặn pipeline; lỗi mềm thì cảnh báo và theo dõi tỉ lệ; dịch chuyển thì điều tra và có thể huấn luyện lại. Trộn cả ba vào một cơ chế là nguồn gốc của báo động giả.

### Độ trễ nhãn | label lag
Khoảng cách giữa lúc dự đoán và lúc biết đáp án thật. Nó quyết định bạn giám sát được gì và huấn luyện lại được nhanh đến đâu — và quan trọng nhất, quyết định bạn có nhìn thấy concept drift hay không.

### Lệch giữa huấn luyện và phục vụ | training–serving skew
Đặc trưng lúc phục vụ được tính khác lúc huấn luyện, thường vì hai bên có hai bản cài đặt riêng. Điểm khó chịu: **nó không tạo ra lỗi nào**, mô hình vẫn chạy, chỉ là kém đi.

### Tính đúng theo thời điểm | point-in-time correctness
Với mỗi dòng huấn luyện có thời điểm dự đoán $t$, giá trị đặc trưng phải là giá trị **quan sát được tại $t$**, không phải giá trị hôm nay. Vi phạm là rò rỉ nhãn.

### Ghép theo thời điểm | as-of join, point-in-time join
Phép ghép lấy bản ghi đặc trưng mới nhất **nhưng không muộn hơn** thời điểm dự đoán của mỗi dòng nhãn.

### Rò rỉ nhãn | label leakage
Dùng thông tin mà lúc dự đoán thật không hề có. Nguy hiểm đặc biệt vì nó **làm chỉ số ngoại tuyến đẹp lên**, nên không ai nghi ngờ cho tới khi đã triển khai.

### Rò rỉ qua đặc trưng thay mặt nhãn | proxy leakage
Đặc trưng hợp lệ về mặt thời gian nhưng là **hệ quả** của nhãn chứ không phải nguyên nhân. Không có kiểm tra máy nào bắt được; chỉ có cách hỏi từng đặc trưng về quan hệ nhân quả.

### Kho đặc trưng | feature store
Hạ tầng gồm kho ngoại tuyến (lịch sử, hỗ trợ ghép theo thời điểm), kho trực tuyến (giá trị mới nhất, độ trễ thấp) và **sổ đăng ký** giữ một định nghĩa duy nhất cho mỗi đặc trưng. Nó **không tự động** xoá skew, chỉ tạo điều kiện.

## Đánh giá

### Đánh giá theo lát cắt | sliced evaluation
Đo chỉ số riêng cho từng nhóm con thay vì chỉ đo chỉ số gộp. Cần thiết vì chỉ số gộp là trung bình có trọng số theo lưu lượng, nên bị nhóm đa số chi phối hoàn toàn.

### Kiểm thử hành vi | behavioral testing
Khẳng định về **hành vi** thay vì về độ chính xác trung bình. Ba họ: **bất biến** (đổi thứ không nên ảnh hưởng thì dự đoán giữ nguyên), **kỳ vọng có hướng** (đổi thứ có hướng rõ thì dự đoán đổi đúng hướng), **chức năng tối thiểu** (các ca đơn giản không được phép sai).

### ML Test Score
Bộ 28 mục kiểm thử chia đều bốn nhóm: dữ liệu, mô hình, hạ tầng, giám sát. 0,5 điểm nếu làm thủ công có ghi chép, 1 điểm nếu tự động định kỳ. **Điểm cuối là giá trị nhỏ nhất trong bốn điểm nhóm** — vì cả bốn đều cần thiết.

### Chỉ số đại diện | proxy metric
Chỉ số ngoại tuyến rẻ, dùng để lặp nhanh, thay cho chỉ số sản phẩm đắt. Phải **kiểm chứng định kỳ rằng nó còn tương quan** với chỉ số sản phẩm; khi tương quan đứt thì nó thành một trò chơi tự sướng.

## Phục vụ và ra mắt

### Ngân sách độ trễ | latency budget
Phân bổ tổng thời gian cho phép của một yêu cầu thành từng phần: mạng, tra đặc trưng, tiền xử lý, suy luận, hậu xử lý, dự phòng. Phần suy luận thường chiếm **chưa tới một nửa**.

### Đuôi khi toả nhánh | tail at scale
Khi một yêu cầu gọi $k$ nhánh song song và đợi đủ cả $k$, thời gian trả lời là **giá trị lớn nhất** của $k$ biến, nên phân vị $q$ của tổng thể ứng với phân vị $q^{1/k}$ của mỗi nhánh. Với $k=100$, 63% số yêu cầu chạm ít nhất một nhánh vượt p99.

### Yêu cầu dự phòng | hedged request
Sau một khoảng chờ chưa có trả lời thì gửi thêm một bản sao tới máy chủ khác và lấy cái nào về trước. Tốn thêm vài phần trăm tài nguyên để cắt hẳn phần đuôi.

### Chạy song song ngầm | shadow deployment
100% lưu lượng chạy qua cả mô hình cũ và mới, nhưng chỉ bản cũ trả lời người dùng. Rủi ro cho người dùng bằng 0; là cách rẻ nhất để đo **training–serving skew** trên lưu lượng thật.

### Ra mắt dần | canary deployment
Tăng dần tỉ lệ lưu lượng 1% → 5% → 25% → 100%, dừng nếu chỉ số xấu đi. **Canary bảo vệ khỏi thảm hoạ; nó không đo được chỉ số sản phẩm** vì ở 1% lưu lượng không đủ mẫu để có ý nghĩa thống kê.

### Lực kiểm định | statistical power
Xác suất phát hiện được một hiệu ứng thật, thường lấy 80%. Cỡ mẫu A/B tỉ lệ nghịch với **bình phương** mức cải thiện: muốn bắt cải thiện nhỏ đi một nửa thì cần gấp bốn lần lưu lượng.

### CUPED | controlled experiment using pre-experiment data
Kỹ thuật giảm phương sai bằng cách dùng dữ liệu trước thí nghiệm của chính người dùng làm hiệp biến. Là đòn bẩy rẻ nhất để rút ngắn A/B test mà **không** đổi mức ý nghĩa hay lực kiểm định.

### Nhìn lén | peeking
Kiểm tra p-value liên tục và dừng ngay khi thấy $p<0{,}05$. Làm tỉ lệ dương tính giả vượt xa 5%. Chữa bằng cố định cỡ mẫu trước hoặc dùng kiểm định tuần tự.

### Chỉ số bảo vệ | guardrail metric
Chỉ số theo dõi để bảo đảm một thay đổi không làm hỏng thứ khác, khác với **chỉ số chính** được khai báo trước. Có nhiều chỉ số bảo vệ là cách đúng để tránh bẫy đa kiểm định.

## Dịch chuyển phân phối

### Dịch chuyển hiệp biến | covariate shift
$P(X)$ đổi, $P(Y \mid X)$ giữ nguyên. **Dò được bằng cách giám sát $X$.**

### Dịch chuyển nhãn | label shift, prior probability shift
$P(Y)$ đổi, $P(X \mid Y)$ giữ nguyên. Dò được gián tiếp qua $X$ vì phân phối biên của $X$ là hỗn hợp. Hại chính là làm hỏng hiệu chuẩn và làm lệch precision/recall.

### Trôi khái niệm | concept drift
$P(Y \mid X)$ đổi, $P(X)$ giữ nguyên. **Không dò được bằng cách chỉ giám sát $X$** — theo đúng định nghĩa. Muốn bắt thì phải có tín hiệu về nhãn.

### PSI | Population Stability Index
$\sum_j (T_j - B_j)\ln(T_j/B_j)$, thực chất là **phân kỳ Jeffreys** $D_{KL}(T\|B) + D_{KL}(B\|T)$. Ngưỡng 0,1 và 0,25 là quy tắc kinh nghiệm từ chấm điểm tín dụng, **không tính tới cỡ mẫu và số bin**.

### Phân kỳ Jeffreys | Jeffreys divergence
Tổng đối xứng của hai chiều KL divergence. Chính là PSI.

### Kiểm định Kolmogorov–Smirnov hai mẫu | two-sample KS test
So sánh hai phân phối một chiều mà không giả định dạng phân phối. Có p-value thật, nên **ngưỡng tự thích ứng theo cỡ mẫu** — ưu điểm quyết định so với PSI.

### MMD | maximum mean discrepancy
Phép đo khoảng cách giữa hai phân phối **nhiều chiều**, kiểm định được bằng hoán vị. Dùng khi dịch chuyển nằm ở tương quan giữa các đặc trưng chứ không ở phân phối biên.

### Bộ phân loại phân biệt miền | domain classifier
Huấn luyện một mô hình phân biệt mẫu nguồn với mẫu đích; AUC của nó là thước đo độ lớn dịch chuyển. Trực giác rõ và xử lý được nhiều chiều.

### Hiệu chỉnh đa kiểm định | multiple testing correction
Bắt buộc khi chạy kiểm định trên hàng trăm đặc trưng. Không hiệu chỉnh thì với 200 đặc trưng ở $\alpha=0{,}01$ kỳ vọng có **2 báo động giả mỗi lần chạy**. Dùng Bonferroni hoặc Benjamini–Hochberg.

## Giám sát

### Chỉ số vận hành | operational metrics
Độ trễ, thông lượng, tỉ lệ lỗi, mức dùng tài nguyên, thời gian hoạt động. Hỏng thì biết ngay — giống mọi dịch vụ khác.

### Chỉ số riêng của ML | ML-specific metrics
Độ chính xác, phân phối dự đoán, phân phối đặc trưng, chất lượng dữ liệu vào. Hỏng thì **không biết ngay** — đó là toàn bộ vấn đề.

### Độ lệch dự đoán | prediction bias
Phân phối nhãn **dự đoán** thường phải bằng phân phối nhãn **quan sát được**. Không phải phép kiểm tra đầy đủ (một mô hình rỗng cũng thoả mãn), nhưng **thay đổi của nó thường là dấu hiệu sự cố**. Ưu điểm: không cần nhãn cho từng dự đoán, chỉ cần tỉ lệ nền theo thời gian.

### Giới hạn hành động | action limit
Trần đặt cho số hành động thật mỗi loại trong mỗi cửa sổ thời gian; chạm trần thì báo động và có người vào xem. Không cần biết mô hình đúng hay sai — chỉ cần biết mức đó là chưa từng xảy ra.

### Nhà cung cấp thượng nguồn | up-stream producer
Hệ thống cấp dữ liệu cho pipeline ML. Phải được giám sát, kiểm thử và đạt một SLO **có tính đến nhu cầu của hệ ML ở hạ nguồn**; mọi cảnh báo phải truyền hai chiều.

### Mệt mỏi vì cảnh báo | alert fatigue
Hậu quả của báo động giả: đội ngừng tin vào cảnh báo. Một hệ cảnh báo không được tin **tệ hơn không có**, vì nó tạo cảm giác an toàn giả.

### SLI và SLO | service level indicator / objective
Chỉ số đo mức dịch vụ và mục tiêu đặt cho chỉ số ấy. Với ML, SLO phải gồm cả chỉ số vận hành lẫn chỉ số chất lượng.

## Vòng phản hồi

### Vòng phản hồi trực tiếp | direct feedback loop
Mô hình ảnh hưởng trực tiếp tới việc chọn dữ liệu huấn luyện tương lai của chính nó. Lời giải đúng về lý thuyết là thuật toán bandit; cách giảm nhẹ khả thi là thêm ngẫu nhiên hoá hoặc cô lập một phần dữ liệu.

### Vòng phản hồi ẩn | hidden feedback loop
Hai hệ thống ảnh hưởng lẫn nhau **gián tiếp qua thế giới**. Khó hơn hẳn vì không có đường dữ liệu nào trong sơ đồ hệ thống cho thấy mối liên hệ.

### Vòng phản hồi thoái hoá | degenerate feedback loop
Dự đoán ảnh hưởng tới phản hồi, phản hồi thành nhãn cho vòng sau, nên hệ thống **tự xác nhận chính mình**. Dữ liệu trông rất đẹp trong khi hệ thống đã khoá cứng vào một phần nhỏ của danh mục.

### Khai thác và khám phá | exploitation vs exploration
Đánh đổi giữa dùng thứ đã biết là tốt và thử thứ chưa biết. Một lượng ngẫu nhiên hoá **nhỏ** (5–10%) thường trả đủ tiền cho chính nó; một lượng lớn là thuế.

### Hiệu chỉnh theo xác suất hiển thị | inverse propensity weighting
Đánh trọng số mỗi mẫu huấn luyện bằng nghịch đảo xác suất nó được hiển thị, để bù thiên lệch phơi nhiễm. Đòi hỏi **ghi lại xác suất hiển thị tại thời điểm hiển thị** — không khôi phục về sau được.

### Nhóm đối chứng | holdout group
Một tỉ lệ nhỏ lưu lượng không bị mô hình tác động. Là **nguồn dữ liệu không thiên lệch duy nhất** và là thước đo trung thực về giá trị thật của hệ thống.

### Thiên lệch vị trí | position bias
Món xếp cao hơn nhận nhiều click hơn chỉ vì được nhìn thấy nhiều hơn. Chữa bằng cách đưa vị trí vào mô hình lúc huấn luyện rồi đặt nó thành hằng số lúc suy luận.

## LLMOps

### Giám khảo LLM | LLM-as-judge
Dùng một mô hình ngôn ngữ chấm điểm đầu ra theo rubric. Rẻ hơn người nhiều lần nhưng **cũng là một mô hình nên cũng trôi**, cần người chấm định kỳ để hiệu chuẩn lại.

### Tập vàng | golden set
Tập câu hỏi khó được tuyển chọn, phải đạt trước mỗi lần phát hành — tức một cửa chất lượng trong CI. **Phải sống**: mỗi sự cố sản xuất sinh ra ít nhất một ca mới.

### Độ bám nguồn | groundedness, faithfulness
Câu trả lời có được chống đỡ bởi ngữ cảnh đã lấy về hay không. Là chỉ số để phát hiện **lỗi sinh**, tách biệt với recall@k dùng để phát hiện **lỗi truy hồi**.

### Rào chắn | guardrails
Các kiểm tra chạy quanh lượt gọi mô hình ở cả hai đầu: lọc prompt injection và dữ liệu cá nhân ở đầu vào; kiểm tra định dạng, nội dung độc hại, độ bám nguồn và **giới hạn hành động** ở đầu ra.

### Chèn lệnh độc | prompt injection
Nội dung do người dùng hoặc tài liệu đưa vào làm mô hình bỏ qua chỉ dẫn gốc. Là lớp rủi ro mà MLOps cổ điển không có.

### Thời gian tới token đầu tiên | TTFT — time to first token
Quyết định cảm nhận về độ nhạy của hệ thống, khác với **thời gian giữa các token** quyết định cảm nhận về tốc độ.

### Bản ghi vết | trace
Bản ghi đầy đủ một lượt: prompt, đoạn đã truy hồi kèm id, đầu ra, điểm đánh giá, độ trễ và **chi phí tách theo từng dòng**. Không ghi id đoạn truy hồi thì sau này không tách được lỗi truy hồi khỏi lỗi sinh.
