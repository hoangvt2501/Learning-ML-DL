# Ngân hàng câu hỏi tự kiểm tra — MLOps

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Nguyên lý CACE phát biểu điều gì?
- [ ] Mọi thay đổi đều cần được rà soát mã
- [x] Đổi bất cứ thứ gì là đổi tất cả — không đầu vào nào thực sự độc lập với đầu vào nào
- [ ] Mô hình phức tạp luôn tốt hơn mô hình đơn giản
- [ ] Cấu hình phải được quản lý phiên bản
> *Changing Anything Changes Everything.* Thêm hay bớt một đặc trưng làm đổi trọng số của **mọi** đặc trưng còn lại. Bài báo nói rõ nó áp cho cả siêu tham số, thiết lập học, cách lấy mẫu và ngưỡng hội tụ. Hệ quả: không tồn tại "thay đổi nhỏ và cô lập" trong một mô hình.

### Con số "5% mã học máy, 95% mã keo" thực chất nói về cái gì?
- [ ] Chú thích của Hình 1 trong bài báo, về tỉ lệ diện tích các hộp
- [x] Tỉ lệ mã keo khi **tái sử dụng các gói đa dụng**, nêu trong mục *Glue Code*
- [ ] Tỉ lệ thời gian kỹ sư bỏ ra cho từng loại công việc
- [ ] Kết quả đo trên một mẫu hệ thống sản xuất của Google
> Bài báo viết có biên: *nhiều nhất* 5% và *ít nhất* 95%. Và kết luận đi kèm hay bị bỏ mất: vì tỉ lệ ấy, đôi khi **viết một lời giải gọn và thuần nội bộ rẻ hơn tái dùng một gói đa dụng**; chiến lược chống glue code là bọc gói hộp đen sau một API chung.

### "Correction cascade" là gì?
- [ ] Chuỗi lỗi lan từ pipeline này sang pipeline khác
- [x] Học một mô hình nhỏ nhận đầu ra của mô hình có sẵn làm đầu vào để sửa cho bài toán hơi khác
- [ ] Nhiều mô hình cùng dự đoán rồi lấy trung bình
- [ ] Sửa nhãn sai trong tập huấn luyện theo tầng
> Nhanh trước mắt, nhưng nó tạo ra một phụ thuộc hệ thống mới vào mô hình gốc, khiến việc cải thiện mô hình gốc từ đó trở nên rất đắt.

### Đặc trưng nào sau đây **không** thuộc nhóm "phụ thuộc dữ liệu ít dùng"?
- [ ] Legacy feature — đưa vào từ sớm, sau bị các đặc trưng mới làm cho thừa
- [ ] Bundled feature — vào cả gói vì áp lực thời hạn
- [ ] ε-feature — tăng độ chính xác một chút xíu nhưng phức tạp lớn
- [x] Unstable feature — lấy từ hệ thống thượng nguồn thay đổi hành vi theo thời gian
> Ba cái đầu là *underutilized data dependency*; cái cuối là *unstable data dependency*, một dạng nợ khác. Cách phát hiện nhóm ít dùng mà bài báo đề xuất là **đánh giá bỏ-từng-đặc-trưng định kỳ**.

## Chương 2

### Ba chữ V của MLOps theo nghiên cứu phỏng vấn của Shankar và cộng sự là gì?
- [ ] Volume, Variety, Velocity
- [x] Velocity, Validation, Versioning
- [ ] Validation, Visualization, Versioning
- [ ] Velocity, Value, Versioning
> Velocity: đi từ ý tưởng tới mô hình đã huấn luyện trong một ngày. Validation: loại ý tưởng tồi càng sớm càng tốt. Versioning: giữ nhiều phiên bản để **quay lui được**.

### CT (Continuous Training) khác CI/CD ở chỗ nào?
- [ ] CT chạy nhanh hơn CI/CD
- [x] CI/CD lo việc **mã** thay đổi; CT lo việc **dữ liệu** thay đổi mà không ai chạm vào repo
- [ ] CT chỉ áp dụng cho mô hình học sâu
- [ ] CT thay thế hoàn toàn CI/CD trong ML
> Trong phần mềm thường, hệ thống chỉ đổi khi có người sửa mã. Trong ML có thêm một nguồn thay đổi: thế giới. Tạo tác của ML gồm **mã + dữ liệu + mô hình**, ba thứ có nhịp thay đổi khác nhau — đó là lý do MLOps tồn tại như một nghề riêng.

### Ở mức 1 của thang trưởng thành MLOps, thứ được bàn giao là gì?
- [ ] Một mô hình đã huấn luyện
- [x] **Toàn bộ pipeline huấn luyện**
- [ ] Một hệ thống CI/CD hoàn chỉnh
- [ ] Một tập dữ liệu đã được kiểm định
> Đây chính là trục của thang đo: mức 0 bàn giao một mô hình, mức 1 bàn giao một pipeline, mức 2 bàn giao một hệ thống tự cập nhật.

### Anti-pattern "Keeping GPUs Warm" sai ở chỗ nào?
- [ ] Nó làm hỏng phần cứng
- [x] Nó tối ưu nhầm thứ khan hiếm — cái khan hiếm là giả thuyết tốt và thời gian đọc kết quả, không phải GPU
- [ ] Nó tốn điện
- [ ] Nó chỉ sai với đội nhỏ
> Chạy càng nhiều thí nghiệm càng tốt nghe như một đức tính, nhưng thí nghiệm không có giả thuyết chỉ sinh ra nhiễu và rất nhiều phiên bản phải quản.

## Chương 3

### Phổ lỗi dữ liệu "cứng → mềm → dịch chuyển" đòi hỏi mấy cách phản ứng?
- [ ] Một — cảnh báo cho cả ba
- [x] Ba — chặn pipeline, cảnh báo và theo dõi tỉ lệ, điều tra và có thể huấn luyện lại
- [ ] Hai — chặn hoặc bỏ qua
- [ ] Không cần phản ứng, chỉ cần ghi log
> Trộn cả ba vào một cơ chế cảnh báo chính là nguồn gốc của báo động giả — nỗi đau được nhắc nhiều nhất trong nghiên cứu phỏng vấn.

### Vì sao lỗi mềm khó bắt hơn lỗi cứng?
- [ ] Vì nó xảy ra ít hơn
- [x] Vì nó vẫn cho ra dự đoán trông hợp lý, nên trôi qua mọi kiểm tra
- [ ] Vì nó chỉ xuất hiện trong dữ liệu huấn luyện
- [ ] Vì nó cần nhãn mới phát hiện được
> Vài trường null hay một đơn vị bị đổi không vi phạm ràng buộc nào, nhưng nó lặng lẽ làm lệch dự đoán. Vì vậy lỗi mềm cần theo dõi **tỉ lệ** chứ không cần chặn.

### Hậu quả nghiêm trọng nhất của báo động giả là gì?
- [ ] Tốn tài nguyên tính toán
- [ ] Làm chậm pipeline
- [x] Đội ngừng tin vào cảnh báo, nên lúc có sự cố thật thì không ai nhìn
- [ ] Làm sai lệch chỉ số chất lượng
> Một hệ cảnh báo không được tin thì **tệ hơn không có**, vì nó tạo cảm giác an toàn giả.

## Chương 4

### Training–serving skew là gì?
- [ ] Mô hình huấn luyện trên GPU nhưng phục vụ trên CPU
- [x] Đặc trưng lúc phục vụ được tính khác với lúc huấn luyện
- [ ] Tập huấn luyện và tập kiểm tra có phân phối khác nhau
- [ ] Độ trễ lúc phục vụ cao hơn lúc huấn luyện
> Thường vì hai bên có hai bản cài đặt riêng: một bản SQL trên kho dữ liệu, một bản viết lại trong dịch vụ cho đủ nhanh. Điểm khó chịu nhất là **nó không tạo ra lỗi nào** — mô hình vẫn chạy, chỉ là kém đi.

### Point-in-time correctness đòi hỏi điều gì?
- [ ] Đặc trưng phải được cập nhật theo thời gian thực
- [x] Giá trị đặc trưng dùng cho một dòng huấn luyện phải là giá trị **quan sát được tại thời điểm cần dự đoán**
- [ ] Tập huấn luyện phải được sắp xếp theo thời gian
- [ ] Mọi bản ghi phải có dấu thời gian
> Vi phạm nó là rò rỉ nhãn. Nguy hiểm đặc biệt vì nó **làm chỉ số ngoại tuyến đẹp lên**, nên không ai nghi ngờ cho tới khi đã triển khai và đã ra vài dự đoán sai.

### Khi nào **không** nên xây kho đặc trưng?
- [ ] Khi đội chưa có kinh nghiệm vận hành
- [x] Khi chỉ có một mô hình, chỉ chấm điểm theo lô, và đặc trưng gần như tĩnh
- [ ] Khi dữ liệu quá lớn
- [ ] Khi chưa có CI/CD
> Ba điều kiện để nó xứng đáng: nhiều mô hình dùng chung đặc trưng, có phục vụ trực tuyến độ trễ thấp, đặc trưng phụ thuộc thời gian. Thiếu cả ba thì một bảng trong kho dữ liệu cộng kỷ luật dùng chung một hàm biến đổi là đủ và rẻ hơn nhiều.

### Loại rò rỉ nào **không** có kiểm tra máy nào bắt được?
- [ ] Rò rỉ theo thời gian
- [ ] Rò rỉ theo thực thể khi chia ngẫu nhiên
- [x] Rò rỉ qua đặc trưng thay mặt cho nhãn (proxy leakage)
- [ ] Rò rỉ do trộn tập huấn luyện và tập kiểm tra
> "Số lần gọi tổng đài sau sự cố" để dự đoán "có gặp sự cố không" là hợp lệ về mặt thời gian nhưng là **hệ quả** của nhãn. Chỉ có cách hỏi từng đặc trưng: tại thời điểm dự đoán, giá trị này đã tồn tại chưa, và nó tồn tại *vì* nhãn hay không?

## Chương 5

### Vì sao `cudnn.benchmark = True` làm mất tính tất định?
- [ ] Vì nó dùng số dấu phẩy động độ chính xác thấp
- [x] Vì nó chọn thuật toán tích chập bằng cách đo thời gian lúc chạy, mà kết quả đo phụ thuộc tải máy
- [ ] Vì nó bỏ qua hạt giống ngẫu nhiên
- [ ] Vì nó chạy song song nhiều luồng
> Đây là một trong ba chỗ hay bị bỏ sót, cùng với hạt giống của worker trong DataLoader và việc **số GPU thực chất là một siêu tham số**.

### Ghi lại "giả thuyết đang kiểm chứng" trước khi chạy thí nghiệm chữa được thói xấu nào?
- [ ] Keeping GPUs Warm
- [x] Retrofitting an Explanation
- [ ] Undocumented Tribal Knowledge
- [ ] Industry–Classroom Mismatch
> Nếu giả thuyết được ghi lại **trước**, thì không thể gắn một lời giải thích vào sau khi đã thấy kết quả.

## Chương 6

### Trong thí nghiệm đánh giá theo lát cắt ở Mục 6.3, điều gì xảy ra?
- [ ] Mọi lát cắt đều cải thiện, chỉ khác mức độ
- [x] Chỉ số gộp tăng 19,8 điểm nhưng nhóm 15% người dùng mới **mất 11,6 điểm** và tụt xuống dưới mô hình cũ
- [ ] Chỉ số gộp giảm nhưng các lát cắt đều tăng
- [ ] Không lát cắt nào đổi đáng kể
> Cơ chế: đặc trưng mới chỉ có tín hiệu với nhóm đa số, nên mô hình học trọng số lớn cho nó; với nhóm thiểu số chính trọng số lớn ấy **bơm nhiễu thẳng vào dự đoán**. Đây là CACE ở dạng đo được.

### ML Test Score được tính bằng cách nào?
- [ ] Tổng điểm của cả 28 mục
- [ ] Trung bình điểm của bốn nhóm
- [x] **Giá trị nhỏ nhất** trong bốn điểm nhóm
- [ ] Tỉ lệ phần trăm số mục đã đạt
> Mỗi mục được 0,5 điểm nếu làm thủ công có ghi chép, 1 điểm nếu tự động định kỳ. Lấy `min` vì cả bốn nhóm đều cần thiết — nên đội có hạ tầng hoàn hảo mà không giám sát gì thì **điểm bằng 0**.

### Ba họ kiểm thử hành vi là gì?
- [ ] Đơn vị, tích hợp, đầu-cuối
- [x] Bất biến, kỳ vọng có hướng, chức năng tối thiểu
- [ ] Chính xác, độ nhạy, độ đặc hiệu
- [ ] Ngoại tuyến, trực tuyến, shadow
> Giá trị của chúng: một mô hình có thể tăng 1 điểm độ chính xác trung bình mà lại mất khả năng xử lý đúng một ca hiển nhiên. Chỉ số trung bình không bao giờ báo chuyện đó.

## Chương 7

### Vì sao phải báo cáo p99 chứ không phải độ trễ trung bình?
- [ ] Vì trung bình khó tính hơn
- [x] Vì phân phối độ trễ lệch phải rất mạnh, và khi một yêu cầu toả ra $k$ nhánh thì thời gian trả lời là **giá trị lớn nhất** của $k$ biến
- [ ] Vì p99 luôn nhỏ hơn trung bình
- [ ] Vì công cụ giám sát chỉ hỗ trợ p99
> Với $k$ nhánh độc lập, phân vị $q$ của max là $F^{-1}(q^{1/k})$ — muốn p99 tổng thể với $k=100$ thì mỗi nhánh phải đạt tới phân vị 99,99.

### Nếu một yêu cầu gọi 100 dịch vụ song song, mỗi dịch vụ "nhanh ở mức p99", bao nhiêu phần trăm yêu cầu chạm phải ít nhất một dịch vụ chậm?
- [ ] 1%
- [ ] 10%
- [x] Khoảng 63%
- [ ] Khoảng 99%
> $1 - 0{,}99^{100} = 63{,}4\%$. Cải thiện độ trễ **trung bình** không giúp gì cho chuyện này; chỉ có ba cách chữa: giảm $k$, cắt đuôi bằng thời hạn chờ và trả lời một phần, hoặc gửi yêu cầu dự phòng.

### Trong một ngân sách độ trễ điển hình, phần suy luận mô hình chiếm bao nhiêu?
- [ ] Gần như toàn bộ
- [x] Thường chưa tới một nửa — phần còn lại là mạng, tra đặc trưng, tiền xử lý và hậu xử lý
- [ ] Không đáng kể
- [ ] Đúng một nửa theo quy ước
> Vì vậy tối ưu mô hình từ 60 ms xuống 30 ms chỉ cải thiện tổng thể 20%, trong khi bỏ một lần tra đặc trưng thừa có thể cho nhiều hơn thế.

## Chương 8

### Canary **không** trả lời được câu hỏi nào?
- [ ] Tỉ lệ lỗi có tăng không
- [ ] Độ trễ có xấu đi không
- [x] Chỉ số sản phẩm có tốt lên không
- [ ] Phân phối dự đoán có lệch không
> Ở 1% lưu lượng, chỉ số sản phẩm gần như chắc chắn không đủ mẫu để có ý nghĩa thống kê. **Canary bảo vệ khỏi thảm hoạ, A/B trả lời câu hỏi về giá trị** — lẫn hai vai trò này là lỗi quy trình phổ biến nhất.

### Với tỉ lệ nền 5%, cần bao nhiêu mẫu mỗi nhánh để bắt một cải thiện tương đối 1% ở lực kiểm định 80%?
- [ ] Khoảng 30 nghìn
- [ ] Khoảng 300 nghìn
- [x] Khoảng 3 triệu
- [ ] Khoảng 30 triệu
> Chính xác là 2 996 694 — tức khoảng một tháng lưu lượng cho **một** ý tưởng nếu mỗi nhánh nhận 100 000 lượt/ngày. Đây là lý do quy trình tốt loại ý tưởng tồi ở giai đoạn rẻ hơn trước khi tiêu lưu lượng.

### Cỡ mẫu A/B phụ thuộc mức cải thiện $\Delta$ như thế nào?
- [ ] Tỉ lệ nghịch với $\Delta$
- [x] Tỉ lệ nghịch với $\Delta^2$
- [ ] Tỉ lệ thuận với $\Delta$
- [ ] Không phụ thuộc $\Delta$
> Muốn bắt cải thiện nhỏ đi một nửa thì cần **gấp bốn lần** lưu lượng. Đòn bẩy rẻ nhất để rút ngắn thí nghiệm là giảm **phương sai** (ví dụ bằng CUPED), không phải giảm lực kiểm định.

### "Peeking" trong A/B test là gì và vì sao nó sai?
- [ ] Xem kết quả của nhánh đối thủ — sai vì thiên lệch
- [x] Kiểm tra p-value liên tục và dừng ngay khi thấy $p<0{,}05$ — sai vì làm tỉ lệ dương tính giả vượt xa 5%
- [ ] Dùng dữ liệu của thí nghiệm trước — sai vì rò rỉ
- [ ] Chạy thí nghiệm quá lâu — sai vì tốn lưu lượng
> Chữa bằng cách cố định cỡ mẫu trước, hoặc dùng phương pháp thiết kế cho theo dõi liên tục (kiểm định tuần tự, ranh giới alpha-spending).

## Chương 9

### Loại dịch chuyển nào **không** dò được bằng cách chỉ giám sát $X$?
- [ ] Covariate shift
- [ ] Label shift
- [x] Concept drift
- [ ] Cả ba đều dò được
> Theo đúng định nghĩa, concept drift giữ nguyên $P(X)$ và chỉ đổi $P(Y|X)$. Thí nghiệm ở Mục 9.2: độ chính xác rơi từ 75,6% xuống **26,9%** trong khi kiểm định KS trên đặc trưng cho $p = 0{,}48$.

### PSI thực chất là đại lượng gì?
- [ ] Khoảng cách Kolmogorov–Smirnov
- [ ] KL divergence một chiều
- [x] Phân kỳ Jeffreys — tổng hai chiều của KL divergence
- [ ] Khoảng cách Wasserstein
> $\sum_j (T_j-B_j)\ln(T_j/B_j) = D_{KL}(T\|B) + D_{KL}(B\|T)$. Nó đối xứng, khác với KL.

### Vì sao ngưỡng PSI 0,25 không dùng thẳng được?
- [ ] Vì nó quá chặt với mọi bài toán
- [x] Vì PSI kỳ vọng khi **không có dịch chuyển** xấp xỉ $2(k-1)/n$, nên ngưỡng cố định vừa báo động giả ở mẫu nhỏ vừa mù ở mẫu lớn
- [ ] Vì nó chỉ đúng cho biến rời rạc
- [ ] Vì nó chỉ đúng khi dùng đúng 10 bin
> Đo thực tế: ở $n=50$, quy tắc PSI > 0,25 báo động **73% số lần khi không có gì xảy ra**; ở $n=1000$ với dịch chuyển thật $0{,}3\sigma$ thì nó **không báo lần nào**, trong khi KS báo 100%.

### Dùng tham chiếu trượt 30 ngày cho phép dò dịch chuyển có nhược điểm gì?
- [ ] Tốn bộ nhớ hơn
- [x] Nó **không bao giờ thấy dịch chuyển chậm**, vì tham chiếu trôi theo
- [ ] Nó chỉ dùng được cho biến liên tục
- [ ] Nó làm tăng báo động giả
> Cách dùng đúng là giữ **cả hai**: tham chiếu cố định để biết khoảng cách so với mô hình, tham chiếu trượt để biết có cú sốc mới không.

## Chương 10

### Trong bốn lớp giám sát, lớp nào là tuyến phòng thủ chính khi nhãn tới trễ?
- [ ] Đầu vào thô
- [ ] Đặc trưng
- [x] Dự đoán
- [ ] Chỉ số chất lượng
> Càng gần nhãn thì càng có ý nghĩa nhưng càng khó lấy. Không có nhãn nhanh thì phân phối dự đoán, tỉ lệ theo lớp và **độ lệch dự đoán** là thứ dùng được ngay.

### Phép kiểm tra "độ lệch dự đoán" (prediction bias) nói gì, và giới hạn của nó là gì?
- [ ] Nó chứng minh mô hình đúng; không có giới hạn nào
- [x] Phân phối nhãn dự đoán thường phải bằng phân phối nhãn quan sát; giới hạn là một mô hình rỗng chỉ đoán tỉ lệ trung bình cũng thoả mãn
- [ ] Nó đo độ lệch giữa tập huấn luyện và tập kiểm tra
- [ ] Nó chỉ dùng được cho hồi quy
> Chính bài báo tự nêu giới hạn ấy, nhưng vẫn khẳng định **thay đổi của chỉ số này thường là dấu hiệu sự cố cần chú ý**. Sức mạnh của nó là không cần nhãn cho từng dự đoán, chỉ cần tỉ lệ nền theo thời gian.

### Nguyên tắc "cảnh báo trên triệu chứng, không trên nguyên nhân" nghĩa là gì?
- [ ] Chỉ cảnh báo khi đã biết nguyên nhân gốc
- [x] "Tỉ lệ chuyển đổi giảm 15%" đáng đánh thức người; "PSI của đặc trưng số 37 vượt 0,2" thì không — đó là manh mối điều tra
- [ ] Cảnh báo phải ghi rõ nguyên nhân
- [ ] Chỉ cảnh báo những gì tự động sửa được
> Kèm ba nguyên tắc nữa: mỗi cảnh báo phải gắn với một hành động; phân tầng trang / vé / bảng điều khiển; và đòi hỏi **độ dai** qua nhiều cửa sổ liên tiếp.

### "Fixed threshold in dynamic system" là dạng nợ gì?
- [ ] Ngưỡng cảnh báo đặt quá thấp
- [x] Ngưỡng quyết định đặt tay không còn hợp lệ sau khi mô hình được huấn luyện lại
- [ ] Ngưỡng học được nhưng không lưu phiên bản
- [ ] Ngưỡng khác nhau giữa các môi trường
> Đây là một ví dụ rất gọn của CACE: huấn luyện lại đã đổi phân phối điểm số đầu ra, nên mọi thứ phụ thuộc thang điểm ấy đều đã đổi. Cách chữa: **học ngưỡng cùng mô hình và coi nó là một phần của tạo tác mô hình.**

## Chương 11

### Trong thí nghiệm về độ cũ ở Mục 11.1, không huấn luyện lại suốt một năm làm mất bao nhiêu?
- [ ] Khoảng 1 điểm phần trăm
- [ ] Khoảng 5 điểm phần trăm
- [x] 13,19 điểm phần trăm trung bình, và tới tuần 52 thì mô hình còn 45,5% — tệ hơn đoán bừa
- [ ] Mô hình giữ nguyên chất lượng
> Đáng sợ nhất là nó tụt **dần**, nên không có ngày nào báo động nổ.

### So với huấn luyện lại hằng quý, huấn luyện lại hằng tuần cho thêm bao nhiêu trong thí nghiệm ấy?
- [ ] Hơn 10 điểm
- [ ] Khoảng 5 điểm
- [x] Chỉ 0,91 điểm, đổi lại chi phí vận hành gấp khoảng 13 lần
- [ ] Không có chênh lệch nào
> Lợi ích giảm dần rất nhanh, nên **nhịp huấn luyện lại là một quyết định kinh tế, không phải kỹ thuật**: đo đường cong này trên bài toán của mình rồi so giá trị một điểm phần trăm với chi phí một chu kỳ.

### Kích hoạt huấn luyện lại "theo dịch chuyển" có điểm mù nào?
- [ ] Nó cần nhãn
- [x] Nó mù với concept drift, vì concept drift giữ nguyên $P(X)$
- [ ] Nó chỉ chạy được theo lô
- [ ] Nó không dùng được cho mô hình học sâu
> Kích hoạt theo chất lượng thì ngược lại: bắt được concept drift nhưng **cần nhãn**, nên bị chặn bởi độ trễ nhãn.

### Cửa nào hay bị quên nhất trong pipeline huấn luyện lại tự động?
- [ ] Kiểm định dữ liệu đầu vào
- [ ] Đánh giá trên tập cố định
- [x] **So với mô hình đang chạy** trước khi thay thế
- [ ] Đăng ký vào sổ mô hình
> Tài liệu Google Cloud tách **kiểm định mô hình** khỏi **đánh giá mô hình** đúng vì lý do này: đánh giá trả lời "mô hình này tốt đến đâu", kiểm định trả lời "có nên thay cái đang chạy bằng nó không".

## Chương 12

### Vòng phản hồi thoái hoá nguy hiểm vì điều gì?
- [ ] Nó làm mô hình chạy chậm đi
- [x] Dữ liệu trông rất đẹp — tỉ lệ click trên những gì được hiển thị vẫn cao, vì hệ thống chỉ hiển thị thứ nó tin là tốt
- [ ] Nó làm tăng chi phí huấn luyện
- [ ] Nó chỉ ảnh hưởng tới người dùng mới
> Mọi chỉ số ngoại tuyến đều xanh trong khi hệ thống đã khoá cứng vào một phần nhỏ của danh mục.

### Trong thí nghiệm ở Mục 12.3, hệ không ngẫu nhiên hoá nhìn thấy bao nhiêu phần danh mục?
- [ ] Gần như toàn bộ
- [ ] Khoảng 50%
- [x] 9,5% — và không bao giờ mở rộng thêm
- [ ] Phụ thuộc số vòng huấn luyện lại
> Nó đạt chất lượng 0,351 so với trần 0,555, tức 63% của mức tốt nhất có thể. Dành 30% số chỗ cho hiển thị ngẫu nhiên nâng lên 0,402 và mở vùng nhìn thấy lên 13,4%.

### Cách chữa nào cho vòng phản hồi vừa cho dữ liệu sạch để huấn luyện, vừa cho thước đo trung thực về giá trị hệ thống?
- [ ] Ngẫu nhiên hoá một phần chỗ hiển thị
- [x] Giữ một **nhóm đối chứng** không bị mô hình tác động
- [ ] Hiệu chỉnh theo xác suất hiển thị
- [ ] Đưa vị trí vào mô hình rồi đặt cứng lúc suy luận
> Cả bốn đều là cách chữa hợp lệ, nhưng nhóm đối chứng là thứ duy nhất cho bạn **nguồn dữ liệu không thiên lệch** đồng thời là thước đo giá trị thật. Nó là khoản bảo hiểm rẻ nhất trong toàn bộ tài liệu.

### Muốn dùng hiệu chỉnh theo xác suất hiển thị (inverse propensity weighting) thì phải chuẩn bị gì **trước**?
- [ ] Một mô hình nhân quả
- [x] Ghi lại **xác suất hiển thị tại thời điểm hiển thị** — không thể khôi phục về sau
- [ ] Một tập kiểm tra riêng
- [ ] Nhãn của toàn bộ danh mục
> Đây là loại quyết định phải làm trước, giống như việc ghi lại id đoạn đã truy hồi trong hệ RAG ở Mục 13.5.

## Chương 13

### Nhà cung cấp đổi bản mô hình mà không báo là dạng nợ nào?
- [ ] Glue code
- [x] Dạng cực đoan của **unstable data dependency**
- [ ] Correction cascade
- [ ] Configuration debt
> Cách chữa cũng giống: **ghim phiên bản mô hình**, và coi mỗi lần đổi phiên bản là một lần phát hành đầy đủ, phải qua shadow và canary.

### Cấu trúc đánh giá LLM ba tầng phân bổ tỉ lệ thế nào?
- [ ] Người chấm 100%, giám khảo 10%, quy tắc 1%
- [x] Kiểm tra quy tắc 100%, giám khảo LLM 10–20% (hoặc 1–5% khi lưu lượng lớn), người chấm mẫu nhỏ định kỳ
- [ ] Cả ba tầng đều chấm 100%
- [ ] Chỉ dùng giám khảo LLM là đủ
> Đây là phiên bản LLM của bốn lớp giám sát: càng gần sự thật thì càng đắt, nên phải lấy mẫu. Tầng người chấm hay bị bỏ nhất, nhưng **giám khảo cũng là một mô hình nên nó cũng trôi**.

### Câu trả lời RAG sai thì bước chẩn đoán đầu tiên là gì?
- [ ] Đổi mô hình sinh
- [x] Hỏi xem **tài liệu đúng có nằm trong ngữ cảnh không** — để tách lỗi truy hồi khỏi lỗi sinh
- [ ] Tăng số đoạn lấy về
- [ ] Sửa prompt
> Không nằm trong ngữ cảnh ⇒ lỗi truy hồi, đo bằng recall@k. Có mà vẫn sai ⇒ lỗi sinh, đo bằng **độ bám nguồn**. Gộp hai loại vào một chỉ số là lý do phổ biến nhất khiến đội không biết sửa chỗ nào.

### Thiên lệch vị trí của giám khảo LLM chữa bằng cách nào?
- [ ] Dùng mô hình giám khảo lớn hơn
- [x] Hỏi hai lần với thứ tự đảo, chỉ tính thắng khi cả hai lần đều thắng
- [ ] Chuẩn hoá điểm theo độ dài
- [ ] Tăng nhiệt độ sinh
> Chuẩn hoá theo độ dài là cách chữa cho **thiên lệch độ dài**; dùng giám khảo khác họ là cách chữa cho **thiên lệch tự ưu ái**.

### Đổi thứ nào sau đây **không** tạo ra một phiên bản mới của hệ thống LLM cần đánh giá lại?
- [ ] Prompt hệ thống
- [ ] Cách chia đoạn tài liệu
- [ ] Ảnh chụp kho tri thức đã lập chỉ mục
- [x] Không có — cả ba đều tạo ra phiên bản mới
> Một phiên bản gồm ít nhất sáu thành phần: prompt, bản mô hình và tham số sinh, cấu hình truy hồi, ảnh chụp corpus, định nghĩa công cụ, và rubric đánh giá. Đây là CACE phát biểu lại cho thời LLM.
