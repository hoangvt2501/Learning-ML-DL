# Ngân hàng câu hỏi tự kiểm tra — MLOps

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Nguyên lý CACE phát biểu điều gì?
- [ ] Mọi thay đổi đều phải được rà soát mã
- [x] Thay đổi bất kỳ thành phần nào của hệ thống học máy cũng có thể thay đổi hành vi của toàn bộ hệ thống
- [ ] Mô hình phức tạp luôn tốt hơn mô hình đơn giản
- [ ] Cấu hình phải được quản lý phiên bản như mã
> *Changing Anything Changes Everything* (Sculley và cộng sự, 2015). Thêm hay bớt một đặc trưng làm trọng số của mọi đặc trưng khác thay đổi; nguyên lý áp dụng cả cho siêu tham số, cách lấy mẫu và ngưỡng hội tụ. Vì vậy không có thay đổi nào là nhỏ và cô lập trong một mô hình.

### Nhận định "5% mã học máy, 95% mã keo" nói về điều gì?
- [ ] Chú thích của Hình 1 trong bài báo, về tỉ lệ diện tích các ô
- [x] Tỉ lệ mã keo trong hệ thống tái sử dụng các thư viện đa dụng, nêu trong mục *Glue Code*
- [ ] Tỉ lệ thời gian kỹ sư dành cho từng loại công việc
- [ ] Kết quả đo trên một mẫu các hệ thống sản xuất của Google
> Bài báo diễn đạt có giới hạn: nhiều nhất 5% mã học máy và ít nhất 95% mã keo; đây là nhận định định tính. Kết luận đi kèm: đôi khi tự viết một giải pháp gọn còn rẻ hơn dùng lại thư viện đa dụng, và nên bọc các thư viện sau một giao diện chung.

### Correction cascade là gì?
- [ ] Chuỗi lỗi lan từ pipeline này sang pipeline khác
- [x] Học một mô hình mới nhận đầu ra của một mô hình có sẵn làm đầu vào, để giải một bài toán gần giống
- [ ] Kết hợp nhiều mô hình bằng cách lấy trung bình dự đoán
- [ ] Sửa nhãn sai trong tập huấn luyện theo nhiều bước
> Cách này nhanh trước mắt nhưng tạo ra phụ thuộc mới vào mô hình gốc, khiến việc cải thiện mô hình gốc trở nên tốn kém vì có thể làm hỏng mô hình phía sau.

### Trường hợp nào **không** thuộc nhóm phụ thuộc dữ liệu ít giá trị (underutilized data dependency)?
- [ ] Đặc trưng cũ đã bị các đặc trưng mới thay thế nhưng chưa được gỡ
- [ ] Một nhóm đặc trưng được thêm cả gói vì áp lực thời hạn
- [ ] Đặc trưng cải thiện độ chính xác rất ít nhưng làm hệ thống phức tạp hơn nhiều
- [x] Đặc trưng lấy từ một hệ thống phía trước thay đổi hành vi theo thời gian
> Ba trường hợp đầu là phụ thuộc dữ liệu ít giá trị; trường hợp cuối là phụ thuộc dữ liệu không ổn định, một dạng nợ khác. Cách phát hiện phụ thuộc ít giá trị mà bài báo đề xuất là định kỳ đánh giá lại mô hình khi bỏ từng đặc trưng.

### Vì sao lỗi của hệ thống học máy thường khó phát hiện hơn lỗi của phần mềm thông thường?
- [ ] Vì mô hình học máy chạy chậm hơn
- [x] Vì mô hình nhận đầu vào có phân phối đã thay đổi vẫn trả về kết quả đúng kiểu, đúng thời hạn, nhưng sai
- [ ] Vì mã học máy không thể kiểm thử đơn vị
- [ ] Vì hệ thống học máy không ghi log
> Phần mềm thông thường thường hỏng một cách dễ thấy (ngoại lệ, lỗi kiểu dữ liệu). Hệ thống học máy thường hỏng một cách im lặng, nên cần các cơ chế giám sát riêng (Chương 9, 10).

## Chương 2

### Ba yếu tố quyết định thành công theo nghiên cứu phỏng vấn của Shankar và cộng sự (2022) là gì?
- [ ] Volume, Variety, Velocity
- [x] Velocity, Validation, Versioning
- [ ] Validation, Visualization, Versioning
- [ ] Velocity, Value, Versioning
> Tốc độ: đi nhanh từ ý tưởng tới mô hình đã huấn luyện. Kiểm định: phát hiện ý tưởng kém và lỗi càng sớm càng tốt. Phiên bản: quản lý nhiều phiên bản mô hình và dữ liệu để quay lui được.

### Huấn luyện liên tục (CT) khác CI/CD ở điểm nào?
- [ ] CT chạy nhanh hơn CI/CD
- [x] CI/CD xử lý thay đổi của mã; CT xử lý thay đổi của dữ liệu, loại thay đổi không đi qua kho mã
- [ ] CT chỉ áp dụng cho mô hình học sâu
- [ ] CT thay thế hoàn toàn CI/CD trong học máy
> Trong phần mềm thông thường, hệ thống chỉ thay đổi khi có người sửa mã. Trong học máy, dữ liệu thay đổi theo thế giới và mô hình cũ dần, nên cần tự động huấn luyện lại. Tạo tác của học máy gồm mã, dữ liệu và mô hình, thay đổi theo ba nhịp khác nhau.

### Ở mức 1 của ba mức tự động hoá, thứ được bàn giao là gì?
- [ ] Một mô hình đã huấn luyện
- [x] Toàn bộ pipeline huấn luyện
- [ ] Một hệ thống CI/CD hoàn chỉnh cho pipeline
- [ ] Một tập dữ liệu đã được kiểm định
> Mức 0 bàn giao một mô hình, mức 1 bàn giao pipeline huấn luyện (có huấn luyện liên tục), mức 2 bàn giao một hệ thống có CI/CD cho chính pipeline.

### Thói quen *keeping GPUs warm* sai ở đâu?
- [ ] Làm hỏng phần cứng
- [x] Tối ưu sai tài nguyên: thứ khan hiếm là giả thuyết tốt và thời gian để đọc kết quả, không phải GPU
- [ ] Tốn điện năng
- [ ] Chỉ sai với nhóm nhỏ
> Chạy càng nhiều thí nghiệm càng tốt nghe như tận dụng tài nguyên, nhưng thí nghiệm không có giả thuyết tạo ra nhiều kết quả khó diễn giải và nhiều phiên bản phải quản lý.

## Chương 3

### Ba loại lỗi dữ liệu (cứng, mềm, dịch chuyển) cần mấy cách phản ứng?
- [ ] Một: cảnh báo cho cả ba
- [x] Ba: dừng pipeline; cảnh báo và theo dõi tỉ lệ; điều tra và có thể huấn luyện lại
- [ ] Hai: dừng hoặc bỏ qua
- [ ] Không cần phản ứng, chỉ cần ghi log
> Dùng chung một cơ chế cảnh báo cho cả ba loại là nguyên nhân phổ biến của báo động giả, khó khăn được nhắc nhiều nhất trong nghiên cứu của Shankar và cộng sự.

### Vì sao lỗi mềm khó phát hiện hơn lỗi cứng?
- [ ] Vì lỗi mềm xảy ra ít hơn
- [x] Vì dự đoán vẫn trông hợp lý, nên vượt qua các kiểm tra đơn giản
- [ ] Vì lỗi mềm chỉ xuất hiện trong dữ liệu huấn luyện
- [ ] Vì lỗi mềm chỉ phát hiện được khi có nhãn
> Một số trường bị thiếu hay một đơn vị tính bị đổi không vi phạm ràng buộc nào, nhưng làm dự đoán lệch dần. Vì vậy lỗi mềm được theo dõi theo tỉ lệ thay vì dừng pipeline.

### Hậu quả nghiêm trọng nhất của báo động giả là gì?
- [ ] Tốn tài nguyên tính toán
- [ ] Làm chậm pipeline
- [x] Người nhận mất niềm tin vào cảnh báo và bỏ qua cả cảnh báo thật
- [ ] Làm sai lệch chỉ số chất lượng
> Khi phần lớn cảnh báo là giả, hệ thống cảnh báo tạo ra cảm giác an toàn không có cơ sở (Mục 10.5).

### Tầng nào của hợp đồng dữ liệu phát hiện được một lô dữ liệu có số dòng chỉ bằng một nửa bình thường?
- [ ] Lược đồ
- [ ] Miền giá trị
- [ ] Không tầng nào; cần kiểm tra thủ công
- [x] Quan hệ giữa các bảng và giữa các lô, ví dụ số dòng so với trung vị 7 ngày
> Mỗi dòng trong lô có thể hoàn toàn hợp lệ về lược đồ và miền giá trị; chỉ khi so sánh với các lô trước mới thấy thiếu dữ liệu. Vi phạm tầng 3 và 4 thường là lỗi mềm, nên cảnh báo thay vì dừng pipeline.

## Chương 4

### Lệch giữa huấn luyện và phục vụ (training–serving skew) là gì?
- [ ] Mô hình huấn luyện trên GPU nhưng phục vụ trên CPU
- [x] Đặc trưng lúc phục vụ được tính khác với lúc huấn luyện
- [ ] Tập huấn luyện và tập kiểm tra có phân phối khác nhau
- [ ] Độ trễ lúc phục vụ cao hơn lúc huấn luyện
> Thường vì có hai bản cài đặt riêng: một truy vấn SQL khi huấn luyện và một đoạn mã viết lại trong dịch vụ phục vụ. Loại lỗi này không gây ra lỗi chương trình nào; mô hình vẫn chạy nhưng kém đi.

### Tính đúng theo thời điểm đòi hỏi điều gì?
- [ ] Đặc trưng phải được cập nhật theo thời gian thực
- [x] Giá trị đặc trưng dùng cho một dòng huấn luyện phải là giá trị quan sát được tại thời điểm cần dự đoán của dòng đó
- [ ] Tập huấn luyện phải được sắp xếp theo thời gian
- [ ] Mọi bản ghi phải có dấu thời gian
> Vi phạm tính chất này là đưa thông tin tương lai vào tập huấn luyện. Nó làm kết quả đánh giá ngoại tuyến tốt lên, nên thường chỉ bị phát hiện sau khi triển khai.

### Khi nào **không** cần xây dựng kho đặc trưng?
- [ ] Khi nhóm chưa có kinh nghiệm vận hành
- [x] Khi chỉ có một mô hình, chỉ chấm điểm theo lô, và đặc trưng gần như tĩnh
- [ ] Khi dữ liệu quá lớn
- [ ] Khi chưa có CI/CD
> Kho đặc trưng đáng đầu tư khi có đồng thời: nhiều mô hình dùng chung đặc trưng, phục vụ trực tuyến với độ trễ thấp, và đặc trưng phụ thuộc thời gian. Thiếu các điều kiện đó, một bảng trong kho dữ liệu cộng với một hàm biến đổi dùng chung là đủ.

### Loại rò rỉ nào **không** có kiểm tra tự động nào phát hiện được?
- [ ] Rò rỉ theo thời gian
- [ ] Rò rỉ theo thực thể khi chia dữ liệu ngẫu nhiên
- [x] Rò rỉ qua đặc trưng là hệ quả của nhãn
- [ ] Rò rỉ do trộn lẫn tập huấn luyện và tập kiểm tra
> Ví dụ "số lần gọi tổng đài sau sự cố" để dự đoán "khách có gặp sự cố không": hợp lệ về thời gian nhưng là hệ quả của nhãn. Chỉ có cách xem xét từng đặc trưng: tại thời điểm dự đoán, giá trị này đã tồn tại chưa, và nó có tồn tại vì nhãn không?

## Chương 5

### Vì sao `cudnn.benchmark = True` làm mất tính tất định?
- [ ] Vì nó dùng số thực dấu phẩy động độ chính xác thấp
- [x] Vì nó chọn thuật toán tích chập bằng cách đo thời gian lúc chạy, và kết quả đo phụ thuộc tải của máy
- [ ] Vì nó bỏ qua hạt giống ngẫu nhiên
- [ ] Vì nó chạy song song nhiều luồng
> Cùng với các tiến trình con của DataLoader và số GPU, đây là những điểm hay bị bỏ sót khi cố định các yếu tố để tái lập kết quả.

### Đổi từ 4 sang 8 GPU mà giữ nguyên kích thước lô trên mỗi GPU thì điều gì thay đổi?
- [ ] Không có gì thay đổi ngoài tốc độ
- [x] Kích thước lô hiệu dụng tăng gấp đôi, tức bài toán tối ưu đã thay đổi
- [ ] Learning rate tự động giảm một nửa
- [ ] Mô hình có gấp đôi số tham số
> Vì vậy số GPU là một siêu tham số. Ngay cả khi giữ kích thước lô hiệu dụng, thứ tự cộng gradient thay đổi, và vì phép cộng số thực dấu phẩy động không có tính kết hợp, kết quả khác đi ở những chữ số cuối.

### Ghi lại giả thuyết cần kiểm tra trước khi chạy thí nghiệm giúp tránh thói quen xấu nào?
- [ ] Keeping GPUs warm
- [x] Retrofitting an explanation
- [ ] Undocumented tribal knowledge
- [ ] Industry–classroom mismatch
> Nếu giả thuyết được ghi lại trước khi chạy, không thể gắn một lời giải thích vào sau khi đã thấy kết quả.

### Thẻ mô hình (model card) chứa những thông tin gì?
- [ ] Chỉ kiến trúc và số tham số
- [x] Mục đích sử dụng, những trường hợp không nên dùng, dữ liệu huấn luyện, và kết quả đánh giá tách theo các nhóm liên quan
- [ ] Mã nguồn huấn luyện
- [ ] Chi phí phần cứng
> Mitchell và cộng sự (2019) đề xuất thẻ mô hình để những hiểu biết về mô hình không chỉ nằm trong đầu người huấn luyện mà đi cùng mô hình trong sổ đăng ký.

## Chương 6

### Trong thí nghiệm đánh giá theo lát cắt ở Mục 6.3, điều gì xảy ra?
- [ ] Mọi lát cắt đều cải thiện, chỉ khác mức độ
- [x] Độ chính xác tổng thể tăng 19,8 điểm phần trăm, nhưng nhóm 15% khách hàng mới giảm 11,6 điểm, xuống dưới mô hình cũ
- [ ] Độ chính xác tổng thể giảm nhưng mọi lát cắt đều tăng
- [ ] Không lát cắt nào thay đổi đáng kể
> Đặc trưng mới chỉ có thông tin với khách hàng cũ, nên mô hình học cho nó trọng số lớn (2,28); với khách hàng mới, trọng số lớn đó đưa nhiễu vào dự đoán. Đây là một ví dụ đo được của nguyên lý CACE.

### ML Test Score cuối cùng được tính thế nào?
- [ ] Tổng điểm của 28 mục
- [ ] Trung bình điểm của bốn nhóm
- [x] Giá trị nhỏ nhất trong bốn điểm nhóm
- [ ] Tỉ lệ phần trăm số mục đã đạt
> Mỗi mục được 0,5 điểm nếu làm thủ công có ghi lại kết quả, 1 điểm nếu tự động và định kỳ. Lấy giá trị nhỏ nhất vì cả bốn nhóm đều cần thiết: hệ thống có hạ tầng hoàn hảo mà không có giám sát vẫn có điểm 0.

### Mục Monitor 6 của ML Test Score kiểm tra điều gì?
- [ ] Chất lượng dự đoán trên dữ liệu thật
- [x] Suy giảm, đột ngột hay từ từ, về tốc độ huấn luyện, độ trễ phục vụ, thông lượng hay bộ nhớ
- [ ] Đặc trưng lúc huấn luyện và lúc phục vụ tính ra cùng giá trị
- [ ] Mô hình có quá cũ không
> Chất lượng dự đoán là Monitor 7; đặc trưng khớp giữa huấn luyện và phục vụ là Monitor 3; độ cũ của mô hình là Monitor 4.

### Ba loại kiểm thử hành vi theo CheckList là gì?
- [ ] Đơn vị, tích hợp, đầu cuối
- [x] Chức năng tối thiểu, bất biến, kỳ vọng có hướng
- [ ] Độ chính xác, độ nhạy, độ đặc hiệu
- [ ] Ngoại tuyến, trực tuyến, shadow
> Một mô hình có thể tăng một điểm độ chính xác trung bình mà sai ở một trường hợp hiển nhiên; chỉ số trung bình không phát hiện được điều đó, kiểm thử hành vi thì có (Ribeiro và cộng sự, 2020).

## Chương 7

### Vì sao theo dõi p99 thay vì độ trễ trung bình?
- [ ] Vì trung bình khó tính hơn
- [x] Vì phân phối độ trễ lệch phải, và khi một yêu cầu toả ra $k$ nhánh, thời gian trả lời là giá trị lớn nhất của $k$ biến
- [ ] Vì p99 luôn nhỏ hơn trung bình
- [ ] Vì công cụ giám sát chỉ hỗ trợ p99
> Với $k$ nhánh độc lập, phân vị $q$ của giá trị lớn nhất là $F^{-1}(q^{1/k})$: để cả yêu cầu đạt p99 với $k = 100$, mỗi nhánh phải đạt phân vị 99,99.

### Một yêu cầu gọi 100 dịch vụ song song, mỗi dịch vụ chậm hơn p99 của nó ở 1% số lần gọi. Bao nhiêu phần trăm yêu cầu gặp ít nhất một dịch vụ chậm?
- [ ] 1%
- [ ] 10%
- [x] Khoảng 63%
- [ ] Khoảng 99%
> $1 - 0{,}99^{100} \approx 63{,}4\%$. Giảm độ trễ trung bình không giải quyết được; các cách xử lý là giảm số nhánh, đặt thời hạn chờ và trả lời với kết quả một phần, hoặc gửi yêu cầu dự phòng (Dean và Barroso, 2013).

### Trong ví dụ ngân sách độ trễ 150 ms ở Mục 7.2, giảm thời gian suy luận từ 60 ms xuống 30 ms làm tổng thời gian giảm bao nhiêu?
- [ ] 50%
- [x] 20%
- [ ] 40%
- [ ] 10%
> 30 ms trên tổng 150 ms. Suy luận mô hình chỉ là một phần của ngân sách; bỏ một lần tra cứu đặc trưng thừa có thể giảm nhiều hơn.

### Khi so sánh mô hình gốc với mô hình đã xuất sang ONNX, vì sao thường dùng dung sai thay vì đòi trùng khớp từng bit?
- [ ] Vì ONNX làm tròn mọi trọng số
- [x] Vì hai bản có thể dùng các kernel khác nhau, với thứ tự phép tính khác nhau
- [ ] Vì ONNX không hỗ trợ số thực 32 bit
- [ ] Vì so sánh từng bit quá chậm
> Phép cộng số thực dấu phẩy động không có tính kết hợp, nên thứ tự tính khác nhau cho kết quả khác nhau ở những chữ số cuối. Kiểm thử so sánh vẫn bắt buộc, chỉ là với một dung sai hợp lý.

## Chương 8

### Canary **không** trả lời được câu hỏi nào?
- [ ] Tỉ lệ lỗi có tăng không
- [ ] Độ trễ có xấu đi không
- [x] Chỉ số sản phẩm có tốt lên không
- [ ] Phân phối dự đoán có khác thường không
> Ở vài phần trăm lưu lượng, chỉ số sản phẩm gần như chắc chắn không đủ mẫu để có ý nghĩa thống kê. Canary bảo vệ khỏi sự cố lớn; A/B test trả lời câu hỏi về giá trị.

### Với tỉ lệ nền 5%, cần khoảng bao nhiêu mẫu mỗi nhánh để phát hiện một cải thiện tương đối 1% với lực kiểm định 80%?
- [ ] Khoảng 30 nghìn
- [ ] Khoảng 300 nghìn
- [x] Khoảng 3 triệu
- [ ] Khoảng 30 triệu
> Chính xác là 2 996 694, tức khoảng một tháng lưu lượng nếu mỗi nhánh nhận 100 000 lượt mỗi ngày. Vì vậy các ý tưởng kém nên được loại bằng những cách rẻ hơn trước khi dùng lưu lượng cho A/B test.

### Cỡ mẫu A/B test phụ thuộc mức chênh lệch $\Delta$ cần phát hiện như thế nào?
- [ ] Tỉ lệ nghịch với $\Delta$
- [x] Tỉ lệ nghịch với $\Delta^2$
- [ ] Tỉ lệ thuận với $\Delta$
- [ ] Không phụ thuộc $\Delta$
> Muốn phát hiện một chênh lệch nhỏ bằng một nửa thì cần gấp bốn lần số mẫu.

### CUPED giảm cỡ mẫu cần thiết bằng cách nào?
- [ ] Giảm lực kiểm định
- [ ] Tăng mức ý nghĩa $\alpha$
- [x] Dùng giá trị của chỉ số trong giai đoạn trước thí nghiệm để giảm phương sai; với tương quan $\rho$, phương sai giảm theo hệ số $1-\rho^2$
- [ ] Loại bỏ những người dùng có giá trị ngoại lai
> Đây là cách rút ngắn thí nghiệm mà không đổi câu hỏi và không giảm độ tin cậy của kết luận (Deng và cộng sự, 2013).

### Xem kết quả giữa chừng (peeking) trong A/B test là gì và vì sao sai?
- [ ] Xem kết quả của nhánh đối chứng trước, sai vì gây thiên lệch
- [x] Kiểm tra p-value liên tục và dừng khi thấy $p < 0{,}05$, sai vì làm tỉ lệ dương tính giả cao hơn 5% nhiều
- [ ] Dùng dữ liệu của thí nghiệm trước, sai vì rò rỉ
- [ ] Chạy thí nghiệm quá lâu, sai vì tốn lưu lượng
> Cách xử lý: cố định cỡ mẫu trước khi chạy, hoặc dùng các phương pháp thiết kế cho việc theo dõi liên tục như kiểm định tuần tự.

### Thí nghiệm thiết kế chia 50/50 nhưng một nhánh có ít hơn nhánh kia 3% số người dùng, với hàng triệu người dùng mỗi nhánh. Nên làm gì?
- [ ] Đọc kết quả bình thường, vì 3% là nhỏ
- [x] Chưa đọc kết quả; đây là tỉ lệ mẫu lệch (sample ratio mismatch), dấu hiệu thí nghiệm có lỗi, ví dụ một nhánh làm mất log
- [ ] Nhân kết quả của nhánh nhỏ với hệ số bù
- [ ] Bỏ bớt người dùng ở nhánh lớn cho bằng nhau
> Với hàng triệu người dùng, chênh lệch 3% lớn hơn rất nhiều so với dao động ngẫu nhiên, và kiểm định khi bình phương sẽ bác bỏ tỉ lệ 50/50. Khi đó kết quả không đáng tin, dù p-value của chỉ số chính là bao nhiêu (Kohavi và cộng sự, 2020).

## Chương 9

### Loại dịch chuyển nào **không** phát hiện được bằng cách chỉ giám sát $X$?
- [ ] Covariate shift
- [ ] Label shift
- [x] Concept drift
- [ ] Cả ba đều phát hiện được
> Concept drift thay đổi $P(Y \mid X)$ và có thể giữ nguyên $P(X)$. Trong thí nghiệm ở Mục 9.2, độ chính xác giảm từ 75,6% xuống 26,9% trong khi kiểm định KS trên đặc trưng cho $p = 0{,}48$.

### Vì sao label shift phát hiện được qua phân phối của $X$?
- [ ] Vì label shift thay đổi $P(Y \mid X)$
- [x] Vì $P(X) = \sum_y P(X \mid y)\,P(y)$ là một hỗn hợp; thay đổi tỉ lệ nhãn làm phân phối của $X$ thay đổi theo
- [ ] Vì label shift luôn đi kèm covariate shift do một nguyên nhân bên ngoài
- [ ] Vì kiểm định KS đo trực tiếp phân phối nhãn
> Điều này đúng khi $X$ mang thông tin về $Y$. Trong thí nghiệm ở Mục 9.2, label shift cho p-value KS $1{,}2 \times 10^{-29}$ trong khi độ chính xác gần như không đổi.

### PSI là đại lượng gì?
- [ ] Khoảng cách Kolmogorov–Smirnov
- [ ] Phân kỳ KL một chiều
- [x] Phân kỳ Jeffreys, tổng hai chiều của phân kỳ KL giữa hai phân phối đã chia bin
- [ ] Khoảng cách Wasserstein
> $\sum_j (T_j - B_j)\ln(T_j/B_j) = D_{\mathrm{KL}}(T \,\|\, B) + D_{\mathrm{KL}}(B \,\|\, T)$. Khác với phân kỳ KL, PSI đối xứng.

### Vì sao không nên dùng ngưỡng PSI 0,25 cho mọi cỡ mẫu?
- [ ] Vì ngưỡng này quá chặt với mọi bài toán
- [x] Vì PSI kỳ vọng khi không có dịch chuyển xấp xỉ $2(k-1)/n$, nên ngưỡng cố định vừa gây báo động giả ở mẫu nhỏ vừa bỏ sót dịch chuyển ở mẫu lớn
- [ ] Vì ngưỡng này chỉ đúng cho biến rời rạc
- [ ] Vì ngưỡng này chỉ đúng khi dùng đúng 10 bin
> Mô phỏng ở Mục 9.5: với $n = 50$, quy tắc PSI > 0,25 báo động 73% số lần khi không có dịch chuyển; với $n = 1\,000$ và dịch chuyển thật 0,3 độ lệch chuẩn, quy tắc này không báo động lần nào, trong khi kiểm định KS báo động ở mọi lần.

### Với hai mẫu cùng cỡ $n = 1\,000$ và $k = 10$ bin, ngưỡng PSI ở mức ý nghĩa 1% xấp xỉ bằng bao nhiêu?
- [ ] 0,25
- [ ] 0,10
- [x] Khoảng 0,043
- [ ] Khoảng 0,0043
> Ngưỡng là $\tfrac{2}{n}\chi^2_{k-1,\,0{,}99} = \tfrac{2}{1000} \times 21{,}67 \approx 0{,}043$. Giá trị 0,0043 ứng với $n = 10\,000$.

### Theo Rabanser và cộng sự (2019), một cách đơn giản mà hiệu quả để phát hiện dịch chuyển là gì?
- [ ] Tính PSI trên từng đặc trưng với ngưỡng 0,25
- [x] Áp dụng kiểm định KS lên đầu ra của mô hình (xác suất dự đoán) kèm hiệu chỉnh Bonferroni
- [ ] Huấn luyện lại mô hình mỗi ngày và so sánh trọng số
- [ ] So sánh trung bình của từng đặc trưng
> Đầu ra của mô hình có ít chiều hơn đầu vào nhiều và tập trung vào những thay đổi ảnh hưởng tới dự đoán.

### Dùng tham chiếu trượt 30 ngày để phát hiện dịch chuyển có nhược điểm gì?
- [ ] Tốn bộ nhớ hơn
- [x] Không phát hiện được dịch chuyển chậm, vì tham chiếu trôi theo dữ liệu
- [ ] Chỉ dùng được cho biến liên tục
- [ ] Làm tăng báo động giả
> Cách hợp lý là dùng cả hai: tham chiếu cố định để biết dữ liệu đã cách xa dữ liệu huấn luyện bao nhiêu, tham chiếu trượt để phát hiện thay đổi đột ngột.

## Chương 10

### Trong bốn lớp giám sát, lớp nào là lớp chính khi nhãn đến trễ?
- [ ] Đầu vào thô
- [ ] Đặc trưng
- [x] Dự đoán
- [ ] Chỉ số chất lượng
> Càng gần nhãn càng có ý nghĩa nhưng càng khó có. Khi nhãn đến trễ, phân phối dự đoán, tỉ lệ từng lớp và độ lệch dự đoán là những gì dùng được ngay.

### Độ lệch dự đoán (prediction bias) là gì, và giới hạn của nó là gì?
- [ ] Nó chứng minh mô hình đúng, không có giới hạn
- [x] Chênh lệch giữa phân phối nhãn dự đoán và phân phối nhãn quan sát; giới hạn là một mô hình luôn dự đoán tỉ lệ trung bình cũng có độ lệch bằng 0
- [ ] Chênh lệch giữa tập huấn luyện và tập kiểm tra
- [ ] Chỉ dùng được cho bài toán hồi quy
> Bài báo của Sculley và cộng sự tự nêu giới hạn này, nhưng lưu ý rằng thay đổi đột ngột của đại lượng này thường là dấu hiệu sự cố. Ưu điểm: không cần nhãn cho từng dự đoán, chỉ cần tỉ lệ nền theo thời gian.

### Nguyên tắc "cảnh báo theo triệu chứng, không theo nguyên nhân" nghĩa là gì?
- [ ] Chỉ cảnh báo khi đã biết nguyên nhân gốc
- [x] "Tỉ lệ chuyển đổi giảm 15%" đáng gọi người trực; "PSI của đặc trưng thứ 37 vượt 0,2" là manh mối để điều tra
- [ ] Mỗi cảnh báo phải ghi rõ nguyên nhân
- [ ] Chỉ cảnh báo những gì tự động sửa được
> Cùng với các nguyên tắc: mỗi cảnh báo gắn với một hành động; phân mức gọi người trực, phiếu xử lý, bảng theo dõi; và yêu cầu tín hiệu kéo dài qua nhiều cửa sổ.

### Ngưỡng cố định trong hệ thống thay đổi (fixed thresholds in dynamic systems) là dạng nợ gì?
- [ ] Ngưỡng cảnh báo đặt quá thấp
- [x] Ngưỡng quyết định đặt bằng tay không còn phù hợp sau khi mô hình được huấn luyện lại
- [ ] Ngưỡng học được nhưng không lưu phiên bản
- [ ] Ngưỡng khác nhau giữa các môi trường
> Huấn luyện lại làm phân phối điểm số thay đổi, nên mọi thứ phụ thuộc vào điểm số cũng thay đổi theo. Cách xử lý: xác định ngưỡng tự động trên tập kiểm định mỗi lần huấn luyện và lưu ngưỡng cùng mô hình.

### Giới hạn hành động (action limits) có đặc điểm gì?
- [ ] Cần biết mô hình đúng hay sai
- [x] Không cần biết mô hình đúng hay sai; chỉ cần biết một mức hành động là bất thường
- [ ] Chỉ áp dụng cho mô hình ngôn ngữ
- [ ] Thay thế được mọi loại giám sát khác
> Ví dụ: khoá 50 000 tài khoản trong một giờ là điều chưa từng xảy ra. Với agent có quyền gọi công cụ có tác động thật, cơ chế này áp dụng nguyên vẹn (Mục 13.7).

## Chương 11

### Trong thí nghiệm ở Mục 11.1, không huấn luyện lại suốt một năm gây thiệt hại bao nhiêu?
- [ ] Khoảng 1 điểm phần trăm
- [ ] Khoảng 5 điểm phần trăm
- [x] 13,19 điểm phần trăm độ chính xác trung bình; tới tuần 52 mô hình chỉ còn đúng 45,5%, thấp hơn đoán ngẫu nhiên
- [ ] Mô hình giữ nguyên chất lượng
> Độ chính xác giảm dần từng chút, nên không có tuần nào có một thay đổi đột ngột đủ để kích hoạt cảnh báo theo ngưỡng.

### So với huấn luyện lại hằng quý, huấn luyện lại hằng tuần tăng thêm bao nhiêu trong thí nghiệm đó?
- [ ] Hơn 10 điểm
- [ ] Khoảng 5 điểm
- [x] Chỉ 0,91 điểm, trong khi số lần huấn luyện gấp 13 lần
- [ ] Không có chênh lệch nào
> Lợi ích giảm dần nhanh, nên nhịp huấn luyện lại là một quyết định kinh tế: đo đường cong trên chính bài toán, rồi so giá trị của một điểm phần trăm với chi phí một lần huấn luyện, kiểm định và triển khai.

### Kích hoạt huấn luyện lại theo dịch chuyển có hạn chế gì?
- [ ] Cần nhãn
- [x] Không phát hiện được concept drift, vì concept drift có thể giữ nguyên $P(X)$
- [ ] Chỉ chạy được theo lô
- [ ] Không dùng được cho mô hình học sâu
> Kích hoạt theo chất lượng thì ngược lại: phát hiện được concept drift nhưng cần nhãn, nên bị giới hạn bởi độ trễ nhãn.

### Bước nào hay bị bỏ qua nhất trong pipeline huấn luyện lại tự động?
- [ ] Kiểm định dữ liệu đầu vào
- [ ] Đánh giá trên tập kiểm tra cố định
- [x] So sánh với mô hình đang chạy trước khi thay thế
- [ ] Đăng ký vào sổ đăng ký mô hình
> Tài liệu của Google Cloud tách kiểm định mô hình khỏi đánh giá mô hình vì lý do này: đánh giá trả lời "mô hình này tốt tới đâu", kiểm định trả lời "có nên thay mô hình đang chạy bằng mô hình này không".

### Vì sao nên mặc định huấn luyện lại từ đầu thay vì cập nhật mô hình cũ?
- [ ] Vì huấn luyện lại từ đầu luôn cho mô hình chính xác hơn
- [x] Vì dễ tái lập và dễ quay lui hơn; cập nhật liên tục làm trạng thái mô hình phụ thuộc toàn bộ chuỗi cập nhật trước đó
- [ ] Vì huấn luyện lại từ đầu rẻ hơn
- [ ] Vì cập nhật mô hình cũ không dùng được với dữ liệu mới
> Cập nhật mô hình cũ rẻ hơn mỗi lần, nhưng có rủi ro quên thảm khốc, và một lô dữ liệu lỗi ảnh hưởng tới mọi phiên bản sau. Chỉ nên chuyển sang khi có ràng buộc thật về dữ liệu hoặc thời gian thích ứng.

## Chương 12

### Vì sao vòng phản hồi thoái hoá khó phát hiện?
- [ ] Vì nó làm mô hình chạy chậm đi
- [x] Vì dữ liệu vẫn trông tốt: tỉ lệ nhấp trên những gì được hiển thị vẫn cao, vì hệ thống chỉ hiển thị những gì nó tin là tốt
- [ ] Vì nó làm tăng chi phí huấn luyện
- [ ] Vì nó chỉ ảnh hưởng tới người dùng mới
> Các chỉ số ngoại tuyến tính trên dữ liệu log không cho thấy vấn đề, trong khi hệ thống chỉ còn hiển thị một phần nhỏ của danh mục.

### Trong thí nghiệm ở Mục 12.3, hệ thống không ngẫu nhiên hoá hiển thị bao nhiêu phần danh mục?
- [ ] Gần như toàn bộ
- [ ] Khoảng 50%
- [x] 9,5%, đúng những món đã xuất hiện ở vòng khởi động, và không mở rộng thêm
- [ ] Tuỳ số vòng huấn luyện lại
> Những món chưa từng được hiển thị có tỉ lệ nhấp quan sát bằng 0 nên không bao giờ vào danh sách. Hệ thống đạt chất lượng 0,351, khoảng 63% mức tối đa 0,555. Dành 30% vị trí cho món ngẫu nhiên nâng chất lượng lên 0,402 và tỉ lệ danh mục được hiển thị lên 13,4%.

### Cách khắc phục nào vừa cung cấp dữ liệu không bị mô hình làm sai lệch, vừa cho thước đo đáng tin cậy về giá trị của hệ thống?
- [ ] Ngẫu nhiên hoá một phần vị trí hiển thị
- [x] Giữ một nhóm đối chứng không chịu tác động của mô hình
- [ ] Hiệu chỉnh theo xác suất hiển thị
- [ ] Đưa vị trí hiển thị vào mô hình
> Cả bốn cách đều hợp lệ, nhưng nhóm đối chứng là cách duy nhất cho cả hai lợi ích cùng lúc.

### Muốn dùng hiệu chỉnh theo xác suất hiển thị thì phải chuẩn bị gì trước?
- [ ] Một mô hình nhân quả
- [x] Ghi lại xác suất hiển thị tại thời điểm hiển thị, vì không khôi phục được về sau
- [ ] Một tập kiểm tra riêng
- [ ] Nhãn của toàn bộ danh mục
> Đây là loại quyết định phải đưa ra từ đầu, giống như việc lưu định danh các đoạn đã truy xuất trong hệ thống RAG (Mục 13.5).

## Chương 13

### Nhà cung cấp cập nhật mô hình sau cùng một tên gọi là dạng nợ nào?
- [ ] Mã keo
- [x] Dạng cực đoan của phụ thuộc dữ liệu không ổn định
- [ ] Correction cascade
- [ ] Nợ cấu hình
> Cách xử lý: dùng định danh phiên bản cố định của mô hình, và coi mỗi lần đổi phiên bản là một lần phát hành đầy đủ, phải qua bộ đánh giá, shadow và canary. Chen, Zaharia và Zou (2023) đo được những thay đổi đáng kể về hành vi của cùng một tên mô hình giữa hai phiên bản.

### Cấu trúc đánh giá ba tầng cho hệ thống dùng LLM gồm những gì?
- [ ] Người chấm trên mọi lượt, giám khảo LLM trên một mẫu, kiểm tra quy tắc trên một mẫu nhỏ
- [x] Kiểm tra theo quy tắc trên mọi lượt, giám khảo LLM trên một mẫu, người chấm trên một mẫu nhỏ định kỳ
- [ ] Cả ba tầng đều chấm mọi lượt
- [ ] Chỉ dùng giám khảo LLM là đủ
> Càng gần đánh giá thật thì càng đắt, nên phải lấy mẫu. Tầng người chấm hay bị bỏ qua nhất, nhưng cần để hiệu chỉnh giám khảo LLM, vốn cũng là một mô hình có sai số và có thể thay đổi.

### Một câu trả lời của hệ thống RAG sai. Bước chẩn đoán đầu tiên là gì?
- [ ] Đổi mô hình sinh
- [x] Kiểm tra tài liệu cần thiết có nằm trong ngữ cảnh không, để tách lỗi truy xuất khỏi lỗi sinh
- [ ] Tăng số đoạn truy xuất
- [ ] Sửa prompt
> Không có trong ngữ cảnh: lỗi truy xuất, đo bằng recall@$k$. Có trong ngữ cảnh mà vẫn sai: lỗi sinh, đo bằng độ trung thành với ngữ cảnh. Gộp hai loại lỗi vào một chỉ số thì không biết cần sửa ở đâu.

### Thiên lệch vị trí của giám khảo LLM được xử lý thế nào?
- [ ] Dùng mô hình giám khảo lớn hơn
- [x] Hỏi hai lần với thứ tự đảo ngược, chỉ tính là thắng khi thắng ở cả hai lần
- [ ] So sánh các câu trả lời có độ dài tương đương
- [ ] Tăng nhiệt độ lấy mẫu
> So sánh các câu trả lời có độ dài tương đương là cách xử lý thiên lệch độ dài; dùng giám khảo khác họ với mô hình đang phục vụ là cách xử lý thận trọng cho khả năng tự ưu tiên (Zheng và cộng sự, 2023).

### Hiện tượng criteria drift (Shankar và cộng sự, 2024) nói gì?
- [ ] Giám khảo LLM thay đổi tiêu chí theo thời gian vì nhà cung cấp cập nhật mô hình
- [x] Người xây dựng tiêu chí chấm thay đổi tiêu chí trong khi đọc và chấm đầu ra thật
- [ ] Tiêu chí chấm tự động lệch về phía câu trả lời dài
- [ ] Người chấm khác nhau dùng tiêu chí khác nhau
> Vì vậy không thể viết xong tiêu chí chấm một lần từ đầu; cần đọc một lượng đầu ra thật trước và xem lại tiêu chí định kỳ.

### Đổi mô hình embedding của một hệ thống RAG đòi hỏi gì?
- [ ] Không cần làm gì, vì vector của các mô hình embedding tương thích với nhau
- [x] Lập chỉ mục lại toàn bộ kho tài liệu, vì vector của hai mô hình khác nhau không so sánh được với nhau
- [ ] Chỉ cần lập chỉ mục lại các tài liệu mới
- [ ] Chỉ cần đổi hàm độ tương đồng
> Việc đổi mô hình embedding nên được làm như một lần di chuyển dữ liệu: lập chỉ mục mới song song, đánh giá, rồi chuyển.

### Thay đổi nào sau đây **không** tạo ra một phiên bản mới của hệ thống LLM cần đánh giá lại?
- [ ] Prompt hệ thống
- [ ] Cách chia đoạn tài liệu
- [ ] Ảnh chụp kho tài liệu đã lập chỉ mục
- [x] Không có; cả ba đều tạo ra phiên bản mới
> Một phiên bản gồm ít nhất sáu thành phần: prompt, phiên bản mô hình và tham số sinh, cấu hình truy xuất, ảnh chụp kho tài liệu, định nghĩa công cụ, tiêu chí chấm và tập đánh giá chuẩn. Đây là nguyên lý CACE áp dụng cho hệ thống LLM.
