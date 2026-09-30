# Ngân hàng câu hỏi tự kiểm tra — Ứng dụng LLM

Cú pháp: `## Chương N` mở một nhóm, `### …` là câu hỏi, `- [x]` đánh dấu đáp án đúng,
dòng `>` là phần giải thích hiện ra sau khi trả lời.

## Chương 1

### Điểm xuất phát điển hình của một kỹ sư AI khác kỹ sư học máy ở chỗ nào?
- [ ] Kỹ sư AI bắt đầu từ dữ liệu có nhãn và huấn luyện mô hình từ đầu
- [x] Kỹ sư AI bắt đầu từ một mô hình nền tảng đã huấn luyện sẵn và xây hệ thống quanh nó
- [ ] Kỹ sư AI chỉ làm việc với dữ liệu ảnh
- [ ] Kỹ sư AI không cần đánh giá mô hình
> Công việc chính của kỹ sư AI là thiết kế prompt và ngữ cảnh, truy xuất, tool, rào chắn và đánh giá quanh một mô hình có sẵn (Mục 1.1, 1.2).

### Trong phát triển dựa trên LLM, vì sao nên xây bộ đánh giá sớm?
- [ ] Vì mô hình không chạy được nếu chưa có bộ đánh giá
- [x] Vì bản thử đầu tiên có ngay rất nhanh, còn biết chắc hệ thống tốt tới mức nào mới là phần khó và cần số liệu
- [ ] Vì nhà cung cấp API yêu cầu
- [ ] Vì bộ đánh giá thay thế được việc giám sát trong sản xuất
> Một prompt tốt trên vài ví dụ thử bằng tay có thể sai ở nhiều trường hợp thật. Quy trình phát triển dựa trên đánh giá đo sau mỗi thay đổi (Mục 1.3).

### Khi một ứng dụng dùng LLM trả lời sai, việc đầu tiên nên làm là gì?
- [ ] Đổi sang mô hình lớn hơn
- [ ] Viết thêm quy tắc vào prompt
- [x] Xác định lỗi nằm ở thành phần nào: truy xuất, tool, ngữ cảnh hay bước sinh của mô hình
- [ ] Tăng temperature
> Mỗi thành phần có kiểu lỗi riêng, và chỉ xác định được nếu từng thành phần được ghi lại riêng trong trace (Mục 1.5, Chương 13).

## Chương 2

### Cùng một văn bản, vì sao bản tiếng Việt tốn gấp 4,8 lần số token so với bản tiếng Anh với tokenizer của GPT-2?
- [ ] Vì tiếng Việt có nhiều từ hơn gấp 4,8 lần
- [x] Vì tokenizer được huấn luyện gần như chỉ trên tiếng Anh nên cắt các ký tự có dấu của tiếng Việt thành từng byte
- [ ] Vì GPT-2 không hỗ trợ Unicode
- [ ] Vì tiếng Việt phải dịch sang tiếng Anh trước
> Với tokenizer đa ngôn ngữ như Qwen2.5 và XLM-RoBERTa, tỉ lệ chỉ còn 1,4 và 1,2 lần. Tokenizer BPE tự huấn luyện trên tiếng Việt cho tỉ lệ 0,64 (Mục 2.1).

### Với Llama 3 8B, KV cache của một chuỗi 128 nghìn token khoảng bao nhiêu?
- [ ] 128 KiB
- [ ] 1 GiB
- [x] 16 GiB, gấp đôi dung lượng trọng số 16 bit của mô hình
- [ ] 128 GiB
> Mỗi token tốn $2 \times 32 \times 8 \times 128 \times 2 = 131\,072$ byte = 128 KiB (Mục 2.2).

### Trong một yêu cầu RAG điển hình ở Mục 2.5, đầu ra chiếm 9,8% số token nhưng chiếm bao nhiêu phần chi phí khi token đầu ra đắt gấp 4 lần?
- [ ] 9,8%
- [x] Khoảng 30%
- [ ] 50%
- [ ] 90%
> $350 \times 4 / (3\,210 + 350 \times 4) \approx 30{,}4\%$. Phần lớn chi phí còn lại đến từ ngữ cảnh do ứng dụng thêm vào, phần kỹ sư AI kiểm soát được.

### Vì sao temperature bằng 0 không bảo đảm kết quả giống hệt nhau giữa các lần gọi?
- [ ] Vì temperature 0 vẫn lấy mẫu ngẫu nhiên
- [x] Vì gộp lô trên máy chủ và phép cộng số thực không có tính kết hợp làm logit khác nhau rất nhỏ, đủ đổi token khi hai token gần bằng xác suất
- [ ] Vì mô hình thay đổi trọng số sau mỗi lần gọi
- [ ] Vì API cố tình thêm nhiễu
> Ứng dụng cần chịu được tính không tất định, và đánh giá cần đủ nhiều ví dụ (Mục 2.3).

## Chương 3

### Khác biệt giữa mô hình mở trọng số và mô hình mã nguồn mở theo định nghĩa của OSI là gì?
- [ ] Không có khác biệt
- [x] Mở trọng số chỉ công bố trọng số; mã nguồn mở theo OSI còn yêu cầu đủ thông tin về dữ liệu và mã huấn luyện để tái tạo
- [ ] Mô hình mở trọng số không được dùng thương mại
- [ ] Mô hình mã nguồn mở không có trọng số
> Giấy phép của mô hình mở trọng số cũng khác nhau; cần đọc giấy phép của đúng phiên bản trước khi dùng thương mại (Mục 3.2).

### Mô hình 8 tỉ tham số lượng tử 4 bit cần khoảng bao nhiêu bộ nhớ cho trọng số?
- [ ] 32 GB
- [ ] 16 GB
- [x] Khoảng 4 tới 5 GB
- [ ] 1 GB
> 4 bit là 0,5 byte mỗi tham số, cộng thêm hệ số tỉ lệ của lượng tử hoá. Ngoài trọng số còn KV cache tăng theo độ dài ngữ cảnh và số yêu cầu đồng thời (Mục 3.3).

### Vì sao không nên chọn mô hình chỉ dựa vào bảng xếp hạng công khai?
- [ ] Vì bảng xếp hạng luôn sai
- [x] Vì dữ liệu đánh giá có thể đã lọt vào dữ liệu huấn luyện, và nhiệm vụ của bộ đánh giá hiếm khi giống nhiệm vụ của ứng dụng
- [ ] Vì mô hình đứng đầu luôn đắt nhất
- [ ] Vì bảng xếp hạng không có mô hình mở
> Bảng xếp hạng dùng để chọn vài ứng viên; bộ đánh giá trên dữ liệu của ứng dụng dùng để quyết định (Mục 3.6).

## Chương 4

### Trong cấu trúc prompt ở Mục 4.1, vì sao nên đặt phần ổn định ở đầu và phần thay đổi theo yêu cầu ở cuối?
- [ ] Vì mô hình chỉ đọc phần đầu
- [x] Vì prompt caching chỉ dùng lại được khi tiền tố giống hệt nhau từng token
- [ ] Vì phần cuối luôn bị cắt bỏ
- [ ] Vì thứ tự không ảnh hưởng gì
> Một dấu thời gian ở dòng đầu làm mất tác dụng của cache cho toàn bộ phần phía sau (Mục 4.6).

### Chain-of-thought hữu ích nhất cho loại nhiệm vụ nào?
- [ ] Phân loại cảm xúc một câu ngắn
- [x] Bài toán nhiều bước như tính toán và suy luận
- [ ] Trích xuất một trường từ văn bản
- [ ] Mọi nhiệm vụ, luôn nên dùng
> Chuỗi suy luận tốn token đầu ra và tăng độ trễ, thường không giúp gì với nhiệm vụ đơn giản (Mục 4.3).

### Giải mã có ràng buộc (constrained decoding) bảo đảm được điều gì?
- [ ] Nội dung của đầu ra luôn đúng
- [x] Đầu ra đúng cú pháp của ngữ pháp hoặc lược đồ, như JSON hợp lệ
- [ ] Đầu ra không có ảo giác
- [ ] Mô hình chạy nhanh hơn
> Giá trị bên trong vẫn có thể sai, nên ràng buộc về nội dung vẫn cần kiểm tra bằng mã (Mục 4.4).

### Khi dùng function calling, ai thực thi hàm?
- [ ] Mô hình tự gọi API
- [x] Ứng dụng, sau khi mô hình đề xuất lời gọi có cấu trúc
- [ ] Nhà cung cấp API
- [ ] Người dùng
> Ứng dụng quyết định có thực thi hay không, với tham số nào và trong phạm vi quyền nào (Mục 4.5, 9.3).

## Chương 5

### Context engineering khác prompt engineering ở chỗ nào?
- [ ] Không khác, hai tên gọi của một việc
- [x] Context engineering quản lý toàn bộ những gì nằm trong cửa sổ ngữ cảnh ở mỗi lần gọi, không chỉ chỉ dẫn
- [ ] Context engineering chỉ dùng cho mô hình ảnh
- [ ] Context engineering là huấn luyện lại mô hình
> Ngữ cảnh gồm chỉ dẫn, định nghĩa tool, tài liệu truy xuất, kết quả tool, lịch sử và bộ nhớ; với agent, nó thay đổi sau mỗi bước (Mục 5.1).

### Hiện tượng "lost in the middle" nói gì?
- [ ] Mô hình luôn bỏ qua phần đầu ngữ cảnh
- [x] Khả năng dùng một thông tin phụ thuộc vào vị trí của nó trong ngữ cảnh, thường kém hơn khi thông tin nằm ở giữa ngữ cảnh dài
- [ ] Mô hình bị lạc khi có quá nhiều tool
- [ ] Mô hình mất kết nối với máy chủ giữa chừng
> Liu và cộng sự (2024) ghi nhận hiện tượng này; Mục 5.3 đo trên một mô hình nhỏ với các vị trí và độ dài khác nhau.

### Vì sao agent con trong hệ nhiều agent giúp quản lý ngữ cảnh?
- [ ] Vì agent con dùng mô hình lớn hơn
- [x] Vì mỗi agent con có cửa sổ ngữ cảnh sạch cho phần việc của nó và chỉ trả về bản tóm tắt
- [ ] Vì agent con không cần chỉ dẫn
- [ ] Vì agent con không tốn token
> Cô lập ngữ cảnh giữ cho ngữ cảnh của agent chính gọn, đổi lại tổng số token tăng nhiều (Mục 5.5, 9.6).

## Chương 6

### Trên kho văn bản tiếng Việt của Mục 6.5, multilingual-e5-small so với BM25 thế nào?
- [ ] Tốt hơn hẳn ở cả hai bộ truy vấn
- [x] Kém hơn ở cả hai bộ truy vấn; kết hợp hai phương pháp bằng RRF tốt nhất ở bộ câu hỏi diễn đạt lại
- [ ] Bằng nhau ở mọi thước đo
- [ ] Không so sánh được
> R@5 tìm đúng mục ở bộ 2: BM25 0,60, e5-small 0,52, RRF 0,62. Nguyên nhân gồm mô hình nhỏ, văn bản chuyên ngành, và 30,7% số mục bị cắt vì dài hơn 512 token.

### Vì sao bộ câu hỏi trắc nghiệm (bộ 1) cho kết quả truy xuất cao hơn nhiều so với bộ câu hỏi diễn đạt lại (bộ 2)?
- [ ] Vì bộ 1 có nhiều câu hơn
- [x] Vì câu trắc nghiệm được viết dựa trên nội dung bài nên dùng lại nhiều từ của tài liệu
- [ ] Vì bộ 2 chỉ có câu tiếng Anh
- [ ] Vì bộ 1 dùng mô hình khác
> Đánh giá bằng câu hỏi sinh ra từ chính tài liệu cho kết quả lạc quan hơn so với câu hỏi của người dùng thật (Mục 6.5, 6.6).

### Khi mọi vector đã được chuẩn hoá về độ dài 1, cosine, tích vô hướng và khoảng cách Euclid quan hệ thế nào?
- [ ] Cho ba thứ hạng khác nhau
- [x] Cho cùng một thứ hạng, vì $\|q - d\|^2 = 2 - 2\cos(q, d)$
- [ ] Chỉ cosine dùng được
- [ ] Khoảng cách Euclid luôn bằng 0
> Vì vậy phần lớn hệ thống chuẩn hoá vector rồi dùng tích vô hướng, phép tính rẻ nhất (Mục 6.3).

### Vì sao RRF không cần đưa điểm của hai phương pháp về cùng thang đo?
- [ ] Vì RRF chỉ dùng một phương pháp
- [x] Vì RRF chỉ dùng thứ hạng, mỗi tài liệu nhận điểm $\sum_r 1/(60 + \text{hạng}_r)$
- [ ] Vì điểm BM25 và cosine luôn cùng thang đo
- [ ] Vì RRF chuẩn hoá điểm về 0 và 1
> Điểm BM25 không bị chặn trên, cosine nằm trong khoảng hẹp; cộng trực tiếp đòi hỏi chuẩn hoá và chọn trọng số (Mục 6.5).

## Chương 7

### Chỉ mục IVF đánh đổi điều gì?
- [ ] Bộ nhớ lấy độ chính xác
- [x] Chỉ so truy vấn với vector trong vài cụm gần nhất để giảm khối lượng tính, đổi lại có thể bỏ sót láng giềng nằm ở cụm khác
- [ ] Tốc độ xây chỉ mục lấy tốc độ truy vấn
- [ ] Không đánh đổi gì
> Ở Mục 7.2, so 6,1% số vector cho recall@10 là 0,888; muốn 0,968 cần so 23,8%.

### Trong thí nghiệm product quantization ở Mục 7.3, nén 64 lần rồi xếp lại 100 ứng viên bằng vector gốc cho recall@10 bao nhiêu?
- [ ] 0,596
- [ ] 0,5
- [x] 0,987
- [ ] 0,1
> Chỉ dùng mã nén cho 0,596, nhưng hầu hết láng giềng thật vẫn nằm trong 100 ứng viên đầu, nên xếp lại khôi phục gần hết độ chính xác.

### Nhược điểm chính của HNSW với kho hàng tỉ vector là gì?
- [ ] Độ chính xác thấp
- [x] Bộ nhớ: vector gốc và các cạnh của đồ thị thường phải nằm trong RAM
- [ ] Không hỗ trợ tích vô hướng
- [ ] Không cập nhật được
> Với kho rất lớn, IVF-PQ hoặc chỉ mục đồ thị trên SSD như DiskANN được dùng thay thế (Mục 7.4).

## Chương 8

### RAG phù hợp hơn fine-tuning trong trường hợp nào?
- [ ] Khi cần đổi giọng văn của mô hình
- [x] Khi cần đưa kiến thức thay đổi thường xuyên vào, trích dẫn nguồn và phân quyền theo tài liệu
- [ ] Khi cần mô hình trả lời nhanh hơn
- [ ] Khi không có tài liệu nào
> Fine-tuning phù hợp để thay đổi hành vi; RAG phù hợp để đưa kiến thức vào (Mục 8.1).

### Trên kho văn bản của lộ trình, cách chia đoạn nào cho kết quả tốt nhất khi kết hợp BM25 và e5-small bằng RRF?
- [ ] Đoạn 50 từ
- [ ] Đoạn 100 từ
- [x] Chia theo mục (cả mục)
- [ ] Kích thước không ảnh hưởng
> RRF đạt 0,750 khi dùng cả mục; BM25 đơn lẻ tốt nhất với đoạn 400 từ (0,738) và cả mục (0,725). Kết quả phụ thuộc cấu trúc tài liệu nên cần đo (Mục 8.3).

### Vì sao cần ghi lại các đoạn đã truy xuất cho mỗi câu trả lời của hệ thống RAG?
- [ ] Để tính chi phí
- [x] Để phân biệt lỗi truy xuất (đoạn đúng không có trong ngữ cảnh) với lỗi sinh (có mà vẫn trả lời sai)
- [ ] Để mô hình nhớ lâu hơn
- [ ] Vì bắt buộc theo giao thức MCP
> Hai loại lỗi cần cách sửa khác nhau và được đo bằng thước đo khác nhau (Mục 8.6).

## Chương 9

### Agent 20 bước, mỗi bước đúng với xác suất 0,95, hoàn thành nhiệm vụ với xác suất khoảng bao nhiêu nếu các bước độc lập?
- [ ] 95%
- [ ] 80%
- [x] 36%
- [ ] 5%
> $0{,}95^{20} \approx 0{,}358$. Bước kiểm tra phát hiện 80% lỗi kèm một lần thử lại nâng con số này lên khoảng 79% (Mục 9.7).

### Workflow khác agent ở điểm nào?
- [ ] Workflow không dùng LLM
- [x] Trong workflow, các bước và thứ tự do mã quyết định; trong agent, mô hình tự quyết định bước tiếp theo
- [ ] Agent không gọi tool
- [ ] Workflow luôn tốn nhiều token hơn
> Nên dùng giải pháp đơn giản nhất đáp ứng yêu cầu; workflow dễ kiểm thử và dự đoán chi phí hơn (Mục 9.1, 9.2).

### Khi một tool thất bại trong vòng lặp của agent, nên xử lý thế nào?
- [ ] Dừng toàn bộ chương trình
- [x] Trả về thông báo lỗi dễ hiểu như một kết quả để mô hình tự sửa lời gọi
- [ ] Bỏ qua và không báo gì cho mô hình
- [ ] Tăng temperature rồi gọi lại
> Lỗi là kết quả, không phải ngoại lệ; kèm theo đó là giới hạn số bước để tránh vòng lặp vô tận (Mục 9.3, 9.8).

## Chương 10

### MCP giảm số tích hợp giữa $N$ ứng dụng và $M$ dịch vụ từ bao nhiêu xuống bao nhiêu?
- [ ] Từ $N + M$ xuống $N \times M$
- [x] Từ $N \times M$ xuống $N + M$
- [ ] Từ $N^2$ xuống $N$
- [ ] Không thay đổi
> Mỗi dịch vụ viết một MCP server, mỗi ứng dụng cài một MCP client (Mục 10.1).

### Trong kiến trúc MCP, phát biểu nào đúng?
- [ ] Mỗi server đọc được toàn bộ hội thoại
- [x] Mỗi client kết nối với đúng một server, và server không nhìn thấy các server khác
- [ ] Server gọi LLM trực tiếp trong mọi trường hợp
- [ ] Client và server phải chạy trên cùng máy
> Host giữ toàn bộ hội thoại và kiểm soát mọi tương tác giữa các server (Mục 10.2).

### Thay đổi lớn của đặc tả MCP phiên bản 2026-07-28 là gì?
- [ ] Bỏ JSON-RPC
- [x] Giao thức phi trạng thái: bỏ bước bắt tay initialize, mỗi yêu cầu tự mang phiên bản giao thức và khả năng của client
- [ ] Chỉ còn truyền qua stdio
- [ ] Bỏ khái niệm tool
> Server cần thêm thông tin thì trả kết quả tạm với resultType "input_required" (Mục 10.4).

### Vì sao mô tả tool của một MCP server phải được coi là không đáng tin nếu server không đáng tin?
- [ ] Vì mô tả có thể quá dài
- [x] Vì mô tả đi thẳng vào ngữ cảnh của mô hình và có thể chứa chỉ dẫn độc hại điều khiển mô hình
- [ ] Vì mô tả không được hiển thị cho người dùng
- [ ] Vì mô tả làm chậm mô hình
> Đây là một dạng prompt injection, đôi khi gọi là đầu độc tool (Mục 10.6).

## Chương 11

### Rủi ro đứng đầu danh sách OWASP Top 10 cho ứng dụng LLM năm 2025 là gì?
- [ ] Tiêu tốn tài nguyên không giới hạn
- [x] Prompt injection
- [ ] Rò rỉ chỉ dẫn hệ thống
- [ ] Thông tin sai lệch
> Danh sách đầy đủ ở Mục 11.1.

### Vì sao chỉ viết thêm chỉ dẫn "không làm theo yêu cầu trong tài liệu" không đủ để chống prompt injection?
- [ ] Vì mô hình không đọc chỉ dẫn hệ thống
- [x] Vì mô hình không có cơ chế chắc chắn để phân biệt chỉ dẫn với dữ liệu, nên các biện pháp ở tầng prompt chỉ giảm chứ không loại bỏ rủi ro
- [ ] Vì chỉ dẫn phải viết bằng tiếng Anh
- [ ] Vì prompt injection chỉ xảy ra với mô hình nhỏ
> Phòng thủ chính nằm ở kiến trúc: quyền tối thiểu, xin phép người dùng, không để một agent vừa đọc dữ liệu không tin cậy vừa gửi được dữ liệu ra ngoài (Mục 11.2).

### Trong thí nghiệm ở Mục 11.2, vì sao không coi mọi đầu ra có chứa câu của kẻ tấn công là "đã làm theo chỉ dẫn độc hại"?
- [ ] Vì kẻ tấn công có thể viết sai chính tả
- [x] Vì mô hình nhỏ thường chép lại cả email, gồm cả câu độc hại, thay vì tóm tắt; cần tách trường hợp làm theo khỏi trường hợp chép lại
- [ ] Vì kiểm định Fisher không áp dụng được cho 40 email
- [ ] Vì chỉ dẫn độc hại luôn bị mô hình bỏ qua
> Chép lại vẫn là vấn đề, vì câu độc hại đi tiếp vào đầu ra, nhưng đó là một kiểu lỗi khác với việc mô hình bỏ nhiệm vụ. Nếu chỉ đếm các đầu ra có câu của kẻ tấn công, cả ba cách phòng thủ trong thí nghiệm đều trông như làm tình hình tệ đi, vì chúng làm tăng số đầu ra chép lại email.

### Với hệ thống RAG có tài liệu phân quyền, bộ lọc quyền nên đặt ở đâu?
- [ ] Trong chỉ dẫn hệ thống
- [x] Ở bước truy xuất, dựa trên danh tính người dùng đã xác thực, trước khi đoạn văn vào ngữ cảnh
- [ ] Ở bước kiểm tra đầu ra
- [ ] Không cần vì mô hình tự biết
> Đoạn văn đã vào ngữ cảnh thì có thể bị lộ qua cách hỏi khéo (Mục 11.3).

## Chương 12

### Với độ chính xác khoảng 80%, bộ đánh giá 100 câu cho khoảng tin cậy 95% rộng cỡ nào?
- [ ] ±1 điểm phần trăm
- [ ] ±3 điểm phần trăm
- [x] ±7,8 điểm phần trăm
- [ ] ±20 điểm phần trăm
> Nửa độ rộng giảm theo $1/\sqrt{n}$: 1 000 câu cho ±2,5 điểm (Mục 12.4).

### Vì sao nên so sánh hai phiên bản prompt trên cùng một bộ câu hỏi?
- [ ] Vì tốn ít chi phí hơn
- [x] Vì kết quả của hai phiên bản trên cùng câu tương quan mạnh, so sánh ghép cặp loại bỏ biến động do độ khó của câu hỏi
- [ ] Vì kiểm định McNemar chỉ dùng được với một phiên bản
- [ ] Vì không có lý do thống kê nào
> Với 1 000 câu, so sánh ghép cặp phát hiện cải thiện 3 điểm phần trăm ở 92% số lần, hai bộ độc lập chỉ 41% (Mục 12.4).

### Các thiên lệch đã biết của LLM làm giám khảo gồm những gì?
- [ ] Không có thiên lệch
- [x] Thiên lệch vị trí, thiên lệch độ dài và tự đề cao câu trả lời của chính mô hình
- [ ] Chỉ chấm được tiếng Anh
- [ ] Luôn cho điểm tối đa
> Biện pháp: đổi vị trí và chấm hai lần, tiêu chí chi tiết, so định kỳ với nhãn do người gán (Mục 12.2).

## Chương 13

### Điều quan trọng nhất cần ghi lại để gỡ lỗi một câu trả lời sai trong sản xuất là gì?
- [ ] Chỉ câu hỏi của người dùng
- [x] Đầu vào đầy đủ đã gửi cho mô hình, cùng phiên bản mô hình, prompt, tài liệu truy xuất và kết quả tool
- [ ] Chỉ số token
- [ ] Chỉ phản hồi của người dùng
> Nguyên nhân thường nằm ở ngữ cảnh; không có đầu vào đầy đủ thì không tái tạo được lỗi (Mục 13.1).

### Vì sao theo dõi độ trễ theo phân vị p95, p99 thay vì trung bình?
- [ ] Vì trung bình khó tính
- [x] Vì trải nghiệm tệ nhất của người dùng nằm ở đuôi phân phối, trung bình che mất nó
- [ ] Vì phân vị luôn nhỏ hơn trung bình
- [ ] Vì nhà cung cấp chỉ báo phân vị
> Tương tự với chi phí: trung bình ổn định có thể che các yêu cầu cá biệt rất đắt như agent rơi vào vòng lặp (Mục 13.3).

## Chương 14

### Với ô 14 × 14 điểm ảnh, một ảnh vuông 448 × 448 cho bao nhiêu token ảnh?
- [ ] 32
- [ ] 256
- [x] 1 024
- [ ] 200 704
> $(448/14)^2 = 32^2 = 1\,024$. Gộp 2 × 2 ô (tương đương ô 28 × 28) giảm còn 256 token (Mục 14.2).

### Thước đo chuẩn cho nhận dạng tiếng nói là gì?
- [ ] BLEU
- [x] Tỉ lệ lỗi từ (WER)
- [ ] Recall@k
- [ ] pass@k
> Số phép thay, xoá, chèn cần để biến bản nhận dạng thành bản chuẩn, chia cho số từ của bản chuẩn (Mục 14.4).

## Chương 15

### Thí nghiệm có đối chứng của METR (2025) với lập trình viên mã nguồn mở giàu kinh nghiệm cho kết quả gì?
- [ ] Họ nhanh hơn 55,8%
- [x] Họ chậm hơn 19% khi dùng công cụ AI, dù tự ước tính mình nhanh hơn khoảng 20%
- [ ] Không có khác biệt
- [ ] Họ viết mã ít lỗi hơn một nửa
> Kết quả phụ thuộc bối cảnh; thí nghiệm của Peng và cộng sự (2023) với một nhiệm vụ mới, độc lập cho kết quả nhanh hơn 55,8% (Mục 15.2).

### Vì sao cần kiểm tra các gói thư viện do trợ lý lập trình gợi ý?
- [ ] Vì gói luôn quá cũ
- [x] Vì mô hình có thể gợi ý tên gói không tồn tại, và kẻ xấu có thể đăng ký gói trùng tên để phát tán mã độc
- [ ] Vì gói luôn có giấy phép sai
- [ ] Vì trình quản lý gói không cài được
> Đây là rủi ro chuỗi cung ứng đặc thù của mã sinh bởi LLM (Mục 15.3).
