# Từ điển thuật ngữ — Ứng dụng LLM

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mỗi định nghĩa khớp với cách dùng trong giáo trình. Thuật ngữ nào giáo trình giữ nguyên tiếng Anh thì tên tiếng Anh đứng trước.

## Khái niệm chung

### Kỹ sư AI | AI engineer
Người xây dựng ứng dụng dựa trên các mô hình nền tảng có sẵn: chọn mô hình, thiết kế prompt và ngữ cảnh, kết nối dữ liệu và tool, bảo đảm an toàn, đánh giá và vận hành.

### Mô hình nền tảng | foundation model
Mô hình lớn được huấn luyện trên lượng dữ liệu rất lớn và dùng được cho nhiều nhiệm vụ khác nhau, thường qua prompt hoặc tinh chỉnh.

### Mô hình ngôn ngữ lớn (LLM) | large language model
Mạng nơ-ron, thường là Transformer, huấn luyện để dự đoán token tiếp theo trên lượng văn bản rất lớn, rồi tinh chỉnh để làm theo chỉ dẫn.

### Suy luận | inference
Chạy mô hình đã huấn luyện để sinh kết quả cho một đầu vào. Chi phí chính của ứng dụng LLM nằm ở suy luận.

### Phát triển dựa trên đánh giá | evaluation-driven development
Quy trình trong đó bộ đánh giá được xây sớm và mọi thay đổi (prompt, mô hình, truy xuất) đều được đo trên bộ đánh giá trước khi chấp nhận.

## Mô hình ngôn ngữ

### Token | token
Đơn vị văn bản mà mô hình xử lý, lấy từ một từ vựng cố định. Mọi giới hạn độ dài và chi phí của LLM tính theo token.

### Tokenizer | tokenizer
Thành phần cắt văn bản thành dãy token. Cùng một nội dung tiếng Việt có thể tốn số token chênh nhau vài lần giữa các tokenizer.

### Byte-pair encoding (BPE) | byte-pair encoding
Thuật toán xây từ vựng tokenizer bằng cách lặp lại việc gộp cặp đơn vị xuất hiện cạnh nhau nhiều nhất trong dữ liệu huấn luyện.

### Cửa sổ ngữ cảnh | context window
Số token tối đa mô hình xử lý trong một lần gọi, tính cả đầu vào và đầu ra.

### KV cache | key-value cache
Các vector key và value của mọi token đã xử lý, lưu lại ở mọi lớp để sinh token tiếp theo không phải tính lại. Dung lượng tăng tuyến tính theo độ dài ngữ cảnh.

### Temperature | temperature
Tham số chia logit trước softmax khi lấy mẫu; nhỏ làm phân phối nhọn và kết quả ổn định, lớn làm kết quả đa dạng hơn.

### Top-k, top-p | top-k, top-p (nucleus) sampling
Hai cách giới hạn tập token được lấy mẫu: giữ $k$ token có xác suất cao nhất, hoặc giữ tập nhỏ nhất có tổng xác suất vượt $p$.

### Phạt lặp lại | repetition penalty
Các tham số giảm xác suất của token đã xuất hiện để hạn chế mô hình lặp lại.

### Mô hình gốc | base model
Mô hình chỉ qua tiền huấn luyện, viết tiếp văn bản chứ chưa biết làm theo chỉ dẫn.

### Mô hình chỉ dẫn | instruct model
Mô hình đã được tinh chỉnh để làm theo chỉ dẫn và trò chuyện; loại dùng trong phần lớn ứng dụng.

### Mô hình suy luận | reasoning model
Mô hình được huấn luyện thêm để sinh một chuỗi suy luận dài trước khi trả lời; tốt hơn ở bài toán nhiều bước, tốn nhiều token hơn.

### Mốc cắt kiến thức | knowledge cutoff
Thời điểm thu thập dữ liệu huấn luyện; mô hình không biết các sự kiện sau mốc này.

### Pha nạp ngữ cảnh, pha sinh | prefill, decode
Hai pha của một lần gọi: xử lý toàn bộ đầu vào cùng lúc, rồi sinh từng token đầu ra. Pha sinh bị giới hạn bởi băng thông bộ nhớ.

### Thời gian tới token đầu tiên | time to first token (TTFT)
Thời gian từ lúc gửi yêu cầu tới khi nhận token đầu ra đầu tiên; chủ yếu do pha nạp ngữ cảnh.

### Ảo giác | hallucination
Nội dung trôi chảy và tự tin nhưng sai sự thật do mô hình sinh ra.

### Mô hình mở trọng số | open-weight model
Mô hình có trọng số được công bố để tải về và tự chạy. Khác với mã nguồn mở theo định nghĩa của OSI, vốn đòi hỏi thêm thông tin về dữ liệu và mã huấn luyện.

### API tương thích OpenAI | OpenAI-compatible API
Giao diện gọi mô hình theo định dạng của API OpenAI, được nhiều công cụ tự triển khai (Ollama, LM Studio, vLLM) và dịch vụ trung gian (OpenRouter) hỗ trợ, cho phép đổi mô hình chỉ bằng cách đổi địa chỉ và tên mô hình.

### Định tuyến mô hình | model routing
Phân loại yêu cầu để gửi yêu cầu đơn giản tới mô hình rẻ và yêu cầu khó tới mô hình mạnh.

## Prompt và ngữ cảnh

### Prompt engineering | prompt engineering
Thiết kế chỉ dẫn, ví dụ và định dạng đầu vào để mô hình thực hiện đúng nhiệm vụ.

### Chỉ dẫn hệ thống | system prompt
Tin nhắn có vai "system" chứa vai trò, quy tắc và định dạng đầu ra; mô hình được huấn luyện để ưu tiên nó hơn tin nhắn của người dùng.

### Zero-shot, few-shot | zero-shot, few-shot prompting
Giao nhiệm vụ chỉ bằng mô tả, hoặc kèm một vài ví dụ đầu vào và đầu ra trong prompt.

### Học trong ngữ cảnh | in-context learning
Khả năng mô hình học cách làm nhiệm vụ từ ví dụ nằm trong prompt mà không cập nhật tham số.

### Chain-of-thought (CoT) | chain-of-thought prompting
Yêu cầu mô hình trình bày các bước trung gian trước khi kết luận; cải thiện bài toán nhiều bước, tốn thêm token đầu ra.

### Self-consistency | self-consistency
Sinh nhiều lời giải độc lập rồi chọn đáp án xuất hiện nhiều nhất.

### Đầu ra có cấu trúc | structured output
Đầu ra theo một định dạng chương trình đọc được, như JSON theo một lược đồ.

### Giải mã có ràng buộc | constrained decoding
Đặt xác suất bằng 0 cho các token vi phạm ngữ pháp ở mỗi bước sinh, bảo đảm đầu ra đúng cú pháp.

### Prompt caching | prompt caching
Lưu KV cache của phần tiền tố đã xử lý để dùng lại cho yêu cầu sau có cùng tiền tố; giảm độ trễ và chi phí.

### Streaming | streaming responses
Trả về từng đoạn token ngay khi sinh ra, giảm thời gian người dùng chờ trước khi thấy phản hồi.

### Context engineering | context engineering
Quản lý toàn bộ thông tin nằm trong cửa sổ ngữ cảnh ở mỗi lần gọi mô hình: chỉ dẫn, tool, tài liệu, kết quả tool, lịch sử, bộ nhớ.

### Lớp ngữ cảnh | context layer
Phần của ứng dụng chịu trách nhiệm thu thập, chọn lọc và sắp xếp thông tin trước mỗi lần gọi mô hình.

### Suy giảm theo độ dài ngữ cảnh | context rot
Hiện tượng khả năng tìm và dùng thông tin giảm khi ngữ cảnh dài ra.

### Nén ngữ cảnh | context compaction
Tóm tắt phần cũ của hội thoại hoặc của trạng thái agent để giải phóng chỗ trong cửa sổ ngữ cảnh.

### Cô lập ngữ cảnh | context isolation
Chia việc cho các agent con có cửa sổ ngữ cảnh riêng, chỉ trả về bản tóm tắt, để giữ ngữ cảnh của agent chính gọn.

## Embedding và truy xuất

### Embedding | embedding
Vector số biểu diễn ý nghĩa của một đoạn văn bản (hoặc ảnh), sao cho nội dung gần nghĩa có vector gần nhau.

### Huấn luyện tương phản | contrastive learning
Huấn luyện để vector của cặp liên quan gần nhau hơn vector của các cặp không liên quan; hàm mất mát InfoNCE là cross-entropy trên một lô.

### Bi-encoder, cross-encoder | bi-encoder, cross-encoder
Bi-encoder mã hoá truy vấn và tài liệu riêng rẽ, tính trước được vector tài liệu; cross-encoder nhận cả cặp, chính xác hơn nhưng phải chạy cho từng cặp, dùng để xếp lại.

### BM25 | BM25
Hàm chấm điểm tìm kiếm theo từ khoá dựa trên tần suất từ (có bão hoà), độ hiếm của từ và độ dài tài liệu; mốc so sánh mạnh của truy xuất.

### Tìm kiếm kết hợp | hybrid search
Kết hợp tìm kiếm theo từ khoá và tìm kiếm bằng embedding, thường bằng RRF.

### Reciprocal rank fusion (RRF) | reciprocal rank fusion
Gộp nhiều bảng xếp hạng bằng cách cộng $1/(k + \text{hạng})$ của mỗi tài liệu qua các bảng; chỉ dùng thứ hạng nên không cần chuẩn hoá điểm.

### Recall@k | recall at k
Tỉ lệ tài liệu liên quan (hoặc tỉ lệ truy vấn có đáp án) nằm trong $k$ kết quả đầu.

### MRR | mean reciprocal rank
Trung bình của nghịch đảo vị trí kết quả đúng đầu tiên.

### nDCG | normalized discounted cumulative gain
Thước đo xếp hạng cho phép nhiều mức liên quan, phạt kết quả tốt đứng thấp theo $1/\log_2(\text{vị trí} + 1)$.

### Tìm kiếm láng giềng gần đúng | approximate nearest neighbor (ANN) search
Tìm các vector gần truy vấn mà không so với mọi vector, chấp nhận bỏ sót một phần nhỏ để tăng tốc.

### IVF | inverted file index
Chỉ mục chia không gian bằng K-means; khi tìm, chỉ so truy vấn với vector trong $n_{\text{probe}}$ cụm gần nhất.

### Product quantization (PQ) | product quantization
Nén vector bằng cách chia thành các đoạn con và thay mỗi đoạn bằng chỉ số của tâm gần nhất trong bảng mã 256 phần tử (1 byte).

### HNSW | hierarchical navigable small world
Chỉ mục đồ thị nhiều tầng: tìm từ tầng thưa ở trên xuống tầng đầy đủ ở dưới. Chính xác và nhanh, tốn bộ nhớ.

### Cơ sở dữ liệu vector | vector database
Hệ thống lưu vector cùng metadata, hỗ trợ tìm kiếm gần đúng, lọc, cập nhật và phân quyền.

### RAG | retrieval-augmented generation
Truy xuất tài liệu liên quan rồi đưa vào ngữ cảnh để mô hình trả lời dựa trên tài liệu, kèm trích dẫn.

### Chia đoạn | chunking
Cắt tài liệu thành các đoạn để lập chỉ mục; kích thước và cách cắt ảnh hưởng trực tiếp tới chất lượng truy xuất.

### Xếp lại | reranking
Dùng mô hình chính xác hơn (thường là cross-encoder) để sắp xếp lại vài chục ứng viên đầu của bước truy xuất.

### Độ trung thành với nguồn | faithfulness
Mức mọi khẳng định trong câu trả lời đều được hỗ trợ bởi ngữ cảnh đã truy xuất; thước đo chính của lỗi sinh trong RAG.

## Agent và MCP

### Agent | agent
Hệ thống trong đó mô hình được gọi lặp lại trong vòng lặp, tự chọn hành động (gọi tool hoặc trả lời) dựa trên kết quả các bước trước.

### Workflow | workflow
Hệ thống dùng LLM và tool theo đường đi định sẵn trong mã.

### Function calling, tool calling | function calling
Cơ chế mô hình trả về lời gọi hàm có cấu trúc (tên và tham số theo JSON Schema) để ứng dụng thực thi.

### ReAct | ReAct
Phương pháp prompt xen kẽ suy nghĩ, hành động và quan sát; nền móng của agent hiện đại.

### Hệ nhiều agent | multi-agent system
Chia nhiệm vụ cho nhiều agent có chỉ dẫn, tool và ngữ cảnh riêng, thường theo mẫu điều phối viên và agent con.

### Quyền tối thiểu | least privilege
Nguyên tắc chỉ cấp cho agent hoặc tool đúng những quyền cần cho nhiệm vụ.

### Model Context Protocol (MCP) | Model Context Protocol
Giao thức mở chuẩn hoá cách ứng dụng AI kết nối với tool và dữ liệu, dùng thông điệp JSON-RPC 2.0.

### Host, client, server (MCP) | MCP host, client, server
Host là ứng dụng người dùng tương tác; mỗi client trong host nối với một server; server cung cấp tools, resources và prompts.

### Tools, resources, prompts (MCP) | MCP tools, resources, prompts
Ba loại thành phần của MCP server: hàm để mô hình gọi, dữ liệu để ứng dụng đọc, mẫu prompt để người dùng chọn.

### Elicitation (MCP) | elicitation
Tính năng cho server yêu cầu thêm thông tin từ người dùng trong lúc xử lý một yêu cầu.

### stdio, Streamable HTTP | stdio, Streamable HTTP transport
Hai cách truyền thông điệp của MCP: qua luồng vào ra của tiến trình con trên cùng máy, hoặc qua HTTP tới server từ xa.

## An toàn

### Prompt injection | prompt injection
Văn bản trong ngữ cảnh chứa chỉ dẫn làm mô hình làm theo kẻ tấn công. Dạng gián tiếp nằm trong dữ liệu như email, trang web, tài liệu truy xuất, kết quả tool.

### Jailbreak | jailbreak
Kỹ thuật prompt nhằm vượt qua các giới hạn an toàn mà mô hình được huấn luyện để tuân theo.

### Đánh dấu dữ liệu | spotlighting, datamarking
Biến đổi dữ liệu không đáng tin (ví dụ thay dấu cách bằng một ký tự đặc biệt) và dặn mô hình không làm theo chỉ dẫn trong văn bản được đánh dấu.

### Đầu độc tool | tool poisoning
Giấu chỉ dẫn độc hại trong mô tả tool của một server để điều khiển mô hình.

### Rào chắn | guardrail
Các kiểm tra ở đầu vào và đầu ra: phát hiện thông tin cá nhân, kiểm duyệt nội dung, kiểm tra định dạng, phát hiện chèn lệnh.

### Kiểm thử đối kháng | adversarial testing, red teaming
Chủ động tấn công hệ thống bằng các đầu vào độc hại hoặc bất thường để tìm lỗ hổng trước khi người khác tìm ra.

## Đánh giá và vận hành

### Đánh giá tất định | deterministic evaluation
Kiểm tra bằng mã: đúng lược đồ, đúng đáp án, chạy qua kiểm thử, có trích dẫn.

### LLM làm giám khảo | LLM-as-a-judge
Dùng một LLM chấm điểm hoặc so sánh câu trả lời theo tiêu chí; có thiên lệch vị trí, độ dài và tự đề cao.

### Kiểm định McNemar | McNemar's test
Kiểm định so sánh hai hệ thống trên cùng bộ ví dụ, chỉ dựa vào các ví dụ hai hệ thống cho kết quả khác nhau.

### pass@k | pass@k
Xác suất ít nhất một trong $k$ lời giải sinh ra là đúng; ước lượng không chệch bằng $1 - \binom{n-c}{k}/\binom{n}{k}$.

### Kiểm thử hồi quy | regression testing
Chạy lại bộ đánh giá sau mỗi thay đổi để phát hiện những gì trước đúng nay sai.

### Bộ vàng | golden set
Tập ví dụ có tiêu chí đúng sai rõ ràng, chia theo nhóm tình huống, dùng làm cửa kiểm tra cho mọi thay đổi.

### Trace, span | trace, span
Trace mô tả toàn bộ một yêu cầu như một cây các span; mỗi span là một bước có thời điểm bắt đầu, kết thúc và thuộc tính.

### OpenTelemetry | OpenTelemetry
Chuẩn mở cho trace, số đo và log, có quy ước thuộc tính riêng cho các lời gọi AI tạo sinh.

## Đa phương thức

### AI đa phương thức | multimodal AI
Hệ thống xử lý nhiều loại dữ liệu: văn bản, ảnh, âm thanh, video.

### Token ảnh | image token
Vector biểu diễn một ô ảnh, đưa vào mô hình ngôn ngữ cạnh token văn bản; số token tăng theo bình phương độ phân giải.

### Tỉ lệ lỗi từ (WER) | word error rate
Thước đo của nhận dạng tiếng nói: số phép thay, xoá, chèn chia cho số từ của bản chuẩn.
