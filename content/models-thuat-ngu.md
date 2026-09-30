# Từ điển thuật ngữ — Học sâu

Cú pháp: `## Nhóm`, rồi `### Tiếng Việt | English` và phần định nghĩa bên dưới.
Mỗi định nghĩa khớp với cách dùng trong giáo trình. Thuật ngữ nào giáo trình giữ nguyên tiếng Anh thì tên tiếng Anh đứng trước.

## Sai số và tổng quát hoá

### Độ chệch | bias (của một ước lượng)
Chênh lệch giữa dự đoán trung bình của mô hình, lấy trung bình trên các tập huấn luyện có thể rút ra, và giá trị thật. Độ chệch cao cho biết họ hàm quá hẹp so với hàm cần học (Mục 2.2). Khác với hệ số chặn, là tham số $b$ trong $Wx + b$.

### Phương sai | variance
Mức dự đoán của mô hình thay đổi khi đổi tập huấn luyện. Phương sai cao cho biết mô hình đủ linh hoạt để khớp cả phần nhiễu riêng của từng tập huấn luyện (Mục 2.2).

### Underfitting | underfitting
Tình trạng độ chệch cao: sai số huấn luyện và sai số xác thực đều cao và gần nhau. Thêm dữ liệu không giúp được; cần mô hình linh hoạt hơn.

### Overfitting | overfitting
Tình trạng phương sai cao: sai số huấn luyện thấp, sai số xác thực cao hơn nhiều. Thêm dữ liệu, tăng regularization hoặc dùng mô hình đơn giản hơn giúp giảm.

### Nhiễu không giảm được | irreducible error
Phần sai số $\sigma^2$ do bản thân dữ liệu, không phụ thuộc mô hình. Là cận dưới của sai số mà không mô hình nào vượt qua được.

### Đường cong học | learning curve
Đồ thị sai số huấn luyện và sai số xác thực theo số điểm dữ liệu huấn luyện hoặc theo số vòng huấn luyện. Công cụ chẩn đoán độ chệch và phương sai dùng được với dữ liệu thật (Mục 2.4).

### Double descent | double descent
Hiện tượng sai số xác thực tăng tới một đỉnh tại ngưỡng mô hình vừa đủ tham số để khớp chính xác dữ liệu huấn luyện, rồi giảm trở lại khi số tham số tiếp tục tăng. Không mâu thuẫn với phân rã độ chệch – phương sai (Mục 2.5).

### Thiên kiến quy nạp | inductive bias
Tập các giả định một thuật toán học dùng để chọn giữa các hàm cùng khớp dữ liệu huấn luyện. Mạng tích chập giả định tính cục bộ và tính đẳng biến với dịch chuyển; Transformer giả định rất ít. Giả định đúng giúp học từ ít dữ liệu; giả định sai gây độ chệch (Mục 1.4).

### Định lý không có bữa trưa miễn phí | no free lunch theorem
Lấy trung bình trên mọi bài toán có thể, mọi thuật toán học có cùng sai số trên các điểm ngoài tập huấn luyện. Một thuật toán chỉ tốt hơn trên những bài toán mà giả định của nó phù hợp (Wolpert, 1996).

## Cây và phương pháp tập hợp

### Cây quyết định | decision tree
Mô hình chia không gian đặc trưng thành các hình hộp bằng các phép chia $x_j \le t$ và dự đoán một hằng số trong mỗi hộp. Bất biến với phép biến đổi đơn điệu của từng đặc trưng; không ngoại suy được (Mục 3.1).

### Độ không thuần nhất Gini | Gini impurity
$1 - \sum_c p_c^2$, với $p_c$ là tỉ lệ lớp $c$ trong một nút. Bằng 0 khi nút chỉ có một lớp. Gần như luôn chọn cùng phép chia với entropy và tính nhanh hơn (Mục 3.2).

### Bagging | bootstrap aggregating
Huấn luyện nhiều mô hình trên các mẫu bootstrap rồi lấy trung bình dự đoán. Phương sai của trung bình $M$ mô hình là $\rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$, nên bagging giảm phương sai, và muốn giảm tiếp phải giảm tương quan $\rho$ (Mục 3.3).

### Rừng ngẫu nhiên | random forest
Bagging với cây quyết định, trong đó mỗi nút chỉ chọn phép chia trong một tập con ngẫu nhiên các đặc trưng để giảm tương quan giữa các cây. Thêm cây không làm kết quả kém đi.

### Gradient boosting | gradient boosting
Cộng dần các cây nông, mỗi cây xấp xỉ gradient âm của hàm mất mát theo dự đoán hiện tại: gradient descent trong không gian hàm. Chủ yếu giảm độ chệch; có thể overfitting nếu có quá nhiều cây (Mục 3.3).

### Permutation importance | permutation importance
Độ quan trọng của một đặc trưng đo bằng mức sai số tăng lên khi xáo trộn ngẫu nhiên giá trị của đặc trưng đó. Đáng tin hơn độ quan trọng tính từ độ giảm độ không thuần nhất của cây.

## Mạng nơ-ron

### Mạng nơ-ron truyền thẳng nhiều lớp | multilayer perceptron, MLP
Chuỗi các lớp affine xen kẽ hàm kích hoạt phi tuyến. Không có hàm phi tuyến thì cả mạng tương đương một phép biến đổi affine (Mục 4.1).

### Hàm kích hoạt | activation function
Hàm phi tuyến áp dụng cho từng phần tử sau mỗi lớp tuyến tính: sigmoid, tanh, ReLU, GELU, SiLU. Đạo hàm của ReLU bằng 1 ở phía dương là lý do chính ReLU thay sigmoid trong mạng sâu (Mục 4.2).

### Đơn vị ReLU chết | dying ReLU
Đơn vị ReLU có đầu vào âm với mọi mẫu nên đạo hàm luôn bằng 0 và không bao giờ được cập nhật lại. Thường do tốc độ học quá lớn.

### Định lý xấp xỉ phổ quát | universal approximation theorem
Mạng một lớp ẩn đủ rộng, với hàm kích hoạt liên tục không phải đa thức, xấp xỉ được mọi hàm liên tục trên tập compact. Chỉ khẳng định sự tồn tại, không cho biết cần bao nhiêu đơn vị hay có học được không (Mục 4.3).

### Lan truyền ngược | backpropagation
Thuật toán tính gradient của hàm mất mát theo mọi tham số bằng quy tắc dây chuyền, đi từ đầu ra ngược về đầu vào và dùng lại các giá trị trung gian của lượt xuôi. Chỉ tính gradient; việc cập nhật tham số do thuật toán tối ưu đảm nhận (Mục 5.1).

### Chế độ ngược và chế độ xuôi | reverse-mode and forward-mode differentiation
Hai thứ tự nhân các ma trận Jacobi. Chế độ ngược tốn chi phí cỡ số đầu ra lần tính hàm, chế độ xuôi cỡ số đầu vào lần; với hàm mất mát là một số và hàng tỉ tham số, chế độ ngược rẻ hơn rất nhiều (Mục 5.2).

### Gradient checkpointing | gradient checkpointing
Chỉ lưu giá trị kích hoạt ở một số lớp mốc và tính lại các lớp khác khi lượt ngược cần. Giảm bộ nhớ kích hoạt từ $O(L)$ xuống $O(\sqrt L)$, đổi lại khoảng một lượt xuôi tính thêm (Mục 5.3).

## Tối ưu

### SGD theo mini-batch | mini-batch stochastic gradient descent
Gradient descent với gradient ước lượng trên một lô nhỏ điểm dữ liệu chọn ngẫu nhiên. Ước lượng không chệch, phương sai giảm theo kích thước lô.

### Momentum | momentum
Cập nhật theo trung bình trượt của các gradient, triệt tiêu dao động và cộng dồn hướng ổn định. Với $\beta = 0{,}9$, bước theo hướng ổn định lớn gấp khoảng 10 lần.

### Adam | Adam
Thuật toán tối ưu giữ trung bình trượt của gradient và của bình phương gradient, chia cái thứ nhất cho căn bậc hai của cái thứ hai để mỗi tham số có bước đi riêng. Cần hai trạng thái cho mỗi tham số (Mục 5.4).

### Hiệu chỉnh độ chệch | bias correction
Phép chia cho $1 - \beta^t$ trong Adam. Vì khởi tạo bằng 0, trung bình trượt $m_t$ có kỳ vọng $(1 - \beta_1^t)\,\mathbb{E}[g]$, tức là ước lượng chệch về 0 ở các bước đầu; phép chia cho ước lượng không chệch.

### AdamW | AdamW
Biến thể của Adam trừ weight decay trực tiếp vào tham số thay vì cộng vào gradient, để thành phần phạt không bị chia cho $\sqrt v$. Lựa chọn mặc định khi huấn luyện Transformer.

### Warmup | warmup
Tăng dần tốc độ học từ gần 0 trong những bước đầu. Cần thiết với Adam, vì các ước lượng ban đầu của $v$ rất nhiễu, và với Transformer dùng post-LN.

### Lịch cosine | cosine schedule
Giảm tốc độ học theo nửa chu kỳ hàm cosin về gần 0 ở cuối quá trình huấn luyện. Lịch phổ biến khi huấn luyện mô hình ngôn ngữ.

## Khởi tạo, chuẩn hoá và regularization

### Khởi tạo Xavier | Xavier (Glorot) initialization
Khởi tạo trọng số với phương sai $2/(n_{\text{in}} + n_{\text{out}})$ để giữ phương sai của tín hiệu ở cả lượt xuôi lẫn lượt ngược. Phù hợp với tanh và sigmoid.

### Khởi tạo He | He initialization
Khởi tạo trọng số với phương sai $2/n_{\text{in}}$. Hệ số 2 bù cho việc ReLU đặt một nửa phân phối bằng 0, vì $\mathbb{E}[\operatorname{ReLU}(z)^2] = \tfrac12\operatorname{Var}(z)$ (Mục 6.2).

### Gradient tiêu biến và bùng nổ | vanishing and exploding gradients
Gradient qua nhiều lớp là tích nhiều ma trận Jacobi, nên co về 0 hoặc tăng vọt theo cấp số mũ nếu mỗi lớp nhân độ lớn với một hệ số lệch khỏi 1 (Mục 6.1).

### BatchNorm | batch normalization
Chuẩn hoá mỗi đặc trưng bằng trung bình và phương sai tính trên các mẫu trong lô. Phụ thuộc kích thước lô, và hành vi khi suy luận khác khi huấn luyện. Phổ biến trong mạng tích chập.

### LayerNorm | layer normalization
Chuẩn hoá mỗi mẫu bằng trung bình và phương sai tính trên các đặc trưng của chính mẫu đó. Không phụ thuộc lô; là lựa chọn của Transformer.

### RMSNorm | RMS normalization
LayerNorm bỏ bước trừ trung bình, chỉ chia cho căn trung bình bình phương. Rẻ hơn, chất lượng tương đương; dùng trong Llama và nhiều mô hình ngôn ngữ khác.

### Kết nối tắt | residual connection, skip connection
Thay $h = F(x)$ bằng $h = x + F(x)$, cho gradient một đường đi không qua ma trận trọng số. Cần đi kèm chuẩn hoá: một mình nó làm gradient bùng nổ trong thí nghiệm ở Mục 6.3.

### Vấn đề suy thoái | degradation problem
Hiện tượng mạng thường sâu hơn có sai số huấn luyện cao hơn mạng nông hơn. Là vấn đề tối ưu, không phải overfitting, và là lý do kết nối tắt ra đời (Mục 6.5).

### Dropout | dropout
Khi huấn luyện, đặt mỗi đơn vị bằng 0 với xác suất $p$ và chia các đơn vị còn lại cho $1 - p$; khi suy luận dùng mọi đơn vị. Xấp xỉ việc lấy trung bình nhiều mạng con.

### Weight decay | weight decay
Phạt chuẩn $\ell_2$ của trọng số, tương đương ước lượng MAP với tiên nghiệm Gauss. Với Adam phải dùng dạng AdamW.

### Làm mượt nhãn | label smoothing
Thay nhãn one-hot $e_y$ bằng $(1 - \epsilon)e_y + \epsilon/K$ để mô hình không bị đẩy tới xác suất 0 và 1 tuyệt đối.

## Mạng tích chập

### Tích chập | convolution
Phép tính tích vô hướng giữa một bộ lọc nhỏ và từng vùng của đầu vào, cùng bộ lọc dùng cho mọi vị trí. Số tham số không phụ thuộc kích thước ảnh; số phép tính thì có (Mục 7.1).

### Bước nhảy và phần đệm | stride, padding
Bước nhảy là khoảng cách giữa hai vị trí liên tiếp của bộ lọc; phần đệm là số ô thêm vào quanh biên. Kích thước đầu ra là $\lfloor (H + 2p - k)/s \rfloor + 1$.

### Trường tiếp nhận | receptive field
Vùng của đầu vào có ảnh hưởng tới một đơn vị. Tăng tuyến tính theo số lớp khi bước nhảy bằng 1; trường tiếp nhận hiệu dụng nhỏ hơn nhiều so với lý thuyết (Mục 7.2).

### Tích chập tách theo chiều sâu | depthwise separable convolution
Tích chập riêng cho từng kênh rồi tích chập $1 \times 1$. Rẻ hơn tích chập thường khoảng $k^2$ lần; nền tảng của MobileNet.

### Vision Transformer | Vision Transformer, ViT
Cắt ảnh thành các mảnh $16 \times 16$ và xử lý chúng như token bằng Transformer. Kém CNN khi ít dữ liệu, vượt CNN khi tiền huấn luyện trên tập rất lớn (Mục 7.4).

## Mạng hồi quy

### Mạng nơ-ron hồi quy | recurrent neural network, RNN
Mạng xử lý chuỗi bằng một trạng thái ẩn cập nhật qua từng bước với cùng một bộ trọng số. Không song song hoá được theo thời gian (Mục 8.1).

### Cắt ngưỡng gradient | gradient clipping
Nếu chuẩn của gradient vượt ngưỡng $\tau$ thì nhân gradient với $\tau/\lVert g\rVert$. Xử lý gradient bùng nổ mà giữ nguyên hướng.

### LSTM | long short-term memory
RNN có thêm trạng thái ô nhớ cập nhật bằng phép cộng có trọng số và ba cổng quên, vào, ra. Dọc ô nhớ, gradient nhân với giá trị cổng quên thay vì với ma trận trọng số (Mục 8.3).

### Cổng quên | forget gate
Cổng của LSTM quyết định giữ lại bao nhiêu phần ô nhớ cũ. Với cổng quên không đổi $f$, bộ nhớ có thang thời gian khoảng $1/(1 - f)$ bước; hệ số chặn của cổng thường khởi tạo bằng 1.

### GRU | gated recurrent unit
Biến thể đơn giản của LSTM với hai cổng, ít hơn khoảng 25% tham số, chất lượng thường tương đương.

## Transformer

### Attention | scaled dot-product attention
$\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$: mỗi truy vấn lấy trung bình có trọng số của các giá trị, với trọng số tính từ độ giống giữa truy vấn và các khoá (Mục 9.1).

### Phép chia cho √dₖ | scaling by √dₖ
Tích vô hướng của hai vector ngẫu nhiên có phương sai $d_k$; chia cho $\sqrt{d_k}$ đưa phương sai về 1, tránh để softmax bão hoà làm gradient gần bằng 0 (Mục 9.2).

### Attention nhiều đầu | multi-head attention
$h$ phép attention song song trên các không gian con $d/h$ chiều, rồi nối lại và chiếu. Tổng chi phí gần bằng attention một đầu; lợi ích là theo dõi nhiều loại quan hệ cùng lúc.

### Mặt nạ nhân quả | causal mask
Che mọi vị trí phía sau khi tính attention cho một vị trí, để mô hình sinh văn bản không thấy tương lai. Cho phép tính dự đoán cho mọi vị trí của chuỗi trong một lượt xuôi.

### Mã hoá vị trí | positional encoding
Cách đưa thông tin thứ tự vào Transformer, vì self-attention không phân biệt thứ tự các token: sin–cos cố định, embedding vị trí học được, hoặc vị trí tương đối như RoPE và ALiBi (Mục 9.6).

### Encoder và decoder | encoder, decoder
Encoder dùng attention hai chiều, tiền huấn luyện bằng cách đoán token bị che (BERT). Decoder dùng mặt nạ nhân quả, tiền huấn luyện bằng cách đoán token tiếp theo (GPT). Encoder–decoder kết hợp hai phần bằng cross-attention (T5) (Mục 9.7).

### Pre-LN và post-LN | pre-norm, post-norm
Post-LN chuẩn hoá sau phép cộng của kết nối tắt; pre-LN chuẩn hoá trước khối con để đường tắt đi thẳng. Pre-LN huấn luyện ổn định hơn và ít cần warmup (Mục 10.2).

### RoPE | rotary position embedding
Quay từng cặp chiều của truy vấn và khoá một góc tỉ lệ với vị trí. Điểm số attention chỉ phụ thuộc khoảng cách tương đối: $\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$ (Mục 10.3).

### KV cache | KV cache
Khoá và giá trị của các token đã xử lý, lưu lại để không phải tính lại khi sinh token mới. Dung lượng $2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times T \times B$ nhân số byte (Mục 10.4).

### MQA và GQA | multi-query attention, grouped-query attention
Nhiều đầu truy vấn dùng chung khoá và giá trị: MQA dùng một cặp cho mọi đầu, GQA dùng một cặp cho mỗi nhóm đầu. Giảm KV cache theo đúng tỉ lệ số đầu; là quyết định vì bộ nhớ khi suy luận.

### SwiGLU | SwiGLU
Khối FFN có cổng, $W_2(\operatorname{SiLU}(Wx) \odot Vx)$, gồm ba ma trận. Để giữ số tham số, số chiều ẩn giảm còn $\tfrac83 d$ (Mục 10.5).

### Mixture of experts | mixture of experts, MoE
Thay khối FFN bằng nhiều chuyên gia và một bộ định tuyến chọn vài chuyên gia cho mỗi token. Tăng số tham số mà không tăng số phép tính mỗi token, nhưng mọi chuyên gia phải nằm trong bộ nhớ (Mục 10.6).

### FlashAttention | FlashAttention
Cách tính chính xác attention theo từng khối trong bộ nhớ trên chip, không lưu ma trận $T \times T$. Bộ nhớ tăng tuyến tính theo độ dài chuỗi và thời gian chạy giảm; số phép tính vẫn bậc hai (Mục 10.7).

## Giải mã

### Giải mã tham lam | greedy decoding
Luôn chọn token có xác suất cao nhất. Tất định về nguyên tắc, phù hợp với bài toán có một đáp án đúng.

### Beam search | beam search
Giữ nhiều chuỗi ứng viên có xác suất cao nhất ở mỗi bước. Phù hợp với dịch máy; với văn bản mở thường cho kết quả lặp và nhạt (Mục 11.4).

### Nhiệt độ | temperature
Chia logit cho $\tau$ trước softmax. $\tau < 1$ làm phân phối nhọn hơn, $\tau > 1$ làm phẳng hơn, $\tau \to 0$ cho greedy.

### Top-k | top-k sampling
Chỉ giữ $k$ token có xác suất cao nhất rồi lấy mẫu. Ngưỡng cố định, không thích ứng theo mức chắc chắn của mô hình.

### Top-p | nucleus sampling
Giữ tập nhỏ nhất các token có tổng xác suất ít nhất $p$ rồi lấy mẫu. Số token giữ lại thay đổi theo phân phối ở từng bước.

### Giải mã suy đoán | speculative decoding
Một mô hình nhỏ đề xuất trước vài token, mô hình lớn kiểm tra trong một lượt xuôi, và một quy tắc chấp nhận–từ chối giữ nguyên phân phối đầu ra. Giảm độ trễ mà không đổi chất lượng.

## Tài nguyên tính toán

### Tham số không tính embedding | non-embedding parameters
Số tham số của các khối Transformer, khoảng $12Ld^2$ với FFN hai ma trận và $d_{\text{ff}} = 4d$. Là $N$ trong các công thức FLOP và quy luật co giãn (Mục 12.2).

### Quy tắc 6ND | the 6ND rule
Huấn luyện tốn khoảng $6ND$ FLOP: $2N$ mỗi token ở lượt xuôi và $4N$ ở lượt ngược (Mục 12.4).

### Mức sử dụng phần cứng | model FLOPs utilization, MFU
Tỉ lệ giữa số FLOP có ích mỗi giây mà quá trình huấn luyện đạt được và tốc độ tối đa của phần cứng. Thường 30% tới 50% với mô hình lớn.

### Quy luật co giãn | scaling laws
Quan hệ dạng luỹ thừa giữa mất mát với số tham số, lượng dữ liệu và lượng tính toán. Kết quả của Chinchilla: với ngân sách cố định, số tham số và số token nên tăng cùng tỉ lệ, khoảng 20 token cho mỗi tham số.

### Bộ nhớ huấn luyện | training memory
Với Adam và độ chính xác hỗn hợp, khoảng 16 byte mỗi tham số cho trọng số, gradient và trạng thái bộ tối ưu, chưa tính giá trị kích hoạt (Mục 12.5).
