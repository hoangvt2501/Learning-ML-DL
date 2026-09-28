# Lời giải chi tiết

Mỗi mục dưới đây ứng với một bài trong Chương 14 của giáo trình. Dòng `@meta` được
script build đọc để gắn nhãn chương, dạng bài và độ khó; nó không hiện ra trên trang.

## Bài 1
@meta chuong=3 | dang=Tính tay | kho=Cơ bản

**Bước 1 — kiểm tra dải.** Dải $[\alpha, \beta] = [-0{,}5;\ 2{,}5]$ đã chứa số 0 nên không phải mở rộng (Mục 3.4).

**Bước 2 — tính $S$.**

$$S = \frac{\beta - \alpha}{q_{max} - q_{min}} = \frac{2{,}5 - (-0{,}5)}{255 - 0} = \frac{3}{255} = \frac{1}{85} \approx 0{,}0117647.$$

Con số này rất dễ chịu: $1/S = 85$ **đúng bằng số nguyên**, nên mọi phép chia phía dưới đều tròn.

**Bước 3 — tính $Z$.**

$$Z = \left\lfloor q_{min} - \frac{\alpha}{S} \right\rceil = \left\lfloor 0 - (-0{,}5) \cdot 85 \right\rceil = \lfloor 42{,}5 \rceil = 42.$$

Đây chính là chỗ gợi ý của đề nhắc: $42{,}5$ rơi đúng vào điểm giữa. NumPy (và cả `round()` của Python) dùng quy tắc **làm tròn nửa về số chẵn**, nên cho 42 chứ không phải 43.

**Hệ quả quan trọng.** Vì $Z$ bị làm tròn xuống, dải thực sự biểu diễn được không còn là $[-0{,}5;\ 2{,}5]$ mà là

$$[\,S(0 - 42),\ S(255 - 42)\,] = \left[-\tfrac{42}{85},\ \tfrac{213}{85}\right] = [-0{,}494118;\ 2{,}505882],$$

tức **toàn bộ lưới bị đẩy sang phải $S/2$**. Đúng hiện tượng đã mô tả ở Mục 3.3 với Hình 4.

**Bước 4 — lượng tử và giải lượng tử.**

| $x$ | $x/S$ | $q = \text{clamp}(\lfloor x/S \rceil + Z)$ | $\hat{x} = S(q - Z)$ | Sai số $\hat{x} - x$ | Loại sai số |
|---|---|---|---|---|---|
| $-0{,}5$ | $-42{,}5$ | $-42 + 42 = 0$ | $-0{,}494118$ | $+0{,}005882$ | làm tròn, đúng bằng $S/2$ |
| $0$ | $0$ | $0 + 42 = 42 = Z$ | $0$ | $0$ | không có |
| $1$ | $85$ | $85 + 42 = 127$ | $1$ | $0$ | không có |
| $3$ | $255$ | $255 + 42 = 297 \to \mathbf{255}$ | $2{,}505882$ | $-0{,}494118$ | **cắt** |

**Trả lời câu hỏi của đề.** Chỉ $x = 3$ chịu **sai số cắt**: phép `clamp` thực sự can thiệp (297 bị kéo về 255) và sai số $0{,}4941$ lớn gấp 84 lần sai số làm tròn tối đa.

Hai giá trị đáng chú ý khác:

- $x = 0$ và $x = 1$ **không sai số chút nào**. Với $x = 0$ là do thiết kế ($q = Z$, Mục 3.4); với $x = 1$ là may mắn số học: $1/S = 85$ nguyên nên 1 nằm đúng trên lưới.
- $x = -0{,}5$ **không bị `clamp`** (vì $q = 0$ vẫn nằm trong $[0, 255]$) nhưng sai số của nó đúng bằng $S/2$ — mức làm tròn lớn nhất có thể. Nó nằm ngay ngoài dải biểu diễn được, đúng ở ranh giới giữa hai loại sai số.

**Biến thể đáng thử.** Nếu thư viện của bạn làm tròn nửa *ra xa số 0* thì $Z = 43$, dải biểu diễn thành $[-0{,}505882;\ 2{,}494118]$, và khi đó $\beta = 2{,}5$ mới là giá trị bị cắt còn $\alpha = -0{,}5$ thì không. Quy ước làm tròn không phải chi tiết vụn vặt.

```python
import numpy as np

S = 3 / 255
Z = int(np.round(0 - (-0.5) / S))          # 42
x = np.array([-0.5, 0.0, 1.0, 3.0])
q = np.clip(np.round(x / S) + Z, 0, 255).astype(np.int32)
xh = S * (q - Z)
print(S, Z); print(q); print(xh); print(xh - x)
```

## Bài 2
@meta chuong=2 | dang=Suy luận | kho=Cơ bản

**Phần a — vì sao `np.float16(2049.0)` bằng 2048.**

FP16 có 10 bit mantissa được lưu, cộng bit ẩn là $p = 11$ bit độ chính xác (Mục 2.2).
Với một số nằm trong $[2^e,\ 2^{e+1})$, hai số FP16 liền kề cách nhau

$$\Delta = 2^{\,e - 10}.$$

Số 2049 nằm trong $[2^{11}, 2^{12}) = [2048, 4096)$, nên $e = 11$ và $\Delta = 2^{1} = 2$. Hai số biểu diễn được kề nó là **2048** và **2050**, còn 2049 rơi **đúng điểm giữa**.

IEEE 754 mặc định làm tròn *về số gần nhất, hoà thì chọn mantissa chẵn*:

- $2048 = 1{,}0000000000_2 \times 2^{11}$ — bit cuối mantissa là **0** (chẵn),
- $2050 = 1{,}0000000001_2 \times 2^{11}$ — bit cuối mantissa là **1** (lẻ).

Nên kết quả là **2048**. Đây cùng một quy tắc đã khiến $Z = \lfloor 42{,}5 \rceil = 42$ ở Bài 1, và khiến `np.float16(1993.5) = 1994` ở Mục 2.3.

**Phần b — số nguyên dương nhỏ nhất không biểu diễn chính xác được.**

Một số nguyên $n$ biểu diễn được chính xác khi và chỉ khi nó viết được dạng $n = m \cdot 2^k$ với $m$ nguyên và $|m| < 2^{p}$. Nói cách khác: $n$ cần không quá $p = 11$ **chữ số có nghĩa** trong hệ nhị phân.

- Mọi $n \le 2048 = 2^{11}$ đều được: $2048 = 1_2 \times 2^{11}$ chỉ cần 1 chữ số có nghĩa.
- $2049 = 2^{11} + 1 = 100000000001_2$ cần **12** chữ số có nghĩa, và không rút gọn được vì nó lẻ.

Vậy đáp số là $\boxed{2049}$, đúng như gợi ý của đề.

**Tổng quát hoá.** Với định dạng có $p$ bit độ chính xác, số nguyên dương nhỏ nhất không biểu diễn chính xác được luôn là $2^{p} + 1$:

| Kiểu | Mantissa lưu | $p$ | $2^p + 1$ |
|---|---|---|---|
| FP32 | 23 | 24 | $16\,777\,217$ |
| FP16 | 10 | 11 | $2049$ |
| BF16 | 7 | 8 | $257$ |
| FP8 E4M3 | 3 | 4 | $17$ |

Bảng này giải thích một sự cố rất hay gặp trong thực tế: **BF16 không đếm nổi quá 256**. Bộ đếm bước huấn luyện, chỉ số token hay ID mà để ở BF16 sẽ đứng yên từ 257 trở đi. Cũng vì vậy mà bộ cộng dồn trong Chương 6 phải là int32 chứ không thể là một kiểu dấu phẩy động ngắn.

```python
import numpy as np
n = 1
while np.float16(n) == n:
    n += 1
print(n)          # 2049
```

## Bài 3
@meta chuong=4 | dang=Chứng minh | kho=Trung bình

**Thiết lập.** Lưới đối xứng *đầy đủ* $[-2^{b-1},\ 2^{b-1}-1]$ có đúng $2^b$ mức, bước nhảy $S$. Mỗi mức "phụ trách" một ô rộng $S$, nên vùng mà lưới phủ được có bề rộng tổng cộng

$$W = 2^b \cdot S.$$

Đề cho dữ liệu phân phối **đều trên đúng dải biểu diễn**, tức đều trên một đoạn rộng $W$.

**Công suất tín hiệu.** Biến ngẫu nhiên đều trên đoạn rộng $W$ (lấy tâm làm gốc) có

$$\mathbb{E}[x^2] = \frac{W^2}{12} = \frac{(2^b S)^2}{12} = \frac{4^{b} S^2}{12}.$$

**Công suất nhiễu.** Mọi giá trị đều nằm trong dải nên không có sai số cắt; chỉ còn sai số làm tròn, phân phối đều trên $[-S/2, S/2]$ (Mục 4.1):

$$\mathbb{E}[e^2] = \frac{S^2}{12}.$$

**Kết quả.**

$$\text{SQNR} = \frac{4^b S^2 / 12}{S^2/12} = 4^{b} = 2^{2b} \;\Longrightarrow\; 10\log_{10} 2^{2b} = \boxed{20\log_{10}(2^b)} \approx 6{,}0206\,b \ \text{dB}.$$

**So sánh với $20\log_{10}(2^b - 2)$ ở Mục 4.2.** Hai giả thiết khác nhau ở đúng hai chỗ, và cả hai đều làm *giảm công suất tín hiệu* chứ không đụng tới nhiễu:

1. **Bỏ bớt một mã.** Lưới hạn chế dùng $[-(2^{b-1}-1),\ 2^{b-1}-1]$, tức vứt giá trị $-2^{b-1}$ để lưới đối xứng hoàn hảo. Còn $2^b - 1$ mức thay vì $2^b$.
2. **Đo dải theo mức chứ không theo ô.** Mục 4.2 lấy dữ liệu đều trên $[-c, c]$ với $c = qS$, $q = 2^{b-1}-1$, tức từ **mức thấp nhất đến mức cao nhất**, bề rộng $2qS = (2^b - 2)S$. So với bề rộng phủ ô $2^b S$, cách này hụt nửa ô ở mỗi đầu.

Chênh lệch giữa hai kết quả vì thế đúng bằng tỉ số biên độ:

$$\Delta = 20\log_{10}\frac{2^b - 2}{2^b} = 20\log_{10}\!\left(1 - 2^{\,1-b}\right).$$

| $b$ | $20\log_{10}(2^b)$ | $20\log_{10}(2^b-2)$ | $\Delta$ |
|---|---|---|---|
| 4 | 24,08 dB | **22,92 dB** | $-1{,}16$ dB |
| 8 | 48,16 dB | **48,09 dB** | $-0{,}07$ dB |
| 16 | 96,33 dB | 96,33 dB | $-0{,}0003$ dB |

Hai cột giữa khớp đúng bảng số liệu ở Mục 4.2.

**Điều cần rút ra.** Hằng số cộng thêm phụ thuộc quy ước, nhưng **độ dốc thì không**: cả hai đều cho $6{,}02$ dB mỗi bit. Khi $b$ lớn, $2^{1-b} \to 0$ và hai công thức trùng nhau. Chênh lệch chỉ đáng kể ở số bit rất thấp — đúng vùng mà mọi thứ khác (chọn ngưỡng, granularity, QAT) cũng bắt đầu quan trọng.

## Bài 4
@meta chuong=5 | dang=Suy luận | kho=Trung bình

**Viết lại phép tích chập.** Với trọng số $W[\,j, c, u, v\,]$ có hình $[C_{out}, C_{in}, k_h, k_w]$:

$$y[n, j, p, q] \;=\; \sum_{c=1}^{C_{in}} \sum_{u=1}^{k_h} \sum_{v=1}^{k_w} W[j, c, u, v]\; x[n, c, \,p s + u,\ q s + v] \;+\; b[j].$$

**Tiêu chuẩn duy nhất** (đúng như lập luận ở Mục 5.2): một hệ số scale đưa được ra ngoài tổng khi và chỉ khi nó **không phụ thuộc chỉ số bị lấy tổng**. Ở đây các chỉ số bị lấy tổng là $c$, $u$, $v$.

Xét từng trục của tensor trọng số:

| Trục | Bị lấy tổng? | Được mang scale riêng? |
|---|---|---|
| $C_{out}$ (chỉ số $j$) | không | **có** — đây chính là per-channel chuẩn |
| $C_{in}$ (chỉ số $c$) | có | không |
| $k_h$ (chỉ số $u$) | có | không |
| $k_w$ (chỉ số $v$) | có | không |

Vậy với tích chập 2D thông thường, **chỉ trục $C_{out}$** mang được scale riêng mà vẫn giữ được bộ cộng dồn thuần số nguyên:

$$y[n,j,p,q] \approx S_{w,j}\, S_x \sum_{c,u,v} q_w[j,c,u,v]\,\bigl(q_x[n,c,\cdot,\cdot] - Z_x\bigr).$$

**Còn activation thì sao?** Chỉ số của $x$ là $[n, c, \text{vị trí}]$. Chỉ số $c$ bị lấy tổng nên **không** được lượng tử activation theo kênh. Chỉ số batch $n$ thì không bị lấy tổng, nên scale theo từng mẫu là hợp lệ — đây đúng là bản sao của "per-token" ở LLM (Mục 11.2).

**Depthwise convolution.** Khi $C_{in}/\text{groups} = 1$, trọng số có hình $[C_{out}, 1, k_h, k_w]$ và mỗi kênh đầu ra $j$ chỉ đọc **đúng một** kênh đầu vào $g(j)$:

$$y[n, j, p, q] \;=\; \sum_{u, v} W[j, 0, u, v]\; x[n,\, g(j),\, ps+u,\, qs+v].$$

Hai thay đổi:

1. **Tổng không còn chạy theo $c$.** Chỉ số kênh đầu vào giờ được xác định bởi $j$ chứ không bị lấy tổng. Về mặt toán học, activation **được phép** mang scale theo kênh ở riêng lớp này, vì $S_{x, g(j)}$ đưa được ra ngoài tổng:
   $$y[n,j,p,q] \approx S_{w,j}\, S_{x,g(j)} \sum_{u,v} q_w[j,0,u,v]\,(q_x - Z_{x,g(j)}).$$
   *Lưu ý thực tế:* đây là điều đúng nhưng hiếm được framework hỗ trợ, vì cùng tensor activation đó thường còn chảy vào các lớp pointwise/Linear phía sau — và ở đó kênh lại bị lấy tổng.
2. **Per-channel trở nên tối quan trọng.** Mỗi kênh đầu ra chỉ có $k_h k_w = 9$ trọng số, và sau khi gộp BatchNorm mỗi kênh bị nhân với $\gamma_j/\sqrt{\sigma_j^2 + \epsilon}$ khác nhau (Mục 7.3). Dải giữa các kênh vì thế chênh nhau rất mạnh — chính là hiện tượng đo được ở Mục 5.3 và Hình 8, và là lý do lượng tử per-tensor 8 bit làm MobileNetV2 sụp đổ.

**Per-group thì sao?** Chia trục $C_{in}$ thành các nhóm liên tiếp (Mục 5.1) *vẫn* đặt scale bên trong tổng. Nó chạy được là vì ta chấp nhận **tách tổng thành từng nhóm**: cộng dồn số nguyên trong mỗi nhóm, rồi nhân scale của nhóm đó và cộng các nhóm lại bằng số thực (hoặc int32 sau requantize). Đổi lại độ chính xác cao hơn là một lần nhân scale cho mỗi nhóm — chấp nhận được khi nhóm đủ lớn (32–128), và đó là lý do per-group là chuẩn cho LLM 4 bit.

## Bài 5
@meta chuong=6 | dang=Lập trình | kho=Trung bình

**Công thức đầy đủ.** Khai triển ở Mục 6.2 không giả định gì về $Z_w$:

$$\sum_k (q_{w,k} - Z_w)(q_{x,k} - Z_x) \;=\; \underbrace{\sum_k q_{w,k} q_{x,k}}_{(1)} \;-\; \underbrace{Z_x \sum_k q_{w,k}}_{(2)} \;-\; \underbrace{Z_w \sum_k q_{x,k}}_{(3)} \;+\; \underbrace{K Z_w Z_x}_{(4)}.$$

Mã trong `code/numpy_experiments.py` bỏ (3) và (4) vì trọng số đối xứng. Muốn dùng trọng số bất đối xứng thì phải cộng lại đúng hai số hạng đó.

**Cài đặt.** Với $X$ hình $[M, K]$ và $W$ hình $[N, K]$:

```python
# Trọng số bất đối xứng: dùng hết miền int8 [-128, 127], Z_w tính như activation
S_w, Z_w = affine_params(W.min(), W.max(), -128, 127)
q_w = np.clip(np.round(W / S_w) + Z_w, -128, 127).astype(np.int32)

row_w = q_w.sum(axis=1)                       # [N] - TÍNH TRƯỚC được (chỉ phụ thuộc W)
row_x = q_x.sum(axis=1, keepdims=True)        # [M,1] - phải tính LÚC CHẠY

acc = (q_x @ q_w.T                            # (1)  M*N*K phép nhân-cộng int8 -> int32
       - Z_x * row_w                          # (2)  tính trước, gộp thẳng vào q_b
       - Z_w * row_x                          # (3)  phải tính lúc chạy
       + K * Z_w * Z_x                        # (4)  hằng số, gộp vào q_b
       + q_b)

M0, shift = quantize_multiplier(S_w * S_x / S_y)
q_y = np.clip(Z_y + rounding_right_shift(acc.astype(np.int64) * M0, 31 + shift), 0, 255)
```

**Mẹo gộp.** Cả (2) và (4) chỉ phụ thuộc trọng số và các hằng $Z_x, Z_w, K$, nên **gộp hẳn vào bias lúc chuyển đổi mô hình**:

$$q_b' = q_b - Z_x \sum_k q_{w,k} + K Z_w Z_x.$$

Sau khi gộp, lúc chạy chỉ còn dư đúng số hạng (3).

**Đếm phép toán phát sinh thêm** (so với trọng số đối xứng):

| Việc | Chi phí | Khi nào |
|---|---|---|
| (2) $Z_x \sum_k q_{w,k}$ | $NK$ phép cộng + $N$ phép nhân | offline, **0 lúc chạy** |
| (4) $K Z_w Z_x$ | 2 phép nhân | offline, **0 lúc chạy** |
| (3) $Z_w \sum_k q_{x,k}$ | $M(K-1)$ cộng + $M$ nhân + $MN$ cộng để phát tán | **lúc chạy** |

So với $MNK$ phép nhân-cộng của số hạng (1), phần dư lúc chạy chiếm tỉ lệ

$$\frac{MK + MN}{MNK} = \frac{1}{N} + \frac{1}{K}.$$

Với lớp $256 \to 64$ trong thí nghiệm ($K = 256$, $N = 64$): $1/64 + 1/256 \approx 1{,}95\%$.

**Nhưng vì sao thực tế vẫn tránh?** Ba lý do mà phép đếm FLOP không thấy:

1. Số hạng (3) buộc phải **quét thêm một lượt toàn bộ ma trận $X$** để lấy tổng theo hàng. Với bài toán memory-bound (Mục 1.3), thêm một lượt đọc đắt hơn nhiều so với 2% phép tính.
2. Tổng theo hàng không gộp được vào vòng lặp GEMM đã tối ưu sẵn của thư viện, nên thường thành một kernel riêng.
3. Trọng số vốn phân bố khá đối xứng quanh 0 (Mục 3.5), nên $Z_w \ne 0$ gần như **không đổi lại được độ chính xác nào**. Trả chi phí mà không nhận được gì.

**Kiểm chứng.** Sau khi sửa, `max |q_y(integer) - q_y(float sim)|` vẫn phải bằng **0 LSB**. Nếu khác 0, gần như chắc chắn bạn quên một trong bốn số hạng, hoặc dùng miền clamp không khớp giữa đường số nguyên và đường mô phỏng số thực (nhớ rằng bất đối xứng dùng hết $[-128, 127]$, còn đối xứng chỉ dùng $[-127, 127]$).

Một nguồn lệch nữa, không liên quan tới bốn số hạng: **quy ước làm tròn của bước dịch phải**. Hàm `rounding_right_shift` trong bài làm tròn nửa về phía $+\infty$, còn gemmlowp/TFLite làm tròn nửa ra xa số 0; hai quy ước chỉ khác nhau ở đúng điểm giữa của giá trị âm và lệch tối đa 1 LSB. Nếu bạn so kết quả với một backend thật mà thấy vài phần tử lệch đúng 1 LSB thì hãy nghi chỗ này trước — xem [ghi chú Mục 6.4](ghi-chu.html#gc-sec-6-4).

## Bài 6
@meta chuong=8 | dang=Thí nghiệm | kho=Trung bình

> **Chưa chạy.** Bài này cần chạy lại thí nghiệm trong `code/torch_experiments.py`, tức cần PyTorch và scikit-learn. Phần dưới vì thế là **phân tích dự đoán kèm cách kiểm chứng**, không phải số liệu đã đo — khác với [Bài 8](#bai-8) là bài duy nhất trong nhóm này đã chạy thật. Hãy chạy rồi đối chiếu với dự đoán.

**Thay đổi cần làm.** Trong phần calibration của `code/torch_experiments.py`, làm bẩn tập calibration:

```python
idx = torch.randperm(len(X_calib))[: max(1, len(X_calib) // 100)]   # 1% số ảnh
X_calib_bad = X_calib.clone()
X_calib_bad[idx] *= 10.0
```

rồi chạy lại cả bốn phương pháp và ghi lại **ngưỡng chọn được** lẫn **độ chính xác W4A4**.

**Dự đoán, xếp theo mức thiệt hại.**

1. **Min–max — hỏng nặng nhất.** Theo định nghĩa nó lấy giá trị lớn nhất trên toàn tập, nên $\beta$ nhảy lên khoảng 10 lần. Bước nhảy $S$ thô lên 10 lần, sai số làm tròn tăng 10 lần, công suất nhiễu tăng 100 lần (Mục 4.1) — tương đương **mất hơn 3 bit** theo quy tắc 6 dB. Ở 4 bit chỉ còn 16 mức, mà 99% dữ liệu giờ nằm gọn trong khoảng 1/10 dưới của dải, tức thực chất chỉ còn 1–2 mức hữu ích. Đây là minh hoạ sống động cho Hình 7: min–max không hề "an toàn".

2. **Percentile 99,99% — hỏng nhiều hơn người ta tưởng.** Phân vị chỉ an toàn khi ta cắt **nhiều hơn** tỉ lệ dữ liệu bị nhiễm. Ở đây 1% số *ảnh* bị nhiễm, và mỗi ảnh bẩn sinh ra rất nhiều activation lớn, nên tỉ lệ *giá trị* nằm trong vùng outlier vượt xa 0,01%. Phân vị 99,99% vì thế vẫn rơi vào vùng nhiễu. Bài học: **con số phân vị phải chọn theo tỉ lệ nhiễm, mà tỉ lệ đó thì ta không biết trước.**

3. **MSE — chịu được một phần.** Nó tối thiểu $\mathbb{E}[(x-\hat{x})^2]$, mà cắt 1% giá trị ở biên độ $10c$ tốn khoảng $0{,}01 \cdot (9c)^2$ sai số bình phương — không nhỏ, nên ngưỡng tối ưu **có** dịch lên. Nhưng đây là một phép đánh đổi trơn chứ không phải yêu cầu cứng "phải phủ hết", nên mức xấu đi nhẹ hơn hẳn min–max.

4. **KL (entropy) — bền nhất.** Nó so sánh *hình dạng phân phối*. 1% khối lượng nằm ở đuôi xa đóng góp rất ít vào $D_{KL}$, trong khi làm thô vùng trung tâm dày dữ liệu thì phạt rất nặng. Ngưỡng chọn được sẽ gần với ngưỡng trên tập sạch.

**Thứ tự dự đoán:** min–max $>$ percentile 99,99% $>$ MSE $>$ KL (thiệt hại giảm dần).

**Hai cảnh báo khi đọc kết quả.**

- Mô hình thí nghiệm rất nhỏ và bài toán rất dễ. Mục 8.2 cho thấy ngưỡng chênh nhau 1,7 lần mà độ chính xác gần như không đổi. Rất có thể độ chính xác **không** phản ánh rõ thứ tự trên; hãy nhìn cả **ngưỡng chọn được** và **MSE của từng lớp**, đó mới là đại lượng nhạy.
- Mỗi ảnh test đổi kết quả 0,19 điểm phần trăm. Đừng kết luận từ chênh lệch dưới 1 điểm nếu chỉ chạy một lần.

**Bài học thực hành (Mục 8.3).** Lỗi thật sự trong bài này không nằm ở phương pháp calibration mà ở **tập calibration**. Một tập bị nhiễm bẩn phá min–max từ rất lâu trước khi nó phá mô hình. Trong quy trình sản xuất, luôn kiểm tra histogram activation trước khi tin vào ngưỡng.

## Bài 7
@meta chuong=9 | dang=Thí nghiệm | kho=Khó

> **Chưa chạy.** Bài này cần huấn luyện lại bằng PyTorch. Phần dưới là **hướng dẫn cài đặt đầy đủ cùng dự đoán**, không phải số liệu đã đo. Khi báo cáo kết quả, hãy ghi rõ đâu là số đo được và đâu là kỳ vọng.

**Ý tưởng LSQ (Mục 9.2).** Thay vì để $S$ bám theo EMA của min/max, ta coi $S$ là **tham số huấn luyện được** và cho nó nhận gradient từ chính hàm mất mát. Với lượng tử đối xứng

$$\hat{x} = S \cdot \text{clamp}\!\left(\left\lfloor x/S \right\rceil,\ -Q_N,\ Q_P\right),$$

đạo hàm theo $S$ (vẫn dùng STE cho phép làm tròn) là

$$\frac{\partial \hat{x}}{\partial S} = \begin{cases} -Q_N & x/S \le -Q_N \\ \lfloor x/S \rceil - x/S & -Q_N < x/S < Q_P \\ Q_P & x/S \ge Q_P \end{cases}$$

**Cài đặt trong PyTorch.** Hai hàm STE nhỏ là đủ, không cần viết `autograd.Function`:

```python
def round_ste(x):                       # lượt xuôi: làm tròn; lượt ngược: đồng nhất
    return (x.round() - x).detach() + x

def grad_scale(x, g):                   # lượt xuôi: x; lượt ngược: nhân gradient với g
    return (x - x * g).detach() + x * g

class LsqQuant(nn.Module):
    def __init__(self, n_bits, signed):
        super().__init__()
        self.Qn = 2 ** (n_bits - 1) if signed else 0
        self.Qp = 2 ** (n_bits - 1) - 1 if signed else 2 ** n_bits - 1
        self.S = nn.Parameter(torch.tensor(1.0))
        self.inited = False

    def forward(self, x):
        if not self.inited:             # khởi tạo theo bài báo LSQ
            self.S.data = 2 * x.detach().abs().mean() / (self.Qp ** 0.5)
            self.inited = True
        g = 1.0 / (x.numel() * self.Qp) ** 0.5      # hệ số chỉnh gradient
        s = grad_scale(self.S, g)
        return torch.clamp(round_ste(x / s), -self.Qn, self.Qp) * s
```

**Ba chi tiết quyết định thành bại.**

1. **Hệ số chỉnh gradient $g = 1/\sqrt{N \cdot Q_P}$**, với $N$ là số phần tử dùng chung scale đó. Không có nó, gradient của $S$ lớn hơn gradient của từng trọng số khoảng $\sqrt{N}$ lần và quá trình huấn luyện dao động rồi phân kỳ. Đây là phần hay bị bỏ sót nhất khi cài lại LSQ.
2. **Khởi tạo.** Dùng $S \leftarrow 2\,\overline{|x|}/\sqrt{Q_P}$ trên batch đầu, hoặc khởi tạo từ ngưỡng PTQ–MSE như `qat()` đang làm. Khởi tạo tệ làm LSQ mất vài epoch chỉ để bò về vùng hợp lý.
3. **QAT là fine-tune.** Giữ nguyên bài học ở Mục 10.1.3: xuất phát từ mô hình FP32 đã hội tụ, learning rate nhỏ ($5 \times 10^{-4}$), vài epoch. Có thể đặt learning rate riêng, lớn hơn một chút, cho các tham số $S$.

**Dự đoán kết quả.** Mốc so sánh là QAT dùng EMA ở Mục 9.3: **94,01%** tại W3A3 và **49,44%** tại W2A2 (FP32: 99,26%).

- **W3A3:** kỳ vọng cải thiện ít. Ở 3 bit, EMA đã cho ngưỡng gần đủ tốt và mạng còn dư sức bù bằng trọng số.
- **W2A2:** kỳ vọng cải thiện rõ nhất. Chỉ còn 4 mức, nên **vị trí đặt mức** quyết định gần như tất cả, và đó đúng là thứ LSQ tối ưu trực tiếp.

**Cách báo cáo cho trung thực.** Chạy **3 hạt giống** như Mục 9.3 và báo cáo trung bình kèm khoảng dao động. Ở W2A2 độ chính xác dao động rất mạnh giữa các lần chạy; một con số đơn lẻ không nói lên điều gì. Nhớ rằng 1 ảnh test = 0,19 điểm phần trăm.

**Hướng mở rộng.** LSQ+ bổ sung thêm **offset học được** (tức $Z$ cũng học), hợp với activation bất đối xứng — đáng thử ở W2A2, nơi mọi mức đều quý.

## Bài 8
@meta chuong=11 | dang=Thí nghiệm | kho=Trung bình

> **Bài này đã được chạy thật.** Script `code/sweep_alpha.py` dựng lại đúng thí nghiệm ở Mục 11.3 — cùng hạt giống 42 và cùng thứ tự rút số ngẫu nhiên — nên nó tái lập chính xác hai mốc đã in trong giáo trình: per-tensor **7,7435%** và SmoothQuant $\alpha = 0{,}5$ cho **1,4430%**. Mọi con số dưới đây lấy từ lần chạy đó, không phải dự đoán.

**Cách quét.**

```python
ax = np.abs(Xl).max(axis=0)          # biên độ activation theo kênh đầu vào
aw = np.abs(Wl).max(axis=0)          # biên độ trọng số theo kênh đầu vào

for alpha in np.arange(0.0, 1.001, 0.05):
    s = ax ** alpha / aw ** (1 - alpha)
    Xs, Ws = Xl / s, Wl * s
    assert np.allclose(Xs @ Ws.T, Yl)                      # biến đổi tương đương
    err = rel(a8_per_tensor(Xs) @ w8_per_channel(Ws).T, Yl)
```

Chạy bằng `python code/sweep_alpha.py`; có matplotlib thì script vẽ luôn đường cong.

**Kết quả đo được.**

| $\alpha$ | 0,00 | 0,20 | 0,30 | 0,40 | 0,50 | **0,60** | 0,70 | 0,80 | 1,00 |
|---|---|---|---|---|---|---|---|---|---|
| Sai số đầu ra | 8,02% | 3,60% | 2,48% | 1,81% | 1,44% | **1,34%** | 1,46% | 1,84% | 3,64% |
| Chênh lệch giữa các kênh $X$ | 91,3× | 37,0× | 23,6× | 15,0× | 9,6× | 6,1× | 3,9× | 2,5× | 1,0× |
| Chênh lệch giữa các cột $W$ | 1,0× | 2,5× | 3,9× | 6,1× | 9,6× | 15,0× | 23,6× | 37,0× | 91,3× |

Đường cong đúng là **chữ U** như trực giác dự đoán. Nhưng hai chi tiết chỉ lộ ra khi đo thật.

### Chi tiết 1 — $\alpha = 0$ không phải "không làm gì"

Không biến đổi gì cả: **7,74%**. Đặt $\alpha = 0$: **8,02%**. Tức là tệ hơn một chút.

Khi $\alpha = 0$ thì $s_j = 1/\max|W_{:,j}|$, nên mỗi kênh activation bị **nhân** với $\max|W_{:,j}|$. Đây là một phép nhân thật, nên:

- Nó **không xoá được** cấu trúc outlier, vì $\max|W_{:,j}|$ chẳng liên quan gì tới việc kênh $j$ có phải outlier hay không. Sáu kênh lớn vẫn lớn.
- Nhưng nó **vẫn làm đổi tỉ lệ** giữa các kênh: tỉ lệ giữa kênh $j$ và kênh $k$ bị nhân thêm $\max|W_{:,j}| / \max|W_{:,k}|$. Ở lần chạy này $\max|W|$ chênh nhau 1,94 lần giữa các cột, và kết quả là chênh lệch giữa các kênh activation **tăng** từ 77,4× lên 91,3×.

> Đây là chỗ dễ nói sai nhất của bài. "Hệ số không tương quan với outlier" **không** đồng nghĩa với "tỉ lệ giữa các kênh không đổi". Không tương quan chỉ có nghĩa là phép nhân ấy không giúp gì; nó vẫn là một phép nhân và vẫn làm xáo trộn tỉ lệ, theo hướng nào thì tuỳ dữ liệu.

Chạy thêm hai hạt giống khác cho 8,48% và 8,43%, so với 8,05% và 8,52% khi không biến đổi. Kết luận **bền vững** là: $\alpha = 0$ gần như không mang lại gì, chứ không phải lúc nào cũng có hại. Toàn bộ độ khó vẫn nằm ở activation, nơi ta chỉ có một scale duy nhất cho cả tensor.

### Chi tiết 2 — cực tiểu nằm ở $\alpha \approx 0{,}60$, không phải 0,50

Ở $\alpha = 0{,}5$, hai vế có **biên độ bằng nhau** đúng như lý thuyết: sau biến đổi cả hai đều bằng

$$\max|X_{:,j}| / s_j \;=\; s_j \max|W_{:,j}| \;=\; \sqrt{\max|X_{:,j}| \cdot \max|W_{:,j}|},$$

tức trung bình nhân — bảng trên cho thấy cả hai dòng đều là 9,6×. Đây đúng là ý tưởng cân bằng giữa các lớp (CLE) ở Mục 9.5, áp dụng giữa activation và trọng số.

**Nhưng biên độ bằng nhau không có nghĩa sai số bằng nhau.** Tách riêng từng nguồn sai số:

| $\alpha$ | chỉ lượng tử $W$ | chỉ lượng tử $X$ | cả hai |
|---|---|---|---|
| 0,0 | 0,66% | 7,99% | 8,02% |
| **0,5** | **0,60%** | **1,31%** | 1,44% |
| 1,0 | 3,55% | 0,80% | 3,64% |

Ở $\alpha = 0{,}5$, activation vẫn là vế **đắt gấp hơn hai lần** trọng số. Vì vậy đẩy thêm chút độ khó sang phía trọng số còn có lợi, và cực tiểu dịch sang 0,60.

**Vì sao trọng số chịu đựng giỏi hơn với cùng mức chênh lệch?** Vì granularity của hai vế khác nhau (Chương 5):

- Trọng số được lượng tử **per-kênh-đầu-ra**: 512 scale, mỗi hàng một cái.
- Activation được lượng tử **per-tensor**: đúng **một** scale cho toàn bộ tensor.

Cùng một mức chênh lệch 9,6×, vế có 512 scale hấp thụ rẻ hơn hẳn vế chỉ có 1 scale. Quy tắc mang đi được: **cực tiểu luôn lệch về phía tensor có granularity mịn hơn.** Nếu đổi activation sang per-token (nhiều scale hơn), cực tiểu sẽ dịch ngược lại gần 0,5.

Cực tiểu ở 0,60 lặp lại y hệt trên cả ba hạt giống đã thử, nên đây không phải chuyện may rủi của một lần rút số.

### Vì sao $\alpha = 1$ cũng không tốt

Khi đó $s_j = \max|X_{:,j}|$:

- $X\,\text{diag}(s)^{-1}$: mọi kênh activation về biên độ 1 — activation **phẳng lý tưởng**, sai số chỉ còn 0,80%.
- $W\,\text{diag}(s)$: cột $j$ của trọng số bị nhân lên $\max|X_{:,j}|$, nên **toàn bộ độ chênh 91,3× chuyển sang trọng số**. Chỗ chết người: độ chênh vừa tạo ra nằm dọc theo **cột** (kênh đầu vào), trong khi scale của trọng số lại theo **hàng** (kênh đầu ra). Vậy là độ chênh nằm gọn *bên trong* phạm vi của một scale, đúng chỗ mà per-channel không với tới. Sai số do trọng số nhảy từ 0,60% lên 3,55%.

### Những gì mang đi được

- **Hình dạng chữ U** và lý do tạo ra nó: hai vế cùng chia nhau một lượng độ khó cố định, ép hết về một phía đều đắt.
- **Đáy khá phẳng**: mọi $\alpha \in [0{,}5;\ 0{,}7]$ đều dưới 1,5%. Chọn 0,5 như bài báo chỉ đắt hơn cực tiểu 8%, không đáng để dò kỹ.
- **Cực tiểu lệch về phía có granularity mịn hơn** — đây mới là quy tắc dùng lại được cho mô hình thật.
- **Số tuyệt đối thì không mang đi được.** Dữ liệu ở đây là tổng hợp, chỉ 6 kênh outlier, biên độ gấp đúng 60 lần. Bài báo SmoothQuant lấy $\alpha = 0{,}5$ làm mặc định và ghi nhận mô hình có outlier nặng hơn cần giá trị lớn hơn; muốn biết con số cho mô hình của mình thì phải quét trên chính mô hình đó.

## Bài 9
@meta chuong=11 | dang=Tính toán | kho=Cơ bản

**Dữ kiện.** 13 tỉ tham số; $n_{\text{layers}} = 40$; $n_{\text{heads}} = 40$; $d_{\text{head}} = 128$; không dùng GQA nên $n_{\text{kv\_heads}} = n_{\text{heads}} = 40$. Batch 4, ngữ cảnh 8192 token.

### a) Dung lượng trọng số ở GGUF `Q4_K`

Bảng ở Mục 11.8 cho số bit thực tế mỗi trọng số của `Q4_K`:

$$\frac{256 \cdot 4 \;+\; 8 \cdot 12 \;+\; 32}{256} = \frac{1024 + 96 + 32}{256} = \frac{1152}{256} = 4{,}5 \ \text{bit/trọng số}.$$

$$13 \times 10^{9} \times 4{,}5 \ \text{bit} = 58{,}5 \times 10^{9} \ \text{bit} = 7{,}3125 \times 10^{9} \ \text{byte} = \boxed{7{,}31\ \text{GB}} = 6{,}81 \ \text{GiB}.$$

*Đối chiếu thực tế:* một file `Q4_K_M` 13B thật thường lớn hơn con số này một chút (quãng 7,5–8 GiB), vì llama.cpp giữ embedding và lớp đầu ra ở độ chính xác cao hơn và trộn `Q6_K` cho một số tensor — đúng như cảnh báo cuối Mục 11.8.

### b) Dung lượng KV cache FP16

Áp công thức ở Mục 11.7:

$$2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{số token} \times \text{batch} \times \text{số byte}$$

Tính theo từng bậc cho dễ kiểm tra:

| Đại lượng | Phép tính | Kết quả |
|---|---|---|
| Mỗi token, mỗi chuỗi | $2 \times 40 \times 40 \times 128 \times 2$ | $819\,200$ B $= 0{,}78$ MiB |
| Một chuỗi 8192 token | $\times\, 8192$ | $6{,}71$ GB $= 6{,}25$ GiB |
| Batch 4 | $\times\, 4$ | $\boxed{26{,}84\ \text{GB}} = 25{,}0$ GiB |

### c) Thành phần nào lớn hơn

**KV cache lớn hơn trọng số 3,67 lần** (26,84 GB so với 7,31 GB). Tổng cộng khoảng **34 GB**, tức không vừa một GPU 24 GB, dù "mô hình 4 bit" nghe như chỉ tốn 7 GB.

**Điều cần rút ra.** Với ngữ cảnh dài và batch không nhỏ, **lượng tử trọng số thôi là chưa đủ**. Các hướng xử lý, theo đúng Mục 11.7:

| Cách | Hiệu quả trong ví dụ này |
|---|---|
| Lượng tử KV cache sang FP8/INT8 | 26,84 GB → 13,4 GB |
| Lượng tử KV cache sang INT4 (kiểu KIVI) | 26,84 GB → 6,7 GB |
| Dùng GQA với 8 KV head | 26,84 GB → 5,4 GB (giảm 5 lần) |
| Giảm batch xuống 1 | 26,84 GB → 6,7 GB |

Chính phép tính này giải thích vì sao **GQA (và trước đó là MQA) trở thành lựa chọn phổ biến** ở các mô hình mở cỡ lớn ra đời từ 2023 trở đi: nó cắt KV cache ngay từ kiến trúc, tức trước khi phải nhờ tới lượng tử hoá. Nhưng đây là một **xu hướng**, không phải quy luật và cũng không phải cách duy nhất — vẫn có mô hình giữ MHA đầy đủ, và các hướng khác như lượng tử KV cache, nén KV cache hay attention thưa đều nhắm vào cùng nút thắt. Khi đọc một mô hình cụ thể, hãy tra $n_{\text{kv\_heads}}$ trong file cấu hình của nó thay vì suy đoán từ năm phát hành.
