// Phòng thí nghiệm: năm công cụ tương tác, cài đặt đúng theo công thức trong giáo trình.

const num = (id, label, value, attrs = '') =>
  '<label class="ctl"><span class="ctl-label">' + label + '</span>' +
  '<input type="number" id="' + id + '" value="' + value + '" ' + attrs + '></label>';

const range = (id, label, value, min, max, step) =>
  '<label class="ctl ctl-range"><span class="ctl-label">' + label +
  ' <output id="' + id + 'Out">' + value + '</output></span>' +
  '<input type="range" id="' + id + '" value="' + value + '" min="' + min +
  '" max="' + max + '" step="' + step + '"></label>';

const select = (id, label, options) =>
  '<label class="ctl"><span class="ctl-label">' + label + '</span><select id="' + id + '">' +
  options.map((o) => '<option value="' + o[0] + '"' + (o[2] ? ' selected' : '') + '>' + o[1] + '</option>').join('') +
  '</select></label>';

const stat = (id, label, hint = '') =>
  '<div class="stat"><span class="stat-label">' + label + '</span>' +
  '<span class="stat-value" id="' + id + '">–</span>' +
  (hint ? '<span class="stat-hint">' + hint + '</span>' : '') + '</div>';

export function buildPlaygroundPage(ctx) {
  const { md, nav, page, write, neighbours, registry, f } = ctx;
  registry.currentFile = f('thuc-hanh.html');

  // Công thức trong phần mô tả được KaTeX dựng sẵn lúc build.
  const t = (s) => md.renderInline(s);

  // ---------------------------------------------------- 1. máy lượng tử affine
  const lab1 =
    '<section class="lab" id="may-luong-tu" data-lab="affine">' +
    '<header class="lab-head"><h3>1 · Máy lượng tử affine</h3><p>' +
    t('Nhập dải và độ rộng bit, xem ngay $S$, $Z$ và toàn bộ hàm bậc thang. Đây chính là ví dụ ' +
      'tính tay ở [Mục 3.3](ch03.html#sec-3-3) và hình dạng ở [Hình 4](ch03.html#hinh-4), ' +
      'nhưng đổi được tham số.') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    num('afAlpha', 'Cận dưới α', '-1', 'step="0.1"') +
    num('afBeta', 'Cận trên β', '3', 'step="0.1"') +
    select('afBits', 'Độ rộng bit b', [
      ['2', '2 bit'], ['3', '3 bit', true], ['4', '4 bit'], ['6', '6 bit'], ['8', '8 bit'],
    ]) +
    select('afMode', 'Kiểu lượng tử', [
      ['unsigned', 'Bất đối xứng, không dấu (uint)', true],
      ['signed', 'Bất đối xứng, có dấu (int)'],
      ['symmetric', 'Đối xứng, miền hạn chế (Z = 0)'],
    ]) +
    num('afX', 'Giá trị x cần lượng tử', '1.5', 'step="0.1"') +
    '<p class="ctl-note">Phép làm tròn dùng quy tắc <b>nửa về số chẵn</b> giống NumPy.</p>' +
    '</div><div class="lab-output">' +
    '<div class="stat-row">' +
    stat('afS', 'Bước nhảy S') + stat('afZ', 'Zero-point Z') + stat('afRange', 'Miền số nguyên') +
    '</div><div class="stat-row">' +
    stat('afQ', 'q của x') + stat('afXhat', 'x̂ khôi phục') + stat('afErr', 'Sai số x̂ − x') +
    '</div>' +
    '<p class="lab-note" id="afNote"></p>' +
    '<div class="chart" id="afChart"></div>' +
    '<p class="chart-cap">' +
    t('Trên: hàm fake quantization $x \\mapsto \\hat{x}$. Dưới: sai số $\\hat{x} - x$. ' +
      'Vùng tô đỏ là vùng bị cắt.') +
    '</p></div></div></section>';

  // ------------------------------------------------ 2. đánh đổi làm tròn – cắt
  const lab2 =
    '<section class="lab" id="lam-tron-cat" data-lab="clip">' +
    '<header class="lab-head"><h3>2 · Đánh đổi giữa làm tròn và cắt</h3><p>' +
    t('Thu hẹp dải làm bước nhảy mịn hơn nhưng cắt nhiều hơn, nên tổng MSE có một cực tiểu. ' +
      'Mô phỏng lại [Mục 4.3](ch04.html#sec-4-3) và [Hình 7](ch04.html#hinh-7) ngay trên trình duyệt.') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    select('clDist', 'Phân phối dữ liệu', [
      ['laplace', 'Laplace (giống trọng số thật)', true],
      ['gauss', 'Gauss chuẩn'],
      ['relu', 'Gauss sau ReLU (một phía)'],
    ]) +
    select('clBits', 'Độ rộng bit', [
      ['2', '2 bit'], ['3', '3 bit'], ['4', '4 bit', true], ['6', '6 bit'], ['8', '8 bit'],
    ]) +
    range('clC', 'Ngưỡng cắt c', '4.8', '0.2', '16', '0.1') +
    '<p class="ctl-note">' +
    t('50 000 mẫu, hạt giống cố định. Con số sẽ *gần* nhưng không trùng khít Hình 7 (dùng $10^6$ mẫu).') +
    '</p></div><div class="lab-output">' +
    '<div class="stat-row">' +
    stat('clMse', 'MSE ở ngưỡng đang chọn') +
    stat('clBest', 'MSE nhỏ nhất', 'tại c*') +
    stat('clMinmax', 'MSE nếu dùng min–max') +
    '</div><div class="stat-row">' +
    stat('clCstar', 'Ngưỡng tối ưu c*') +
    stat('clMaxAbs', 'max |x| của mẫu') +
    stat('clRatio', 'min–max tệ hơn bao nhiêu lần') +
    '</div>' +
    '<p class="lab-note" id="clNote"></p>' +
    '<div class="chart" id="clChart"></div>' +
    '<p class="chart-cap">' +
    t('MSE theo ngưỡng cắt $c$. Chấm là vị trí đang chọn, vạch đứt là $c^*$, vạch chấm là ' +
      '$\\max|x|$ (tức dải min–max).') +
    '</p></div></div></section>';

  // ------------------------------------------------------ 3. soi bit số thực
  const lab3 =
    '<section class="lab" id="soi-bit" data-lab="float">' +
    '<header class="lab-head"><h3>3 · Soi bit một số dấu phẩy động</h3><p>' +
    t('Tách ba trường dấu – exponent – mantissa và xem giá trị thực sự được lưu. Công cụ cho ' +
      '[Mục 2.2](ch02.html#sec-2-2), [Mục 2.3](ch02.html#sec-2-3) và [Bài 2](bai-tap.html#bai-2).') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    '<label class="ctl"><span class="ctl-label">Giá trị</span>' +
    '<input type="text" id="flX" value="1993" inputmode="decimal" autocomplete="off"></label>' +
    select('flFmt', 'Định dạng', [
      ['fp32', 'FP32 — 8 exp / 23 man', true],
      ['fp16', 'FP16 — 5 exp / 10 man'],
      ['bf16', 'BF16 — 8 exp / 7 man'],
      ['e4m3', 'FP8 E4M3 (OCP) — 4 exp / 3 man'],
      ['e5m2', 'FP8 E5M2 — 5 exp / 2 man'],
    ]) +
    '<div class="chip-row">' +
    ['1993', '0.1', '2049', '-6.5', '70000', '448']
      .map((v) => '<button class="chip" type="button" data-float-preset="' + v + '">' + v + '</button>')
      .join('') +
    '</div></div><div class="lab-output">' +
    '<div class="bits" id="flBits"></div>' +
    '<div class="stat-row">' +
    stat('flStored', 'Giá trị lưu được') + stat('flErr', 'Sai số tuyệt đối') +
    stat('flRel', 'Sai số tương đối') +
    '</div><div class="stat-row">' +
    stat('flExp', 'Số mũ thật e') + stat('flSpacing', 'Khoảng cách lưới tại đây') +
    stat('flEps', 'ε máy của định dạng') +
    '</div>' +
    '<p class="lab-note" id="flNote"></p>' +
    '</div></div></section>';

  // ------------------------------------------------------- 4. requantization
  const lab4 =
    '<section class="lab" id="requantize" data-lab="requant">' +
    '<header class="lab-head"><h3>4 · Requantization bằng dấu chấm tĩnh</h3><p>' +
    t('Nhân với hệ số thực $M = S_w S_x / S_y$ mà không dùng một phép toán số thực nào. ' +
      'Cài đặt đúng theo [Mục 6.4](ch06.html#sec-6-4).') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    num('rqSw', 'S_w — scale trọng số', '0.001661', 'step="0.0001"') +
    num('rqSx', 'S_x — scale activation vào', '0.038963', 'step="0.001"') +
    num('rqSy', 'S_y — scale activation ra', '0.031995', 'step="0.001"') +
    num('rqAcc', 'Giá trị bộ cộng dồn (int32)', '66876', 'step="1"') +
    '<p class="ctl-note">' +
    t('Giá trị mặc định lấy đúng từ kết quả in ra ở [Mục 6.5](ch06.html#sec-6-5). Lưu ý các ' +
      'scale ở đó chỉ in 6 chữ số thập phân, nên $M$ dựng lại từ chúng lệch với con số ' +
      '$0{,}0020223740$ của sách ở vài chữ số cuối.') +
    '</p></div><div class="lab-output">' +
    '<div class="stat-row">' +
    stat('rqM', 'M = S_w·S_x / S_y') + stat('rqM0', 'M₀ dạng int32') + stat('rqShift', 'Tổng dịch phải') +
    '</div><div class="stat-row">' +
    stat('rqInt', 'Kết quả đường số nguyên') + stat('rqFloat', 'Kết quả dùng số thực') +
    stat('rqDiff', 'Chênh lệch') +
    '</div>' +
    '<div class="formula-box" id="rqFormula"></div>' +
    '<p class="lab-note" id="rqNote"></p>' +
    '</div></div></section>';

  // ----------------------------------------------------- 5. tính dung lượng
  const lab5 =
    '<section class="lab" id="dung-luong" data-lab="size">' +
    '<header class="lab-head"><h3>5 · Tính dung lượng thật</h3><p>' +
    t('Số bit thực tế mỗi trọng số ([Mục 11.8](ch11.html#sec-11-8)) và dung lượng KV cache ' +
      '([Mục 11.7](ch11.html#sec-11-7)) — hai phép tính quyết định mô hình có vừa bộ nhớ hay không. ' +
      'Giá trị mặc định là đề [Bài 9](bai-tap.html#bai-9).') +
    '</p></header>' +
    '<div class="lab-grid lab-grid-2"><div class="lab-panel">' +
    '<h4 class="lab-sub">Trọng số</h4>' +
    num('szParams', 'Số tham số (tỉ)', '13', 'step="0.1" min="0.001"') +
    select('szFmt', 'Định dạng', [
      ['fp16', 'FP16 / BF16 — 16 bit'],
      ['int8', 'INT8 nhóm 128, scale FP16'],
      ['q8_0', 'GGUF Q8_0 — khối 32'],
      ['int4', 'INT4 nhóm 128, scale FP16'],
      ['q4_0', 'GGUF Q4_0 — khối 32'],
      ['q4_k', 'GGUF Q4_K — siêu khối 256', true],
      ['nf4', 'NF4 + double quantization (QLoRA)'],
    ]) +
    '<div class="stat-row">' +
    stat('szBits', 'Bit thực tế / trọng số') + stat('szSize', 'Dung lượng') +
    '</div>' +
    '<p class="lab-note" id="szNote"></p>' +
    '</div><div class="lab-panel">' +
    '<h4 class="lab-sub">KV cache</h4>' +
    '<div class="ctl-pair">' +
    num('kvLayers', 'Số lớp', '40', 'step="1" min="1"') +
    num('kvHeads', 'Số KV head', '40', 'step="1" min="1"') +
    '</div><div class="ctl-pair">' +
    num('kvDim', 'd_head', '128', 'step="1" min="1"') +
    num('kvTokens', 'Số token', '8192', 'step="1" min="1"') +
    '</div><div class="ctl-pair">' +
    num('kvBatch', 'Batch', '4', 'step="1" min="1"') +
    select('kvDtype', 'Kiểu số', [
      ['2', 'FP16 / BF16 — 2 byte', true],
      ['1', 'FP8 / INT8 — 1 byte'],
      ['0.5', 'INT4 — 0,5 byte'],
    ]) +
    '</div>' +
    '<div class="stat-row">' +
    stat('kvPerToken', 'Mỗi token, mỗi chuỗi') + stat('kvTotal', 'Tổng KV cache') +
    '</div>' +
    '<p class="lab-note" id="kvNote"></p>' +
    '</div></div>' +
    '<div class="verdict" id="szVerdict"></div>' +
    '</section>';

  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Phòng thí nghiệm</span>' +
    '<span class="kicker-time">5 công cụ</span></div>' +
    '<h1>Phòng thí nghiệm</h1>' +
    '<p class="chapter-lede">Năm công cụ chạy hoàn toàn trong trình duyệt, cài đặt đúng theo công thức của giáo trình. Đổi tham số rồi quan sát là cách nhanh nhất để thấy vì sao mỗi lựa chọn lại quan trọng.</p>' +
    '<nav class="ex-jump">' +
    '<a href="#may-luong-tu">1 · Máy lượng tử</a>' +
    '<a href="#lam-tron-cat">2 · Làm tròn và cắt</a>' +
    '<a href="#soi-bit">3 · Soi bit</a>' +
    '<a href="#requantize">4 · Requantization</a>' +
    '<a href="#dung-luong">5 · Dung lượng</a>' +
    '</nav>' +
    lab1 + lab2 + lab3 + lab4 + lab5 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm — Quantization trong Deep Learning',
      description:
        'Năm công cụ tương tác: máy lượng tử affine, đánh đổi làm tròn – cắt, soi bit số thực, requantization dấu chấm tĩnh, tính dung lượng mô hình và KV cache.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/playground.js'],
    })
  );
}
