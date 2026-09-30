// Phòng thí nghiệm của giáo trình Học sâu.
import { range, num, check, select, stat, canvas, khoi, mo } from './lab-ui.mjs';

export function buildModelsLabPage(ctx) {
  const { md, nav, page, write, neighbours, registry, f, docTitle } = ctx;
  registry.currentFile = f('thuc-hanh.html');
  const t = (s) => md.renderInline(s);
  const K = khoi(t);

  // ------------------------------------------------------ 1. vì sao chia √d_k
  const lab1 = K('can-dk', 1,
    'Vì sao phải chia cho $\\sqrt{d_k}$',
    'Nếu các thành phần của $q$ và $k$ độc lập, trung bình 0, phương sai 1 thì $q \\cdot k$ có phương sai **đúng bằng** $d_k$. Tăng $d_k$ rồi xem softmax bão hoà ra sao, và bật phép chia để thấy entropy đứng yên. Lý thuyết và số đo ở Mục 9.2.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    range('atDk', 'd_k', 0, 4, 2, 1, 'atDkOut') +
    check('atScale', 'Chia cho √d_k', false) +
    stat('atVar', 'Phương sai điểm số đo được') +
    stat('atVarLt', 'Phương sai theo lý thuyết') +
    stat('atH', 'Entropy (nat)', 'tối đa ln 64 = 4,159') +
    stat('atMax', 'Trọng số lớn nhất') +
    '</div><div class="lab-output lab-output--2">' +
    canvas('atW', 240) + canvas('atE', 240) +
    '</div></div>', 'atNote');

  // ------------------------------------- 2. máy tính tham số / FLOP / bộ nhớ
  const lab2 = K('may-tinh', 2,
    'Máy tính tham số, FLOP và bộ nhớ',
    'Toàn bộ Chương 12 gói vào một biểu mẫu. Chọn sẵn một cấu hình hoặc tự gõ. Con số quan trọng nhất là ô cuối: tỉ lệ $T/(6d)$ quyết định khi nào chi phí bậc hai của attention mới thật sự chi phối.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    select('pcPreset', 'Cấu hình có sẵn', [
      ['12,768,50257,12,12,1024,0.3', 'GPT-2 small', true],
      ['36,1280,50257,20,20,1024,0.3', 'GPT-2 large'],
      ['32,4096,32000,32,32,4096,2', 'Llama-2 7B'],
      ['80,8192,32000,64,8,4096,2', 'Llama-2 70B (GQA 8 nhóm)'],
    ]) +
    num('pcL', 'Số lớp L', 12, 'min="1" step="1"') +
    num('pcD', 'Chiều mô hình d', 768, 'min="8" step="8"') +
    num('pcV', 'Từ vựng V', 50257, 'min="1" step="1"') +
    num('pcH', 'Số đầu query', 12, 'min="1" step="1"') +
    num('pcKV', 'Số đầu KV', 12, 'min="1" step="1"') +
    num('pcT', 'Độ dài ngữ cảnh T', 1024, 'min="1" step="1"') +
    num('pcB', 'Cỡ lô khi suy luận', 8, 'min="1" step="1"') +
    num('pcTok', 'Token huấn luyện (tỉ)', 0.3, 'min="0.001" step="0.1"') +
    '</div><div class="lab-output lab-stats">' +
    stat('pcPhiEmb', 'Tham số phi-embedding') +
    stat('pcCheck', 'Đối chiếu 12·L·d²', 'phải trùng ô trên') +
    stat('pcTong', 'Tổng tham số') +
    stat('pcEmbPct', 'Embedding chiếm') +
    stat('pcFfnPct', 'FFN chiếm (trong khối)', 'lý thuyết 2/3') +
    stat('pcFlop', 'C ≈ 6ND (FLOP)') +
    stat('pcGio', 'Giờ-GPU ở 40% MFU', 'A100 bf16') +
    stat('pcKV', 'KV cache (FP16)') +
    stat('pcKVTok', 'KV mỗi token mỗi chuỗi') +
    stat('pcTiLe', 'Tỉ lệ T/(6d)', '> 1 thì attention chi phối') +
    stat('pcNguong', 'Ngưỡng 6d (token)') +
    '</div></div>', 'pcNote');

  // -------------------------------------------- 3. gradient qua nhiều lớp
  const lab3 = K('gradient-nhieu-lop', 3,
    'Gradient qua nhiều lớp: ba cách chữa, và chúng cộng dồn',
    'Đường đậm là cấu hình bạn đang chọn; các đường mờ là năm cấu hình ở Mục 6.3 để so. Hãy bật **kết nối tắt mà không bật chuẩn hoá** — đó là chỗ hay bị nói sai nhất, và bạn sẽ thấy nó làm gradient **bùng nổ** chứ không phải ổn định.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('ggGain', 'Hệ số khởi tạo', 30, 250, 141, 1, 'ggGainOut') +
    range('ggDepth', 'Số lớp', 5, 80, 40, 1, 'ggDepthOut') +
    check('ggNorm', 'Chuẩn hoá lớp', false) +
    check('ggRes', 'Kết nối tắt', false) +
    stat('ggTiLe', 'Tỉ lệ lớp đầu / lớp cuối') +
    stat('ggKl', 'Kết luận') +
    '</div><div class="lab-output">' + canvas('ggPlot', 300) + '</div></div>', 'ggNote');

  const body =
    '<article class="prose lab-page" data-lab-models>' +
    mo('Học sâu · thực hành',
      'Ba công cụ cho ba chỗ mà một bảng số không đủ: phép chia $\\sqrt{d_k}$, ' +
      'phép đếm tài nguyên của một Transformer, và chuyện gì xảy ra với gradient qua nhiều lớp.') +
    lab1 + lab2 + lab3 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm — ' + docTitle,
      description:
        'Ba công cụ tương tác: softmax bão hoà theo d_k và tác dụng của phép chia căn d_k, ' +
        'máy tính tham số / FLOP / KV cache cho Transformer, và độ lớn gradient qua nhiều lớp ' +
        'theo khởi tạo, chuẩn hoá và kết nối tắt.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/viz.js', 'assets/lab-models.js'],
    })
  );
}
