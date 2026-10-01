// Phòng thí nghiệm của giáo trình Ứng dụng LLM: bốn công cụ tính toán.
import { range, num, check, select, stat, canvas, khoi, mo } from './lab-ui.mjs';

export function buildUngdungLabPage(ctx) {
  const { md, nav, page, write, neighbours, registry, f, docTitle } = ctx;
  registry.currentFile = f('thuc-hanh.html');
  const t = (s) => md.renderInline(s);
  const K = khoi(t);

  // ------------------------------------------------------ 1. chi phí theo token
  const lab1 = K('chi-phi', 1,
    'Chi phí của một yêu cầu',
    'Tách chi phí của một yêu cầu thành từng phần của ngữ cảnh, theo cách tính ở Mục 2.5. ' +
    'Giá trị mặc định là ví dụ trong bài: một yêu cầu RAG với đơn giá giả định 1 đơn vị tiền cho mỗi triệu token đầu vào ' +
    'và 4 đơn vị cho mỗi triệu token đầu ra. Đơn giá thật thay đổi theo nhà cung cấp và theo thời gian.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    num('cpHeThong', 'Token chỉ dẫn hệ thống', 600, 'min="0" step="10"') +
    num('cpSoDoan', 'Số đoạn truy xuất', 5, 'min="0" step="1"') +
    num('cpDoan', 'Token mỗi đoạn', 350, 'min="0" step="10"') +
    num('cpLichSu', 'Token lịch sử hội thoại', 800, 'min="0" step="10"') +
    num('cpCauHoi', 'Token câu hỏi', 60, 'min="0" step="5"') +
    num('cpRa', 'Token đầu ra', 350, 'min="0" step="10"') +
    num('cpGiaVao', 'Đơn giá đầu vào (mỗi triệu token)', 1, 'min="0" step="0.1"') +
    num('cpGiaRa', 'Đơn giá đầu ra (mỗi triệu token)', 4, 'min="0" step="0.1"') +
    check('cpCache', 'Chỉ dẫn hệ thống đọc từ bộ đệm prompt', false) +
    num('cpGiaCache', 'Giá phần đọc từ bộ đệm (% giá thường)', 10, 'min="0" max="100" step="1"') +
    num('cpSoYeuCau', 'Số yêu cầu mỗi tháng', 660000, 'min="0" step="1000"') +
    '</div><div class="lab-output">' +
    '<div class="lab-stats">' +
    stat('cpTongVao', 'Tổng token đầu vào') +
    stat('cpMoiYC', 'Chi phí mỗi yêu cầu') +
    stat('cpTiLeRa', 'Phần chi phí do đầu ra') +
    stat('cpTiLeHoi', 'Câu hỏi chiếm trong đầu vào') +
    stat('cpTietKiem', 'Bộ đệm prompt giảm được', 'so với không dùng bộ đệm') +
    stat('cpThang', 'Chi phí mỗi tháng') +
    '</div>' +
    canvas('cpPlot', 170) +
    '</div></div>', 'cpNote');

  // ------------------------------------------------------ 2. KV cache
  const lab2 = K('kv-cache', 2,
    'Bộ nhớ KV cache',
    'Mỗi token trong ngữ cảnh cần lưu key và value ở mọi lớp: $2 \\times L \\times n_{kv} \\times d_{head} \\times$ số byte mỗi giá trị (Mục 2.2). ' +
    'Chọn một cấu hình công bố hoặc tự nhập, rồi so dung lượng KV cache với dung lượng trọng số của mô hình.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    select('kvPreset', 'Cấu hình có sẵn', [
      ['32,32,8,128,8.03', 'Llama 3 8B', true],
      ['80,64,8,128,70.6', 'Llama 3 70B'],
      ['28,28,4,128,7.62', 'Qwen2.5-7B'],
      ['24,14,2,64,0.494', 'Qwen2.5-0.5B'],
    ]) +
    num('kvL', 'Số lớp L', 32, 'min="1" step="1"') +
    num('kvQ', 'Số đầu query', 32, 'min="1" step="1"') +
    num('kvKV', 'Số đầu key/value', 8, 'min="1" step="1"') +
    num('kvD', 'Số chiều mỗi đầu', 128, 'min="1" step="1"') +
    num('kvP', 'Số tham số (tỉ)', 8.03, 'min="0.01" step="0.01"') +
    select('kvByte', 'Kiểu số của KV cache', [
      ['2', 'FP16 / BF16 (2 byte)', true], ['1', 'FP8 / INT8 (1 byte)'], ['0.5', 'INT4 (0,5 byte)'],
    ]) +
    num('kvT', 'Độ dài ngữ cảnh (token)', 131072, 'min="1" step="1024"') +
    num('kvB', 'Số chuỗi phục vụ đồng thời', 1, 'min="1" step="1"') +
    '</div><div class="lab-output">' +
    '<div class="lab-stats">' +
    stat('kvMoiToken', 'KV cache mỗi token') +
    stat('kvMoiChuoi', 'KV cache một chuỗi') +
    stat('kvTong', 'KV cache tất cả các chuỗi') +
    stat('kvTrongSo', 'Trọng số (2 byte mỗi tham số)') +
    stat('kvMHA', 'Nếu không dùng GQA', 'số đầu KV bằng số đầu query') +
    stat('kvTiLe', 'KV cache / trọng số') +
    '</div>' +
    canvas('kvPlot', 220) +
    '</div></div>', 'kvNote');

  // ------------------------------------------------------ 3. độ tin cậy của agent
  const lab3 = K('agent', 3,
    'Độ tin cậy của agent nhiều bước',
    'Mỗi bước đúng với xác suất $p$, độc lập với các bước khác. Một bước kiểm tra phát hiện bước sai với xác suất $c$, ' +
    'và mỗi bước được thử lại tối đa $r$ lần khi bị phát hiện sai. Công thức ở Mục 9.7: ' +
    '$q = \\sum_{j=0}^{r} p\\,((1-p)c)^j$ và $P(\\text{hoàn thành}) = q^n$.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('agP', 'Xác suất đúng mỗi bước p', 500, 999, 950, 1, 'agPOut') +
    range('agN', 'Số bước n', 1, 100, 20, 1, 'agNOut') +
    range('agC', 'Tỉ lệ phát hiện lỗi c', 0, 100, 80, 1, 'agCOut') +
    range('agR', 'Số lần thử lại tối đa r', 0, 5, 1, 1, 'agROut') +
    stat('agKhong', 'Hoàn thành, không kiểm tra', 'pⁿ') +
    stat('agCo', 'Hoàn thành, có kiểm tra', 'qⁿ') +
    stat('agGoi', 'Lời gọi trung bình mỗi bước') +
    '</div><div class="lab-output">' + canvas('agPlot', 280) + '</div></div>', 'agNote');

  // ------------------------------------------------------ 4. cỡ bộ đánh giá
  const lab4 = K('bo-danh-gia', 4,
    'Cần bao nhiêu câu hỏi để so sánh hai phiên bản',
    'Phiên bản A đúng một tỉ lệ $p_A$ số câu. Phiên bản B giữ đúng một phần số câu A đúng và sửa được một phần số câu A sai (Mục 12.4). ' +
    'Công cụ tính khoảng tin cậy của độ chính xác và xác suất phát hiện B tốt hơn A ở mức ý nghĩa 5%, ' +
    'khi chấm hai phiên bản trên cùng bộ câu hỏi (kiểm định McNemar chính xác) và khi dùng hai bộ câu hỏi độc lập.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    num('dgA', 'Độ chính xác của A (%)', 80, 'min="1" max="99" step="1"') +
    num('dgGiu', 'B giữ đúng (% số câu A đúng)', 97, 'min="0" max="100" step="0.5"') +
    num('dgSua', 'B sửa được (% số câu A sai)', 27, 'min="0" max="100" step="0.5"') +
    num('dgN', 'Số câu hỏi n', 1000, 'min="10" max="5000" step="10"') +
    '</div><div class="lab-output">' +
    '<div class="lab-stats">' +
    stat('dgB', 'Độ chính xác của B') +
    stat('dgKTC', 'Nửa độ rộng KTC 95% của A', 'đơn vị: điểm phần trăm') +
    stat('dgLucCap', 'Xác suất phát hiện, ghép cặp', 'McNemar chính xác') +
    stat('dgLucRieng', 'Xác suất phát hiện, hai bộ độc lập', 'xấp xỉ chuẩn') +
    stat('dgNCap', 'Số câu cần cho 80%, ghép cặp', 'xấp xỉ chuẩn') +
    stat('dgNRieng', 'Số câu cần cho 80%, hai bộ độc lập', 'xấp xỉ chuẩn') +
    '</div>' +
    canvas('dgPlot', 220) +
    '</div></div>', 'dgNote');

  const body =
    '<article class="prose lab-page" data-lab-ungdung>' +
    mo('Ứng dụng LLM · thực hành',
      'Bốn công cụ tính toán đi kèm giáo trình: chi phí của một yêu cầu theo số token, bộ nhớ KV cache theo độ dài ngữ cảnh, ' +
      'độ tin cậy của một agent nhiều bước, và số câu hỏi cần để so sánh hai phiên bản của một ứng dụng.') +
    '<nav class="ex-jump">' +
    '<a href="#chi-phi">1 · Chi phí theo token</a>' +
    '<a href="#kv-cache">2 · KV cache</a>' +
    '<a href="#agent">3 · Agent nhiều bước</a>' +
    '<a href="#bo-danh-gia">4 · Cỡ bộ đánh giá</a>' +
    '</nav>' +
    lab1 + lab2 + lab3 + lab4 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm — ' + docTitle,
      description:
        'Bốn công cụ tương tác: tách chi phí của một yêu cầu LLM theo token, bộ nhớ KV cache theo độ dài ngữ cảnh, ' +
        'xác suất hoàn thành của agent nhiều bước có kiểm tra và thử lại, và số câu hỏi cần cho một bộ đánh giá.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/viz.js', 'assets/lab-ungdung.js'],
    })
  );
}
