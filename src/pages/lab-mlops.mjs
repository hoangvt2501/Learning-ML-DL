// Phòng thí nghiệm của giáo trình MLOps: bốn công cụ tính toán dùng được trong việc thật.

const num = (id, label, value, attrs = '') =>
  '<label class="ctl"><span class="ctl-label">' + label + '</span>' +
  '<input type="number" id="' + id + '" value="' + value + '" ' + attrs + '></label>';

const select = (id, label, options) =>
  '<label class="ctl"><span class="ctl-label">' + label + '</span><select id="' + id + '">' +
  options.map((o) => '<option value="' + o[0] + '"' + (o[2] ? ' selected' : '') + '>' + o[1] + '</option>').join('') +
  '</select></label>';

const stat = (id, label, hint = '') =>
  '<div class="stat"><span class="stat-label">' + label + '</span>' +
  '<span class="stat-value" id="' + id + '">–</span>' +
  (hint ? '<span class="stat-hint">' + hint + '</span>' : '') + '</div>';

const TESTS = {
  'Dữ liệu': [
    'Kỳ vọng về đặc trưng được ghi thành schema',
    'Mọi đặc trưng đều có ích',
    'Không đặc trưng nào quá đắt so với lợi ích',
    'Đặc trưng tuân thủ yêu cầu chính sách',
    'Pipeline có kiểm soát quyền riêng tư',
    'Thêm đặc trưng mới được nhanh',
    'Toàn bộ mã tính đặc trưng được kiểm thử',
  ],
  'Mô hình': [
    'Đặc tả mô hình qua rà soát mã và quản phiên bản',
    'Chỉ số đại diện tương quan với tác động thật',
    'Mọi siêu tham số đã được tinh chỉnh',
    'Đã biết ảnh hưởng của độ cũ của mô hình',
    'Một mô hình đơn giản hơn không tốt hơn',
    'Chất lượng đủ tốt trên mọi lát cắt quan trọng',
    'Đã kiểm tra tính bao hàm / công bằng',
  ],
  'Hạ tầng': [
    'Huấn luyện lặp lại được',
    'Mã đặc tả mô hình có kiểm thử đơn vị',
    'Toàn bộ pipeline có kiểm thử tích hợp',
    'Chất lượng được kiểm định trước khi phục vụ',
    'Gỡ lỗi được bằng cách quan sát từng bước',
    'Có quy trình canary trước khi ra toàn bộ',
    'Quay lui được nhanh và an toàn',
  ],
  'Giám sát': [
    'Thay đổi ở phụ thuộc thượng nguồn sinh thông báo',
    'Bất biến dữ liệu giữ đúng ở cả hai phía',
    'Đặc trưng lúc huấn luyện và phục vụ tính ra cùng giá trị',
    'Mô hình không quá cũ',
    'Mô hình ổn định về mặt số học',
    'Không bị tụt chất lượng đột ngột',
    'Không bị tụt chất lượng do hồi quy',
  ],
};

export function buildMlopsLabPage(ctx) {
  const { md, nav, page, write, neighbours, registry, f } = ctx;
  registry.currentFile = f('thuc-hanh.html');
  const t = (s) => md.renderInline(s);

  // ------------------------------------------------------ 1. cỡ mẫu A/B
  const lab1 =
    '<section class="lab" id="co-mau" data-lab="ab">' +
    '<header class="lab-head"><h3>1 · Máy tính cỡ mẫu A/B</h3><p>' +
    t('Trả lời câu "thí nghiệm này phải chạy bao lâu". Cỡ mẫu tỉ lệ nghịch với **bình phương** ' +
      'mức cải thiện, nên đây thường là ràng buộc quyết định cả quy trình ra mắt ' +
      '([Mục 8.4](mlops-ch08.html#sec-8-4)).') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    num('abBase', 'Tỉ lệ nền (%)', '5', 'step="0.1" min="0.01" max="99"') +
    num('abLift', 'Cải thiện tương đối cần bắt (%)', '2', 'step="0.1" min="0.05"') +
    select('abPower', 'Lực kiểm định', [['0.8', '80%', true], ['0.9', '90%'], ['0.95', '95%']]) +
    select('abAlpha', 'Mức ý nghĩa α (hai phía)', [['0.05', '0,05', true], ['0.01', '0,01']]) +
    num('abTraffic', 'Lưu lượng mỗi nhánh mỗi ngày', '100000', 'step="1000" min="1"') +
    num('abCuped', 'Giảm phương sai nhờ CUPED (%)', '0', 'step="5" min="0" max="80"') +
    '</div><div class="lab-output">' +
    '<div class="stat-row">' +
    stat('abN', 'Cỡ mẫu mỗi nhánh') + stat('abDays', 'Số ngày cần chạy') +
    stat('abAbs', 'Chênh lệch tuyệt đối') +
    '</div><div class="stat-row">' +
    stat('abTotal', 'Tổng mẫu cả thí nghiệm') + stat('abWeeks', 'Làm tròn lên tuần trọn') +
    stat('abSaved', 'CUPED tiết kiệm được') +
    '</div>' +
    '<p class="lab-note" id="abNote"></p>' +
    '<div class="chart" id="abChart"></div>' +
    '<p class="chart-cap">' +
    t('Số ngày cần chạy theo mức cải thiện cần bắt. Chấm là cấu hình đang chọn.') +
    '</p></div></div></section>';

  // -------------------------------------------------- 2. ngưỡng PSI
  const lab2 =
    '<section class="lab" id="nguong-psi" data-lab="psi">' +
    '<header class="lab-head"><h3>2 · Hiệu chuẩn ngưỡng PSI</h3><p>' +
    t('Ngưỡng 0,25 là quy tắc kinh nghiệm không tính tới cỡ mẫu và số bin. Công cụ này tính ' +
      'ngưỡng **đúng** cho cấu hình của bạn, dựa trên kết quả $\\tfrac{n}{2}\\mathrm{PSI} \\sim \\chi^2_{k-1}$ ' +
      'ở [Mục 9.5](mlops-ch09.html#sec-9-5).') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    num('psiN', 'Cỡ mẫu n mỗi phía', '500', 'step="50" min="20"') +
    num('psiK', 'Số bin k', '10', 'step="1" min="2" max="100"') +
    num('psiFeat', 'Số đặc trưng theo dõi', '200', 'step="10" min="1"') +
    num('psiPeriod', 'Số lần đo mỗi ngày', '1', 'step="1" min="1"') +
    num('psiAlarm', 'Báo động giả chấp nhận được mỗi ngày', '0.1', 'step="0.05" min="0.001"') +
    '</div><div class="lab-output">' +
    '<div class="stat-row">' +
    stat('psiMean', 'PSI kỳ vọng khi KHÔNG dịch chuyển') +
    stat('psiP99', 'Phân vị 99% của PSI') +
    stat('psiThresh', 'Ngưỡng nên dùng', 'cho mức báo động giả đã chọn') +
    '</div><div class="stat-row">' +
    stat('psiFa25', 'Báo động giả/ngày nếu dùng 0,25') +
    stat('psiFa10', 'Báo động giả/ngày nếu dùng 0,10') +
    stat('psiNeed', 'n để PSI kỳ vọng < 0,01') +
    '</div>' +
    '<p class="lab-note" id="psiNote"></p>' +
    '<div class="chart" id="psiChart"></div>' +
    '<p class="chart-cap">' +
    t('Phân bố của PSI khi hai mẫu **cùng** một phân phối. Vạch đứt là ngưỡng đề nghị, ' +
      'vạch chấm là quy tắc 0,25.') +
    '</p></div></div></section>';

  // ------------------------------------------------ 3. đuôi độ trễ
  const lab3 =
    '<section class="lab" id="duoi-do-tre" data-lab="tail">' +
    '<header class="lab-head"><h3>3 · Đuôi độ trễ khi toả nhánh</h3><p>' +
    t('Một yêu cầu gọi $k$ nhánh song song và đợi đủ cả $k$. Công cụ tính p99 của tổng thể và ' +
      'phân vị mà **mỗi** nhánh phải đạt ([Mục 7.3](mlops-ch07.html#sec-7-3)).') +
    '</p></header>' +
    '<div class="lab-grid"><div class="lab-controls">' +
    num('tlP50', 'Trung vị một nhánh (ms)', '20', 'step="1" min="0.1"') +
    num('tlP99', 'p99 một nhánh (ms)', '100', 'step="5" min="0.2"') +
    num('tlK', 'Số nhánh gọi song song', '25', 'step="1" min="1" max="500"') +
    num('tlBudget', 'Ngân sách p99 tổng thể (ms)', '150', 'step="10" min="1"') +
    '</div><div class="lab-output">' +
    '<div class="stat-row">' +
    stat('tlP99max', 'p99 của thời gian trả lời') +
    stat('tlSlow', '% yêu cầu chạm nhánh chậm') +
    stat('tlNeed', 'Phân vị mỗi nhánh phải đạt') +
    '</div><div class="stat-row">' +
    stat('tlOne', 'Chỉ 1 trong bao nhiêu được chậm') +
    stat('tlKmax', 'Số nhánh tối đa trong ngân sách') +
    stat('tlMedian', 'Trung vị của thời gian trả lời') +
    '</div>' +
    '<p class="lab-note" id="tlNote"></p>' +
    '<div class="chart" id="tlChart"></div>' +
    '<p class="chart-cap">' +
    t('p99 của thời gian trả lời theo số nhánh. Vạch ngang là ngân sách, chấm là cấu hình đang chọn.') +
    '</p></div></div></section>';

  // --------------------------------------------- 4. ML Test Score
  const groups = Object.entries(TESTS)
    .map(([name, items], gi) =>
      '<div class="mts-group" data-mts-group="' + gi + '">' +
      '<div class="mts-head"><h4>' + name + '</h4>' +
      '<span class="mts-score" data-mts-score="' + gi + '">0</span></div>' +
      '<ul class="mts-list">' +
      items.map((it, i) =>
        '<li class="mts-item"><span class="mts-text">' + it + '</span>' +
        '<span class="mts-opts">' +
        ['0', '0.5', '1'].map((v, j) =>
          '<button type="button" class="mts-opt' + (j === 0 ? ' is-on' : '') +
          '" data-g="' + gi + '" data-i="' + i + '" data-v="' + v + '">' +
          ['không', 'thủ công', 'tự động'][j] + '</button>').join('') +
        '</span></li>').join('') +
      '</ul></div>')
    .join('');

  const lab4 =
    '<section class="lab" id="ml-test-score" data-lab="mts">' +
    '<header class="lab-head"><h3>4 · Tự chấm ML Test Score</h3><p>' +
    t('Chấm 28 mục của [Mục 6.5](mlops-ch06.html#sec-6-5): 0,5 điểm nếu làm thủ công có ghi chép, ' +
      '1 điểm nếu có hệ thống chạy tự động định kỳ. Điểm cuối là **giá trị nhỏ nhất** trong bốn nhóm.') +
    '</p></header>' +
    '<div class="mts-wrap">' +
    '<div class="stat-row mts-total">' +
    stat('mtsScore', 'ML Test Score', 'min của bốn nhóm') +
    stat('mtsWeak', 'Nhóm yếu nhất') +
    stat('mtsSum', 'Tổng điểm cả 28 mục', 'chỉ để tham khảo') +
    '</div>' +
    '<div class="verdict" id="mtsVerdict"></div>' +
    '<div class="mts-grid">' + groups + '</div>' +
    '<div class="drill-actions"><button class="btn btn-ghost" type="button" id="mtsReset">Đặt lại</button></div>' +
    '</div></section>';

  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Phòng thí nghiệm</span>' +
    '<span class="kicker-time">4 công cụ</span></div>' +
    '<h1>Phòng thí nghiệm MLOps</h1>' +
    '<p class="chapter-lede">Bốn công cụ trả lời bốn câu hỏi hay phải trả lời trong việc thật, và cũng hay bị hỏi trong phỏng vấn. Mọi công thức đều là công thức đã dùng trong giáo trình, không có hằng số bịa.</p>' +
    '<nav class="ex-jump">' +
    '<a href="#co-mau">1 · Cỡ mẫu A/B</a>' +
    '<a href="#nguong-psi">2 · Ngưỡng PSI</a>' +
    '<a href="#duoi-do-tre">3 · Đuôi độ trễ</a>' +
    '<a href="#ml-test-score">4 · ML Test Score</a>' +
    '</nav>' +
    lab1 + lab2 + lab3 + lab4 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm MLOps — ' + ctx.docTitle,
      description:
        'Bốn công cụ tương tác: tính cỡ mẫu A/B có CUPED, hiệu chuẩn ngưỡng PSI theo cỡ mẫu và số bin, phân tích đuôi độ trễ khi toả nhánh, và tự chấm ML Test Score.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/lab-mlops.js'],
    })
  );
}
