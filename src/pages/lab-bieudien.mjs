// Phòng thí nghiệm của giáo trình Biểu diễn, Sinh và Căn chỉnh.
import { range, num, check, stat, canvas, khoi, mo } from './lab-ui.mjs';

export function buildBieudienLabPage(ctx) {
  const { md, nav, page, write, neighbours, registry, f, docTitle } = ctx;
  registry.currentFile = f('thuc-hanh.html');
  const t = (s) => md.renderInline(s);
  const K = khoi(t);

  // ---------------------------------------------------------- 1. RLHF beta
  const lab1 = K('rlhf-beta', 1,
    'RLHF: $\\beta$ là vị trí trên một đường đánh đổi',
    'Nghiệm tối ưu có dạng đóng $\\pi^{*} \\propto \\pi_{\\text{ref}}\\,e^{r/\\beta}$ — tức **chính sách tham chiếu đánh trọng số lại theo hàm mũ của phần thưởng**. Trượt $\\beta$ và xem hai việc cùng lúc: bên trái là phân phối bị kéo đi thế nào, bên phải là vị trí hiện tại trên mặt đánh đổi. Lý thuyết ở Mục 14.4.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    range('rlBeta', 'β', -30, 24, 0, 1, 'rlBetaOut') +
    stat('rlKL', 'KL(π* ‖ π_ref)') +
    stat('rlEr', 'E[r] đạt được') +
    stat('rlMax', 'Xác suất lớn nhất của π*') +
    '</div><div class="lab-output lab-output--2">' +
    canvas('rlBar', 250) + canvas('rlTrade', 250) +
    '</div></div>', 'rlNote');

  // --------------------------------------------------- 2. lịch nhiễu khuếch tán
  const lab2 = K('lich-nhieu', 2,
    'Lịch nhiễu: vì sao $t$ nhỏ học chi tiết còn $t$ lớn học bố cục',
    'Dạng đóng $q(x_t \\mid x_0) = \\mathcal{N}(\\sqrt{\\bar\\alpha_t}\\,x_0,\\ (1-\\bar\\alpha_t)I)$ cho phép nhảy thẳng tới bước $t$ bất kỳ. Trượt $t$ và xem đồng thời hai thứ: tín hiệu còn lại theo lịch (trái) và một "ảnh" một chiều đang tan dần (phải). Số liệu ở Mục 11.3.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    range('dfT', 'Bước t', 0, 999, 200, 1, 'dfTOut') +
    stat('dfAn', 'ᾱ_t') +
    stat('dfSnr', 'SNR = ᾱ/(1−ᾱ)') +
    stat('dfBd', 'Biên độ tín hiệu còn lại') +
    '</div><div class="lab-output lab-output--2">' +
    canvas('dfSig', 250) + canvas('dfX', 250) +
    '</div></div>', 'dfNote');

  // ------------------------------------------------------- 3. bộ nhớ LoRA
  const lab3 = K('lora-bo-nho', 3,
    'LoRA: cái nó thật sự tiết kiệm',
    'LoRA **không** làm mô hình chạy nhanh hơn — khi suy luận thì gộp $BA$ vào $W_0$ nên tốc độ bằng bản gốc. Thứ nó đổi là **bộ nhớ lúc huấn luyện**. Gõ cấu hình vào và so hai cột cuối. Lý thuyết và số đo ở Mục 7.3.',
    '<div class="lab-grid"><div class="lab-controls">' +
    num('loD', 'Chiều mô hình d', 4096, 'min="8" step="8"') +
    num('loL', 'Số lớp L', 32, 'min="1" step="1"') +
    num('loR', 'Hạng r của LoRA', 8, 'min="1" step="1"') +
    num('loN', 'Tham số phi-embedding (tỉ)', 6.476, 'min="0.01" step="0.1"') +
    check('loQV', 'Chỉ gắn vào W_Q và W_V', true) +
    '</div><div class="lab-output lab-stats">' +
    stat('loTiLeLop', 'Tỉ lệ 2r/d của một lớp') +
    stat('loTong', 'Tham số huấn luyện') +
    stat('loPct', 'Bằng bao nhiêu % mô hình') +
    stat('loAdam', 'Trạng thái Adam · LoRA') +
    stat('loAdamFull', 'Trạng thái Adam · toàn phần') +
    stat('loLan', 'Chênh nhau') +
    stat('loFull', 'Tổng bộ nhớ khi tinh chỉnh toàn phần', 'trọng số + gradient + Adam') +
    stat('loVua', 'Một GPU 80 GB') +
    '</div></div>', 'loNote');

  const body =
    '<article class="prose lab-page" data-lab-bieudien>' +
    mo('Biểu diễn &amp; Căn chỉnh · thực hành',
      'Ba công cụ cho ba chỗ mà công thức nhìn thì hiểu nhưng chưa thấy: ' +
      'phép đánh trọng số lại của RLHF, lịch nhiễu của mô hình khuếch tán, ' +
      'và con số bộ nhớ đứng sau LoRA.') +
    lab1 + lab2 + lab3 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm — ' + docTitle,
      description:
        'Ba công cụ tương tác: nghiệm RLHF có ràng buộc KL và mặt đánh đổi của nó, ' +
        'lịch nhiễu cùng SNR của mô hình khuếch tán, và phép đếm bộ nhớ huấn luyện của LoRA.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/viz.js', 'assets/lab-bieudien.js'],
    })
  );
}
