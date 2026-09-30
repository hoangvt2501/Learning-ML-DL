// Phòng thí nghiệm của giáo trình Nền tảng: sáu công cụ tương tác.
//
// Lối trình bày học theo ba trang được nhắc ở phần tài liệu: chữ kẹp giữa các
// hình tương tác (setosa.io), đổi một tham số rồi xem kết quả đổi theo ngay
// (distill.pub), và mỗi công cụ nối thẳng về mục đã trình bày lý thuyết.

const range = (id, label, min, max, value, step, out) =>
  '<label class="ctl ctl-range"><span class="ctl-label">' + label +
  '<b class="ctl-out" id="' + out + '">–</b></span>' +
  '<input type="range" id="' + id + '" min="' + min + '" max="' + max +
  '" value="' + value + '" step="' + (step || 1) + '"></label>';

const check = (id, label, checked) =>
  '<label class="ctl ctl-check"><input type="checkbox" id="' + id + '"' +
  (checked ? ' checked' : '') + '><span>' + label + '</span></label>';

const stat = (id, label, hint = '') =>
  '<div class="stat"><span class="stat-label">' + label + '</span>' +
  '<span class="stat-value" id="' + id + '">–</span>' +
  (hint ? '<span class="stat-hint">' + hint + '</span>' : '') + '</div>';

const btn = (id, label) =>
  '<button class="lab-btn" id="' + id + '" type="button">' + label + '</button>';

const canvas = (id, h) =>
  '<canvas class="viz" id="' + id + '" style="height:' + h + 'px"></canvas>';

export function buildNentangLabPage(ctx) {
  const { md, nav, page, write, neighbours, registry, f, docTitle } = ctx;
  registry.currentFile = f('thuc-hanh.html');
  const t = (s) => md.renderInline(s);

  const khoi = (id, so, ten, dan, than, ghi) =>
    '<section class="lab" id="' + id + '">' +
    '<header class="lab-head"><h3>' + so + ' · ' + t(ten) + '</h3><p>' + t(dan) + '</p></header>' +
    than +
    '<p class="lab-note" id="' + ghi + '"></p>' +
    '</section>';

  // ------------------------------------------------ 1. độ chệch – phương sai
  const lab1 = khoi('thien-lech-phuong-sai', 1,
    'Độ chệch và phương sai',
    'Kéo thanh trượt để đổi bậc đa thức. Bên trái là 14 đường khớp trên 14 tập huấn luyện khác nhau, đúng các tập dữ liệu mà định nghĩa độ chệch và phương sai ở Mục 3.6 lấy kỳ vọng trên đó. Bên phải là sai số tách thành bình phương độ chệch, phương sai và nhiễu. Phép đo trên 400 tập huấn luyện nằm ở [Chương 2 của *Học sâu*](models-ch02.html).',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    range('bvDeg', 'Bậc đa thức', 1, 14, 5, 1, 'bvDegOut') +
    stat('bvB2', 'Độ chệch²') + stat('bvVar', 'Phương sai') + stat('bvTot', 'Tổng MSE') +
    '</div><div class="lab-output lab-output--2">' +
    canvas('bvFit', 230) + canvas('bvBar', 230) +
    '</div></div>', 'bvNote');

  // ------------------------------------------------------- 2. ridge và lasso
  const lab2 = khoi('ridge-lasso', 2,
    'Ridge và lasso: số hệ số bằng 0',
    'Mười hai hệ số, trong đó mô hình sinh dữ liệu chỉ có ba hệ số khác 0. Trượt $\\lambda$ và đếm số hệ số mà mỗi phương pháp đưa về đúng 0. Lý thuyết ở Mục 9.4.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('rlLam', 'λ', 0, 60, 30, 1, 'rlLamOut') +
    stat('rlZR', 'Hệ số = 0 · ridge') +
    stat('rlZL', 'Hệ số = 0 · lasso', 'mô hình thật có 9') +
    stat('rlNR', '‖w‖₂ · ridge') + stat('rlNL', '‖w‖₂ · lasso') +
    '</div><div class="lab-output">' + canvas('rlPath', 270) + '</div></div>', 'rlNote');

  // ----------------------------------------------------- 3. gradient descent
  const lab3 = khoi('gradient-descent', 3,
    'Gradient descent và số điều kiện',
    'Hàm $f(x) = \\tfrac12(x_0^2 + \\kappa x_1^2)$. Trượt $\\kappa$ để xem số vòng lặp thay đổi. Đặt tốc độ học vượt ngưỡng $2/\\lambda_{\\max}$ để thấy thuật toán phân kỳ. Bật momentum để thấy số vòng lặp giảm từ bậc $\\kappa$ xuống bậc $\\sqrt\\kappa$ (Mục 5.3).',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('gdKap', 'Số điều kiện κ', 0, 40, 20, 1, 'gdKapOut') +
    range('gdEta', 'Tốc độ học (% của ngưỡng 2/λ_max)', 5, 130, 60, 5, 'gdEtaOut') +
    check('gdMom', 'Dùng momentum (heavy ball)', false) +
    stat('gdIter', 'Số vòng tới sai số 10⁻⁸') +
    stat('gdEta2', 'η đang dùng') +
    stat('gdOpt', 'η tối ưu về lý thuyết', '2/(L+m)') +
    '</div><div class="lab-output">' + canvas('gdPath', 270) + '</div></div>', 'gdNote');

  // ------------------------------------------------------------ 4. SVM lề mềm
  const lab4 = khoi('svm-le-mem', 4,
    'SVM lề mềm: ba nhóm điểm theo $C$',
    'Theo điều kiện KKT (Mục 13.4), mỗi điểm thuộc một trong ba nhóm: nằm ngoài lề (không ảnh hưởng tới nghiệm), nằm đúng trên lề, hoặc vi phạm lề. Trượt $C$ để xem số điểm trong mỗi nhóm thay đổi. Điểm vẽ nhỏ là điểm nằm ngoài lề.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('svmC', 'C', -25, 25, 0, 1, 'svmCOut') +
    stat('svmLe', 'Lề 2/‖w‖') +
    stat('svmNgoai', 'Ngoài lề · α = 0') +
    stat('svmTren', 'Đúng trên lề · 0 < α < C') +
    stat('svmViPham', 'Vi phạm lề · α = C') +
    stat('svmSai', 'Sai số huấn luyện') +
    '</div><div class="lab-output">' + canvas('svmPlot', 300) + '</div></div>', 'svmNote');

  // ------------------------------------------------------- 5. PCA so với LDA
  const lab5 = khoi('pca-lda', 5,
    'PCA và LDA: kéo thả các điểm',
    'Hai lớp được tô hai màu. Kéo một điểm bất kỳ để xem hai hướng được tính lại. Thử kéo để dữ liệu trải rộng theo phương ngang trong khi hai lớp vẫn tách nhau theo phương dọc: thành phần chính thứ nhất sẽ nằm ngang và AUC của nó giảm về gần 0,5, như ở Mục 14.4.',
    '<div class="lab-grid"><div class="lab-controls">' +
    btn('pcaReset', 'Đặt lại vị trí điểm') +
    stat('pcaVar', 'Phương sai giữ lại bởi PC₁') +
    stat('pcaAucPC', 'AUC khi chiếu lên PC₁') +
    stat('pcaAucLD', 'AUC khi chiếu lên LDA') +
    '</div><div class="lab-output">' + canvas('pcaPlot', 320) + '</div></div>', 'pcaNote');

  // ------------------------------------------------------------- 6. K-means
  const lab6 = khoi('k-means', 6,
    'K-means: chạy từng bước',
    'Bấm "Một bước" để chạy một vòng lặp: gán mỗi điểm cho tâm gần nhất, rồi dời mỗi tâm về trung bình cụm. Bấm "Khởi tạo dồn một chỗ" vài lần để thấy thuật toán dừng ở các nghiệm tồi khác nhau, hiện tượng đo được ở Mục 15.2.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('kmK', 'Số cụm k', 2, 8, 4, 1, 'kmKOut') +
    '<div class="lab-btns">' + btn('kmStep', 'Một bước') + btn('kmRun', 'Chạy tới hội tụ') + '</div>' +
    '<div class="lab-btns">' + btn('kmNew', 'Khởi tạo ngẫu nhiên') + btn('kmBad', 'Khởi tạo dồn một chỗ') + '</div>' +
    stat('kmBuoc', 'Số bước đã chạy') +
    stat('kmQT', 'Inertia', 'luôn giảm theo k') +
    '</div><div class="lab-output">' + canvas('kmPlot', 320) + '</div></div>', 'kmNote');

  const body =
    '<article class="prose lab-page" data-lab-nentang>' +
    '<header class="hero hero--tight">' +
    '<p class="hero-kicker">Nền tảng · thực hành</p>' +
    '<h1>Phòng thí nghiệm</h1>' +
    '<p class="hero-lede">Sáu công cụ tương tác, mỗi công cụ gắn với một mục trong giáo trình và cài đúng công thức ở mục đó. Thay đổi tham số và quan sát kết quả thay đổi theo.</p>' +
    '<p class="hero-lede">Các công cụ chạy trong trình duyệt nên dùng ít điểm dữ liệu và ít vòng lặp hơn thí nghiệm ngoại tuyến. Con số hiển thị gần với bảng trong giáo trình nhưng không trùng khít, trừ những chỗ có ghi rõ, như số hệ số bằng 0 của lasso.</p>' +
    '</header>' +
    lab1 + lab2 + lab3 + lab4 + lab5 + lab6 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm — ' + docTitle,
      description:
        'Sáu công cụ tương tác: độ chệch và phương sai, ridge và lasso, ' +
        'gradient descent theo số điều kiện, ba nhóm điểm của SVM lề mềm, ' +
        'PCA và LDA có kéo thả điểm, K-means chạy từng bước.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/viz.js', 'assets/lab-nentang.js'],
    })
  );
}
