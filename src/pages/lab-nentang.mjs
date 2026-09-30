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

  // ------------------------------------------------ 1. thiên lệch – phương sai
  const lab1 = khoi('thien-lech-phuong-sai', 1,
    'Thiên lệch và phương sai, tách ra bằng mắt',
    'Kéo thanh trượt để đổi bậc đa thức. Bên trái là 14 lần khớp trên 14 **tập huấn luyện khác nhau** — đó chính là thứ mà định nghĩa ở Mục 2.2 lấy kỳ vọng lên. Bên phải là phân rã sai số. Đối chiếu với bảng đo được ở Mục 2.3.',
    '<div class="lab-grid lab-grid--wide"><div class="lab-controls">' +
    range('bvDeg', 'Bậc đa thức', 1, 14, 5, 1, 'bvDegOut') +
    stat('bvB2', 'Thiên lệch²') + stat('bvVar', 'Phương sai') + stat('bvTot', 'Tổng MSE') +
    '</div><div class="lab-output lab-output--2">' +
    canvas('bvFit', 230) + canvas('bvBar', 230) +
    '</div></div>', 'bvNote');

  // ------------------------------------------------------- 2. ridge và lasso
  const lab2 = khoi('ridge-lasso', 2,
    'Ridge và lasso: vì sao chỉ một cái cho hệ số bằng 0',
    'Mười hai hệ số, nhưng mô hình sinh dữ liệu chỉ có **ba** hệ số khác 0. Trượt $\\lambda$ và đếm xem mỗi phương pháp đưa được bao nhiêu hệ số về **đúng** 0. Lý thuyết ở Mục 9.4.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('rlLam', 'λ', 0, 60, 30, 1, 'rlLamOut') +
    stat('rlZR', 'Hệ số = 0 · ridge') +
    stat('rlZL', 'Hệ số = 0 · lasso', 'mô hình thật có 9') +
    stat('rlNR', '‖w‖₂ · ridge') + stat('rlNL', '‖w‖₂ · lasso') +
    '</div><div class="lab-output">' + canvas('rlPath', 270) + '</div></div>', 'rlNote');

  // ----------------------------------------------------- 3. gradient descent
  const lab3 = khoi('gradient-descent', 3,
    'Gradient descent: số điều kiện, không phải số chiều',
    'Hàm $f(x) = \\tfrac12(x_0^2 + \\kappa x_1^2)$. Trượt $\\kappa$ rồi xem số vòng lặp. Trượt tốc độ học vượt $2/\\lambda_{\\max}$ để thấy nó **phân kỳ**, không phải chậm đi. Bật quán tính để thấy bậc đổi từ $O(\\kappa)$ xuống $O(\\sqrt\\kappa)$ — số liệu ở Mục 5.3.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('gdKap', 'Số điều kiện κ', 0, 40, 20, 1, 'gdKapOut') +
    range('gdEta', 'Tốc độ học (% của ngưỡng 2/λ_max)', 5, 130, 60, 5, 'gdEtaOut') +
    check('gdMom', 'Thêm quán tính (heavy ball)', false) +
    stat('gdIter', 'Số vòng tới sai số 10⁻⁸') +
    stat('gdEta2', 'η đang dùng') +
    stat('gdOpt', 'η tối ưu về lý thuyết', '2/(L+m)') +
    '</div><div class="lab-output">' + canvas('gdPath', 270) + '</div></div>', 'gdNote');

  // ------------------------------------------------------------ 4. SVM lề mềm
  const lab4 = khoi('svm-le-mem', 4,
    'SVM lề mềm: $C$ chia các điểm làm đúng ba nhóm',
    'Điều kiện KKT ở Mục 12.4 nói rằng mỗi điểm rơi vào đúng một trong ba nhóm: **ngoài lề** (không ảnh hưởng gì tới nghiệm), **đúng trên lề**, hoặc **vi phạm lề**. Trượt $C$ và xem ba con số ấy đổi theo. Điểm nhỏ là điểm ngoài lề.',
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
    'PCA so với LDA — **kéo thả các điểm**',
    'Đây là công cụ đáng nghịch nhất trang này. Hai lớp, màu khác nhau. **Kéo một điểm bất kỳ** rồi xem hai trục tự xoay theo. Hãy thử kéo cho dữ liệu trải rất rộng theo chiều ngang trong khi hai lớp vẫn tách nhau theo chiều dọc — PCA sẽ chọn trục ngang và AUC của nó rơi về 0,5, đúng hiện tượng đo được ở Mục 14.4.',
    '<div class="lab-grid"><div class="lab-controls">' +
    btn('pcaReset', 'Đặt lại vị trí điểm') +
    stat('pcaVar', 'Phương sai trục PCA₁ giữ được') +
    stat('pcaAucPC', 'AUC khi chiếu lên PCA₁') +
    stat('pcaAucLD', 'AUC khi chiếu lên LDA') +
    '</div><div class="lab-output">' + canvas('pcaPlot', 320) + '</div></div>', 'pcaNote');

  // ------------------------------------------------------------- 6. K-means
  const lab6 = khoi('k-means', 6,
    'K-means: bấm từng bước để thấy nó hội tụ, và kẹt ở đâu',
    'Bấm **một bước** để chạy đúng một vòng lặp: gán điểm cho tâm gần nhất, rồi dời tâm về trung bình cụm. Bấm **khởi tạo dồn một chỗ** vài lần để thấy nó kẹt ở những nghiệm tồi khác nhau — hiện tượng đo được ở Mục 15.2, nơi khởi tạo ngẫu nhiên thuần kẹt 71,5% số lần.',
    '<div class="lab-grid"><div class="lab-controls">' +
    range('kmK', 'Số cụm k', 2, 8, 4, 1, 'kmKOut') +
    '<div class="lab-btns">' + btn('kmStep', 'Một bước') + btn('kmRun', 'Chạy tới hội tụ') + '</div>' +
    '<div class="lab-btns">' + btn('kmNew', 'Khởi tạo ngẫu nhiên') + btn('kmBad', 'Khởi tạo dồn một chỗ') + '</div>' +
    stat('kmBuoc', 'Số bước đã chạy') +
    stat('kmQT', 'Quán tính', 'luôn giảm theo k') +
    '</div><div class="lab-output">' + canvas('kmPlot', 320) + '</div></div>', 'kmNote');

  const body =
    '<article class="prose lab-page" data-lab-nentang>' +
    '<header class="hero hero--tight">' +
    '<p class="hero-kicker">Nền tảng · thực hành</p>' +
    '<h1>Phòng thí nghiệm</h1>' +
    '<p class="hero-lede">Sáu công cụ cho sáu chỗ mà đọc chữ không đủ. Đổi tham số rồi xem kết quả đổi theo — đó là cách nhanh nhất để thấy một cơ chế, thay vì tin vào một câu khẳng định.</p>' +
    '<p class="hero-lede">Mỗi công cụ gắn với một mục trong giáo trình và cài đúng công thức ở đó. Nhưng chúng chạy trong trình duyệt nên dùng ít mẫu và ít vòng lặp hơn các thí nghiệm ngoại tuyến, vì vậy con số hiện ra <b>sát chứ không trùng khít</b> với bảng đã in. Chỗ nào phải trùng đúng — như số hệ số bằng 0 của lasso — thì có ghi rõ.</p>' +
    '</header>' +
    lab1 + lab2 + lab3 + lab4 + lab5 + lab6 +
    '</article>';

  write(
    f('thuc-hanh.html'),
    page({
      title: 'Phòng thí nghiệm — ' + docTitle,
      description:
        'Sáu công cụ tương tác: tách thiên lệch và phương sai, so ridge với lasso, ' +
        'gradient descent theo số điều kiện, ba nhóm KKT của SVM lề mềm, ' +
        'PCA so với LDA có kéo thả điểm, và K-means chạy từng bước.',
      body,
      nav,
      file: f('thuc-hanh.html'),
      ...neighbours(f('thuc-hanh.html')),
      scripts: ['assets/viz.js', 'assets/lab-nentang.js'],
    })
  );
}
