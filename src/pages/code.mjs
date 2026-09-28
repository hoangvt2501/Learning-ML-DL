// Trang Mã nguồn: hiển thị 6 script sinh ra mọi hình và mọi con số trong tài liệu.

import fs from 'node:fs';
import path from 'node:path';
import hljs from 'highlight.js';

const SCRIPTS = [
  {
    file: 'fig_basics.py',
    title: 'fig_basics.py',
    note: 'Sinh Hình 1–7, 12 và 16: chi phí năng lượng, bố cục bit, lưới FP so với INT, hàm bậc thang, đối xứng so với bất đối xứng, SQNR, đánh đổi làm tròn – cắt, STE và NF4.',
  },
  {
    file: 'fig_diagrams.py',
    title: 'fig_diagrams.py',
    note: 'Sinh Hình 9, 10 và 17: luồng dữ liệu thuần số nguyên, đồ thị có nút fake-quant, và sơ đồ chọn phương pháp.',
  },
  {
    file: 'numpy_experiments.py',
    title: 'numpy_experiments.py',
    note: 'Thí nghiệm ở Mục 6.5 (đường tính toán số nguyên trùng khớp từng bit), 11.3 (SmoothQuant) và 11.4 (GPTQ so với RTN). Sinh Hình 14 và 15.',
  },
  {
    file: 'torch_experiments.py',
    title: 'torch_experiments.py',
    note: 'Thí nghiệm ở Mục 5.3 (granularity), 7.3 (gộp BatchNorm), 8.2 (calibration), 9.3 (PTQ so với QAT theo độ rộng bit) và 10.1 (PyTorch). Sinh Hình 8, 11 và 13. Chạy vài phút trên CPU.',
  },
  {
    file: 'torch_qat_eager.py',
    title: 'torch_qat_eager.py',
    note: 'QAT bằng eager mode của PyTorch, Mục 10.1.3.',
  },
  {
    file: 'sensitivity.py',
    title: 'sensitivity.py',
    note: 'Phân tích độ nhạy theo lớp ở Mục 12.2: lượng tử từng lớp một xuống 3 bit, giữ các lớp khác ở FP32.',
  },
];

function highlight(code, lang) {
  try {
    return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
  } catch {
    return code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
}

export function buildCodePage(ctx) {
  const { nav, page, escapeHtml, write, neighbours, registry, ROOT, md } = ctx;
  registry.currentFile = 'ma-nguon.html';

  const codeDir = path.join(ROOT, 'code');
  const outDir = path.join(ROOT, 'docs', 'code');
  fs.mkdirSync(outDir, { recursive: true });
  for (const f of fs.readdirSync(codeDir)) {
    fs.copyFileSync(path.join(codeDir, f), path.join(outDir, f));
  }

  const blocks = SCRIPTS.map((s) => {
    const src = fs.readFileSync(path.join(codeDir, s.file), 'utf8');
    const outFile = s.file.replace(/\.py$/, '_output.txt');
    const outPath = path.join(codeDir, outFile);
    const output = fs.existsSync(outPath) ? fs.readFileSync(outPath, 'utf8') : null;
    const lines = src.split('\n').length;

    return (
      '<section class="src-card" id="' + s.file.replace(/\./g, '-') + '">' +
      '<header class="src-head">' +
      '<h3 class="src-title"><code>code/' + escapeHtml(s.file) + '</code></h3>' +
      '<span class="src-meta">' + lines + ' dòng</span>' +
      '<a class="src-dl" href="code/' + escapeHtml(s.file) + '" download>Tải về</a>' +
      '</header>' +
      '<p class="src-note">' + escapeHtml(s.note) + '</p>' +
      '<details class="src-details"><summary>Xem mã nguồn</summary>' +
      '<div class="code-block" data-lang="python">' +
      '<div class="code-head"><span class="code-lang">Python</span>' +
      '<button class="code-copy" type="button" data-copy>Sao chép</button></div>' +
      '<pre><code class="hljs language-python">' + highlight(src, 'python') + '</code></pre>' +
      '</div></details>' +
      (output
        ? '<details class="src-details"><summary>Kết quả chạy — <code>' +
          escapeHtml(outFile) + '</code></summary>' +
          '<div class="code-block is-output"><div class="code-head">' +
          '<span class="code-lang">Kết quả chạy</span></div>' +
          '<pre><code>' + escapeHtml(output) + '</code></pre></div></details>'
        : '') +
      '</section>'
    );
  }).join('');

  const runInstructions = md.render(
    'Các script đặt hạt giống cố định nên độ chính xác lặp lại được trên cùng phiên bản thư viện. ' +
      'Riêng độ trễ phụ thuộc phần cứng và dao động giữa các lần chạy.\n\n' +
      '```bash\npip install numpy scipy matplotlib scikit-learn torch torchao\n' +
      'python code/fig_basics.py\npython code/fig_diagrams.py\npython code/numpy_experiments.py\n' +
      'python code/torch_experiments.py        # vài phút trên CPU\n' +
      'python code/torch_qat_eager.py\npython code/sensitivity.py\n```\n\n' +
      'Môi trường đã dùng để sinh mọi số liệu trong tài liệu: Python 3.12, NumPy 2.4, ' +
      'PyTorch 2.14 (CPU), torchao 0.18, scikit-learn 1.8, CPU Intel Xeon 2,1 GHz ' +
      '(có AVX-512 VNNI và AMX-INT8), 1 luồng.'
  );

  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Tra cứu</span>' +
    '<span class="kicker-time">' + SCRIPTS.length + ' script</span></div>' +
    '<h1>Mã nguồn thí nghiệm</h1>' +
    '<p class="chapter-lede">Mọi hình và mọi con số thực nghiệm trong giáo trình đều do các script dưới đây sinh ra. Chúng được chép nguyên vẹn vào site nên tải về là chạy được ngay.</p>' +
    '<h3 class="heading" id="cach-chay">Cách chạy lại toàn bộ<a class="anchor" href="#cach-chay">#</a></h3>' +
    runInstructions +
    '<h3 class="heading" id="danh-sach">Từng script<a class="anchor" href="#danh-sach">#</a></h3>' +
    '<div class="src-list">' + blocks + '</div>' +
    '</article>';

  write(
    'ma-nguon.html',
    page({
      title: 'Mã nguồn thí nghiệm — Quantization trong Deep Learning',
      description: 'Sáu script Python sinh ra toàn bộ hình vẽ và số liệu của giáo trình, kèm kết quả in ra.',
      body,
      nav,
      file: 'ma-nguon.html',
      ...neighbours('ma-nguon.html'),
    })
  );
}
