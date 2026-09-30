// Trang Mã nguồn: hiển thị các script sinh ra hình và số liệu của một giáo trình.

import fs from 'node:fs';
import path from 'node:path';
import hljs from 'highlight.js';

function highlight(code, lang) {
  try {
    return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
  } catch {
    return code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
}

export function buildCodePage(ctx) {
  const { nav, page, escapeHtml, write, read, neighbours, registry, ROOT, md, f, book } = ctx;
  registry.currentFile = f('ma-nguon.html');

  const SCRIPTS = JSON.parse(read(book.scripts));
  const codeDir = path.join(ROOT, book.codeDir);
  const outDir = path.join(ROOT, 'docs', book.codeDir);
  fs.mkdirSync(outDir, { recursive: true });
  for (const entry of fs.readdirSync(codeDir, { withFileTypes: true })) {
    if (entry.isFile()) fs.copyFileSync(path.join(codeDir, entry.name), path.join(outDir, entry.name));
  }

  const blocks = SCRIPTS.map((s) => {
    const src = fs.readFileSync(path.join(codeDir, s.file), 'utf8');
    const outFile = s.file.replace(/\.py$/, '_output.txt');
    const outPath = path.join(codeDir, outFile);
    const output = fs.existsSync(outPath) ? fs.readFileSync(outPath, 'utf8') : null;
    const lines = src.split('\n').length;
    const href = book.codeDir + '/' + s.file;

    return (
      '<section class="src-card" id="' + s.file.replace(/\./g, '-') + '">' +
      '<header class="src-head">' +
      '<h3 class="src-title"><code>' + escapeHtml(href) + '</code></h3>' +
      (s.added ? '<span class="src-badge">viết thêm cho repo này</span>' : '') +
      '<span class="src-meta">' + lines + ' dòng</span>' +
      '<a class="src-dl" href="' + href + '" download>Tải về</a>' +
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

  const runInstructions = md.render(read(book.runbook || 'content/runbook-' + book.id + '.md'));

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
    f('ma-nguon.html'),
    page({
      title: 'Mã nguồn thí nghiệm — ' + ctx.docTitle,
      description: SCRIPTS.length + ' script Python sinh ra toàn bộ hình vẽ và số liệu của giáo trình, kèm kết quả in ra.',
      body,
      nav,
      file: f('ma-nguon.html'),
      ...neighbours(f('ma-nguon.html')),
    })
  );
}
