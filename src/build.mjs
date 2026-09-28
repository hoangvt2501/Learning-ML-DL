// Dựng site tĩnh từ content/quantization.md và các tệp nội dung đi kèm.
// Chạy: npm run build   ->   toàn bộ kết quả nằm trong docs/

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMarkdownIt, sectionId, figureId, slugify } from './markdown.mjs';
import { page, escapeHtml } from './layout.mjs';
import { buildExercisePage } from './pages/exercises.mjs';
import { buildPlaygroundPage } from './pages/playground.mjs';
import { buildGlossaryPage } from './pages/glossary.mjs';
import { buildFiguresPage } from './pages/figures.mjs';
import { buildCodePage } from './pages/code.mjs';
import { buildHomePage } from './pages/home.mjs';
import { parseQuizzes, renderQuiz } from './pages/quiz.mjs';
import { parseNotes, injectNotes, buildNotesPage } from './pages/notes.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'docs');

const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const write = (rel, data) => {
  const p = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, data);
};

// ---------------------------------------------------------------- 1. nội dung
const source = read('content/quantization.md');
const chapterMeta = JSON.parse(read('content/chapters.json'));

/** Cắt tài liệu thành các phần theo tiêu đề cấp 2. */
function splitChapters(md) {
  const lines = md.split('\n');
  const parts = [];
  let head = [];
  let cur = null;
  let inFence = false;

  for (const line of lines) {
    if (/^```/.test(line)) inFence = !inFence;
    const h2 = !inFence && /^## (?!#)/.test(line) ? line.slice(3).trim() : null;
    if (h2) {
      if (cur) parts.push(cur);
      cur = { title: h2, lines: [] };
    } else if (cur) {
      cur.lines.push(line);
    } else {
      head.push(line);
    }
  }
  if (cur) parts.push(cur);
  return { head: head.join('\n'), parts };
}

const { head, parts } = splitChapters(source);

// Tiêu đề chính + đoạn dẫn nhập nằm trước mục lục.
const docTitle = (head.match(/^# (.+)$/m) || [, 'Quantization'])[1].trim();
const intro = head
  .split('\n')
  .filter((l) => l.startsWith('>'))
  .join('\n');

// Gắn từng phần đã cắt vào metadata chương.
const chapters = chapterMeta.map((meta) => {
  const part = parts.find((p) => {
    if (meta.num === 'PL') return p.title.startsWith('Phụ lục');
    return p.title.startsWith(meta.num + '. ');
  });
  if (!part) throw new Error('Không tìm thấy chương ' + meta.num + ' trong quantization.md');
  return { ...meta, title: part.title, markdown: part.lines.join('\n').trim() };
});

// ------------------------------------------------- 2. sổ tra cứu mục và hình
const registry = { sections: new Map(), figures: new Map(), currentFile: '' };

for (const ch of chapters) {
  const target = ch.num === '14' ? 'bai-tap.html' : ch.file;
  // tiêu đề chương
  if (ch.num !== 'PL') {
    registry.sections.set(ch.num, {
      href: target + '#' + sectionId(ch.num),
      title: ch.title,
      file: target,
    });
  }
  // các mục con
  const re = /^#{3,4} (\d+(?:\.\d+)+)\.\s*(.+)$/gm;
  let m;
  while ((m = re.exec(ch.markdown)) !== null) {
    registry.sections.set(m[1], {
      href: target + '#' + sectionId(m[1]),
      title: m[2].trim(),
      file: target,
    });
  }
  // hình
  const fr = /!\[Hình (\d+)\]\(figs\/([^)]+)\)\s*\n\s*\n\*\*Hình \1\.\*\*\s*([^\n]+)/g;
  let f;
  while ((f = fr.exec(ch.markdown)) !== null) {
    registry.figures.set(f[1], {
      href: target + '#' + figureId(f[1]),
      src: 'figs/' + f[2],
      caption: f[3].trim(), // markdown thô, có cả công thức
      file: target,
      chapter: ch.num,
      chapterTitle: ch.title,
    });
  }
}

const md = createMarkdownIt(registry);

// ---------------------------------------------------------------- 3. điều hướng
const nav = {
  chapters: chapters.map((c) => ({
    file: c.num === '14' ? 'bai-tap.html' : c.file,
    num: c.num,
    label: c.label,
  })),
  extras: [
    { file: 'thuc-hanh.html', icon: '⚙', label: 'Phòng thí nghiệm' },
    { file: 'hinh-anh.html', icon: '◳', label: 'Thư viện 17 hình' },
    { file: 'thuat-ngu.html', icon: '¶', label: 'Từ điển thuật ngữ' },
    { file: 'ghi-chu.html', icon: '!', label: 'Ghi chú biên tập' },
    { file: 'ma-nguon.html', icon: '{}', label: 'Mã nguồn thí nghiệm' },
    { file: 'toan-van.html', icon: '≡', label: 'Toàn văn một trang' },
  ],
};

const navOrder = [{ file: 'index.html', label: 'Trang chủ' }, ...nav.chapters, ...nav.extras];
const neighbours = (file) => {
  const i = navOrder.findIndex((x) => x.file === file);
  return { prev: i > 0 ? navOrder[i - 1] : null, next: i >= 0 ? navOrder[i + 1] : null };
};

// ---------------------------------------------------------------- 4. tiện ích
function readingMinutes(markdown) {
  const words = markdown.replace(/```[\s\S]*?```/g, ' ').split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 170));
}

/** Mục lục bên phải, lấy từ các thẻ h3 (và h4 nếu có) trong HTML đã render. */
function extractToc(html) {
  const re = /<h([34]) id="([^"]+)" class="heading">([\s\S]*?)<a class="anchor"/g;
  const items = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    const text = m[3].replace(/<[^>]+>/g, '').trim();
    if (text) items.push({ level: +m[1], id: m[2], text });
  }
  if (items.length < 2) return '';
  return (
    '<ul class="toc-list">' +
    items
      .map(
        (i) =>
          '<li class="toc-l' + i.level + '"><a href="#' + i.id + '">' + escapeHtml(i.text) + '</a></li>'
      )
      .join('') +
    '</ul>'
  );
}

/** Chỉ mục tìm kiếm: mỗi mục con là một bản ghi. */
const searchIndex = [];
function indexChapter(ch, targetFile) {
  const blocks = ch.markdown.split(/^(#{3,4} .+)$/m);
  let currentHeading = ch.title;
  let currentId = sectionId(ch.num === 'PL' ? '0' : ch.num);
  if (ch.num === 'PL') currentId = slugify(ch.title);

  // Mỗi mục con được cắt thành vài đoạn ~600 ký tự để kết quả tìm kiếm
  // trỏ tới đúng chỗ và đoạn trích luôn chứa từ khoá.
  const push = (heading, id, body) => {
    const text = body
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
      .replace(/[#*_`>|$\\]/g, ' ')
      .replace(/[ \t]+/g, ' ')
      .trim();
    if (!text) return;

    const chunks = [];
    let buf = '';
    for (const para of text.split(/\n{2,}/)) {
      const p = para.replace(/\s+/g, ' ').trim();
      if (!p) continue;
      if (buf && buf.length + p.length > 600) { chunks.push(buf); buf = p; }
      else buf = buf ? buf + ' ' + p : p;
    }
    if (buf) chunks.push(buf);

    for (const chunk of chunks) {
      searchIndex.push({ c: ch.num, t: heading, h: targetFile + '#' + id, x: chunk });
    }
  };

  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    const hm = b.match(/^#{3,4} (.+)$/);
    if (hm) {
      currentHeading = hm[1].trim();
      const num = currentHeading.match(/^(\d+(?:\.\d+)*)\./);
      currentId = num ? sectionId(num[1]) : slugify(currentHeading);
    } else {
      push(currentHeading, currentId, b);
    }
  }
}

// ---------------------------------------------------------------- 5. dựng trang
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

const rendered = new Map();

for (const ch of chapters) {
  const targetFile = ch.num === '14' ? 'bai-tap.html' : ch.file;
  registry.currentFile = targetFile;
  const heading =
    ch.num === 'PL'
      ? '## ' + ch.title
      : '## ' + ch.title;
  const html = md.render(heading + '\n\n' + ch.markdown);
  rendered.set(ch.num, html);
  indexChapter(ch, targetFile);
}

const minutes = Object.fromEntries(chapters.map((c) => [c.num, readingMinutes(c.markdown)]));
const quizzes = parseQuizzes(read('content/trac-nghiem.md'));
const notes = parseNotes(read('content/ghi-chu.md'));

for (const ch of chapters) {
  if (ch.num === '14') continue; // trang bài tập dựng riêng ở bước sau
  const html = injectNotes(rendered.get(ch.num), ch.num, notes, md);
  const { prev, next } = neighbours(ch.file);
  const kicker = ch.num === 'PL' ? 'Phụ lục' : 'Chương ' + ch.num;
  const quiz = quizzes.has(ch.num) ? renderQuiz(md, quizzes.get(ch.num), 'chapter') : '';
  // Đặt câu tóm tắt ngay sau tiêu đề chương chứ không phải trước nó.
  const lede = '<p class="chapter-lede">' + escapeHtml(ch.summary) + '</p>';
  const withLede = html.includes('</h2>')
    ? html.replace('</h2>', '</h2>' + lede)
    : lede + html;
  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">' + kicker + '</span>' +
    '<span class="kicker-time">' + minutes[ch.num] + ' phút đọc</span></div>' +
    withLede +
    quiz +
    '</article>';

  write(
    ch.file,
    page({
      title: ch.title + ' — ' + docTitle,
      description: ch.summary,
      body,
      nav,
      file: ch.file,
      prev,
      next,
      toc: extractToc(html),
    })
  );
}

// --------------------------------------------------- 6. các trang chuyên biệt
const ctx = {
  md,
  registry,
  nav,
  neighbours,
  page,
  escapeHtml,
  read,
  write,
  chapters,
  minutes,
  docTitle,
  intro,
  extractToc,
  searchIndex,
  ROOT,
  quizCount: [...quizzes.values()].reduce((a, g) => a + g.questions.length, 0),
  noteCount: notes.size,
};

buildHomePage(ctx);
buildExercisePage(ctx, rendered.get('14'), chapters.find((c) => c.num === '14'));
buildPlaygroundPage(ctx);
buildGlossaryPage(ctx);
buildFiguresPage(ctx);
buildNotesPage(ctx, notes);
buildCodePage(ctx);

// ------------------------------------------------------- 7. trang toàn văn
{
  registry.currentFile = 'toan-van.html';
  const all = chapters
    .map((ch) => {
      const body = injectNotes(md.render('## ' + ch.title + '\n\n' + ch.markdown), ch.num, notes, md);
      return '<section class="fulltext-chapter">' + body + '</section>';
    })
    .join('\n');
  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Toàn văn</span>' +
    '<span class="kicker-time">~' +
    Object.values(minutes).reduce((a, b) => a + b, 0) +
    ' phút đọc</span></div>' +
    '<h1>' + escapeHtml(docTitle) + '</h1>' +
    '<p class="chapter-lede">Toàn bộ 15 chương trên một trang — tiện cho Ctrl+F, đọc ngoại tuyến hoặc in ra giấy.</p>' +
    md.render(intro) +
    all +
    '</article>';
  write(
    'toan-van.html',
    page({
      title: 'Toàn văn — ' + docTitle,
      description: 'Toàn bộ giáo trình trên một trang.',
      body,
      nav,
      file: 'toan-van.html',
      ...neighbours('toan-van.html'),
      bodyClass: 'is-fulltext',
    })
  );
}

// ---------------------------------------------------------------- 8. tài nguyên
function copyDir(from, to, filter) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const s = path.join(from, entry.name);
    const d = path.join(to, entry.name);
    if (entry.isDirectory()) copyDir(s, d, filter);
    else if (!filter || filter(entry.name)) fs.copyFileSync(s, d);
  }
}

copyDir(path.join(ROOT, 'figs'), path.join(OUT, 'figs'));
copyDir(path.join(ROOT, 'src/assets'), path.join(OUT, 'assets'));

// KaTeX: chỉ lấy css + font woff2 cho nhẹ, và bỏ các nguồn font không kèm theo.
{
  const kdist = path.join(ROOT, 'node_modules/katex/dist');
  let css = fs.readFileSync(path.join(kdist, 'katex.min.css'), 'utf8');
  css = css.replace(/,url\([^)]*\)format\("(woff|truetype)"\)/g, '');
  write('assets/katex.min.css', css);
  const fontsOut = path.join(OUT, 'assets/fonts');
  fs.mkdirSync(fontsOut, { recursive: true });
  for (const f of fs.readdirSync(path.join(kdist, 'fonts'))) {
    if (f.endsWith('.woff2')) fs.copyFileSync(path.join(kdist, 'fonts', f), path.join(fontsOut, f));
  }
}

// Chỉ mục tìm kiếm nạp theo yêu cầu, dạng script để mở bằng file:// vẫn chạy.
write('assets/search-index.js', 'window.QZ_INDEX=' + JSON.stringify(searchIndex) + ';');

// Bản markdown gốc để tải về.
fs.copyFileSync(path.join(ROOT, 'content/quantization.md'), path.join(OUT, 'quantization.md'));
write('.nojekyll', '');

console.log(
  'Đã dựng ' +
    fs.readdirSync(OUT).filter((f) => f.endsWith('.html')).length +
    ' trang HTML, ' +
    registry.figures.size +
    ' hình, ' +
    registry.sections.size +
    ' mục tra cứu, ' +
    searchIndex.length +
    ' bản ghi tìm kiếm.'
);
