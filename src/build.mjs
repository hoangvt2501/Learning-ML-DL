// Dựng site tĩnh cho các giáo trình khai báo trong content/books.json.
// Chạy: npm run build   ->   toàn bộ kết quả nằm trong docs/

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMarkdownIt, sectionId, figureId, slugify } from './markdown.mjs';
import { buildSprite, icon } from './icons.mjs';
import { page, escapeHtml } from './layout.mjs';
import { buildExercisePage } from './pages/exercises.mjs';
import { buildCurriculumPage } from './pages/curriculum.mjs';
import { buildPlaygroundPage } from './pages/playground.mjs';
import { buildMlopsLabPage } from './pages/lab-mlops.mjs';
import { buildNentangLabPage } from './pages/lab-nentang.mjs';
import { buildModelsLabPage } from './pages/lab-models.mjs';
import { buildBieudienLabPage } from './pages/lab-bieudien.mjs';
import { buildUngdungLabPage } from './pages/lab-ungdung.mjs';
import { buildGlossaryPage } from './pages/glossary.mjs';
import { buildFiguresPage } from './pages/figures.mjs';
import { buildCodePage } from './pages/code.mjs';
import { buildHomePage } from './pages/home.mjs';
import { parseQuizzes, renderQuiz } from './pages/quiz.mjs';
import { parseNotes, injectNotes, buildNotesPage } from './pages/notes.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'docs');

// Chuẩn hoá xuống dòng về LF: mọi bộ phân tích bên dưới đều tách đoạn bằng \n{2,},
// nên một tệp lưu kiểu CRLF sẽ lặng lẽ không tách được đoạn nào.
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8').replace(/\r\n?/g, '\n');
const write = (rel, data) => {
  const p = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, data);
};

// ------------------------------------------------------------ tiện ích chung

/** Cắt tài liệu thành các phần theo tiêu đề cấp 2. */
function splitChapters(mdText) {
  const parts = [];
  const head = [];
  let cur = null;
  let inFence = false;

  for (const line of mdText.split('\n')) {
    if (/^```/.test(line)) inFence = !inFence;
    const h2 = !inFence && /^## (?!#)/.test(line) ? line.slice(3).trim() : null;
    if (h2) {
      if (cur) parts.push(cur);
      cur = { title: h2, lines: [] };
    } else if (cur) cur.lines.push(line);
    else head.push(line);
  }
  if (cur) parts.push(cur);
  return { head: head.join('\n'), parts };
}

function readingMinutes(markdown) {
  const words = markdown.replace(/```[\s\S]*?```/g, ' ').split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 170));
}

/** Mục lục bên phải, lấy từ các thẻ h3/h4 trong HTML đã render. */
function extractToc(html) {
  const re = /<h([34]) id="([^"]+)" class="heading">([\s\S]*?)<a class="anchor"/g;
  const items = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    const so = (m[3].match(/<span class="hnum">([^<]+)<\/span>/) || [])[1] || '';
    const text = m[3].replace(/<span class="hnum">[^<]*<\/span>/, '').replace(/<[^>]+>/g, '').trim();
    if (text) items.push({ level: +m[1], id: m[2], text, so });
  }
  if (items.length < 2) return '';
  return (
    '<ul class="toc-list">' +
    items
      .map((i) =>
        '<li class="toc-l' + i.level + '"><a href="#' + i.id + '">' +
        (i.so ? '<span class="toc-num">' + i.so + '</span>' : '') +
        '<span>' + escapeHtml(i.text) + '</span></a></li>')
      .join('') +
    '</ul>'
  );
}

const searchIndex = [];

// ============================================================ dựng một giáo trình

function buildBook(spec, books) {
  const f = (name) => spec.slug + name;                 // tên file có tiền tố của sách
  const source = read(spec.source);
  const chapterMeta = JSON.parse(read(spec.chapters));
  const { head, parts } = splitChapters(source);

  const docTitle = (head.match(/^# (.+)$/m) || [, spec.short])[1].trim();
  const intro = head.split('\n').filter((l) => l.startsWith('>')).join('\n');

  const chapters = chapterMeta.map((meta) => {
    const part = parts.find((p) =>
      meta.num === 'PL' ? p.title.startsWith('Phụ lục') : p.title.startsWith(meta.num + '. '));
    if (!part) throw new Error(`[${spec.id}] Không tìm thấy chương ${meta.num} trong ${spec.source}`);
    return { ...meta, title: part.title, markdown: part.lines.join('\n').trim() };
  });

  // Chương nào là trang bài tập thì điều hướng trỏ thẳng sang đó.
  const fileOf = (ch) => (ch.kind === 'exercises' ? f('bai-tap.html') : ch.file);

  // ---------------------------------------------- sổ tra cứu mục và hình
  const registry = { sections: new Map(), figures: new Map(), currentFile: '' };

  for (const ch of chapters) {
    const target = fileOf(ch);
    if (ch.num !== 'PL') {
      registry.sections.set(ch.num, {
        href: target + '#' + sectionId(ch.num), title: ch.title, file: target,
      });
    }
    const re = /^#{3,4} (\d+(?:\.\d+)+)\.\s*(.+)$/gm;
    let m;
    while ((m = re.exec(ch.markdown)) !== null) {
      registry.sections.set(m[1], {
        href: target + '#' + sectionId(m[1]), title: m[2].trim(), file: target,
      });
    }
    const fr = /!\[Hình (\d+)\]\((figs\/[^)]+)\)\s*\n\s*\n\*\*Hình \1\.\*\*\s*([^\n]+)/g;
    let fig;
    while ((fig = fr.exec(ch.markdown)) !== null) {
      registry.figures.set(fig[1], {
        href: target + '#' + figureId(fig[1]),
        src: fig[2],
        caption: fig[3].trim(),
        file: target,
        chapter: ch.num,
      });
    }
  }

  const md = createMarkdownIt(registry);

  // ---------------------------------------------------------- điều hướng
  const extras = [];
  if (spec.lab) extras.push({ file: f('thuc-hanh.html'), icon: 'thi-nghiem', label: 'Phòng thí nghiệm' });
  if (spec.figures && registry.figures.size) {
    extras.push({ file: f('hinh-anh.html'), icon: 'hinh', label: `Thư viện ${registry.figures.size} hình` });
  }
  extras.push({ file: f('thuat-ngu.html'), icon: 'thuat-ngu', label: 'Từ điển thuật ngữ' });
  if (spec.notes) extras.push({ file: f('ghi-chu.html'), icon: 'ghi-chu', label: 'Ghi chú biên tập' });
  if (spec.code) extras.push({ file: f('ma-nguon.html'), icon: 'ma-nguon', label: 'Mã nguồn thí nghiệm' });
  extras.push({ file: f('toan-van.html'), icon: 'toan-van', label: 'Toàn văn một trang' });

  const nav = {
    book: spec,
    books,
    groupLabel: spec.kicker + ' · ' + spec.short,
    chapters: chapters.map((c) => ({
      file: fileOf(c), num: c.num, label: c.label, part: c.part || '',
    })),
    extras,
  };

  const navOrder = [{ file: 'index.html', label: 'Trang chủ' }, ...nav.chapters, ...extras];
  const neighbours = (file) => {
    const i = navOrder.findIndex((x) => x.file === file);
    return { prev: i > 0 ? navOrder[i - 1] : null, next: i >= 0 ? navOrder[i + 1] : null };
  };

  // ------------------------------------------------------ chỉ mục tìm kiếm
  function indexChapter(ch, targetFile) {
    const blocks = ch.markdown.split(/^(#{3,4} .+)$/m);
    let heading = ch.title;
    let id = ch.num === 'PL' ? slugify(ch.title) : sectionId(ch.num);

    const push = (h, anchor, body) => {
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
        searchIndex.push({ b: spec.short, c: ch.num, t: h, h: targetFile + '#' + anchor, x: chunk });
      }
    };

    for (const b of blocks) {
      const hm = b.match(/^#{3,4} (.+)$/);
      if (hm) {
        heading = hm[1].trim();
        const num = heading.match(/^(\d+(?:\.\d+)*)\./);
        id = num ? sectionId(num[1]) : slugify(heading);
      } else push(heading, id, b);
    }
  }

  // ------------------------------------------------------------ dựng trang
  const rendered = new Map();
  for (const ch of chapters) {
    const target = fileOf(ch);
    registry.currentFile = target;
    rendered.set(ch.num, md.render('## ' + ch.title + '\n\n' + ch.markdown));
    indexChapter(ch, target);
  }

  const minutes = Object.fromEntries(chapters.map((c) => [c.num, readingMinutes(c.markdown)]));
  const quizzes = spec.quiz ? parseQuizzes(read(spec.quiz)) : new Map();
  const notes = spec.notes ? parseNotes(read(spec.notes)) : new Map();

  for (const ch of chapters) {
    if (ch.kind === 'exercises') continue;               // dựng riêng ở bước sau
    const html = injectNotes(rendered.get(ch.num), ch.num, notes, md);
    const kicker = ch.num === 'PL' ? 'Phụ lục' : 'Chương ' + ch.num;
    const quiz = quizzes.has(ch.num) ? renderQuiz(md, quizzes.get(ch.num), 'chapter') : '';
    // Không chèn dòng tóm tắt dưới tiêu đề: mỗi chương tự mở đầu bằng đoạn dẫn của nó,
    // và dòng tóm tắt (dùng cho mục lục và thẻ mô tả) chỉ lặp lại đoạn đó.
    const withLede = html;

    write(ch.file, page({
      title: ch.title + ' — ' + docTitle,
      description: ch.summary,
      body:
        '<article class="prose">' +
        '<div class="chapter-kicker"><span class="kicker-badge">' + kicker + '</span>' +
        '<span class="kicker-time">' + icon('dong-ho') + minutes[ch.num] + ' phút đọc</span>' +
        (ch.part ? '<span class="kicker-part">' + escapeHtml(ch.part) + '</span>' : '') +
        '</div>' +
        withLede + quiz + '</article>',
      nav,
      file: ch.file,
      ...neighbours(ch.file),
      toc: extractToc(html),
    }));
  }

  // ------------------------------------------------------ trang chuyên biệt
  const ctx = {
    md, registry, nav, neighbours, page, escapeHtml, read, write, f,
    book: spec, chapters, minutes, docTitle, intro, extractToc, searchIndex, ROOT,
    quizCount: [...quizzes.values()].reduce((a, g) => a + g.questions.length, 0),
    noteCount: notes.size,
  };

  const exChapter = chapters.find((c) => c.kind === 'exercises');
  if (exChapter) buildExercisePage(ctx, exChapter);
  if (spec.lab === 'quantization') buildPlaygroundPage(ctx);
  if (spec.lab === 'mlops') buildMlopsLabPage(ctx);
  if (spec.lab === 'nentang') buildNentangLabPage(ctx);
  if (spec.lab === 'models') buildModelsLabPage(ctx);
  if (spec.lab === 'bieudien') buildBieudienLabPage(ctx);
  if (spec.lab === 'ungdung') buildUngdungLabPage(ctx);
  if (spec.glossary) buildGlossaryPage(ctx);
  if (spec.figures && registry.figures.size) buildFiguresPage(ctx);
  if (spec.notes) buildNotesPage(ctx, notes);
  if (spec.code) buildCodePage(ctx);

  // ------------------------------------------------------------ toàn văn
  registry.currentFile = f('toan-van.html');
  const all = chapters
    .map((ch) => '<section class="fulltext-chapter">' +
      injectNotes(md.render('## ' + ch.title + '\n\n' + ch.markdown), ch.num, notes, md) +
      '</section>')
    .join('\n');
  write(f('toan-van.html'), page({
    title: 'Toàn văn — ' + docTitle,
    description: 'Toàn bộ giáo trình ' + spec.short + ' trên một trang.',
    body:
      '<article class="prose">' +
      '<div class="chapter-kicker"><span class="kicker-badge">Toàn văn</span>' +
      '<span class="kicker-time">~' + Object.values(minutes).reduce((a, b) => a + b, 0) +
      ' phút đọc</span></div>' +
      '<h1>' + escapeHtml(docTitle) + '</h1>' +
      '<p class="chapter-lede">Toàn bộ giáo trình trên một trang — tiện cho Ctrl+F, đọc ngoại tuyến hoặc in ra giấy.</p>' +
      (intro ? md.render(intro) : '') + all + '</article>',
    nav,
    file: f('toan-van.html'),
    ...neighbours(f('toan-van.html')),
    bodyClass: 'is-fulltext',
  }));

  // Bản markdown gốc để tải về.
  fs.copyFileSync(path.join(ROOT, spec.source), path.join(OUT, path.basename(spec.source)));

  return {
    spec, docTitle, intro, chapters, minutes, nav, fileOf,
    figures: registry.figures.size,
    sections: registry.sections.size,
    quizCount: ctx.quizCount,
    noteCount: ctx.noteCount,
    exercises: exChapter ? 1 : 0,
    md,
  };
}

// ==================================================================== chạy

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

buildSprite(ROOT);
const allBooks = JSON.parse(read('content/books.json'));
// Bỏ qua giáo trình chưa có tệp nguồn, để repo vẫn dựng được khi đang viết dở.
const books = allBooks.filter((spec) => {
  const ok = fs.existsSync(path.join(ROOT, spec.source));
  if (!ok) console.warn('Bỏ qua giáo trình "' + spec.id + '": chưa có ' + spec.source);
  return ok;
});
const built = books.map((spec) => buildBook(spec, books));

buildCurriculumPage({ books: built, page, escapeHtml, write, searchIndex, read });
buildHomePage({ books: built, page, escapeHtml, write, read, ROOT, searchIndex });

// ---------------------------------------------------------------- tài nguyên
function copyDir(from, to) {
  if (!fs.existsSync(from)) return;
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const s = path.join(from, entry.name);
    const d = path.join(to, entry.name);
    if (entry.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
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
  for (const file of fs.readdirSync(path.join(kdist, 'fonts'))) {
    if (file.endsWith('.woff2')) {
      fs.copyFileSync(path.join(kdist, 'fonts', file), path.join(fontsOut, file));
    }
  }
}

write('assets/search-index.js', 'window.QZ_INDEX=' + JSON.stringify(searchIndex) + ';');
write('.nojekyll', '');

const pages = fs.readdirSync(OUT).filter((x) => x.endsWith('.html')).length;
console.log(
  'Đã dựng ' + pages + ' trang HTML cho ' + built.length + ' giáo trình:\n' +
  built.map((b) =>
    '  · ' + (b.spec.short + '  ').padEnd(22) + b.chapters.length + ' chương, ' +
    b.figures + ' hình, ' + b.sections + ' mục tra cứu, ' + b.quizCount + ' câu trắc nghiệm'
  ).join('\n') +
  '\n  → ' + searchIndex.length + ' bản ghi tìm kiếm.'
);
