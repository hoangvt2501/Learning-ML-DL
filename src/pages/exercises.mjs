// Trang Bài tập: 9 bài của giáo trình + lời giải, bài luyện tính tay, và trắc nghiệm theo chương.

import { parseQuizzes, renderQuiz } from './quiz.mjs';

/** Tách chương 14 thành từng bài: đề + gợi ý. */
function parseExercises(markdown) {
  return markdown
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter((b) => /^\*\*Bài \d+/.test(b))
    .map((block) => {
      const head = block.match(/^\*\*Bài (\d+)(?:\s*\(([^)]*)\))?\.\*\*/);
      const lines = block.split('\n');
      const hintLine = lines.findIndex((l) => /^\*Gợi ý[:：]/.test(l.trim()));
      const hint = hintLine >= 0 ? lines[hintLine].trim().replace(/^\*|\*$/g, '') : '';
      const statement = (hintLine >= 0 ? lines.slice(0, hintLine) : lines)
        .join('\n')
        .replace(/^\*\*Bài \d+(?:\s*\([^)]*\))?\.\*\*\s*/, '')
        .trim();
      return { num: head[1], tag: (head[2] || '').trim(), statement, hint };
    });
}

/** Tách content/loi-giai.md thành lời giải theo số bài, kèm dòng @meta. */
function parseSolutions(raw) {
  const out = new Map();
  const blocks = raw.split(/^## Bài (\d+)\s*$/m);
  for (let i = 1; i < blocks.length; i += 2) {
    const num = blocks[i];
    let body = blocks[i + 1];
    const meta = {};
    body = body.replace(/^@meta\s+(.+)$/m, (_, line) => {
      for (const pair of line.split('|')) {
        const [k, v] = pair.split('=').map((s) => s.trim());
        if (k) meta[k] = v;
      }
      return '';
    });
    out.set(num, { meta, markdown: body.trim() });
  }
  return out;
}

const DRILLS = [
  { id: 'sz', label: 'Tính S và Z từ dải' },
  { id: 'quant', label: 'Lượng tử một giá trị' },
  { id: 'bits', label: 'Số bit thực tế mỗi trọng số' },
  { id: 'kv', label: 'Dung lượng KV cache' },
];

export function buildExercisePage(ctx, exChapter) {
  const { md, nav, page, escapeHtml, write, read, neighbours, registry, searchIndex, f, book } = ctx;

  registry.currentFile = f('bai-tap.html');

  const exercises = parseExercises(exChapter.markdown);
  const solutions = parseSolutions(read(book.solutions));
  const quizzes = parseQuizzes(read(book.quiz));

  if (!exercises.length) throw new Error('[' + book.id + '] Không đọc được bài tập nào');

  // Đề bài và lời giải cũng nên tìm được qua ô tìm kiếm.
  for (const ex of exercises) {
    const sol = solutions.get(ex.num);
    const plain = (s) => s.replace(/```[\s\S]*?```/g, ' ').replace(/[#*_`>|$\\]/g, ' ').replace(/\s+/g, ' ').trim();
    searchIndex.push({
      c: '14',
      t: 'Bài ' + ex.num + (ex.tag ? ' — ' + ex.tag : ''),
      h: f('bai-tap.html') + '#bai-' + ex.num,
      x: plain(ex.statement + ' ' + ex.hint + ' ' + (sol ? sol.markdown : '')).slice(0, 900),
    });
  }

  // ------------------------------------------------------------ 9 bài tập
  const cards = exercises
    .map((ex) => {
      const sol = solutions.get(ex.num);
      if (!sol) throw new Error('Thiếu lời giải cho Bài ' + ex.num);
      const meta = sol.meta || {};
      const chapterRef = meta.chuong
        ? '<a class="ex-chip ex-chip-link" href="ch' + String(meta.chuong).padStart(2, '0') +
          '.html">Chương ' + meta.chuong + '</a>'
        : '';
      const kind = ex.tag || meta.dang || '';
      const hint = ex.hint
        ? '<details class="ex-hint"><summary>Gợi ý của đề</summary><div class="ex-hint-body">' +
          md.renderInline(ex.hint) + '</div></details>'
        : '';

      return (
        '<article class="ex-card" id="bai-' + ex.num + '">' +
        '<header class="ex-head">' +
        '<span class="ex-num">Bài ' + ex.num + '</span>' +
        '<div class="ex-chips">' +
        (kind ? '<span class="ex-chip">' + escapeHtml(kind) + '</span>' : '') +
        (meta.kho ? '<span class="ex-chip ex-chip-' +
          ({ 'Cơ bản': 'easy', 'Trung bình': 'mid', 'Khó': 'hard' }[meta.kho] || 'mid') +
          '">' + escapeHtml(meta.kho) + '</span>' : '') +
        chapterRef +
        '</div>' +
        '<a class="ex-anchor" href="#bai-' + ex.num + '" aria-label="Liên kết tới bài này">#</a>' +
        '</header>' +
        '<div class="ex-statement">' + md.render(ex.statement) + '</div>' +
        hint +
        '<details class="ex-solution"><summary><span>Xem lời giải chi tiết</span></summary>' +
        '<div class="ex-solution-body prose">' + md.render(sol.markdown) + '</div></details>' +
        '</article>'
      );
    })
    .join('');

  // --------------------------------------------------------- luyện tính tay
  const hasDrills = book.drills !== false;
  const drillOptions = DRILLS.map(
    (d, i) =>
      '<button class="drill-tab' + (i === 0 ? ' is-active' : '') +
      '" type="button" data-drill="' + d.id + '">' + escapeHtml(d.label) + '</button>'
  ).join('');

  const drill =
    '<section class="drill" id="luyen-tinh-tay" data-drill-root>' +
    '<div class="drill-tabs" role="tablist">' + drillOptions + '</div>' +
    '<div class="drill-body">' +
    '<p class="drill-question" data-drill-question></p>' +
    '<div class="drill-fields" data-drill-fields></div>' +
    '<div class="drill-actions">' +
    '<button class="btn btn-primary" type="button" data-drill-check>Kiểm tra</button>' +
    '<button class="btn btn-ghost" type="button" data-drill-new>Câu khác</button>' +
    '<button class="btn btn-ghost" type="button" data-drill-show>Xem đáp án</button>' +
    '</div>' +
    '<div class="drill-feedback" data-drill-feedback aria-live="polite"></div>' +
    '</div></section>';

  // ---------------------------------------------------------- trắc nghiệm
  const quizChapters = [...quizzes.keys()];
  const filter =
    '<div class="quiz-filter" data-quiz-filter>' +
    '<button class="chip is-active" type="button" data-filter="all">Tất cả</button>' +
    quizChapters
      .map((c) => '<button class="chip" type="button" data-filter="' + c + '">Chương ' + c + '</button>')
      .join('') +
    '</div>';

  const quizBlocks = quizChapters
    .map((c) => {
      const chapterMeta = ctx.chapters.find((x) => x.num === c);
      const title = 'Chương ' + c + (chapterMeta ? ' — ' + chapterMeta.label : '');
      return (
        '<div class="quiz-wrap" data-quiz-chapter="' + c + '">' +
        renderQuiz(md, quizzes.get(c), 'page', title) +
        '</div>'
      );
    })
    .join('');

  const totalQuestions = [...quizzes.values()].reduce((a, g) => a + g.questions.length, 0);

  // ------------------------------------------------------------------ trang
  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Chương ' + exChapter.num + '</span>' +
    '<span class="kicker-time">' + exercises.length + ' bài · ' + totalQuestions + ' câu trắc nghiệm</span></div>' +
    '<h2 class="heading" id="sec-' + exChapter.num + '">' + escapeHtml(exChapter.title) + '<a class="anchor" href="#sec-' + exChapter.num + '">#</a></h2>' +
    '<p class="chapter-lede">' + escapeHtml(exChapter.summary) + '</p>' +
    '<nav class="ex-jump">' +
    exercises
      .map((e) => '<a href="#bai-' + e.num + '">Bài ' + e.num + '</a>')
      .join('') +
    (hasDrills ? '<a href="#luyen-tinh-tay">Luyện tính tay</a>' : '') +
    '<a href="#trac-nghiem">Trắc nghiệm</a>' +
    '</nav>' +
    '<div class="ex-list">' + cards + '</div>' +

    (hasDrills
      ? '<h3 class="heading" id="luyen-tinh-tay-h">Luyện tính tay' +
        '<a class="anchor" href="#luyen-tinh-tay-h">#</a></h3>' +
        '<p>Mỗi lần bấm <b>Câu khác</b> sẽ sinh một đề mới với số liệu ngẫu nhiên. Mọi phép làm tròn đều theo quy tắc <i>nửa về số chẵn</i> giống NumPy.</p>' +
        drill
      : '') +

    '<h3 class="heading" id="trac-nghiem">Trắc nghiệm theo chương' +
    '<a class="anchor" href="#trac-nghiem">#</a></h3>' +
    '<p>' + totalQuestions + ' câu bám sát nội dung và số liệu của từng chương. Đáp án được xáo lại mỗi lần tải trang.</p>' +
    filter +
    '<div class="quiz-groups">' + quizBlocks + '</div>' +
    '</article>';

  write(
    f('bai-tap.html'),
    page({
      title: 'Bài tập — ' + ctx.docTitle,
      description:
        exercises.length + ' bài tập có lời giải chi tiết và ' + totalQuestions +
        ' câu trắc nghiệm theo chương.',
      body,
      nav,
      file: f('bai-tap.html'),
      ...neighbours(f('bai-tap.html')),
      scripts: [book.drills === false ? 'assets/quiz-filter.js' : 'assets/exercises.js'],
    })
  );

  return quizzes;
}
