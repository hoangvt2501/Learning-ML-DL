// Trang chủ của site: giới thiệu và mục lục cho mọi giáo trình có trong repo.

// Tiêu đề và câu mở đầu sinh ra từ books.json: thêm một giáo trình thì không phải sửa ở đây.
const SO_CHU = ['Không', 'Một', 'Hai', 'Ba', 'Bốn', 'Năm', 'Sáu'];
const demSach = (n) => SO_CHU[n] || String(n);

function lietKe(ten) {
  if (ten.length === 1) return ten[0];
  return ten.slice(0, -1).join(', ') + ' và ' + ten[ten.length - 1];
}

const siteTitle = (books) =>
  demSach(books.length) + ' giáo trình tự học: ' + lietKe(books.map((b) => b.spec.short));

function bookSection(b, escapeHtml) {
  const spec = b.spec;
  const f = (name) => spec.slug + name;
  const byNum = new Map(b.chapters.map((c) => [c.num, c]));

  const pathCards = (spec.paths || [])
    .map((p) => {
      const chips = p.steps
        .map((num) => {
          const c = byNum.get(num);
          if (!c) return '';
          return (
            '<a class="path-chip" href="' + b.fileOf(c) + '"><b>' + num + '</b>' +
            escapeHtml(c.label) + '</a>'
          );
        })
        .join('<span class="path-arrow" aria-hidden="true">→</span>');
      return (
        '<article class="path-card">' +
        '<p class="path-tag">' + escapeHtml(p.tag) + '</p>' +
        '<h4>' + escapeHtml(p.title) + '</h4>' +
        '<p class="path-note">' + escapeHtml(p.note) + '</p>' +
        '<div class="path-steps">' + chips + '</div></article>'
      );
    })
    .join('');

  const chapterCards = b.chapters
    .map((c) => {
      const kicker = c.num === 'PL' ? 'Phụ lục' : 'Chương ' + c.num;
      return (
        '<a class="ch-card" href="' + b.fileOf(c) + '">' +
        '<span class="ch-num">' + (c.num === 'PL' ? '·' : c.num) + '</span>' +
        '<span class="ch-body"><span class="ch-kicker">' + kicker + ' · ' + b.minutes[c.num] +
        ' phút</span><h4>' + escapeHtml(c.label) + '</h4>' +
        '<p>' + escapeHtml(c.summary) + '</p></span></a>'
      );
    })
    .join('');

  const totalMinutes = Object.values(b.minutes).reduce((a, x) => a + x, 0);
  const stats = [
    { n: String(b.chapters.length), l: 'chương' },
    { n: String(b.figures), l: 'hình sinh bằng mã' },
    { n: String(b.quizCount), l: 'câu trắc nghiệm' },
    { n: '~' + Math.round(totalMinutes / 60) + 'h', l: 'thời gian đọc' },
  ].filter((s) => s.n !== '0');

  const tools = b.nav.extras
    .map((e) =>
      '<a class="tool-pill" href="' + e.file + '"><span aria-hidden="true">' + e.icon + '</span>' +
      escapeHtml(e.label) + '</a>')
    .join('');

  return (
    '<section class="book-block" id="' + spec.id + '">' +
    '<header class="book-head">' +
    '<p class="book-kicker">' + escapeHtml(spec.kicker) + '</p>' +
    '<h2><a href="' + b.fileOf(b.chapters[1] || b.chapters[0]) + '">' +
    escapeHtml(b.docTitle) + '</a></h2>' +
    '<p class="book-blurb">' + escapeHtml(spec.blurb) + '</p>' +
    '<dl class="hero-stats">' +
    stats.map((s) => '<div><dt>' + s.n + '</dt><dd>' + escapeHtml(s.l) + '</dd></div>').join('') +
    '</dl>' +
    '<div class="tool-pills">' + tools + '</div>' +
    '</header>' +
    (pathCards ? '<h3 class="home-h3">Nên bắt đầu từ đâu</h3>' +
      '<div class="path-grid">' + pathCards + '</div>' : '') +
    '<h3 class="home-h3">Toàn bộ chương</h3>' +
    '<div class="ch-grid">' + chapterCards + '</div>' +
    '</section>'
  );
}

export function buildHomePage(ctx) {
  const { books, page, escapeHtml, write } = ctx;

  const nav = {
    books: books.map((b) => b.spec),
    book: null,
    groupLabel: 'Giáo trình',
    chapters: books.map((b) => ({
      file: b.fileOf(b.chapters[1] || b.chapters[0]),
      num: b.spec.kicker.replace(/\D+/g, '') || '·',
      label: b.spec.short,
    })),
    // Trên trang chủ, nhãn phải kèm tên sách vì hai sách có cùng loại trang.
    extras: books.flatMap((b) =>
      b.nav.extras
        .filter((e) => /thuc-hanh\.html$/.test(e.file))
        .map((e) => ({ ...e, label: e.label + ' · ' + b.spec.short }))
        .concat([{
          file: b.spec.slug + 'bai-tap.html',
          icon: '✎',
          label: 'Bài tập · ' + b.spec.short,
        }])),
  };

  const jump =
    '<div class="hero-actions">' +
    books
      .map((b, i) =>
        '<a class="btn ' + (i === 0 ? 'btn-primary' : 'btn-ghost') + '" href="#' + b.spec.id + '">' +
        escapeHtml(b.spec.short) + ' →</a>')
      .join('') +
    '</div>';

  const body =
    '<article class="prose home">' +
    '<header class="hero">' +
    '<p class="hero-kicker">Tự học · tiếng Việt · mọi con số đều chạy lại được</p>' +
    '<h1>' + escapeHtml(siteTitle(books)) + '</h1>' +
    '<p class="hero-lede">' + demSach(books.length) + ' giáo trình viết theo cùng một lối: <b>động cơ → định nghĩa → suy luận → ví dụ số → thí nghiệm kiểm chứng</b>. Mỗi chương có phần tự kiểm tra, mỗi bài tập có lời giải đầy đủ, và mọi khẳng định đều truy được về bài báo gốc hoặc về mã chạy lại được.</p>' +
    jump +
    '<p class="home-spine">Bốn giáo trình viết để đọc nối nhau — <a href="lo-trinh.html">xem lộ trình cả bộ</a>, nơi toàn bộ các bài được xếp thành một dãy đánh số liên tục.</p>' +
    '</header>' +
    books.map((b) => bookSection(b, escapeHtml)).join('') +
    '</article>';

  write(
    'index.html',
    page({
      title: siteTitle(books),
      description:
        'Giáo trình tự học bằng tiếng Việt về quantization trong deep learning và về MLOps: lý thuyết, hình sinh bằng mã, bài tập có lời giải và công cụ tương tác.',
      body,
      nav,
      file: 'index.html',
      bodyClass: 'is-home',
    })
  );
}
