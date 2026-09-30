// Khung HTML dùng chung cho mọi trang của site.

const FONTS =
  'https://fonts.googleapis.com/css2?' +
  'family=Be+Vietnam+Pro:wght@400;500;600;700&' +
  'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&' +
  'family=JetBrains+Mono:wght@400;500&display=swap';

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Thanh điều hướng bên trái: dựng từ danh sách chương + các trang phụ. */
function renderSidebar(nav, currentFile) {
  const item = (entry) => {
    const active = entry.file === currentFile ? ' class="is-active" aria-current="page"' : '';
    const num = entry.num != null
      ? '<span class="nav-num">' + entry.num + '</span>'
      : '<span class="nav-num nav-num--icon">' + (entry.icon || '') + '</span>';
    return (
      '<li><a href="' + entry.file + '"' + active + '>' + num +
      '<span class="nav-label">' + escapeHtml(entry.label) + '</span></a></li>'
    );
  };

  // Bộ chuyển giáo trình: chỉ hiện khi site có từ hai giáo trình trở lên.
  const switcher =
    nav.books && nav.books.length > 1
      ? '<div class="book-switch" role="group" aria-label="Chọn giáo trình">' +
        nav.books
          .map((b) => {
            const active = nav.book && b.id === nav.book.id;
            if (active) {
              return '<span class="book-tab is-active" aria-current="true">' +
                escapeHtml(b.short) + '</span>';
            }
            return '<a class="book-tab" href="' + b.slug + 'ch01.html">' +
              escapeHtml(b.short) + '</a>';
          })
          .join('') +
        '</div>'
      : '';

  return (
    '<nav class="sidebar" id="sidebar" aria-label="Mục lục giáo trình">' +
    '<div class="sidebar-inner">' +
    switcher +
    // Bỏ hẳn nhóm rỗng: trang Lộ trình không thuộc giáo trình nào nên không có
    // danh sách chương, và một tiêu đề nhóm trống thì chỉ gây rối.
    (nav.chapters.length ? renderChapterNav(nav, item) : '') +
    (nav.extras.length
      ? '<p class="nav-group">Luyện tập &amp; tra cứu</p><ul class="nav-list">' +
        nav.extras.map(item).join('') + '</ul>'
      : '') +
    '<p class="nav-group">Toàn site</p><ul class="nav-list">' +
    '<li><a href="index.html"><span class="nav-num nav-num--icon">⌂</span>' +
    '<span class="nav-label">Trang chủ</span></a></li>' +
    '<li><a href="lo-trinh.html"><span class="nav-num nav-num--icon">↗</span>' +
    '<span class="nav-label">Lộ trình cả bộ</span></a></li>' +
    '</ul>' +
    '</div></nav>'
  );
}

/**
 * Danh sách chương, chia theo phần. Mỗi giáo trình gồm nhiều phần nhỏ thay vì
 * một danh sách dài phẳng lì; các chương tra cứu (bài tập, phụ lục) không thuộc
 * phần nào nên được gom xuống cuối dưới một nhãn riêng.
 */
function renderChapterNav(nav, item) {
  const khoi = [];
  let phanHienTai = null;
  for (const ch of nav.chapters) {
    const p = ch.part || '';
    if (!khoi.length || p !== phanHienTai) {
      khoi.push({ part: p, items: [] });
      phanHienTai = p;
    }
    khoi[khoi.length - 1].items.push(ch);
  }
  const coPhan = khoi.some((k) => k.part);
  return khoi
    .map((k, i) => {
      const nhan = k.part
        ? '<p class="nav-group nav-group--part">' +
          '<span class="nav-part-no">' + (i + 1) + '</span>' + escapeHtml(k.part) + '</p>'
        : '<p class="nav-group">' +
          escapeHtml(coPhan ? 'Tra cứu trong giáo trình' : (nav.groupLabel || 'Giáo trình')) +
          '</p>';
      return nhan + '<ul class="nav-list">' + k.items.map(item).join('') + '</ul>';
    })
    .join('');
}

function renderPager(prev, next) {
  if (!prev && !next) return '';
  const side = (entry, dir, label) => {
    if (!entry) return '<span></span>';
    return (
      '<a class="pager-link pager-' + dir + '" href="' + entry.file + '">' +
      '<span class="pager-dir">' + label + '</span>' +
      '<span class="pager-title">' + escapeHtml(entry.label) + '</span></a>'
    );
  };
  return (
    '<nav class="pager" aria-label="Chuyển trang">' +
    side(prev, 'prev', '← Trước') +
    side(next, 'next', 'Tiếp →') +
    '</nav>'
  );
}

/**
 * @param {object} o
 * @param {string} o.title      tiêu đề trang (dùng cho <title>)
 * @param {string} o.description mô tả ngắn
 * @param {string} o.body       HTML phần nội dung chính
 * @param {object} o.nav        {chapters, extras}
 * @param {string} o.file       tên file hiện tại
 * @param {object} [o.prev]     mục điều hướng trước
 * @param {object} [o.next]     mục điều hướng sau
 * @param {string} [o.toc]      HTML mục lục bên phải
 * @param {string} [o.bodyClass]
 * @param {string[]} [o.scripts] script bổ sung
 */
export function page(o) {
  const scripts = (o.scripts || []).map((s) => '<script src="' + s + '" defer></script>').join('');
  const tocAside = o.toc
    ? '<aside class="toc" aria-label="Mục lục trang này"><div class="toc-inner">' +
      '<p class="toc-head">Trong trang này</p>' + o.toc + '</div></aside>'
    : '';

  return `<!doctype html>
<html lang="vi" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(o.title)}</title>
<meta name="description" content="${escapeHtml(o.description || '')}">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="${FONTS}">
<link rel="stylesheet" href="assets/katex.min.css">
<link rel="stylesheet" href="assets/style.css">
<script>
// Đặt theme trước khi vẽ để không bị nháy trắng.
(function () {
  try {
    var t = localStorage.getItem('qz-theme');
    if (!t) t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.dataset.theme = t;
  } catch (e) {}
})();
</script>
</head>
<body class="${o.bodyClass || ''}">
<a class="skip-link" href="#main">Bỏ qua điều hướng</a>
<div class="read-progress" id="readProgress" aria-hidden="true"></div>

<header class="topbar">
  <button class="icon-btn menu-btn" id="menuBtn" type="button" aria-label="Mở mục lục" aria-expanded="false">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
  </button>
  <a class="brand" href="index.html">
    <span class="brand-mark" aria-hidden="true">${o.nav.book ? escapeHtml(o.nav.book.short[0]) : 'G'}</span>
    <span class="brand-text"><b>${o.nav.book ? escapeHtml(o.nav.book.short) : 'Giáo trình'}</b><i>${
      o.nav.book ? 'giáo trình tự học'
        : escapeHtml((o.nav.books || []).map((b) => b.short).join(' · '))}</i></span>
  </a>
  <button class="search-btn" id="searchBtn" type="button">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
    <span>Tìm trong giáo trình</span><kbd>/</kbd>
  </button>
  <button class="icon-btn" id="themeBtn" type="button" aria-label="Đổi giao diện sáng/tối">
    <svg class="ic-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/></svg>
    <svg class="ic-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.2A8.2 8.2 0 1 1 9.8 4a6.6 6.6 0 0 0 10.2 10.2z"/></svg>
  </button>
</header>

<div class="shell">
${renderSidebar(o.nav, o.file)}
<div class="scrim" id="scrim" hidden></div>
<main class="content" id="main">
${o.body}
${renderPager(o.prev, o.next)}
</main>
${tocAside}
</div>

<div class="lightbox" id="lightbox" hidden>
  <button class="lightbox-close" type="button" aria-label="Đóng">×</button>
  <img alt="">
  <p class="lightbox-cap"></p>
</div>

<div class="searchpanel" id="searchPanel" hidden>
  <div class="searchpanel-box" role="dialog" aria-modal="true" aria-label="Tìm kiếm">
    <div class="searchpanel-field">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
      <input type="search" id="searchInput" placeholder="Gõ từ khoá: scale, zero-point, GPTQ, SQNR…" autocomplete="off" spellcheck="false">
      <kbd>Esc</kbd>
    </div>
    <div class="searchpanel-results" id="searchResults"></div>
  </div>
</div>

<script src="assets/app.js" defer></script>
<script src="assets/router.js" defer></script>
<script src="assets/preview.js" defer></script>
${scripts}
</body>
</html>
`;
}
