// Thư viện toàn bộ hình minh hoạ, gom từ sổ tra cứu dựng lúc build.

export function buildFiguresPage(ctx) {
  const { md, nav, page, escapeHtml, write, neighbours, registry, chapters, f, book } = ctx;
  registry.currentFile = f('hinh-anh.html');

  const labelOf = (numChapter) => {
    const c = chapters.find((x) => x.num === numChapter);
    return c ? c.label : '';
  };

  const figures = [...registry.figures.entries()]
    .map(([num, fg]) => ({ num: +num, ...fg }))
    .sort((a, b) => a.num - b.num);

  const cards = figures
    .map((fg) => {
      const chapterLabel = labelOf(fg.chapter);
      return (
        '<figure class="gal-card" id="hinh-' + fg.num + '">' +
        '<div class="figure-frame" role="button" tabindex="0" data-zoom aria-label="Phóng to Hình ' +
        fg.num + '">' +
        '<img src="' + fg.src + '" alt="Hình ' + fg.num +
        '" loading="lazy" decoding="async"></div>' +
        '<figcaption>' +
        '<p class="gal-num">Hình ' + fg.num + '</p>' +
        '<p class="gal-cap">' + md.renderInline(fg.caption) + '</p>' +
        '<a class="gal-link" href="' + fg.href + '">Chương ' + fg.chapter +
        (chapterLabel ? ' · ' + escapeHtml(chapterLabel) : '') + ' →</a>' +
        '</figcaption></figure>'
      );
    })
    .join('');

  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Tra cứu</span>' +
    '<span class="kicker-time">' + figures.length + ' hình</span></div>' +
    '<h1>Thư viện hình minh hoạ</h1>' +
    '<p class="chapter-lede">Toàn bộ ' + figures.length +
    ' hình của giáo trình. ' + escapeHtml(book.figuresNote || '') + ' Xem <a href="' + f('ma-nguon.html') + '">mã nguồn sinh hình</a>. Bấm vào ảnh để phóng to.</p>' +
    '<div class="gal-grid">' + cards + '</div>' +
    '</article>';

  write(
    f('hinh-anh.html'),
    page({
      title: 'Thư viện hình minh hoạ — ' + ctx.docTitle,
      description: 'Toàn bộ ' + figures.length + ' hình minh hoạ của giáo trình, kèm chú thích và liên kết về đúng mục đã dùng.',
      body,
      nav,
      file: f('hinh-anh.html'),
      ...neighbours(f('hinh-anh.html')),
    })
  );
}
