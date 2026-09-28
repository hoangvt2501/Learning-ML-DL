// Thư viện toàn bộ hình minh hoạ, gom từ sổ tra cứu dựng lúc build.

export function buildFiguresPage(ctx) {
  const { md, nav, page, escapeHtml, write, neighbours, registry, chapters } = ctx;
  registry.currentFile = 'hinh-anh.html';

  const labelOf = (numChapter) => {
    const c = chapters.find((x) => x.num === numChapter);
    return c ? c.label : '';
  };

  const figures = [...registry.figures.entries()]
    .map(([num, f]) => ({ num: +num, ...f }))
    .sort((a, b) => a.num - b.num);

  const cards = figures
    .map((f) => {
      const chapterLabel = labelOf(f.chapter);
      return (
        '<figure class="gal-card" id="hinh-' + f.num + '">' +
        '<div class="figure-frame" role="button" tabindex="0" data-zoom aria-label="Phóng to Hình ' +
        f.num + '">' +
        '<img src="' + f.src + '" alt="Hình ' + f.num +
        '" loading="lazy" decoding="async"></div>' +
        '<figcaption>' +
        '<p class="gal-num">Hình ' + f.num + '</p>' +
        '<p class="gal-cap">' + md.renderInline(f.caption) + '</p>' +
        '<a class="gal-link" href="' + f.href + '">Chương ' + f.chapter +
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
    ' hình của giáo trình. Trừ Hình 1 dùng số liệu đã công bố của Horowitz (ISSCC 2014), mọi hình còn lại đều do mã nguồn trong <a href="ma-nguon.html">thư mục <code>code/</code></a> sinh ra, chạy thật và chạy lại được. Bấm vào ảnh để phóng to.</p>' +
    '<div class="gal-grid">' + cards + '</div>' +
    '</article>';

  write(
    'hinh-anh.html',
    page({
      title: 'Thư viện hình minh hoạ — Quantization trong Deep Learning',
      description: 'Toàn bộ ' + figures.length + ' hình minh hoạ của giáo trình, kèm chú thích và liên kết về đúng mục đã dùng.',
      body,
      nav,
      file: 'hinh-anh.html',
      ...neighbours('hinh-anh.html'),
    })
  );
}
