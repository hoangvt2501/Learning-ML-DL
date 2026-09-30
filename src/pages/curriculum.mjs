// Trang Lộ trình: xếp chương của mọi giáo trình thành các chặng theo roadmap
// AI Engineer (roadmap.sh), có thêm phần kiến thức nền mà roadmap giả định người học
// đã có. Số bài chạy liên tục qua mọi chặng; tên tệp giữ nguyên nên không liên kết
// cũ nào bị hỏng. Cấu trúc chặng và bảng đối chiếu nằm ở content/lo-trinh.json.

import { icon } from '../icons.mjs';
import { sectionId } from '../markdown.mjs';

/** Chương tra cứu, không phải bài học. */
const TRA_CUU = /^(Bài tập|Ôn phỏng vấn|Câu hỏi phỏng vấn|Tài liệu tham khảo|Ký hiệu|Kiến thức nền)/i;

export function buildCurriculumPage(ctx) {
  const { books, page, escapeHtml, write, searchIndex, read } = ctx;
  const LT = JSON.parse(read('content/lo-trinh.json'));
  const byId = new Map(books.map((b) => [b.spec.id, b]));

  const chuongCua = (b, num) => b.chapters.find((c) => c.num === String(num));

  /** Một tham chiếu ["sach", "3"] hoặc ["sach", "3.4"] -> liên kết, hoặc null nếu sách chưa có. */
  function thamChieu(ref) {
    const b = byId.get(ref[0]);
    if (!b) return null;
    const [so, con] = ref[1].split('.');
    const ch = chuongCua(b, so);
    if (!ch) return null;
    if (!con) {
      return { href: b.fileOf(ch), nhan: 'Chương ' + so, sach: b.spec.short, ten: ch.label };
    }
    const m = ch.markdown.match(new RegExp('^### ' + so + '\\.' + con + '\\.\\s*(.+)$', 'm'));
    return {
      href: b.fileOf(ch) + '#' + sectionId(ref[1]),
      nhan: 'Mục ' + ref[1], sach: b.spec.short, ten: m ? m[1].replace(/\$[^$]*\$/g, '').trim() : '',
    };
  }

  let soBai = 0, tongPhut = 0, tongHinh = 0;
  const daDung = new Set();
  const chang = [];

  LT.chang.forEach((c, i) => {
    const rows = [];
    const sachCo = new Set();
    for (const [id, tu, den] of c.bai) {
      const b = byId.get(id);
      if (!b) continue;
      sachCo.add(b.spec.short);
      for (const ch of b.chapters) {
        const n = Number(ch.num);
        if (!(n >= Number(tu) && n <= Number(den)) || TRA_CUU.test(ch.label)) continue;
        daDung.add(id + ':' + ch.num);
        soBai++;
        tongPhut += (b.minutes && b.minutes[ch.num]) || 0;
        searchIndex.push({
          t: ch.label + ' — ' + b.spec.short, h: b.fileOf(ch),
          k: 'Lộ trình · Chặng ' + (i + 1), x: (ch.summary || '').slice(0, 300),
        });
        rows.push(
          '<li class="lt-row"><span class="lt-num">' + soBai + '</span>' +
          '<a class="lt-link" href="' + b.fileOf(ch) + '">' +
          '<span class="lt-title">' + escapeHtml(ch.label) + '</span>' +
          '<span class="lt-sub">Chương ' + escapeHtml(ch.num) + ' · ' + escapeHtml(b.spec.short) + '</span>' +
          (ch.summary ? '<span class="lt-sum">' + escapeHtml(ch.summary) + '</span>' : '') +
          '</a></li>');
      }
    }
    if (!rows.length) return;
    const nhan = (c.roadmap || []).map((r) => '<span class="lt-tag">' + escapeHtml(r) + '</span>').join('');
    chang.push({
      id: 'chang-' + (chang.length + 1), ten: c.ten, icon: c.icon,
      html:
        '<section class="lt-chang" id="chang-' + (chang.length + 1) + '">' +
        '<header class="lt-chang-head">' +
        '<span class="lt-chang-icon" aria-hidden="true">' + icon(c.icon) + '</span>' +
        '<div class="lt-chang-text">' +
        '<p class="lt-phan-kicker">Chặng ' + (chang.length + 1) + ' · ' + escapeHtml([...sachCo].join(', ')) + '</p>' +
        '<h2>' + escapeHtml(c.ten) + '</h2>' +
        '<p class="lt-phan-blurb">' + escapeHtml(c.mo_ta) + '</p>' +
        (nhan ? '<p class="lt-tags"><span class="lt-tags-lead">Roadmap:</span>' + nhan + '</p>' : '') +
        '</div></header>' +
        '<ol class="lt-list">' + rows.join('') + '</ol></section>',
    });
  });

  for (const b of books) tongHinh += b.figures;

  // Dải chặng ở đầu trang: nhìn một lượt cả lộ trình
  const dai =
    '<nav class="lt-strip" aria-label="Các chặng">' +
    chang.map((c, i) =>
      '<a class="lt-strip-item" href="#' + c.id + '">' +
      '<span class="lt-strip-icon" aria-hidden="true">' + icon(c.icon) + '</span>' +
      '<span class="lt-strip-no">' + (i + 1) + '</span>' +
      '<span class="lt-strip-name">' + escapeHtml(c.ten) + '</span></a>').join('') +
    '</nav>';

  // Bảng đối chiếu với roadmap AI Engineer
  const dongDC = LT.doi_chieu.map((d) => {
    const lk = d.ref.map(thamChieu).filter(Boolean);
    if (!lk.length) return '';
    return '<tr><td class="lt-map-node">' + escapeHtml(d.node) + '</td>' +
      '<td>' + escapeHtml(d.vi) + '</td><td class="lt-map-refs">' +
      lk.map((l) => '<a class="lt-ref" href="' + l.href + '" title="' + escapeHtml(l.ten) + '">' +
        escapeHtml(l.nhan) + ' <span>' + escapeHtml(l.sach) + '</span></a>').join('') +
      '</td></tr>';
  }).join('');

  // Tra cứu: chương không nằm trong mạch bài học của mỗi giáo trình
  const traCuu = books.map((b) => {
    const ds = b.chapters.filter((ch) => !daDung.has(b.spec.id + ':' + ch.num));
    if (!ds.length) return '';
    return '<div class="lt-aux-book"><p class="lt-aux-head">' + escapeHtml(b.spec.short) + '</p><ul>' +
      ds.map((ch) => '<li><a href="' + b.fileOf(ch) + '">' +
        (ch.num === 'PL' ? '' : '<span class="lt-aux-no">' + escapeHtml(ch.num) + '</span>') +
        escapeHtml(ch.label) + '</a></li>').join('') + '</ul></div>';
  }).join('');

  const body =
    '<article class="prose lt-page">' +
    '<header class="hero hero--tight">' +
    '<p class="hero-kicker">Lộ trình kỹ sư AI</p>' +
    '<h1>Lộ trình</h1>' +
    '<p class="hero-lede">Lộ trình đi theo các chặng của roadmap <b>AI Engineer</b> trên roadmap.sh, ' +
    'bắt đầu bằng phần kiến thức nền mà roadmap đó giả định người học đã có: toán, học máy cổ điển, ' +
    'học sâu và cách mô hình ngôn ngữ được huấn luyện. ' + books.length + ' giáo trình được xếp thành ' +
    chang.length + ' chặng với ' + soBai + ' bài đánh số liên tục.</p>' +
    '<p class="hero-lede">Thứ tự này ít phải quay lại nhất, nhưng không bắt buộc: mỗi bài có liên kết tới ' +
    'đúng mục kiến thức cũ mà nó cần. Bảng đối chiếu ở cuối trang cho biết mỗi chủ đề của roadmap được ' +
    'trình bày ở đâu.</p>' +
    '<p class="lt-tong">' + chang.length + ' chặng · ' + soBai + ' bài · ' + tongHinh +
    ' hình sinh bằng mã · khoảng ' + Math.round(tongPhut / 60) + ' giờ đọc</p>' +
    '</header>' +
    dai +
    '<figure class="figure lt-figure">' +
    '<img src="figs/nt13_mach.png" alt="Sáu giáo trình của lộ trình và thứ tự đọc" loading="lazy">' +
    '<figcaption>Sáu giáo trình và ba nhóm lớn: kiến thức nền, mô hình ngôn ngữ và ứng dụng, tối ưu và vận hành.</figcaption>' +
    '</figure>' +
    chang.map((c) => c.html).join('') +
    '<section class="lt-map" id="doi-chieu-roadmap">' +
    '<h2 class="lt-map-title">' + icon('lo-trinh') + '<span>Đối chiếu với roadmap AI Engineer</span></h2>' +
    '<p class="lt-phan-blurb">Mỗi chủ đề lớn của roadmap và nơi nó được trình bày trong lộ trình.</p>' +
    '<div class="table-wrap"><table><thead><tr><th>Chủ đề trên roadmap</th><th>Nội dung</th>' +
    '<th>Ở đâu</th></tr></thead><tbody>' + dongDC + '</tbody></table></div></section>' +
    '<section class="lt-aux" id="tra-cuu"><h2 class="lt-map-title">' + icon('tai-lieu') +
    '<span>Tra cứu trong từng giáo trình</span></h2>' +
    '<p class="lt-phan-blurb">Ký hiệu, bài tập, câu hỏi phỏng vấn, tài liệu tham khảo và phụ lục.</p>' +
    '<div class="lt-aux-grid">' + traCuu + '</div></section>' +
    '</article>';

  write('lo-trinh.html', page({
    title: 'Lộ trình kỹ sư AI',
    description: 'Lộ trình theo roadmap AI Engineer: ' + chang.length + ' chặng, ' + soBai +
      ' bài từ toán và học máy cổ điển tới RAG, agent, MCP, đánh giá và vận hành.',
    body,
    nav: { books: books.map((b) => b.spec), chapters: [], extras: [] },
    file: 'lo-trinh.html',
    bodyClass: 'is-home',
  }));

  searchIndex.push({
    t: 'Lộ trình kỹ sư AI', h: 'lo-trinh.html', k: 'Toàn site',
    x: 'Các chặng theo roadmap AI Engineer, từ toán và học máy cổ điển tới RAG, agent, MCP, ' +
       'đánh giá, lượng tử hoá và vận hành.',
  });
  return { soChang: chang.length, soBai };
}
