// Trang Lộ trình: gom toàn bộ chương của mọi giáo trình thành MỘT mạch đánh số
// liên tục. Đây là câu trả lời cho việc bốn giáo trình đứng cạnh nhau thì dễ đọc
// như bốn thứ rời rạc — số bài chạy suốt từ đầu tới cuối, còn tên tệp giữ nguyên
// nên không liên kết cũ nào bị hỏng.

/** Các chương "phụ" không nằm trong mạch đọc chính. */
const NGOAI_MACH = new Set(['PL']);

/** Chương nào là phần tra cứu chứ không phải bài học. */
function laPhuLuc(ch) {
  return NGOAI_MACH.has(ch.num) ||
    /^(Bài tập|Ôn phỏng vấn|Câu hỏi phỏng vấn|Tài liệu tham khảo)$/i.test(ch.label.trim());
}

const SO_CHU_LT = ['Không', 'Một', 'Hai', 'Ba', 'Bốn', 'Năm', 'Sáu', 'Bảy'];
const demChu = (n) => SO_CHU_LT[n] || String(n);

export function buildCurriculumPage(ctx) {
  const { books, page, escapeHtml, write, searchIndex } = ctx;

  let soBai = 0;
  let soPhan = 0;
  const phanHtml = [];
  let tongBai = 0, tongHinh = 0, tongPhut = 0;

  // Gom chương thành các PHẦN. Một phần là một dãy chương liên tiếp trong cùng
  // giáo trình, khai báo ở trường "part" của tệp *-chapters.json. Các chương tra
  // cứu (bài tập, ôn phỏng vấn, tài liệu, phụ lục) không thuộc phần nào và được
  // gom riêng ở cuối mỗi giáo trình.
  for (const b of books) {
    const spec = b.spec;
    const f = (name) => spec.slug + name;
    const khoi = [];
    let cur = null;

    for (const ch of b.chapters) {
      const key = laPhuLuc(ch) ? '\u0000tracuu' : (ch.part || '\u0000khac');
      if (!cur || cur.key !== key) {
        cur = { key, part: laPhuLuc(ch) ? '' : (ch.part || ''), items: [] };
        khoi.push(cur);
      }
      cur.items.push(ch);
    }

    tongHinh += b.figures;
    tongPhut += Object.values(b.minutes || {}).reduce((x, y) => x + y, 0);

    for (const k of khoi) {
      const laTraCuu = k.key === '\u0000tracuu';
      const rows = k.items.map((ch) => {
        const file = ch.kind === 'exercises' ? f('bai-tap.html') : ch.file;
        const num = laTraCuu
          ? '<span class="lt-num lt-num--aux">' + escapeHtml(ch.num) + '</span>'
          : '<span class="lt-num">' + (++soBai) + '</span>';
        if (!laTraCuu) tongBai++;

        searchIndex.push({
          t: ch.label + ' — ' + spec.short,
          h: file,
          k: 'Lộ trình · ' + (k.part || spec.short),
          x: (ch.summary || '').slice(0, 300),
        });

        return (
          '<li class="lt-row' + (laTraCuu ? ' lt-row--aux' : '') + '">' +
          num +
          '<a class="lt-link" href="' + file + '">' +
          '<span class="lt-title">' + escapeHtml(ch.label) + '</span>' +
          '<span class="lt-sub">Chương ' + escapeHtml(ch.num) + ' · ' +
          escapeHtml(spec.short) + '</span>' +
          (ch.summary ? '<span class="lt-sum">' + escapeHtml(ch.summary) + '</span>' : '') +
          '</a></li>'
        );
      });

      if (laTraCuu) {
        phanHtml.push(
          '<section class="lt-phan lt-phan--aux">' +
          '<p class="lt-aux-head">Tra cứu trong ' + escapeHtml(spec.short) + '</p>' +
          '<ol class="lt-list">' + rows.join('') + '</ol>' +
          '</section>'
        );
      } else {
        soPhan++;
        phanHtml.push(
          '<section class="lt-phan" id="phan-' + soPhan + '">' +
          '<header class="lt-phan-head">' +
          '<p class="lt-phan-kicker">Phần ' + soPhan + ' · ' + escapeHtml(spec.short) + '</p>' +
          '<h2>' + escapeHtml(k.part) + '</h2>' +
          '<p class="lt-phan-meta">' + k.items.length + ' chương</p>' +
          '</header>' +
          '<ol class="lt-list">' + rows.join('') + '</ol>' +
          '</section>'
        );
      }
    }
  }

  const lede =
    '<article class="prose lt-page">' +
    '<header class="hero hero--tight">' +
    '<p class="hero-kicker">Một mạch từ đầu tới cuối</p>' +
    '<h1>Lộ trình</h1>' +
    '<p class="hero-lede">' + demChu(books.length) + ' giáo trình trong repo này được viết để đọc nối nhau. ' +
    'Trang này xếp toàn bộ <b>' + tongBai + ' bài</b> của chúng thành <b>' + soPhan + ' phần</b> nhỏ, đánh số liên tục, để thấy rõ cái gì dẫn tới cái gì. Các trang bài tập, ôn phỏng vấn và tài liệu tham khảo được để riêng vì chúng là phần tra cứu chứ không phải bài học.</p>' +
    '<p class="hero-lede">Thứ tự này là thứ tự <b>ít phải quay lại nhất</b>, không phải ' +
    'thứ tự bắt buộc. Mỗi phần tự đứng vững được, và mỗi giáo trình có trang chủ riêng ' +
    'với các lối đi ngắn hơn tuỳ mục đích.</p>' +
    '<p class="lt-tong">' + soPhan + ' phần · ' + tongBai + ' bài · ' + tongHinh +
    ' hình sinh bằng mã · khoảng ' + Math.round(tongPhut / 60) + ' giờ đọc</p>' +
    '</header>' +
    '<figure class="figure lt-figure">' +
    '<img src="figs/nt13_mach.png" alt="Bản đồ bốn giáo trình" loading="lazy">' +
    '<figcaption>' + demChu(books.length) + ' giáo trình và mối nối giữa chúng. Mũi tên là thứ tự đề nghị.</figcaption>' +
    '</figure>' +
    phanHtml.join('') +
    '</article>';

  write(
    'lo-trinh.html',
    page({
      title: 'Lộ trình — bốn giáo trình đọc như một mạch',
      description: 'Toàn bộ ' + tongBai + ' bài của bốn giáo trình, xếp thành một dãy ' +
        'đánh số liên tục từ nền tảng tới vận hành.',
      body: lede,
      nav: { books: books.map((b) => b.spec), chapters: [], extras: [] },
      file: 'lo-trinh.html',
      bodyClass: 'is-home',
    })
  );

  searchIndex.push({
    t: 'Lộ trình — bốn giáo trình đọc như một mạch',
    h: 'lo-trinh.html',
    k: 'Toàn site',
    x: 'Toàn bộ ' + tongBai + ' bài xếp thành một dãy đánh số liên tục, từ nền tảng ' +
       'toán học tới vận hành mô hình trong sản xuất.',
  });
}
