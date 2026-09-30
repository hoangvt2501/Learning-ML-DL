// Trang Lộ trình: gom toàn bộ chương của mọi giáo trình thành MỘT mạch đánh số
// liên tục. Đây là câu trả lời cho việc bốn giáo trình đứng cạnh nhau thì dễ đọc
// như bốn thứ rời rạc — số bài chạy suốt từ đầu tới cuối, còn tên tệp giữ nguyên
// nên không liên kết cũ nào bị hỏng.

/** Các chương "phụ" không nằm trong mạch đọc chính. */
const NGOAI_MACH = new Set(['PL']);

/** Chương nào là phần tra cứu chứ không phải bài học. */
function laPhuLuc(ch) {
  return NGOAI_MACH.has(ch.num) ||
    /^(Bài tập|Ôn phỏng vấn|Tài liệu tham khảo)$/i.test(ch.label.trim());
}

export function buildCurriculumPage(ctx) {
  const { books, page, escapeHtml, write, searchIndex } = ctx;

  let soBai = 0;
  const phanHtml = [];
  const laMa = ['I', 'II', 'III', 'IV', 'V', 'VI'];

  let tongBai = 0, tongHinh = 0, tongPhut = 0;

  books.forEach((b, i) => {
    const spec = b.spec;
    const f = (name) => spec.slug + name;
    const rows = [];

    for (const ch of b.chapters) {
      const phuLuc = laPhuLuc(ch);
      const file = ch.kind === 'exercises' ? f('bai-tap.html') : ch.file;
      const num = phuLuc
        ? '<span class="lt-num lt-num--aux">' + escapeHtml(ch.num) + '</span>'
        : '<span class="lt-num">' + (++soBai) + '</span>';
      if (!phuLuc) tongBai++;

      rows.push(
        '<li class="lt-row' + (phuLuc ? ' lt-row--aux' : '') + '">' +
        num +
        '<a class="lt-link" href="' + file + '">' +
        '<span class="lt-title">' + escapeHtml(ch.label) + '</span>' +
        '<span class="lt-sub">Chương ' + escapeHtml(ch.num) + ' · ' +
        escapeHtml(spec.short) + '</span>' +
        (ch.summary ? '<span class="lt-sum">' + escapeHtml(ch.summary) + '</span>' : '') +
        '</a></li>'
      );

      searchIndex.push({
        t: ch.label + ' — ' + spec.short,
        h: file,
        k: 'Lộ trình · Phần ' + laMa[i],
        x: (ch.summary || '').slice(0, 300),
      });
    }

    tongHinh += b.figures;
    // b.minutes là một đối tượng {số chương: số phút}, không phải một con số.
    tongPhut += Object.values(b.minutes || {}).reduce((x, y) => x + y, 0);

    phanHtml.push(
      '<section class="lt-phan" id="phan-' + (i + 1) + '">' +
      '<header class="lt-phan-head">' +
      '<p class="lt-phan-kicker">Phần ' + laMa[i] + '</p>' +
      '<h2><a href="' + f('ch00.html') + '">' + escapeHtml(spec.short) + '</a></h2>' +
      '<p class="lt-phan-blurb">' + escapeHtml(spec.blurb || '') + '</p>' +
      '<p class="lt-phan-meta">' + b.chapters.length + ' chương · ' +
      b.figures + ' hình sinh bằng mã · ' + b.quizCount + ' câu trắc nghiệm</p>' +
      '</header>' +
      '<ol class="lt-list">' + rows.join('') + '</ol>' +
      '</section>'
    );
  });

  const lede =
    '<article class="prose lt-page">' +
    '<header class="hero hero--tight">' +
    '<p class="hero-kicker">Một mạch từ đầu tới cuối</p>' +
    '<h1>Lộ trình</h1>' +
    '<p class="hero-lede">Bốn giáo trình trong repo này được viết để đọc nối nhau. ' +
    'Trang này xếp toàn bộ <b>' + tongBai + ' bài</b> của chúng thành một dãy đánh số ' +
    'liên tục, để thấy rõ cái gì dẫn tới cái gì. Các trang bài tập, ôn phỏng vấn và ' +
    'tài liệu tham khảo được để riêng vì chúng là phần tra cứu chứ không phải bài học.</p>' +
    '<p class="hero-lede">Thứ tự này là thứ tự <b>ít phải quay lại nhất</b>, không phải ' +
    'thứ tự bắt buộc. Mỗi phần tự đứng vững được, và mỗi giáo trình có trang chủ riêng ' +
    'với các lối đi ngắn hơn tuỳ mục đích.</p>' +
    '<p class="lt-tong">' + tongBai + ' bài · ' + tongHinh +
    ' hình sinh bằng mã · khoảng ' + Math.round(tongPhut / 60) + ' giờ đọc</p>' +
    '</header>' +
    '<figure class="figure lt-figure">' +
    '<img src="figs/nt13_mach.png" alt="Bản đồ bốn giáo trình" loading="lazy">' +
    '<figcaption>Bốn giáo trình và mối nối giữa chúng. Mũi tên là thứ tự đề nghị.</figcaption>' +
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
