// Trang chủ: giới thiệu, lộ trình học, và lưới thẻ 17 chương.

const PATHS = [
  {
    tag: 'Lần đầu tiếp xúc',
    title: 'Hiểu bản chất trước đã',
    note: 'Bốn chương liền mạch, đọc hết là nắm trọn phần lõi toán học.',
    steps: ['1', '2', '3', '4'],
  },
  {
    tag: 'Triển khai CNN trên thiết bị',
    title: 'Từ mô hình FP32 tới file INT8',
    note: 'Đi theo đúng thứ tự một quy trình thật: chuẩn bị, calibrate, đo, sửa.',
    steps: ['3', '5', '6', '7', '8', '9', '10', '12'],
  },
  {
    tag: 'Chỉ quan tâm LLM',
    title: 'Vì sao LLM cần kỹ thuật riêng',
    note: 'Nắm phần lõi vừa đủ rồi rẽ thẳng sang GPTQ, AWQ, NF4 và KV cache.',
    steps: ['1', '3', '5', '11'],
  },
];

export function buildHomePage(ctx) {
  const { md, nav, page, escapeHtml, write, chapters, minutes, docTitle, intro, neighbours } = ctx;

  const byNum = new Map(chapters.map((c) => [c.num, c]));
  const fileOf = (c) => (c.num === '14' ? 'bai-tap.html' : c.file);

  const stats = [
    { n: '15', l: 'chương' },
    { n: '17', l: 'hình sinh bằng mã' },
    { n: '9', l: 'bài tập có lời giải' },
    { n: '42', l: 'câu trắc nghiệm' },
    { n: '5', l: 'công cụ tương tác' },
  ];

  const pathCards = PATHS.map((p) => {
    const chips = p.steps
      .map((num) => {
        const c = byNum.get(num);
        return (
          '<a class="path-chip" href="' + fileOf(c) + '"><b>' + num + '</b>' +
          escapeHtml(c.label) + '</a>'
        );
      })
      .join('<span class="path-arrow" aria-hidden="true">→</span>');
    return (
      '<article class="path-card">' +
      '<p class="path-tag">' + escapeHtml(p.tag) + '</p>' +
      '<h3>' + escapeHtml(p.title) + '</h3>' +
      '<p class="path-note">' + escapeHtml(p.note) + '</p>' +
      '<div class="path-steps">' + chips + '</div>' +
      '</article>'
    );
  }).join('');

  const chapterCards = chapters
    .map((c) => {
      const kicker = c.num === 'PL' ? 'Phụ lục' : 'Chương ' + c.num;
      return (
        '<a class="ch-card" href="' + fileOf(c) + '">' +
        '<span class="ch-num">' + (c.num === 'PL' ? '·' : c.num) + '</span>' +
        '<span class="ch-body"><span class="ch-kicker">' + kicker + ' · ' + minutes[c.num] +
        ' phút</span><h3>' + escapeHtml(c.label) + '</h3>' +
        '<p>' + escapeHtml(c.summary) + '</p></span></a>'
      );
    })
    .join('');

  const tools = [
    {
      href: 'thuc-hanh.html',
      icon: '⚙',
      title: 'Phòng thí nghiệm',
      note: 'Năm công cụ chạy ngay trên trình duyệt: máy lượng tử affine, đánh đổi làm tròn–cắt, soi bit số thực, requantization, tính dung lượng.',
    },
    {
      href: 'bai-tap.html',
      icon: '✎',
      title: 'Bài tập và trắc nghiệm',
      note: 'Chín bài của giáo trình kèm lời giải đầy đủ, 42 câu trắc nghiệm theo chương và các bài luyện tính tay có tự chấm.',
    },
    {
      href: 'hinh-anh.html',
      icon: '◳',
      title: 'Thư viện 17 hình',
      note: 'Toàn bộ hình minh hoạ ở một chỗ, kèm chú thích và liên kết về đúng mục đã dùng nó.',
    },
    {
      href: 'thuat-ngu.html',
      icon: '¶',
      title: 'Từ điển thuật ngữ',
      note: 'Hơn 60 thuật ngữ đối chiếu Việt – Anh, có định nghĩa ngắn và tìm kiếm tại chỗ.',
    },
    {
      href: 'ma-nguon.html',
      icon: '{}',
      title: 'Mã nguồn thí nghiệm',
      note: 'Bảy script sinh ra mọi hình và mọi con số trong tài liệu, kèm kết quả in ra của từng script.',
    },
    {
      href: 'toan-van.html',
      icon: '≡',
      title: 'Toàn văn một trang',
      note: 'Cả giáo trình trên một trang — tiện Ctrl+F, đọc ngoại tuyến hoặc in ra giấy.',
    },
  ]
    .map(
      (t) =>
        '<a class="tool-card" href="' + t.href + '">' +
        '<span class="tool-icon" aria-hidden="true">' + t.icon + '</span>' +
        '<h3>' + escapeHtml(t.title) + '</h3><p>' + escapeHtml(t.note) + '</p></a>'
    )
    .join('');

  const body =
    '<article class="prose home">' +
    '<header class="hero">' +
    '<p class="hero-kicker">Giáo trình tự học · tiếng Việt</p>' +
    '<h1>' + escapeHtml(docTitle) + '</h1>' +
    '<p class="hero-lede">Từ công thức <b>S</b> và <b>Z</b> cho tới GPTQ, NF4 và KV cache — viết theo lối bài giảng, có hình vẽ sinh bằng mã, bài tập kèm lời giải và công cụ tự thử ngay trên trang.</p>' +
    '<div class="hero-actions">' +
    '<a class="btn btn-primary" href="ch01.html">Bắt đầu từ Chương 1 →</a>' +
    '<a class="btn btn-ghost" href="thuc-hanh.html">Thử công cụ tương tác</a>' +
    '</div>' +
    '<dl class="hero-stats">' +
    stats.map((s) => '<div><dt>' + s.n + '</dt><dd>' + escapeHtml(s.l) + '</dd></div>').join('') +
    '</dl>' +
    '</header>' +

    '<section class="home-section"><h2 class="home-h2">Về độ tin cậy của tài liệu</h2>' +
    '<div class="callout callout-note">' + md.render(intro.replace(/^>\s?/gm, '')) + '</div>' +
    '</section>' +

    '<section class="home-section"><h2 class="home-h2">Nên bắt đầu từ đâu</h2>' +
    '<p class="home-sub">Ba lộ trình tuỳ theo việc bạn đang cần làm. Chương 0 luôn nên xem lướt trước để quen ký hiệu.</p>' +
    '<div class="path-grid">' + pathCards + '</div>' +
    '</section>' +

    '<section class="home-section"><h2 class="home-h2">Công cụ đi kèm</h2>' +
    '<div class="tool-grid">' + tools + '</div>' +
    '</section>' +

    '<section class="home-section"><h2 class="home-h2">Toàn bộ chương</h2>' +
    '<div class="ch-grid">' + chapterCards + '</div>' +
    '</section>' +
    '</article>';

  write(
    'index.html',
    page({
      title: docTitle,
      description:
        'Giáo trình tự học quantization trong deep learning bằng tiếng Việt: 15 chương, 17 hình sinh bằng mã, bài tập có lời giải và công cụ tương tác.',
      body,
      nav,
      file: 'index.html',
      ...neighbours('index.html'),
      bodyClass: 'is-home',
    })
  );
}
