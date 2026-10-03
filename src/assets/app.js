/* Tương tác dùng chung cho mọi trang: giao diện sáng/tối, phông chữ, mục lục,
   thu gọn phần và tiến độ "đã học" ở thanh bên, tìm kiếm, phóng to hình, sao
   chép mã, thanh tiến độ đọc, thời gian đọc còn lại và phần trắc nghiệm. */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ------------------------------------------------------------ theme */
  var themeBtn = $('#themeBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('qz-theme', next); } catch (e) {}
    });
  }

  /* ------------------------------------------------------- phông chữ */
  // Bài đọc mặc định dùng phông có chân; nút này đổi sang phông không chân của
  // giao diện và nhớ lựa chọn. Thuộc tính data-font được đặt sẵn trong <head>.
  var fontBtn = $('#fontBtn');
  function capNhatNutPhong() {
    if (fontBtn) fontBtn.setAttribute('aria-pressed', String(document.documentElement.dataset.font === 'sans'));
  }
  if (fontBtn) {
    capNhatNutPhong();
    fontBtn.addEventListener('click', function () {
      var sans = document.documentElement.dataset.font !== 'sans';
      if (sans) document.documentElement.dataset.font = 'sans';
      else delete document.documentElement.dataset.font;
      try { localStorage.setItem('qz-font', sans ? 'sans' : 'serif'); } catch (e) {}
      capNhatNutPhong();
    });
  }

  /* Đọc/ghi một đối tượng JSON trong localStorage; trình duyệt chặn thì bỏ qua. */
  function docKho(khoa) {
    try { return JSON.parse(localStorage.getItem(khoa)) || {}; } catch (e) { return {}; }
  }
  function ghiKho(khoa, giaTri) {
    try { localStorage.setItem(khoa, JSON.stringify(giaTri)); } catch (e) {}
  }

  /* ---------------------------------------------------------- sidebar */
  var sidebar = $('#sidebar');
  var menuBtn = $('#menuBtn');
  var scrim = $('#scrim');
  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove('is-open');
    if (scrim) scrim.hidden = true;
    if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
  }
  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', function () {
      var open = sidebar.classList.toggle('is-open');
      if (scrim) scrim.hidden = !open;
      menuBtn.setAttribute('aria-expanded', String(open));
    });
  }
  if (scrim) scrim.addEventListener('click', closeSidebar);

  // Giữ mục đang đọc trong tầm nhìn của thanh bên.
  function cuonToiMucDangDoc() {
    var cur = $('.nav-list a.is-active');
    var sb = $('#sidebar');
    if (!cur || !sb) return;
    var r = cur.getBoundingClientRect();
    var sr = sb.getBoundingClientRect();
    if (r.top < sr.top || r.bottom > sr.bottom) cur.scrollIntoView({ block: 'center' });
  }

  /* ------------------------------------------ thu gọn phần ở thanh bên */
  // Mặc định chỉ mở phần chứa chương đang đọc. Phần nào người đọc tự mở thì được
  // nhớ lại, khoá theo đường dẫn chương đầu tiên của phần (duy nhất trên site).
  var KHOA_PHAN = 'qz-phan-mo';

  function khoaPhan(btn) {
    var a = btn.nextElementSibling && btn.nextElementSibling.querySelector('a');
    return a ? a.getAttribute('href') : '';
  }
  function datPhan(btn, mo) {
    btn.setAttribute('aria-expanded', String(mo));
    if (btn.nextElementSibling) btn.nextElementSibling.hidden = !mo;
  }
  function khoiTaoPhan() {
    var daMo = docKho(KHOA_PHAN);
    $$('[data-part-toggle]').forEach(function (btn) {
      var dangDoc = btn.nextElementSibling && btn.nextElementSibling.querySelector('a.is-active');
      datPhan(btn, Boolean(dangDoc) || Boolean(daMo[khoaPhan(btn)]));
    });
  }
  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('[data-part-toggle]');
    if (!btn) return;
    var mo = btn.getAttribute('aria-expanded') !== 'true';
    datPhan(btn, mo);
    var daMo = docKho(KHOA_PHAN);
    if (mo) daMo[khoaPhan(btn)] = 1;
    else delete daMo[khoaPhan(btn)];
    ghiKho(KHOA_PHAN, daMo);
  });

  /* ------------------------------------------------ đánh dấu đã học */
  // Chương được đánh dấu khi người đọc cuộn tới phần tự kiểm tra ở cuối, hoặc khi
  // bấm nút ở cuối chương. Chỉ tính các chương thuộc một phần, không tính bài tập,
  // câu hỏi phỏng vấn, tài liệu tham khảo và phụ lục. Dữ liệu chỉ nằm trong
  // trình duyệt này.
  var KHOA_HOC = 'qz-da-hoc';
  var daHoc = docKho(KHOA_HOC);

  function chuongDangDoc() {
    var a = $('[data-part-list] a.is-active');
    return a && $('#main .chapter-kicker') ? a.getAttribute('href') : '';
  }
  function datDaHoc(file, co) {
    daHoc = docKho(KHOA_HOC);
    if (co) daHoc[file] = Date.now();
    else delete daHoc[file];
    ghiKho(KHOA_HOC, daHoc);
    veTienDo();
  }
  function veTienDo() {
    var links = $$('[data-part-list] a');
    var xong = 0;
    links.forEach(function (a) {
      var r = Boolean(daHoc[a.getAttribute('href')]);
      a.classList.toggle('is-read', r);
      if (r) xong++;
    });
    $$('[data-part-toggle]').forEach(function (btn) {
      var ls = btn.nextElementSibling ? $$('a', btn.nextElementSibling) : [];
      var n = ls.filter(function (a) { return a.classList.contains('is-read'); }).length;
      var c = $('[data-part-count]', btn);
      if (!c) return;
      c.textContent = n ? n + '/' + ls.length : '';
      c.classList.toggle('is-full', n > 0 && n === ls.length);
    });
    var box = $('[data-nav-progress]');
    if (box && links.length) {
      box.hidden = false;
      box.innerHTML =
        '<div class="nav-progress-row"><span>Đã học <b>' + xong + '</b>/' + links.length + ' chương</span>' +
        (xong ? '<button type="button" class="nav-progress-reset" data-progress-reset>Xoá tiến độ</button>' : '') +
        '</div><div class="nav-progress-bar"><span style="width:' +
        (100 * xong / links.length).toFixed(1) + '%"></span></div>';
    }
    veNutDaHoc();
  }
  // Dòng "đã học" ở cuối chương, ngay trên nút chuyển trang.
  function veNutDaHoc() {
    var file = chuongDangDoc();
    var khoi = $('#main .read-mark');
    if (!file) { if (khoi) khoi.remove(); return; }
    if (!khoi) {
      khoi = document.createElement('div');
      khoi.className = 'read-mark';
      var main = $('#main');
      main.insertBefore(khoi, $('.pager', main));
    }
    var co = Boolean(daHoc[file]);
    khoi.classList.toggle('is-read', co);
    khoi.innerHTML = co
      ? '<span>✓ Đã học chương này.</span><button type="button" data-read-toggle>Bỏ đánh dấu</button>'
      : '<span>Đọc xong chương này?</span><button type="button" data-read-toggle>Đánh dấu đã học</button>';
  }
  // Mỗi lần mở một chương chỉ tự đánh dấu một lần, và không tự đánh dấu lại
  // nếu người đọc vừa bấm nút: bỏ đánh dấu ở cuối chương thì phải giữ nguyên.
  var daXuLy = '';
  function kiemTraDocXong() {
    var file = chuongDangDoc();
    if (!file || daXuLy === file || daHoc[file] || window.scrollY < 1) return;
    // Lúc router đang thay trang, nội dung mới nằm tạm ở vị trí cuộn của trang cũ.
    if (document.documentElement.classList.contains('is-navigating')) return;
    var moc = $('#main .quiz') || $('#main .read-mark');
    if (moc && moc.getBoundingClientRect().top < window.innerHeight) {
      daXuLy = file;
      datDaHoc(file, true);
    }
  }
  document.addEventListener('click', function (e) {
    if (!e.target.closest) return;
    if (e.target.closest('[data-read-toggle]')) {
      var file = chuongDangDoc();
      if (file) { daXuLy = file; datDaHoc(file, !daHoc[file]); }
    } else if (e.target.closest('[data-progress-reset]')) {
      if (!window.confirm('Xoá tiến độ đã học của giáo trình này?')) return;
      daHoc = docKho(KHOA_HOC);
      $$('[data-part-list] a').forEach(function (a) { delete daHoc[a.getAttribute('href')]; });
      ghiKho(KHOA_HOC, daHoc);
      veTienDo();
    }
  });

  khoiTaoPhan();
  veTienDo();
  cuonToiMucDangDoc();

  /* ------------------------------------------------------ sao chép mã */
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-copy]');
    if (!btn) return;
    var code = btn.closest('.code-block').querySelector('code');
    if (!code) return;
    var done = function () {
      var old = btn.textContent;
      btn.textContent = 'Đã chép';
      btn.classList.add('is-done');
      setTimeout(function () { btn.textContent = old; btn.classList.remove('is-done'); }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(code.textContent).then(done, function () {});
    } else {
      var ta = document.createElement('textarea');
      ta.value = code.textContent;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); done(); } catch (err) {}
      document.body.removeChild(ta);
    }
  });

  /* --------------------------------------------------------- lightbox */
  var lightbox = $('#lightbox');
  if (lightbox) {
    var lbImg = lightbox.querySelector('img');
    var lbCap = lightbox.querySelector('.lightbox-cap');
    var openLightbox = function (frame) {
      var img = frame.querySelector('img');
      if (!img) return;
      lbImg.src = img.src;
      lbImg.alt = img.alt;
      var fig = frame.closest('figure');
      var cap = fig ? fig.querySelector('figcaption') : null;
      lbCap.textContent = cap ? cap.textContent.replace(/\s+/g, ' ').trim() : img.alt;
      lightbox.hidden = false;
      document.body.style.overflow = 'hidden';
    };
    var closeLightbox = function () {
      lightbox.hidden = true;
      lbImg.src = '';
      document.body.style.overflow = '';
    };
    document.addEventListener('click', function (e) {
      var frame = e.target.closest('[data-zoom]');
      if (frame) { openLightbox(frame); return; }
      if (e.target.closest('.lightbox')) closeLightbox();
    });
    document.addEventListener('keydown', function (e) {
      var frame = e.target.closest && e.target.closest('[data-zoom]');
      if (frame && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openLightbox(frame); }
      if (e.key === 'Escape' && !lightbox.hidden) closeLightbox();
    });
  }

  /* ------------------------------------------ thanh tiến độ + mục lục */
  var progress = $('#readProgress');
  var tocLinks = [];
  var headings = [];

  function docLaiMucLuc() {
    tocLinks = $$('.toc-list a');
    headings = tocLinks
      .map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); })
      .filter(Boolean);
  }
  docLaiMucLuc();

  // Thời gian đọc còn lại, hiện dưới tiêu đề mục lục bên phải của các chương.
  function capNhatThoiGianCon(p) {
    var o = $('[data-time-left]');
    if (!o) return;
    var t = $('#main .kicker-time[data-minutes]');
    if (!t) { o.hidden = true; return; }
    var con = Math.max(1, Math.round(+t.getAttribute('data-minutes') * (1 - p)));
    o.hidden = false;
    o.textContent = p < 0.98 ? 'Còn khoảng ' + con + ' phút đọc' : 'Đã tới cuối chương';
  }

  var ticking = false;
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var p = max > 0 ? Math.min(1, h.scrollTop / max) : 1;
    if (progress) progress.style.width = (max > 0 ? p * 100 : 0) + '%';
    capNhatThoiGianCon(p);
    if (headings.length) {
      var y = window.scrollY + 140;
      var idx = 0;
      for (var i = 0; i < headings.length; i++) {
        if (headings[i].offsetTop <= y) idx = i;
      }
      tocLinks.forEach(function (a, i) { a.classList.toggle('is-current', i === idx); });
    }
    ticking = false;
  }
  window.addEventListener('scroll', function () {
    if (!ticking) {
      ticking = true;
      // Chỉ đánh dấu "đã học" khi người đọc thật sự cuộn, không đánh dấu lúc mới mở trang.
      requestAnimationFrame(function () { onScroll(); kiemTraDocXong(); });
    }
  }, { passive: true });
  // Trang có thể dài thêm khi hình và phông nạp xong mà không có sự kiện cuộn nào;
  // khi đó tính lại thanh tiến độ và thời gian còn lại.
  if ('ResizeObserver' in window) {
    new ResizeObserver(function () {
      if (!ticking) { ticking = true; requestAnimationFrame(onScroll); }
    }).observe(document.body);
  }
  onScroll();

  /* --------------------------------------------------------- tìm kiếm */
  var panel = $('#searchPanel');
  var input = $('#searchInput');
  var results = $('#searchResults');
  var searchBtn = $('#searchBtn');
  var indexLoading = false;

  function loadIndex(cb) {
    if (window.QZ_INDEX) { cb(); return; }
    if (indexLoading) return;
    indexLoading = true;
    var s = document.createElement('script');
    s.src = 'assets/search-index.js';
    s.onload = function () { indexLoading = false; cb(); };
    s.onerror = function () {
      indexLoading = false;
      results.innerHTML = '<p class="sr-empty">Không nạp được chỉ mục tìm kiếm.</p>';
    };
    document.head.appendChild(s);
  }

  function norm(s) {
    return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd').toLowerCase();
  }

  function highlightText(text, terms) {
    var esc = text.replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; });
    var n = norm(esc);
    var hits = [];
    terms.forEach(function (t) {
      var from = 0, at;
      while ((at = n.indexOf(t, from)) !== -1) { hits.push([at, at + t.length]); from = at + t.length; }
    });
    if (!hits.length) return esc;
    hits.sort(function (a, b) { return a[0] - b[0]; });
    var out = '', last = 0;
    hits.forEach(function (h) {
      if (h[0] < last) return;
      out += esc.slice(last, h[0]) + '<mark>' + esc.slice(h[0], h[1]) + '</mark>';
      last = h[1];
    });
    return out + esc.slice(last);
  }

  function search(q) {
    var terms = norm(q).split(/\s+/).filter(function (t) { return t.length > 1; });
    if (!terms.length) {
      results.innerHTML = '<p class="sr-empty">Gõ ít nhất hai ký tự. Bỏ dấu cũng tìm được.</p>';
      return;
    }
    var scored = [];
    (window.QZ_INDEX || []).forEach(function (rec) {
      var title = norm(rec.t);
      var body = norm(rec.x);
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i];
        var inTitle = title.indexOf(t) !== -1;
        var at = body.indexOf(t);
        if (!inTitle && at === -1) return;
        if (inTitle) score += 12;
        if (at !== -1) score += 3 + Math.max(0, 3 - at / 120);
      }
      scored.push({ rec: rec, score: score });
    });
    scored.sort(function (a, b) { return b.score - a.score; });

    // Một mục con được cắt thành nhiều đoạn nên có thể khớp nhiều lần;
    // chỉ giữ đoạn hợp nhất cho mỗi đích đến.
    var seen = {};
    scored = scored.filter(function (s) {
      if (seen[s.rec.h]) return false;
      seen[s.rec.h] = true;
      return true;
    });

    if (!scored.length) {
      results.innerHTML = '<p class="sr-empty">Không tìm thấy “' +
        q.replace(/[&<>]/g, '') + '”.</p>';
      return;
    }
    results.innerHTML = scored.slice(0, 30).map(function (s) {
      var rec = s.rec;
      var body = norm(rec.x);
      var at = body.indexOf(terms[0]);
      var start = Math.max(0, at - 60);
      var snippet = (start > 0 ? '…' : '') + rec.x.slice(start, start + 190);
      var badge = rec.c === 'PL' ? 'Phụ lục' : rec.c === 'TN' ? 'Thuật ngữ' : 'Ch. ' + rec.c;
      return '<a class="sr-item" href="' + rec.h + '">' +
        '<span class="sr-head"><span class="sr-ch">' + badge +
        '</span><span class="sr-title">' + highlightText(rec.t, terms) + '</span></span>' +
        '<p class="sr-snip">' + highlightText(snippet, terms) + '</p></a>';
    }).join('');
  }

  function openSearch() {
    if (!panel) return;
    panel.hidden = false;
    document.body.style.overflow = 'hidden';
    input.focus();
    input.select();
    loadIndex(function () { if (input.value.trim()) search(input.value); });
    if (!input.value.trim()) {
      results.innerHTML = '<p class="sr-empty">Tìm theo khái niệm, công thức hay tên kỹ thuật. Bỏ dấu cũng tìm được.</p>';
    }
  }
  function closeSearch() {
    if (!panel) return;
    panel.hidden = true;
    document.body.style.overflow = '';
  }

  if (searchBtn) searchBtn.addEventListener('click', openSearch);
  if (panel) {
    panel.addEventListener('click', function (e) { if (e.target === panel) closeSearch(); });
    input.addEventListener('input', function () {
      loadIndex(function () { search(input.value); });
      if (window.QZ_INDEX) search(input.value);
    });
    input.addEventListener('keydown', function (e) {
      var items = $$('.sr-item', results);
      var cur = items.findIndex(function (el) { return el.classList.contains('is-active'); });
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!items.length) return;
        var next = e.key === 'ArrowDown'
          ? Math.min(items.length - 1, cur + 1)
          : Math.max(0, cur - 1);
        items.forEach(function (el) { el.classList.remove('is-active'); });
        items[next].classList.add('is-active');
        items[next].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter') {
        var target = items[cur >= 0 ? cur : 0];
        if (target) { e.preventDefault(); window.location.href = target.getAttribute('href'); }
      }
    });
  }

  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    var typing = tag === 'input' || tag === 'textarea' || tag === 'select';
    if (e.key === 'Escape') { closeSearch(); closeSidebar(); }
    if (!typing && (e.key === '/' || ((e.ctrlKey || e.metaKey) && e.key === 'k'))) {
      e.preventDefault();
      openSearch();
    }
  });

  /* -------------------------------------------------------- trắc nghiệm */
  function shuffle(nodes) {
    for (var i = nodes.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      nodes[i].parentNode.insertBefore(nodes[j], nodes[i].nextSibling);
      var tmp = nodes[i]; nodes[i] = nodes[j]; nodes[j] = tmp;
    }
  }

  function capNhatDiem(quiz) {
    var questions = $$('[data-q]', quiz);
    var score = $('[data-score]', quiz);
    var total = questions.length;
    var answered = questions.filter(function (q) { return q.dataset.answered; }).length;
    var right = questions.filter(function (q) { return q.dataset.answered === 'right'; }).length;
    score.textContent = answered === total
      ? 'Xong: đúng ' + right + '/' + total + ' câu'
      : 'Đã trả lời ' + answered + '/' + total;
    score.classList.toggle('is-done', answered === total);
  }

  // Router thay nội dung trang mà không tải lại, nên sự kiện được uỷ quyền cho
  // document; mỗi khối trắc nghiệm mới chỉ cần trộn đáp án và đếm điểm một lần.
  function khoiTaoTracNghiem() {
    $$('[data-quiz]').forEach(function (quiz) {
      if (quiz.dataset.ready) return;
      quiz.dataset.ready = '1';
      $$('[data-q]', quiz).forEach(function (q) { shuffle($$('.quiz-opt', q)); });
      capNhatDiem(quiz);
    });
  }

  document.addEventListener('click', function (e) {
    var quiz = e.target.closest && e.target.closest('[data-quiz]');
    if (!quiz) return;
    var opt = e.target.closest('.quiz-opt');
    if (opt && !opt.disabled) {
      var q = opt.closest('[data-q]');
      var correct = opt.dataset.correct === '1';
      $$('.quiz-opt', q).forEach(function (o) {
        o.disabled = true;
        if (o.dataset.correct === '1') o.classList.add('is-correct');
      });
      if (!correct) opt.classList.add('is-wrong');
      $('.quiz-why', q).hidden = false;
      q.dataset.answered = correct ? 'right' : 'wrong';
      capNhatDiem(quiz);
      return;
    }
    if (e.target.closest('[data-reset]')) {
      $$('[data-q]', quiz).forEach(function (q) {
        delete q.dataset.answered;
        $('.quiz-why', q).hidden = true;
        var opts = $$('.quiz-opt', q);
        opts.forEach(function (o) {
          o.disabled = false;
          o.classList.remove('is-correct', 'is-wrong');
        });
        shuffle(opts);
      });
      capNhatDiem(quiz);
    }
  });

  khoiTaoTracNghiem();

  /* Đóng thanh bên khi bấm vào một mục điều hướng trên màn hình hẹp. */
  // Uỷ quyền cho document thay vì gắn vào từng thẻ: sau khi router thay thanh
  // bên thì các thẻ cũ biến mất, mà handler uỷ quyền thì vẫn sống.
  document.addEventListener('click', function (e) {
    if (e.target.closest && e.target.closest('.nav-list a')) closeSidebar();
  });

  /* Router gọi lại hàm này mỗi khi thay nội dung trang. */
  window.QZ_BIND = function () {
    daXuLy = '';
    docLaiMucLuc();
    khoiTaoPhan();
    veTienDo();
    khoiTaoTracNghiem();
    cuonToiMucDangDoc();
    onScroll();
  };
})();
