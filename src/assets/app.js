/* Tương tác dùng chung cho mọi trang: giao diện sáng/tối, mục lục, tìm kiếm,
   phóng to hình, sao chép mã, thanh tiến độ đọc và phần trắc nghiệm. */
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
  var activeNav = $('.nav-list a.is-active');
  if (activeNav && sidebar) {
    var r = activeNav.getBoundingClientRect();
    var sr = sidebar.getBoundingClientRect();
    if (r.top < sr.top || r.bottom > sr.bottom) {
      activeNav.scrollIntoView({ block: 'center' });
    }
  }

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
  var tocLinks = $$('.toc-list a');
  var headings = tocLinks
    .map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); })
    .filter(Boolean);

  var ticking = false;
  function onScroll() {
    if (progress) {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      progress.style.width = (max > 0 ? Math.min(1, h.scrollTop / max) * 100 : 0) + '%';
    }
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
    if (!ticking) { ticking = true; requestAnimationFrame(onScroll); }
  }, { passive: true });
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

  $$('[data-quiz]').forEach(function (quiz) {
    var questions = $$('[data-q]', quiz);
    var score = $('[data-score]', quiz);
    var total = questions.length;

    questions.forEach(function (q) { shuffle($$('.quiz-opt', q)); });

    function refresh() {
      var answered = questions.filter(function (q) { return q.dataset.answered; }).length;
      var right = questions.filter(function (q) { return q.dataset.answered === 'right'; }).length;
      score.textContent = answered === total
        ? 'Xong: đúng ' + right + '/' + total + ' câu'
        : 'Đã trả lời ' + answered + '/' + total;
      score.classList.toggle('is-done', answered === total);
    }

    quiz.addEventListener('click', function (e) {
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
        refresh();
        return;
      }
      if (e.target.closest('[data-reset]')) {
        questions.forEach(function (q) {
          delete q.dataset.answered;
          $('.quiz-why', q).hidden = true;
          var opts = $$('.quiz-opt', q);
          opts.forEach(function (o) {
            o.disabled = false;
            o.classList.remove('is-correct', 'is-wrong');
          });
          shuffle(opts);
        });
        refresh();
      }
    });

    refresh();
  });

  /* Đóng thanh bên khi bấm vào một mục điều hướng trên màn hình hẹp. */
  $$('.nav-list a').forEach(function (a) { a.addEventListener('click', closeSidebar); });
})();
