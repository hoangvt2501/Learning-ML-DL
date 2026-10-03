/* Điều hướng trong trang: bấm sang chương khác thì chỉ thay phần nội dung chứ
   không tải lại cả trang. Site vẫn là HTML tĩnh — mỗi trang vẫn tồn tại đầy đủ
   và mở trực tiếp được — nên đây chỉ là lớp tăng tốc đặt lên trên.

   Khi fetch không dùng được (mở bằng file://, hoặc trình duyệt chặn) thì router
   tự tắt và trình duyệt điều hướng như bình thường. */
(function () {
  'use strict';

  // fetch trên file:// bị chặn vì lý do CORS, nên chỉ bật router khi chạy qua
  // http(s). Mở trực tiếp bằng file:// vẫn đọc được, chỉ là tải lại mỗi lần bấm.
  var batDuoc = typeof window.fetch === 'function' &&
    (location.protocol === 'http:' || location.protocol === 'https:');
  if (!batDuoc) return;

  var bộNhớ = Object.create(null);     // url -> Document đã phân tích
  var đangTải = Object.create(null);   // url -> Promise
  var scriptĐãNạp = Object.create(null);
  var lượt = 0;

  Array.prototype.slice.call(document.querySelectorAll('script[src]'))
    .forEach(function (s) { scriptĐãNạp[chuẩnHoá(s.getAttribute('src'))] = true; });

  function chuẩnHoá(href) {
    var a = document.createElement('a');
    a.href = href;
    return a.pathname + a.search;
  }

  function nộiBộ(a) {
    if (!a || a.target || a.hasAttribute('download')) return false;
    if (a.origin !== location.origin) return false;
    var p = a.pathname;
    return /\.html$/.test(p) || /\/$/.test(p);
  }

  function tải(url) {
    if (bộNhớ[url]) return Promise.resolve(bộNhớ[url]);
    if (đangTải[url]) return đangTải[url];
    đangTải[url] = fetch(url, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.text();
      })
      .then(function (html) {
        var doc = new DOMParser().parseFromString(html, 'text/html');
        bộNhớ[url] = doc;
        delete đangTải[url];
        return doc;
      })
      .catch(function (e) {
        delete đangTải[url];
        throw e;
      });
    return đangTải[url];
  }

  function nạpScript(src) {
    var key = chuẩnHoá(src);
    if (scriptĐãNạp[key]) return Promise.resolve();
    scriptĐãNạp[key] = true;
    return new Promise(function (ok) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = s.onerror = function () { ok(); };
      document.body.appendChild(s);
    });
  }

  function thayThế(đích, nguồn) {
    var cũ = document.querySelector(đích);
    var mới = nguồn.querySelector(đích);
    if (cũ && mới) {
      cũ.innerHTML = mới.innerHTML;
      return true;
    }
    if (cũ && !mới) cũ.remove();
    return false;
  }

  function ápDụng(doc, url, hash) {
    document.title = doc.title;

    var môTả = doc.querySelector('meta[name="description"]');
    var môTảCũ = document.querySelector('meta[name="description"]');
    if (môTả && môTảCũ) môTảCũ.setAttribute('content', môTả.getAttribute('content') || '');

    document.body.className = doc.body.className;

    thayThế('#main', doc);
    thayThế('.sidebar-inner', doc);

    // Cột mục lục bên phải có trang có, có trang không.
    var tocCũ = document.querySelector('.toc');
    var tocMới = doc.querySelector('.toc');
    if (tocMới && tocCũ) tocCũ.innerHTML = tocMới.innerHTML;
    else if (tocMới && !tocCũ) {
      document.querySelector('.shell').appendChild(tocMới.cloneNode(true));
    } else if (!tocMới && tocCũ) tocCũ.remove();

    // Script riêng của trang mới (bài tập, phòng thí nghiệm, từ điển…).
    var cần = Array.prototype.slice.call(doc.querySelectorAll('script[src]'))
      .map(function (s) { return s.getAttribute('src'); })
      .filter(function (src) { return src && !/app\.js$|router\.js$/.test(src); });

    return Promise.all(cần.map(nạpScript)).then(function () {
      // Mọi script trang đều đăng ký hàm khởi tạo vào window.QZ_INIT để gọi lại
      // được sau khi DOM bị thay. Nút cũ chết theo DOM cũ nên không lo trùng.
      var reg = window.QZ_INIT || {};
      Object.keys(reg).forEach(function (k) {
        try { reg[k](); } catch (e) { /* một công cụ hỏng không được làm hỏng cả trang */ }
      });
      if (typeof window.QZ_BIND === 'function') window.QZ_BIND();

      // Trang có scroll-behavior: smooth; sang trang mới thì nhảy thẳng tới vị trí
      // cần đến, không lướt qua cả trang mới từ vị trí cuộn của trang cũ.
      cuộnNgay(hash && document.getElementById(hash.slice(1)));
    });
  }

  function cuộnNgay(đích) {
    try {
      if (đích) đích.scrollIntoView({ block: 'start', behavior: 'instant' });
      else window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    } catch (e) {
      // Trình duyệt cũ không nhận 'instant' thì cuộn theo cách mặc định.
      if (đích) đích.scrollIntoView();
      else window.scrollTo(0, 0);
    }
  }

  function đi(href, đẩyLịchSử) {
    var a = document.createElement('a');
    a.href = href;
    var url = a.pathname + a.search;
    var hash = a.hash;
    var n = ++lượt;

    document.documentElement.classList.add('is-navigating');

    return tải(url)
      .then(function (doc) {
        if (n !== lượt) return;                 // đã bấm sang trang khác rồi
        if (đẩyLịchSử) history.pushState({ url: url + hash }, '', url + hash);
        return ápDụng(doc, url, hash);
      })
      .catch(function () {
        location.href = href;                   // hỏng thì để trình duyệt tự đi
      })
      .then(function () {
        document.documentElement.classList.remove('is-navigating');
      });
  }

  document.addEventListener('click', function (e) {
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var a = e.target.closest ? e.target.closest('a[href]') : null;
    if (!nộiBộ(a)) return;

    // Liên kết neo trong cùng trang thì để trình duyệt xử lý.
    if (a.pathname === location.pathname && a.hash) return;

    e.preventDefault();
    đi(a.href, true);
  });

  // Tải trước khi rê chuột: tới lúc bấm thì trang thường đã nằm sẵn trong bộ nhớ.
  var hẹn;
  function ngắm(e) {
    var a = e.target.closest ? e.target.closest('a[href]') : null;
    if (!nộiBộ(a)) return;
    var url = a.pathname + a.search;
    if (bộNhớ[url] || đangTải[url]) return;
    clearTimeout(hẹn);
    hẹn = setTimeout(function () { tải(url).catch(function () {}); }, 65);
  }
  document.addEventListener('mouseover', ngắm, { passive: true });
  document.addEventListener('touchstart', ngắm, { passive: true });

  window.addEventListener('popstate', function (e) {
    var url = (e.state && e.state.url) || location.pathname + location.search + location.hash;
    var a = document.createElement('a');
    a.href = url;
    document.documentElement.classList.add('is-navigating');
    tải(a.pathname + a.search)
      .then(function (doc) { return ápDụng(doc, a.pathname, a.hash); })
      .catch(function () { location.reload(); })
      .then(function () {
        document.documentElement.classList.remove('is-navigating');
      });
  });

  history.replaceState({ url: location.pathname + location.search + location.hash }, '');
})();
