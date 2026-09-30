/* Xem trước khi rê chuột lên một liên kết chéo.

   Rê chuột lên "Mục 6.2" thì hiện một ô nhỏ có tiêu đề mục ấy và vài dòng đầu,
   nên không phải rời trang để nhớ ra nó nói gì. Ý này học từ Wikipedia và nLab.

   Dùng fetch nên chỉ chạy qua http(s); mở bằng file:// thì lớp này tự tắt và
   liên kết vẫn bấm được như thường. */
(function () {
  'use strict';

  var batDuoc = typeof window.fetch === 'function' &&
    (location.protocol === 'http:' || location.protocol === 'https:');
  if (!batDuoc || matchMedia('(hover: none)').matches) return;

  var bo = Object.create(null);     // url -> Document
  var noiDung = Object.create(null); // url#id -> {tieuDe, chu} hoặc null
  var hen, oHien, dangChoUrl;

  function hop() {
    if (oHien) return oHien;
    oHien = document.createElement('div');
    oHien.className = 'xpreview';
    oHien.hidden = true;
    oHien.innerHTML = '<p class="xpreview-title"></p><p class="xpreview-body"></p>' +
      '<p class="xpreview-hint">Bấm để mở</p>';
    document.body.appendChild(oHien);
    oHien.addEventListener('mouseenter', function () { clearTimeout(hen); });
    oHien.addEventListener('mouseleave', an);
    return oHien;
  }

  function an() {
    clearTimeout(hen);
    dangChoUrl = null;
    if (oHien) oHien.hidden = true;
  }

  function tai(url) {
    if (bo[url]) return Promise.resolve(bo[url]);
    return fetch(url, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error(String(r.status));
        return r.text();
      })
      .then(function (html) {
        bo[url] = new DOMParser().parseFromString(html, 'text/html');
        return bo[url];
      });
  }

  /** Lấy tiêu đề mục và một đoạn văn đầu tiên sau nó. */
  function trich(doc, id) {
    var el = doc.getElementById(id);
    if (!el) return null;
    var tieuDe = el.textContent.replace(/#\s*$/, '').trim();
    var chu = '';
    var n = el.nextElementSibling;
    var buoc = 0;
    while (n && buoc < 6 && chu.length < 320) {
      if (n.tagName === 'P' && !n.classList.contains('figure-cap')) {
        chu += (chu ? ' ' : '') + n.textContent.trim();
      } else if (/^H[1-6]$/.test(n.tagName)) {
        break;
      }
      n = n.nextElementSibling;
      buoc++;
    }
    if (chu.length > 340) chu = chu.slice(0, 340).replace(/\s+\S*$/, '') + '…';
    return { tieuDe: tieuDe, chu: chu };
  }

  function dat(a) {
    var o = hop();
    var r = a.getBoundingClientRect();
    o.hidden = false;
    var w = o.offsetWidth, h = o.offsetHeight;
    var x = r.left + r.width / 2 - w / 2;
    x = Math.max(10, Math.min(x, window.innerWidth - w - 10));
    var duoi = r.bottom + 10;
    var tren = r.top - h - 10;
    var y = (duoi + h < window.innerHeight - 10 || tren < 10) ? duoi : tren;
    o.style.left = Math.round(x) + 'px';
    o.style.top = Math.round(y + window.scrollY) + 'px';
  }

  function hien(a, url, id) {
    var key = url + '#' + id;
    if (noiDung[key] === null) return;             // đã thử và không có gì
    if (noiDung[key]) {
      var o = hop();
      o.querySelector('.xpreview-title').textContent = noiDung[key].tieuDe;
      o.querySelector('.xpreview-body').textContent = noiDung[key].chu;
      dat(a);
      return;
    }
    dangChoUrl = key;
    tai(url)
      .then(function (doc) {
        noiDung[key] = trich(doc, id);
        if (dangChoUrl === key && noiDung[key]) hien(a, url, id);
      })
      .catch(function () { noiDung[key] = null; });
  }

  document.addEventListener('mouseover', function (e) {
    var a = e.target.closest ? e.target.closest('a[href*="#"]') : null;
    if (!a || a.origin !== location.origin) return;
    if (a.closest('.toc') || a.closest('.sidebar')) return;   // mục lục thì khỏi
    var id = a.hash.slice(1);
    if (!id) return;
    clearTimeout(hen);
    hen = setTimeout(function () { hien(a, a.pathname + a.search, id); }, 280);
  });

  document.addEventListener('mouseout', function (e) {
    var a = e.target.closest ? e.target.closest('a[href*="#"]') : null;
    if (!a) return;
    clearTimeout(hen);
    hen = setTimeout(an, 180);
  });

  document.addEventListener('click', an);
  window.addEventListener('scroll', an, { passive: true });
})();
