/* Lọc tại chỗ cho trang từ điển thuật ngữ. Bỏ dấu tiếng Việt khi so khớp. */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['glossary'] = function () {
  'use strict';

  var input = document.getElementById('termFilter');
  if (!input) return;

  var counter = document.querySelector('[data-term-count]');
  var terms = Array.prototype.slice.call(document.querySelectorAll('[data-term]'));
  var groups = Array.prototype.slice.call(document.querySelectorAll('[data-group]'));
  var jump = document.querySelector('.ex-jump');
  var total = terms.length;

  function norm(s) {
    return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd').toLowerCase();
  }

  // Chuẩn hoá sẵn một lần để gõ phím không phải tính lại.
  terms.forEach(function (t) { t._key = norm(t.dataset.key); });

  function apply() {
    var q = norm(input.value.trim());
    var shown = 0;

    terms.forEach(function (t) {
      var hit = !q || t._key.indexOf(q) !== -1;
      t.hidden = !hit;
      if (hit) shown++;
    });

    groups.forEach(function (g) {
      var any = Array.prototype.some.call(g.querySelectorAll('[data-term]'), function (t) {
        return !t.hidden;
      });
      g.hidden = !any;
    });

    if (jump) jump.hidden = !!q;

    if (counter) {
      counter.textContent = q
        ? 'Hiện ' + shown + ' / ' + total + ' thuật ngữ'
        : total + ' thuật ngữ';
    }
  }

  input.addEventListener('input', apply);
  apply();
};
window.QZ_INIT['glossary']();
