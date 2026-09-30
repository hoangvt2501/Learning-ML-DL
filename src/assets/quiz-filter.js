/* Lọc trắc nghiệm theo chương, dùng cho trang bài tập không có phần luyện tính tay. */
(function () {
  'use strict';
  var filter = document.querySelector('[data-quiz-filter]');
  if (!filter) return;
  filter.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-filter]');
    if (!btn) return;
    Array.prototype.forEach.call(filter.querySelectorAll('[data-filter]'), function (b) {
      b.classList.remove('is-active');
    });
    btn.classList.add('is-active');
    var want = btn.dataset.filter;
    Array.prototype.forEach.call(document.querySelectorAll('[data-quiz-chapter]'), function (w) {
      w.hidden = want !== 'all' && w.dataset.quizChapter !== want;
    });
  });
})();
