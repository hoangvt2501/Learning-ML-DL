// Đọc content/trac-nghiem.md và dựng khối câu hỏi tự kiểm tra.

import { escapeHtml } from '../layout.mjs';

/**
 * Cú pháp của tệp nguồn:
 *   ## Chương 3
 *   ### Câu hỏi?
 *   - [ ] đáp án sai
 *   - [x] đáp án đúng
 *   > giải thích
 */
export function parseQuizzes(raw) {
  const groups = new Map();
  let group = null;
  let q = null;

  const flushQ = () => {
    if (group && q && q.options.length) group.questions.push(q);
    q = null;
  };

  for (const line of raw.split('\n')) {
    const h2 = line.match(/^## Chương (\S+)\s*$/);
    if (h2) {
      flushQ();
      group = { chapter: h2[1], questions: [] };
      groups.set(h2[1], group);
      continue;
    }
    if (!group) continue;

    const h3 = line.match(/^### (.+)$/);
    if (h3) {
      flushQ();
      q = { prompt: h3[1].trim(), options: [], why: '' };
      continue;
    }
    if (!q) continue;

    const opt = line.match(/^- \[([ xX])\]\s+(.+)$/);
    if (opt) {
      q.options.push({ correct: opt[1].toLowerCase() === 'x', text: opt[2].trim() });
      continue;
    }
    const why = line.match(/^>\s?(.*)$/);
    if (why) q.why += (q.why ? ' ' : '') + why[1].trim();
  }
  flushQ();

  for (const g of groups.values()) {
    for (const question of g.questions) {
      if (!question.options.some((o) => o.correct)) {
        throw new Error('Câu hỏi thiếu đáp án đúng: ' + question.prompt);
      }
    }
  }
  return groups;
}

/** Dựng HTML một khối trắc nghiệm. `variant` = 'chapter' (cuối chương) hoặc 'page'. */
export function renderQuiz(md, group, variant = 'chapter', title) {
  if (!group || !group.questions.length) return '';
  const n = group.questions.length;

  const questions = group.questions
    .map((q, i) => {
      const opts = q.options
        .map(
          (o) =>
            '<button class="quiz-opt" type="button" data-correct="' +
            (o.correct ? '1' : '0') +
            '"><span class="quiz-mark" aria-hidden="true"></span>' +
            '<span class="quiz-opt-text">' + md.renderInline(o.text) + '</span></button>'
        )
        .join('');
      return (
        '<li class="quiz-q" data-q>' +
        // Phần chữ phải nằm trong một span riêng: .quiz-prompt là flex nên mọi
        // nút con trực tiếp (kể cả <a> trong câu) sẽ thành một flex item riêng.
        '<p class="quiz-prompt"><span class="quiz-idx">' + (i + 1) + '</span>' +
        '<span class="quiz-text">' + md.renderInline(q.prompt) + '</span></p>' +
        '<div class="quiz-opts">' + opts + '</div>' +
        '<div class="quiz-why" hidden><b>Vì sao:</b> ' + md.renderInline(q.why) + '</div>' +
        '</li>'
      );
    })
    .join('');

  const heading =
    variant === 'chapter'
      ? '<h3 class="quiz-title">Tự kiểm tra</h3>' +
        '<p class="quiz-sub">' + n + ' câu. Chọn đáp án để xem ngay lời giải thích.</p>'
      : '<h3 class="quiz-title">' + escapeHtml(title || 'Chương ' + group.chapter) + '</h3>' +
        '<p class="quiz-sub">' + n + ' câu</p>';

  return (
    '<section class="quiz" data-quiz data-chapter="' + escapeHtml(group.chapter) + '">' +
    '<div class="quiz-head">' + heading +
    '<button class="quiz-reset btn-ghost" type="button" data-reset>Làm lại</button></div>' +
    '<ol class="quiz-list">' + questions + '</ol>' +
    '<p class="quiz-score" data-score aria-live="polite">Đã trả lời 0/' + n + '</p>' +
    '</section>'
  );
}
