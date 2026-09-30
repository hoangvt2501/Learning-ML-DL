// Trang từ điển thuật ngữ Việt – Anh, có lọc tại chỗ.

import { slugify } from '../markdown.mjs';

function parseGlossary(raw) {
  const groups = [];
  let group = null;
  let term = null;

  const flush = () => {
    if (group && term && term.def.trim()) group.terms.push(term);
    term = null;
  };

  for (const line of raw.split('\n')) {
    const h2 = line.match(/^## (.+)$/);
    if (h2) {
      flush();
      group = { name: h2[1].trim(), terms: [] };
      groups.push(group);
      continue;
    }
    const h3 = line.match(/^### (.+)$/);
    if (h3 && group) {
      flush();
      const [vi, en] = h3[1].split('|').map((s) => s.trim());
      term = { vi, en: en || '', def: '' };
      continue;
    }
    if (term) term.def += line + '\n';
  }
  flush();
  return groups.filter((g) => g.terms.length);
}

export function buildGlossaryPage(ctx) {
  const { md, nav, page, escapeHtml, write, read, neighbours, registry, searchIndex, f, book } = ctx;
  registry.currentFile = f('thuat-ngu.html');

  const groups = parseGlossary(read(book.glossary));
  const total = groups.reduce((a, g) => a + g.terms.length, 0);

  // Đưa thuật ngữ vào chỉ mục tìm kiếm chung.
  for (const g of groups) {
    for (const t of g.terms) {
      searchIndex.push({
        c: 'TN', b: book.short,
        t: (t.vi + (t.en ? ' · ' + t.en : '')).replace(/\$/g, '').trim(),
        h: f('thuat-ngu.html') + '#tn-' + slugify(t.vi),
        x: (t.en ? t.en + '. ' : '') + t.def.replace(/[$*_`\\]/g, ' ').replace(/\s+/g, ' ').trim(),
      });
    }
  }

  const sections = groups
    .map((g) => {
      const id = 'nhom-' + groups.indexOf(g);
      const items = g.terms
        .map((t) => {
          const key = (t.vi + ' ' + t.en + ' ' + t.def).toLowerCase();
          return (
            '<div class="term" id="tn-' + slugify(t.vi) + '" data-term data-key="' +
            escapeHtml(key) + '">' +
            '<dt class="term-head"><span class="term-vi">' + md.renderInline(t.vi) + '</span>' +
            (t.en ? '<span class="term-en">' + md.renderInline(t.en) + '</span>' : '') +
            '</dt>' +
            '<dd class="term-def">' + md.render(t.def.trim()) + '</dd>' +
            '</div>'
          );
        })
        .join('');
      return (
        '<section class="term-group" data-group id="' + id + '">' +
        '<h3 class="heading" id="' + id + '-h">' + escapeHtml(g.name) +
        '<a class="anchor" href="#' + id + '-h">#</a></h3>' +
        '<dl class="term-list">' + items + '</dl></section>'
      );
    })
    .join('');

  const jump =
    '<nav class="ex-jump">' +
    groups
      .map((g, i) => '<a href="#nhom-' + i + '">' + escapeHtml(g.name) + '</a>')
      .join('') +
    '</nav>';

  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Tra cứu</span>' +
    '<span class="kicker-time">' + total + ' thuật ngữ</span></div>' +
    '<h1>Từ điển thuật ngữ</h1>' +
    '<p class="chapter-lede">Đối chiếu Việt – Anh cho toàn bộ thuật ngữ dùng trong giáo trình. Định nghĩa bám sát đúng cách dùng trong bài, không phải định nghĩa từ điển chung chung.</p>' +
    '<div class="term-filter">' +
    '<input type="search" id="termFilter" placeholder="Lọc theo từ tiếng Việt, tiếng Anh hoặc nội dung định nghĩa…" autocomplete="off" spellcheck="false">' +
    '<p class="term-count" data-term-count aria-live="polite"></p>' +
    '</div>' +
    jump +
    sections +
    '</article>';

  write(
    f('thuat-ngu.html'),
    page({
      title: 'Từ điển thuật ngữ — ' + ctx.docTitle,
      description: total + ' thuật ngữ ' + book.short + ' đối chiếu Việt – Anh, kèm định nghĩa ngắn.',
      body,
      nav,
      file: f('thuat-ngu.html'),
      ...neighbours(f('thuat-ngu.html')),
      scripts: ['assets/glossary.js'],
    })
  );
}
