// Ghi chú biên tập: giữ nguyên văn giáo trình, chèn thêm khung ghi chú ở cuối mục liên quan.

import { sectionId } from '../markdown.mjs';

/** Cú pháp nguồn: `## 5.2 — Tiêu đề ghi chú` rồi phần thân Markdown. */
export function parseNotes(raw) {
  const notes = new Map();
  const parts = raw.split(/^## (\d+(?:\.\d+)*)\s+—\s+(.+)$/m);
  for (let i = 1; i < parts.length; i += 3) {
    notes.set(parts[i], {
      section: parts[i],
      title: parts[i + 1].trim(),
      markdown: parts[i + 2].trim(),
    });
  }
  return notes;
}

/** Số mục "10.1.1" thuộc chương "10". */
const chapterOf = (section) => section.split('.')[0];

function noteHtml(md, note, opts = {}) {
  const href = opts.standalone ? '' : ' href="ghi-chu.html#gc-' + sectionId(note.section) + '"';
  const tag = opts.standalone
    ? '<span class="note-tag">Mục ' + note.section + '</span>'
    : '<a class="note-tag"' + href + '>Ghi chú biên tập · Mục ' + note.section + '</a>';
  return (
    '<aside class="editor-note" id="gc-' + sectionId(note.section) + '">' +
    '<div class="note-head">' + tag + '</div>' +
    '<h4 class="note-title">' + md.renderInline(note.title) + '</h4>' +
    '<div class="note-body">' + md.render(note.markdown) + '</div>' +
    '</aside>'
  );
}

/**
 * Chèn ghi chú vào cuối đúng mục của nó trong HTML đã render.
 * Cuối một mục = vị trí tiêu đề kế tiếp có cấp bằng hoặc cao hơn.
 */
export function injectNotes(html, chapterNum, notes, md) {
  const mine = [...notes.values()].filter((n) => chapterOf(n.section) === chapterNum);
  if (!mine.length) return html;

  // Xử lý từ cuối lên đầu để các vị trí đã tính không bị lệch.
  mine.sort((a, b) => b.section.localeCompare(a.section, undefined, { numeric: true }));

  for (const note of mine) {
    const id = sectionId(note.section);
    const openRe = new RegExp('<h([234]) id="' + id + '" class="heading">');
    const open = openRe.exec(html);
    if (!open) {
      throw new Error('Ghi chú trỏ tới mục không có thật: ' + note.section);
    }
    const level = Number(open[1]);

    // Tìm tiêu đề kế tiếp có cấp <= cấp của mục này.
    const after = open.index + open[0].length;
    const headingRe = /<h([234]) id="[^"]+" class="heading">/g;
    headingRe.lastIndex = after;
    let at = html.length;
    let m;
    while ((m = headingRe.exec(html)) !== null) {
      if (Number(m[1]) <= level) { at = m.index; break; }
    }
    html = html.slice(0, at) + noteHtml(md, note) + html.slice(at);
  }
  return html;
}

export function buildNotesPage(ctx, notes) {
  const { md, nav, page, escapeHtml, write, neighbours, registry, chapters, searchIndex } = ctx;
  registry.currentFile = 'ghi-chu.html';

  const list = [...notes.values()].sort((a, b) =>
    a.section.localeCompare(b.section, undefined, { numeric: true })
  );

  for (const n of list) {
    searchIndex.push({
      c: 'GC',
      t: 'Ghi chú Mục ' + n.section + ' — ' + n.title,
      h: 'ghi-chu.html#gc-' + sectionId(n.section),
      x: n.markdown.replace(/```[\s\S]*?```/g, ' ').replace(/[#*_`>|$\\]/g, ' ')
        .replace(/\s+/g, ' ').trim().slice(0, 900),
    });
  }

  const labelOf = (num) => {
    const c = chapters.find((x) => x.num === num);
    return c ? c.label : '';
  };

  const jump =
    '<nav class="ex-jump">' +
    list
      .map((n) => '<a href="#gc-' + sectionId(n.section) + '">Mục ' + n.section + '</a>')
      .join('') +
    '</nav>';

  const cards = list
    .map((n) => {
      const ch = chapterOf(n.section);
      const file = 'ch' + String(ch).padStart(2, '0') + '.html';
      return (
        '<article class="note-card" id="gc-' + sectionId(n.section) + '">' +
        '<header class="note-card-head">' +
        '<span class="note-tag">Mục ' + n.section + '</span>' +
        '<a class="note-src" href="' + file + '#' + sectionId(n.section) + '">' +
        'Chương ' + ch + ' · ' + escapeHtml(labelOf(ch)) + ' →</a>' +
        '</header>' +
        '<h3 class="heading" id="gc-h-' + sectionId(n.section) + '">' +
        md.renderInline(n.title) +
        '<a class="anchor" href="#gc-h-' + sectionId(n.section) + '">#</a></h3>' +
        '<div class="note-body">' + md.render(n.markdown) + '</div>' +
        '</article>'
      );
    })
    .join('');

  const body =
    '<article class="prose">' +
    '<div class="chapter-kicker"><span class="kicker-badge">Tra cứu</span>' +
    '<span class="kicker-time">' + list.length + ' ghi chú</span></div>' +
    '<h1>Ghi chú biên tập</h1>' +
    '<p class="chapter-lede">Nguyên văn giáo trình được giữ nguyên. Những chỗ dưới đây phát biểu đúng về ý nhưng rộng hơn mức chứng minh được, hoặc có một chi tiết kỹ thuật đáng nói thêm — mỗi ghi chú cũng hiện ngay cuối mục tương ứng khi bạn đọc chương.</p>' +
    jump +
    '<div class="note-list">' + cards + '</div>' +
    '</article>';

  write(
    'ghi-chu.html',
    page({
      title: 'Ghi chú biên tập — Quantization trong Deep Learning',
      description:
        'Những chỗ trong giáo trình cần phát biểu chặt hơn, kèm kiểm chứng bằng mã: ràng buộc trục thu gọn, phạm vi của quantize_multiplier, ý nghĩa của kết quả trùng khớp từng bit, và giá trị α tối ưu của SmoothQuant.',
      body,
      nav,
      file: 'ghi-chu.html',
      ...neighbours('ghi-chu.html'),
    })
  );
}
