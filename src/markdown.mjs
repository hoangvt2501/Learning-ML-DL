// Bộ chuyển Markdown -> HTML dùng riêng cho giáo trình này.
// Ba việc nó làm thêm so với markdown-it mặc định:
//   1. Gộp ảnh + đoạn chú thích "**Hình N.** ..." thành một <figure> có id tra cứu được.
//   2. Biến mọi cụm "Mục 6.2", "Chương 11", "Hình 7" trong bài thành liên kết thật.
//   3. Tô màu cú pháp sẵn lúc build (không cần tải highlight.js ở trình duyệt).

import MarkdownIt from 'markdown-it';
import katexPlugin from '@vscode/markdown-it-katex';
import hljs from 'highlight.js';

/** Bỏ dấu tiếng Việt và chuyển một tiêu đề thành slug dùng cho id/anchor. */
export function slugify(text) {
  return text
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'D')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/** "6.2" -> "sec-6-2" */
export const sectionId = (num) => 'sec-' + num.replace(/\./g, '-');
/** 7 -> "hinh-7" */
export const figureId = (num) => 'hinh-' + num;

/**
 * registry = {
 *   sections: Map<"6.2", {href, title}>,
 *   figures:  Map<"7",   {href, caption}>,
 *   currentFile: string   (đặt lại trước mỗi lần render)
 * }
 */
/**
 * Trích dẫn khối được dùng xuyên suốt làm "chú thích quan trọng". Ở đây ta gắn
 * cho nó một lớp theo NỘI DUNG mở đầu, để nó hiện ra như một hộp có kiểu thay vì
 * một khối chữ nghiêng lẫn vào bài — đây là cách các trang tài liệu kỹ thuật làm.
 */
const LOAI_HOP = [
  // [biểu thức nhận dạng chữ mở đầu, loại, nhãn hiện ra, icon]
  [/^(định nghĩa)/i, 'dinh-nghia', 'Định nghĩa', 'dinh-nghia'],
  [/^(định lý|mệnh đề|bổ đề|hệ quả toán)/i, 'dinh-ly', 'Định lý', 'dinh-ly'],
  [/^(chứng minh)/i, 'chung-minh', 'Chứng minh', 'chung-minh'],
  [/^(ví dụ)/i, 'vi-du', 'Ví dụ', 'vi-du'],
  [/^(lưu ý|cảnh báo|cẩn thận|chú ý|đừng|không nên|sai lầm)/i, 'luu-y', 'Lưu ý', 'luu-y'],
  [/^(câu hỏi phỏng vấn|khi phỏng vấn|trả lời phỏng vấn)/i, 'phong-van', 'Phỏng vấn', 'phong-van'],
  [/^(trả lời|gợi ý trả lời|đáp án)/i, 'tra-loi', 'Trả lời', 'tra-loi'],
];
const HOP_MAC_DINH = ['nhan-xet', 'Nhận xét', 'nhan-xet'];

/**
 * Khối trích dẫn trong bài được dựng thành hộp kiểu giáo trình — Định nghĩa,
 * Định lý, Chứng minh, Ví dụ, Nhận xét, Lưu ý — nhận dạng theo chữ in đậm mở đầu.
 * Khối không có nhãn thì là Nhận xét.
 */
/**
 * Chữ in đậm mở đầu hộp kiểu tiêu đề chạy ("**Định nghĩa 2.1 (Tích vô hướng).**")
 * được nhấc lên làm dòng tiêu đề của hộp và bỏ khỏi đoạn văn, để không lặp
 * "Định nghĩa" hai lần. Chỉ nhấc khi chữ đậm kết thúc bằng dấu chấm hoặc hai
 * chấm, tức đúng là một tiêu đề chạy chứ không phải một cụm được nhấn mạnh.
 */
function nhacTieuDe(tokens, idx, md, options, env) {
  const inline = tokens[idx + 2];
  if (!inline || inline.type !== 'inline' || !inline.children) return null;
  const ch = inline.children;
  // bỏ qua các mẩu chữ rỗng mà các luật inline khác để lại ở đầu
  let a = 0;
  while (a < ch.length && ch[a].type === 'text' && !ch[a].content.trim()) a++;
  if (a >= ch.length || ch[a].type !== 'strong_open') return null;
  let k = a + 1;
  while (k < ch.length && ch[k].type !== 'strong_close') k++;
  if (k >= ch.length) return null;
  const trong = ch.slice(a + 1, k).map((t) => t.content).join('');
  const sau = ch[k + 1] && ch[k + 1].type === 'text' ? ch[k + 1].content : '';
  if (!/[.:]\s*$/.test(trong) && !/^\s*[.:]/.test(sau)) return null;
  const html = md.renderer.renderInline(ch.slice(a + 1, k), options, env)
    .replace(/[.:]\s*$/, '').trim();
  ch.splice(0, k + 1);
  if (ch.length && ch[0].type === 'text') {
    ch[0].content = ch[0].content.replace(/^\s*[.:]?\s*/, '');
    if (!ch[0].content) ch.shift();
  }
  if (ch.length && ch[0].type === 'softbreak') ch.shift();
  // Đoạn chỉ có mỗi tiêu đề (theo sau là danh sách, công thức...) thì bỏ luôn thẻ <p> rỗng.
  if (!ch.length && tokens[idx + 1].type === 'paragraph_open') {
    tokens[idx + 1].hidden = true;
    if (tokens[idx + 3] && tokens[idx + 3].type === 'paragraph_close') tokens[idx + 3].hidden = true;
  }
  return html;
}

function loaiHop(tokens, idx) {
  for (let i = idx + 1; i < tokens.length; i++) {
    const tk = tokens[i];
    if (tk.type === 'blockquote_close') break;
    if (tk.type === 'inline') {
      const txt = tk.content.replace(/^[*_\s]+/, '').trim();
      for (const [re, loai, nhan, ic] of LOAI_HOP) if (re.test(txt)) return [loai, nhan, ic];
      return HOP_MAC_DINH;
    }
  }
  return HOP_MAC_DINH;
}

import { icon } from './icons.mjs';

export function createMarkdownIt(registry) {
  const md = new MarkdownIt({ html: false, linkify: false, typographer: false });

  md.use(katexPlugin.default ?? katexPlugin, {
    // Công thức trong bài có \text{...} chứa tiếng Việt; KaTeX không có metric cho
    // các ký tự đó nên sẽ cảnh báo. Chữ vẫn hiện đúng nhờ font dự phòng khai báo
    // trong style.css, nên tắt cảnh báo cho gọn log.
    katexOptions: { strict: 'ignore', throwOnError: false },
    strict: 'ignore',
    throwOnError: false,
  });
  // Riêng cảnh báo "No character metrics" được KaTeX in thẳng bằng console.warn,
  // không theo tuỳ chọn strict, nên phải lọc ở đây.
  if (!console.warn.__locKatex) {
    const warn = console.warn;
    console.warn = (...a) => {
      if (typeof a[0] === 'string' && a[0].startsWith('No character metrics')) return;
      warn(...a);
    };
    console.warn.__locKatex = true;
  }

  // ---------------------------------------------------------------- tiêu đề
  md.renderer.rules.heading_open = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const inline = tokens[idx + 1];
    const text = inline ? inline.content : '';
    const num = text.match(/^(\d+(?:\.\d+)*)\.\s*/);
    const id = num ? sectionId(num[1]) : slugify(text) || 'phan-' + idx;
    token.attrSet('id', id);
    token.attrJoin('class', 'heading');
    // Số mục tách ra một ô riêng, bỏ dấu chấm cuối: "4.1. Tiêu đề" -> "4.1  Tiêu đề",
    // đúng lối đánh số của giáo trình in.
    if (num && inline && inline.children && inline.children.length) {
      const dau = inline.children[0];
      if (dau.type === 'text' && dau.content.startsWith(num[0])) {
        dau.content = dau.content.slice(num[0].length);
      }
      return self.renderToken(tokens, idx, options) +
        '<span class="hnum">' + num[1] + '</span><span class="htext">';
    }
    return self.renderToken(tokens, idx, options) + '<span class="htext">';
  };

  md.renderer.rules.heading_close = (tokens, idx) => {
    const open = findOpen(tokens, idx);
    const id = open ? open.attrGet('id') : null;
    const anchor = id
      ? '<a class="anchor" href="#' + id + '" aria-label="Liên kết tới mục này">#</a>'
      : '';
    return anchor + '</span></' + tokens[idx].tag + '>\n';
  };

  // Khối trích dẫn -> hộp chú thích có kiểu, nhận dạng theo chữ mở đầu.
  md.renderer.rules.blockquote_open = (tokens, idx, options, env, self) => {
    const [loai, nhan, ic] = loaiHop(tokens, idx);
    const tieuDe = nhacTieuDe(tokens, idx, md, options, env);
    tokens[idx].attrJoin('class', 'callout callout--' + loai);
    return self.renderToken(tokens, idx, options) +
      '<p class="callout-label">' + icon(ic) + '<span>' + (tieuDe || nhan) + '</span></p>';
  };

  // ---------------------------------------------------------------- bảng
  md.renderer.rules.table_open = () => '<div class="table-wrap"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';

  // ---------------------------------------------------------------- ảnh
  md.renderer.rules.image = (tokens, idx) => {
    const token = tokens[idx];
    const src = token.attrGet('src');
    const alt = token.content || '';
    return (
      '<img src="' + md.utils.escapeHtml(src) + '" alt="' +
      md.utils.escapeHtml(alt) + '" loading="lazy" decoding="async">'
    );
  };

  // ---------------------------------------------------------------- code
  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const info = (token.info || '').trim().split(/\s+/)[0];
    const code = token.content;

    if (!info) {
      // Khối không khai báo ngôn ngữ. Trong giáo trình đây luôn là kết quả in ra
      // của đoạn mã đứng trước, trừ sơ đồ cây thư mục ở phần Phụ lục.
      const isTree = /[├└│]/.test(code);
      const label = isTree ? 'Cấu trúc thư mục' : 'Kết quả chạy';
      const cls = isTree ? 'tree' : 'output';
      return (
        '<div class="code-block is-' + cls + '">' +
        '<div class="code-head"><span class="code-lang">' + label + '</span></div>' +
        '<pre><code>' + md.utils.escapeHtml(code) + '</code></pre></div>\n'
      );
    }

    let highlighted;
    try {
      highlighted = hljs.highlight(code, { language: info, ignoreIllegals: true }).value;
    } catch {
      highlighted = md.utils.escapeHtml(code);
    }
    const names = { python: 'Python', bash: 'Bash', text: 'Văn bản' };
    return (
      '<div class="code-block" data-lang="' + md.utils.escapeHtml(info) + '">' +
      '<div class="code-head"><span class="code-lang">' + (names[info] || info) + '</span>' +
      '<button class="code-copy" type="button" data-copy>Sao chép</button></div>' +
      '<pre><code class="hljs language-' + md.utils.escapeHtml(info) + '">' +
      highlighted + '</code></pre></div>\n'
    );
  };

  // ------------------------------------------------- gộp ảnh + chú thích
  md.core.ruler.push('figures', (state) => groupFigures(state, registry));
  // ------------------------------------------------- liên kết chéo trong bài
  md.core.ruler.push('xref', (state) => linkCrossReferences(state, registry));

  return md;
}

function findOpen(tokens, closeIdx) {
  for (let i = closeIdx - 1; i >= 0; i--) {
    if (tokens[i].type === 'heading_open') return tokens[i];
  }
  return null;
}

/** Nhận ra cặp [đoạn chỉ có ảnh] + [đoạn mở đầu bằng **Hình N.**] và gói vào <figure>. */
function groupFigures(state, registry) {
  const toks = state.tokens;
  for (let i = 0; i < toks.length - 5; i++) {
    if (toks[i].type !== 'paragraph_open') continue;
    const imgInline = toks[i + 1];
    if (!imgInline || imgInline.type !== 'inline') continue;
    const kids = imgInline.children.filter((t) => t.type !== 'text' || t.content.trim());
    if (kids.length !== 1 || kids[0].type !== 'image') continue;
    if (toks[i + 2].type !== 'paragraph_close') continue;
    if (toks[i + 3].type !== 'paragraph_open') continue;
    const capInline = toks[i + 4];
    if (!capInline || capInline.type !== 'inline') continue;

    const m = capInline.content.match(/^\*\*Hình (\d+)\.\*\*/);
    if (!m) continue;
    const num = m[1];

    const Token = state.Token;
    const figOpen = new Token('figure_open', 'figure', 1);
    figOpen.block = true;
    figOpen.attrSet('class', 'figure');
    figOpen.attrSet('id', figureId(num));

    const frameOpen = new Token('figframe_open', 'div', 1);
    frameOpen.block = true;
    frameOpen.attrSet('class', 'figure-frame');
    frameOpen.attrSet('role', 'button');
    frameOpen.attrSet('tabindex', '0');
    frameOpen.attrSet('data-zoom', '');
    frameOpen.attrSet('aria-label', 'Phóng to Hình ' + num);
    const frameClose = new Token('figframe_close', 'div', -1);
    frameClose.block = true;

    const capOpen = new Token('figcaption_open', 'figcaption', 1);
    capOpen.block = true;
    const capClose = new Token('figcaption_close', 'figcaption', -1);
    capClose.block = true;

    const figClose = new Token('figure_close', 'figure', -1);
    figClose.block = true;

    const replacement = [
      figOpen, frameOpen, imgInline, frameClose, capOpen, capInline, capClose, figClose,
    ];
    toks.splice(i, 6, ...replacement);

    // Sổ tra cứu đã được dựng sẵn ở bước quét trước khi render; chỉ ghi
    // ở đây khi render một nguồn nằm ngoài giáo trình.
    if (registry && !registry.figures.has(num)) {
      registry.figures.set(num, {
        href: registry.currentFile + '#' + figureId(num),
        caption: capInline.content
          .replace(/^\*\*Hình \d+\.\*\*\s*/, '')
          .replace(/[*_$\\]/g, ''),
        src: kids[0].attrGet('src'),
        file: registry.currentFile,
      });
    }
    i += replacement.length - 1;
  }
}

// "Mục 6.2", "Mục 11.2–11.3", "Chương 6", "Hình 9"
const XREF_RE = /\b(Mục|Chương|Hình)\s+(\d+(?:\.\d+)*)((?:\s*(?:,|–|—|và|-)\s*\d+(?:\.\d+)*)*)/g;

function linkCrossReferences(state, registry) {
  if (!registry) return;
  // Bỏ qua tiêu đề (đã có anchor riêng) và chú thích hình (nếu không thì
  // "**Hình 9.**" mở đầu chú thích sẽ thành liên kết trỏ về chính nó).
  let skip = 0;
  for (const tok of state.tokens) {
    if (tok.type === 'heading_open' || tok.type === 'figcaption_open') skip++;
    else if (tok.type === 'heading_close' || tok.type === 'figcaption_close') skip--;
    else if (tok.type === 'inline' && skip === 0) linkifyInline(tok, registry, state);
  }
}

function linkifyInline(inlineToken, registry, state) {
  const children = inlineToken.children;
  if (!children) return;
  const out = [];
  let linkDepth = 0;

  for (const child of children) {
    if (child.type === 'link_open') linkDepth++;
    if (child.type === 'link_close') linkDepth--;
    if (child.type !== 'text' || linkDepth > 0) {
      out.push(child);
      continue;
    }
    out.push(...splitText(child, registry, state));
  }
  inlineToken.children = out;
}

function splitText(textToken, registry, state) {
  const text = textToken.content;
  XREF_RE.lastIndex = 0;
  if (!XREF_RE.test(text)) return [textToken];
  XREF_RE.lastIndex = 0;

  const Token = state.Token;
  const pieces = [];
  let last = 0;
  let m;

  const pushText = (s) => {
    if (!s) return;
    const t = new Token('text', '', 0);
    t.content = s;
    pieces.push(t);
  };
  const pushLink = (href, label) => {
    const open = new Token('link_open', 'a', 1);
    open.attrSet('href', href);
    open.attrSet('class', 'xref');
    pieces.push(open);
    pushText(label);
    pieces.push(new Token('link_close', 'a', -1));
  };

  while ((m = XREF_RE.exec(text)) !== null) {
    const full = m[0];
    const kind = m[1];
    const firstNum = m[2];
    const tail = m[3];
    const resolve = (n) =>
      kind === 'Hình' ? registry.figures.get(n) : registry.sections.get(n);

    const firstTarget = resolve(firstNum);
    if (!firstTarget) continue; // chưa có mục tiêu thì để nguyên chữ

    pushText(text.slice(last, m.index));
    pushLink(firstTarget.href, kind + ' ' + firstNum);

    // phần đuôi dạng "–11.3" hoặc ", 11"
    if (tail) {
      const tailRe = /(\s*(?:,|–|—|và|-)\s*)(\d+(?:\.\d+)*)/g;
      let tm;
      while ((tm = tailRe.exec(tail)) !== null) {
        pushText(tm[1]);
        const t = resolve(tm[2]);
        if (t) pushLink(t.href, tm[2]);
        else pushText(tm[2]);
      }
    }
    last = m.index + full.length;
  }
  pushText(text.slice(last));
  return pieces.length ? pieces : [textToken];
}
