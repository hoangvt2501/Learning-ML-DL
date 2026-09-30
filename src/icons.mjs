// Bộ icon của site, lấy từ Lucide (giấy phép ISC, https://lucide.dev).
//
// Chỉ nhúng đúng những icon được dùng, dưới dạng một SVG sprite chèn thẳng vào
// trang. Không tải gì từ mạng, nên site vẫn chạy khi mở bằng file://.
// Gọi icon('ten') để lấy thẻ <svg> tham chiếu vào sprite.

import fs from 'node:fs';
import path from 'node:path';

/** Tên dùng trong site -> tên tệp của Lucide. */
const BANG = {
  // điều hướng chung
  'nha': 'house',
  'lo-trinh': 'route',
  'tim': 'search',
  'sang': 'sun',
  'toi': 'moon',
  'menu': 'menu',
  'truoc': 'arrow-left',
  'sau': 'arrow-right',
  'dong-ho': 'clock-3',
  'chep': 'copy',
  'xong': 'check',
  'ngoai': 'arrow-up-right',
  'mo-rong': 'maximize-2',

  // trang phụ của mỗi giáo trình
  'chuong': 'book-open',
  'thi-nghiem': 'flask-conical',
  'hinh': 'images',
  'thuat-ngu': 'book-a',
  'ghi-chu': 'notebook-pen',
  'ma-nguon': 'file-code-2',
  'toan-van': 'scroll-text',
  'bai-tap': 'pencil-line',
  'trac-nghiem': 'list-checks',
  'phong-van': 'messages-square',
  'tai-lieu': 'library',
  'phu-luc': 'paperclip',

  // hộp trong bài
  'dinh-nghia': 'book-marked',
  'dinh-ly': 'sigma',
  'vi-du': 'lightbulb',
  'nhan-xet': 'info',
  'luu-y': 'triangle-alert',
  'chung-minh': 'square-check-big',
  'tra-loi': 'message-circle-reply',

  // các chặng của lộ trình
  'nen-tang': 'sigma',
  'hoc-may': 'chart-scatter',
  'mang-no-ron': 'network',
  'llm': 'cpu',
  'prompt': 'message-square-text',
  'ngu-canh': 'layers',
  'mo-hinh': 'boxes',
  'embedding': 'waypoints',
  'vector-db': 'database',
  'rag': 'file-search',
  'agent': 'bot',
  'mcp': 'plug',
  'an-toan': 'shield-check',
  'danh-gia': 'gauge',
  'da-phuong-thuc': 'image-play',
  'tinh-chinh': 'sliders-horizontal',
  'mo-hinh-sinh': 'sparkles',
  'luong-tu': 'binary',
  'van-hanh': 'server-cog',
  'gioi-thieu': 'compass',
};

let _sprite = null;

/** Đọc các tệp SVG của Lucide và gộp thành một sprite <symbol>. */
export function buildSprite(ROOT) {
  const dir = path.join(ROOT, 'node_modules/lucide-static/icons');
  const symbols = [];
  for (const [ten, tep] of Object.entries(BANG)) {
    const p = path.join(dir, tep + '.svg');
    if (!fs.existsSync(p)) throw new Error('Không có icon Lucide: ' + tep);
    const svg = fs.readFileSync(p, 'utf8');
    const than = svg
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(/^[\s\S]*?<svg[^>]*>/, '')
      .replace(/<\/svg>\s*$/, '')
      .replace(/\s+/g, ' ')
      .trim();
    symbols.push('<symbol id="i-' + ten + '" viewBox="0 0 24 24">' + than + '</symbol>');
  }
  _sprite =
    '<svg xmlns="http://www.w3.org/2000/svg" style="display:none" aria-hidden="true">' +
    '<defs>' + symbols.join('') + '</defs></svg>';
  return _sprite;
}

export function sprite() {
  if (!_sprite) throw new Error('Chưa dựng sprite icon');
  return _sprite;
}

/** Một icon; `cls` để chỉnh cỡ, `nhan` để đọc màn hình đọc được. */
export function icon(ten, cls = '', nhan = '') {
  if (!BANG[ten]) throw new Error('Icon chưa khai báo: ' + ten);
  const a11y = nhan ? ' role="img" aria-label="' + nhan + '"' : ' aria-hidden="true"';
  return '<svg class="ic' + (cls ? ' ' + cls : '') + '"' + a11y +
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"' +
    ' stroke-linejoin="round"><use href="#i-' + ten + '"></use></svg>';
}

export const TEN_ICON = Object.keys(BANG);
