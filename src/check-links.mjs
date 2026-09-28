// Kiểm tra mọi liên kết nội bộ trong docs/ đều trỏ tới tệp và neo có thật.
// Chạy: npm run check   (sau khi đã npm run build)

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs');

if (!fs.existsSync(OUT)) {
  console.error('Chưa có docs/. Chạy "npm run build" trước.');
  process.exit(1);
}

const pages = fs.readdirSync(OUT).filter((f) => f.endsWith('.html'));
const idsOf = {};
for (const f of pages) {
  const html = fs.readFileSync(path.join(OUT, f), 'utf8');
  idsOf[f] = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
}

const broken = [];
let checked = 0;

for (const f of pages) {
  const html = fs.readFileSync(path.join(OUT, f), 'utf8');
  for (const m of html.matchAll(/href="([^"]+)"/g)) {
    const href = m[1];
    if (/^(https?:|mailto:|data:)/.test(href)) continue;

    checked++;
    if (href.startsWith('#')) {
      if (!idsOf[f].has(href.slice(1))) broken.push(f + ' -> ' + href + ' (không có neo)');
      continue;
    }
    const [file, anchor] = href.split('#');
    if (!fs.existsSync(path.join(OUT, file))) {
      broken.push(f + ' -> ' + href + ' (thiếu tệp)');
      continue;
    }
    if (anchor && file.endsWith('.html') && !idsOf[file].has(anchor)) {
      broken.push(f + ' -> ' + href + ' (không có neo)');
    }
  }
}

console.log('Đã kiểm ' + checked + ' liên kết nội bộ trên ' + pages.length + ' trang.');
if (broken.length) {
  console.error('Hỏng ' + broken.length + ':');
  broken.slice(0, 40).forEach((b) => console.error('  ' + b));
  process.exit(1);
}
console.log('Không có liên kết hỏng.');
