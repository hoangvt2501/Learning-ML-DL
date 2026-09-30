// Các mảnh HTML dùng chung cho mọi trang phòng thí nghiệm.
// Tách ra để ba trang lab không chép đi chép lại cùng một đoạn.

export const range = (id, label, min, max, value, step, out) =>
  '<label class="ctl ctl-range"><span class="ctl-label">' + label +
  '<b class="ctl-out" id="' + out + '">–</b></span>' +
  '<input type="range" id="' + id + '" min="' + min + '" max="' + max +
  '" value="' + value + '" step="' + (step || 1) + '"></label>';

export const num = (id, label, value, attrs = '') =>
  '<label class="ctl"><span class="ctl-label">' + label + '</span>' +
  '<input type="number" id="' + id + '" value="' + value + '" ' + attrs + '></label>';

export const check = (id, label, checked) =>
  '<label class="ctl ctl-check"><input type="checkbox" id="' + id + '"' +
  (checked ? ' checked' : '') + '><span>' + label + '</span></label>';

export const select = (id, label, options) =>
  '<label class="ctl"><span class="ctl-label">' + label + '</span><select id="' + id + '">' +
  options.map((o) => '<option value="' + o[0] + '"' + (o[2] ? ' selected' : '') + '>' +
    o[1] + '</option>').join('') + '</select></label>';

export const stat = (id, label, hint = '') =>
  '<div class="stat"><span class="stat-label">' + label + '</span>' +
  '<span class="stat-value" id="' + id + '">–</span>' +
  (hint ? '<span class="stat-hint">' + hint + '</span>' : '') + '</div>';

export const btn = (id, label) =>
  '<button class="lab-btn" id="' + id + '" type="button">' + label + '</button>';

export const canvas = (id, h) =>
  '<canvas class="viz" id="' + id + '" style="height:' + h + 'px"></canvas>';

/** Một công cụ: tiêu đề, phần dẫn, thân, và dòng ghi chú động ở cuối. */
export const khoi = (t) => (id, so, ten, dan, than, ghi) =>
  '<section class="lab" id="' + id + '">' +
  '<header class="lab-head"><h3>' + so + ' · ' + t(ten) + '</h3><p>' + t(dan) + '</p></header>' +
  than +
  '<p class="lab-note" id="' + ghi + '"></p>' +
  '</section>';

/** Phần mở đầu chung, kèm lời cảnh báo về độ chính xác của số liệu trong trình duyệt. */
export const mo = (kicker, lede) =>
  '<header class="hero hero--tight">' +
  '<p class="hero-kicker">' + kicker + '</p>' +
  '<h1>Phòng thí nghiệm</h1>' +
  '<p class="hero-lede">' + lede + '</p>' +
  '<p class="hero-lede">Mỗi công cụ gắn với một mục trong giáo trình và cài đúng công thức ở đó. ' +
  'Nhưng chúng chạy trong trình duyệt nên dùng ít mẫu và ít vòng lặp hơn các thí nghiệm ngoại tuyến, ' +
  'vì vậy con số hiện ra <b>sát chứ không trùng khít</b> với bảng đã in. ' +
  'Chỗ nào phải trùng đúng thì có ghi rõ.</p>' +
  '</header>';
