/* Trang Bài tập: bộ luyện tính tay tự chấm và bộ lọc trắc nghiệm theo chương. */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* --------------------------------------------------------- tiện ích */

  // Làm tròn nửa về số chẵn, giống NumPy và giống quy tắc IEEE 754 mặc định.
  function rne(v) {
    var f = Math.floor(v);
    var d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }
  function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }
  function rnd(lo, hi, step) {
    var n = Math.round((hi - lo) / step);
    return +(lo + Math.floor(Math.random() * (n + 1)) * step).toFixed(6);
  }
  function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

  // Hiển thị số theo lối Việt Nam (dấu phẩy thập phân) cho phần lời giải.
  function fmt(v, d) {
    if (!isFinite(v)) return '–';
    var s = d === undefined ? String(+v.toPrecision(6)) : v.toFixed(d);
    return s.replace('.', ',');
  }
  // Đọc số người dùng nhập: chấp nhận cả dấu chấm lẫn dấu phẩy thập phân.
  function parseNum(s) {
    if (typeof s !== 'string') return NaN;
    s = s.trim().replace(/\s/g, '').replace(/,/g, '.');
    if (!s) return NaN;
    return Number(s);
  }

  function intRange(bits, mode) {
    if (mode === 'unsigned') return [0, Math.pow(2, bits) - 1];
    return [-Math.pow(2, bits - 1), Math.pow(2, bits - 1) - 1];
  }

  /* ------------------------------------------------------- các dạng bài */

  var DRILLS = {
    sz: function () {
      var alpha = rnd(-5, -0.2, 0.1);
      var beta = rnd(0.5, 6, 0.1);
      var bits = pick([4, 8]);
      var mode = pick(['unsigned', 'signed']);
      var r = intRange(bits, mode);
      var S = (beta - alpha) / (r[1] - r[0]);
      var Z = clamp(rne(r[0] - alpha / S), r[0], r[1]);
      var typeName = (mode === 'unsigned' ? 'uint' : 'int') + bits;

      return {
        question:
          'Lượng tử dải <b>[' + fmt(alpha, 1) + '; ' + fmt(beta, 1) + ']</b> sang <b>' +
          typeName + '</b>, tức miền số nguyên [' + r[0] + ', ' + r[1] + ']. Tính bước nhảy ' +
          '<i>S</i> và zero-point <i>Z</i>.',
        fields: [
          { key: 'S', label: 'S (ít nhất 4 chữ số có nghĩa)' },
          { key: 'Z', label: 'Z (số nguyên)' },
        ],
        answers: {
          S: { value: S, tol: 0.002 },
          Z: { value: Z, exact: true },
        },
        explain:
          '<code>S = (β − α) / (q_max − q_min) = (' + fmt(beta, 1) + ' − (' + fmt(alpha, 1) +
          ')) / (' + r[1] + ' − ' + r[0] + ') = ' + fmt(S) + '</code><br>' +
          '<code>Z = round(q_min − α/S) = round(' + r[0] + ' − (' + fmt(alpha, 1) + ')/' +
          fmt(S) + ') = round(' + fmt(r[0] - alpha / S, 4) + ') = ' + Z + '</code><br>' +
          'Nhớ quy tắc làm tròn nửa về số chẵn nếu kết quả rơi đúng điểm giữa.',
      };
    },

    quant: function () {
      var alpha = rnd(-3, -0.2, 0.1);
      var beta = rnd(0.5, 4, 0.1);
      var bits = pick([4, 8]);
      var r = intRange(bits, 'unsigned');
      var S = (beta - alpha) / (r[1] - r[0]);
      var Z = clamp(rne(r[0] - alpha / S), r[0], r[1]);
      // Một phần ba số đề cố tình cho x nằm ngoài dải để luyện nhận ra sai số cắt.
      var x = Math.random() < 0.3
        ? rnd(beta + 0.2, beta + 2, 0.1)
        : rnd(alpha, beta, 0.05);
      var q = clamp(rne(x / S) + Z, r[0], r[1]);
      var xh = S * (q - Z);
      var clipped = rne(x / S) + Z !== q;

      return {
        question:
          'Với dải <b>[' + fmt(alpha, 1) + '; ' + fmt(beta, 1) + ']</b> sang <b>uint' + bits +
          '</b> ta có <i>S</i> = ' + fmt(S) + ' và <i>Z</i> = ' + Z +
          '. Lượng tử giá trị <b>x = ' + fmt(x, 2) + '</b>: tìm <i>q</i> và <i>x̂</i>.',
        fields: [
          { key: 'q', label: 'q (số nguyên)' },
          { key: 'xh', label: 'x̂ (ít nhất 4 chữ số có nghĩa)' },
        ],
        answers: {
          q: { value: q, exact: true },
          xh: { value: xh, tol: 0.002, abs: S / 4 },
        },
        explain:
          '<code>x/S = ' + fmt(x, 2) + ' / ' + fmt(S) + ' = ' + fmt(x / S, 4) + '</code><br>' +
          '<code>q = clamp(round(' + fmt(x / S, 4) + ') + ' + Z + ', ' + r[0] + ', ' + r[1] +
          ') = ' + q + '</code><br>' +
          '<code>x̂ = S·(q − Z) = ' + fmt(S) + ' · (' + q + ' − ' + Z + ') = ' + fmt(xh) + '</code><br>' +
          (clipped
            ? '<b>Giá trị này bị cắt</b>: phép clamp đã can thiệp, nên sai số ' +
              fmt(Math.abs(xh - x)) + ' lớn hơn hẳn mức làm tròn tối đa S/2 = ' + fmt(S / 2) + '.'
            : 'Không bị cắt: sai số ' + fmt(Math.abs(xh - x)) +
              ' nằm trong mức làm tròn tối đa S/2 = ' + fmt(S / 2) + '.'),
      };
    },

    bits: function () {
      var eb = pick([4, 8]);
      var group = pick([32, 64, 128, 256]);
      var scaleBits = pick([16, 32]);
      var hasZp = Math.random() < 0.5;
      var zpBits = hasZp ? eb : 0;
      var params = pick([1.5, 3, 7, 8, 13, 34, 70]);
      var bpw = eb + (scaleBits + zpBits) / group;
      var bytes = params * 1e9 * bpw / 8;

      return {
        question:
          'Một mô hình <b>' + fmt(params, 1) + ' tỉ tham số</b> được lượng tử <b>INT' + eb +
          '</b> theo nhóm <b>' + group + '</b> phần tử. Mỗi nhóm lưu một scale <b>' + scaleBits +
          ' bit</b>' + (hasZp ? ' và một zero-point <b>' + zpBits + ' bit</b>' : ' (đối xứng, không có zero-point)') +
          '. Tính số bit thực tế mỗi trọng số và dung lượng trọng số theo GB (1 GB = 10⁹ byte).',
        fields: [
          { key: 'bpw', label: 'Bit thực tế / trọng số' },
          { key: 'gb', label: 'Dung lượng (GB)' },
        ],
        answers: {
          bpw: { value: bpw, tol: 0.005 },
          gb: { value: bytes / 1e9, tol: 0.01 },
        },
        explain:
          '<code>bit/trọng số = ' + eb + ' + (' + scaleBits + (hasZp ? ' + ' + zpBits : '') +
          ') / ' + group + ' = ' + fmt(bpw) + '</code><br>' +
          '<code>dung lượng = ' + fmt(params, 1) + 'e9 × ' + fmt(bpw) + ' / 8 = ' +
          fmt(bytes / 1e9) + ' GB</code><br>' +
          'Đây chính là cách bảng ở Mục 11.8 được lập. Phần phụ trội cho scale là lý do ' +
          '“mô hình 4 bit” thực tế luôn tốn hơn 4 bit mỗi trọng số.',
      };
    },

    kv: function () {
      var layers = pick([24, 28, 32, 40, 60, 80]);
      var heads = pick([8, 16, 32, 40]);
      var dim = pick([64, 128]);
      var tokens = pick([1024, 2048, 4096, 8192, 16384]);
      var batch = pick([1, 2, 4, 8]);
      var bytes = pick([1, 2]);
      var perToken = 2 * layers * heads * dim * bytes;
      var total = perToken * tokens * batch;

      return {
        question:
          'Một mô hình có <b>' + layers + ' lớp</b>, <b>' + heads + ' KV head</b>, ' +
          '<b>d_head = ' + dim + '</b>. Sinh văn bản với ngữ cảnh <b>' + tokens + ' token</b>, ' +
          'batch <b>' + batch + '</b>, KV cache lưu ở <b>' + (bytes === 2 ? 'FP16 (2 byte)' : 'FP8/INT8 (1 byte)') +
          '</b>. Tính dung lượng mỗi token cho một chuỗi (MiB) và tổng KV cache (GB, 1 GB = 10⁹ byte).',
        fields: [
          { key: 'per', label: 'Mỗi token, mỗi chuỗi (MiB)' },
          { key: 'total', label: 'Tổng KV cache (GB)' },
        ],
        answers: {
          per: { value: perToken / Math.pow(2, 20), tol: 0.01 },
          total: { value: total / 1e9, tol: 0.01 },
        },
        explain:
          '<code>mỗi token = 2 × ' + layers + ' × ' + heads + ' × ' + dim + ' × ' + bytes +
          ' = ' + perToken.toLocaleString('vi-VN') + ' byte = ' +
          fmt(perToken / Math.pow(2, 20)) + ' MiB</code><br>' +
          '<code>tổng = ' + perToken.toLocaleString('vi-VN') + ' × ' + tokens + ' × ' + batch +
          ' = ' + fmt(total / 1e9) + ' GB</code><br>' +
          'Hệ số 2 ở đầu là vì phải lưu <b>cả key lẫn value</b>. Xem Mục 11.7.',
      };
    },
  };

  /* ------------------------------------------------------------- điều khiển */

  var root = $('[data-drill-root]');
  if (!root) return;

  var qEl = $('[data-drill-question]', root);
  var fieldsEl = $('[data-drill-fields]', root);
  var fbEl = $('[data-drill-feedback]', root);
  var current = null;
  var currentType = 'sz';

  function render() {
    current = DRILLS[currentType]();
    qEl.innerHTML = current.question;
    fieldsEl.innerHTML = current.fields
      .map(function (f) {
        return '<div class="drill-field" data-field="' + f.key + '">' +
          '<label for="drill-' + f.key + '">' + f.label + '</label>' +
          '<input id="drill-' + f.key + '" type="text" inputmode="decimal" autocomplete="off"></div>';
      })
      .join('');
    fbEl.innerHTML = '';
  }

  function check() {
    if (!current) return;
    var allRight = true;
    var missing = false;

    current.fields.forEach(function (f) {
      var wrap = $('[data-field="' + f.key + '"]', fieldsEl);
      var val = parseNum($('input', wrap).value);
      var spec = current.answers[f.key];
      wrap.classList.remove('is-ok', 'is-bad');
      if (!isFinite(val)) { missing = true; allRight = false; return; }

      var ok;
      if (spec.exact) {
        ok = Math.round(val) === Math.round(spec.value);
      } else {
        var relTol = Math.abs(spec.value) * spec.tol;
        var tol = Math.max(relTol, spec.abs || 0, 1e-12);
        ok = Math.abs(val - spec.value) <= tol;
      }
      wrap.classList.add(ok ? 'is-ok' : 'is-bad');
      if (!ok) allRight = false;
    });

    if (missing) {
      fbEl.innerHTML = '<span class="bad">Chưa nhập đủ.</span>';
      return;
    }
    fbEl.innerHTML = allRight
      ? '<span class="ok">Chính xác.</span><span class="why">' + current.explain + '</span>'
      : '<span class="bad">Chưa đúng.</span> Ô viền đỏ là ô sai — thử lại, hoặc bấm ' +
        '<b>Xem đáp án</b>.';
  }

  function reveal() {
    if (!current) return;
    current.fields.forEach(function (f) {
      var wrap = $('[data-field="' + f.key + '"]', fieldsEl);
      var spec = current.answers[f.key];
      $('input', wrap).value = spec.exact ? String(Math.round(spec.value)) : fmt(spec.value);
      wrap.classList.remove('is-bad');
      wrap.classList.add('is-ok');
    });
    fbEl.innerHTML = '<span class="why">' + current.explain + '</span>';
  }

  $$('.drill-tab', root).forEach(function (tab) {
    tab.addEventListener('click', function () {
      $$('.drill-tab', root).forEach(function (t) { t.classList.remove('is-active'); });
      tab.classList.add('is-active');
      currentType = tab.dataset.drill;
      render();
    });
  });

  $('[data-drill-check]', root).addEventListener('click', check);
  $('[data-drill-new]', root).addEventListener('click', render);
  $('[data-drill-show]', root).addEventListener('click', reveal);
  fieldsEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); check(); }
  });

  render();

  /* ----------------------------------------------- lọc trắc nghiệm */
  var filter = $('[data-quiz-filter]');
  if (filter) {
    filter.addEventListener('click', function (e) {
      var btn = e.target.closest('[data-filter]');
      if (!btn) return;
      $$('[data-filter]', filter).forEach(function (b) { b.classList.remove('is-active'); });
      btn.classList.add('is-active');
      var want = btn.dataset.filter;
      $$('[data-quiz-chapter]').forEach(function (wrap) {
        wrap.hidden = want !== 'all' && wrap.dataset.quizChapter !== want;
      });
    });
  }
})();
