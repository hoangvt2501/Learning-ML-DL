/* Năm công cụ của Phòng thí nghiệm.
   Mọi công thức cài đặt đúng theo giáo trình; phép làm tròn dùng quy tắc
   "nửa về số chẵn" giống NumPy và IEEE 754. */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['playground'] = function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  /* =====================================================  tiện ích chung */

  function rne(v) {                       // round-half-to-even
    var f = Math.floor(v);
    var d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }
  function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

  /** Số theo lối Việt Nam: dấu phẩy thập phân. */
  function fmt(v, digits) {
    if (!isFinite(v)) return '–';
    var s;
    if (digits !== undefined) s = v.toFixed(digits);
    else if (v !== 0 && (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e7)) s = v.toExponential(4);
    else s = String(+v.toPrecision(6));
    return s.replace('.', ',').replace(/e([+-])/, '·10^$1');
  }
  function fmtInt(v) { return Math.round(v).toLocaleString('vi-VN'); }

  function setStat(id, text, cls) {
    var el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className = 'stat-value' + (cls ? ' ' + cls : '');
  }

  /* ---- SVG ---- */
  function svg(w, h, inner) {
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" role="img" preserveAspectRatio="xMidYMid meet">' +
      inner + '</svg>';
  }
  function path(points, cls) {
    if (!points.length) return '';
    return '<path class="' + cls + '" d="M' +
      points.map(function (p) { return p[0].toFixed(2) + ' ' + p[1].toFixed(2); }).join('L') + '"/>';
  }
  function line(x1, y1, x2, y2, cls) {
    return '<line class="' + cls + '" x1="' + x1.toFixed(2) + '" y1="' + y1.toFixed(2) +
      '" x2="' + x2.toFixed(2) + '" y2="' + y2.toFixed(2) + '"/>';
  }
  function text(x, y, s, anchor) {
    return '<text class="chart-label" x="' + x.toFixed(1) + '" y="' + y.toFixed(1) +
      '" text-anchor="' + (anchor || 'middle') + '">' + s + '</text>';
  }
  function rect(x, y, w, h, cls) {
    if (w <= 0) return '';
    return '<rect class="' + cls + '" x="' + x.toFixed(2) + '" y="' + y.toFixed(2) +
      '" width="' + w.toFixed(2) + '" height="' + h.toFixed(2) + '"/>';
  }

  /* ==========================================  1. máy lượng tử affine */

  function affineParams(alpha, beta, bits, mode) {
    var qmin, qmax, S, Z;
    if (mode === 'symmetric') {
      qmax = Math.pow(2, bits - 1) - 1;
      qmin = -qmax;
      var c = Math.max(Math.abs(alpha), Math.abs(beta));
      S = c / qmax;
      Z = 0;
    } else {
      if (mode === 'unsigned') { qmin = 0; qmax = Math.pow(2, bits) - 1; }
      else { qmin = -Math.pow(2, bits - 1); qmax = Math.pow(2, bits - 1) - 1; }
      var a = Math.min(alpha, 0);      // dải luôn phải chứa số 0 (Mục 3.4)
      var b = Math.max(beta, 0);
      S = (b - a) / (qmax - qmin);
      Z = clamp(rne(qmin - a / S), qmin, qmax);
    }
    return { qmin: qmin, qmax: qmax, S: S, Z: Z };
  }

  function quantize(x, p) { return clamp(rne(x / p.S) + p.Z, p.qmin, p.qmax); }
  function dequantize(q, p) { return p.S * (q - p.Z); }

  function drawAffine(p, x) {
    var W = 560, H = 320, L = 42, R = 12;
    var top = { y0: 14, y1: 190 };
    var bot = { y0: 214, y1: 296 };

    var repLo = dequantize(p.qmin, p);
    var repHi = dequantize(p.qmax, p);
    var span = repHi - repLo || 1;
    var lo = repLo - 0.22 * span;
    var hi = repHi + 0.22 * span;

    var sx = function (v) { return L + (v - lo) / (hi - lo) * (W - L - R); };

    var N = 460;
    var xs = [], ys = [], errs = [];
    for (var i = 0; i <= N; i++) {
      var v = lo + (hi - lo) * i / N;
      var xh = dequantize(quantize(v, p), p);
      xs.push(v); ys.push(xh); errs.push(xh - v);
    }

    var yMin = Math.min(lo, Math.min.apply(null, ys));
    var yMax = Math.max(hi, Math.max.apply(null, ys));
    var sy = function (v) { return top.y1 - (v - yMin) / (yMax - yMin || 1) * (top.y1 - top.y0); };

    var eAbs = Math.max.apply(null, errs.map(Math.abs)) || 1;
    var se = function (v) { return (bot.y0 + bot.y1) / 2 - v / eAbs * ((bot.y1 - bot.y0) / 2 - 4); };

    var parts = [];

    // vùng bị cắt
    parts.push(rect(sx(lo), top.y0, sx(repLo) - sx(lo), top.y1 - top.y0, 'chart-clip'));
    parts.push(rect(sx(repHi), top.y0, sx(hi) - sx(repHi), top.y1 - top.y0, 'chart-clip'));
    parts.push(rect(sx(lo), bot.y0, sx(repLo) - sx(lo), bot.y1 - bot.y0, 'chart-clip'));
    parts.push(rect(sx(repHi), bot.y0, sx(hi) - sx(repHi), bot.y1 - bot.y0, 'chart-clip'));

    // trục
    if (yMin <= 0 && yMax >= 0) parts.push(line(L, sy(0), W - R, sy(0), 'chart-grid'));
    if (lo <= 0 && hi >= 0) parts.push(line(sx(0), top.y0, sx(0), top.y1, 'chart-grid'));
    parts.push(line(L, top.y1, W - R, top.y1, 'chart-axis'));
    parts.push(line(L, bot.y1, W - R, bot.y1, 'chart-axis'));
    parts.push(line(L, se(0), W - R, se(0), 'chart-grid'));

    // ±S/2
    parts.push(line(L, se(p.S / 2), W - R, se(p.S / 2), 'chart-mark2'));
    parts.push(line(L, se(-p.S / 2), W - R, se(-p.S / 2), 'chart-mark2'));

    // đường lý tưởng x̂ = x
    parts.push(path([[sx(Math.max(lo, yMin)), sy(Math.max(lo, yMin))],
                     [sx(Math.min(hi, yMax)), sy(Math.min(hi, yMax))]], 'chart-ideal'));

    // bậc thang + sai số
    parts.push(path(xs.map(function (v, i) { return [sx(v), sy(ys[i])]; }), 'chart-line'));
    parts.push(path(xs.map(function (v, i) { return [sx(v), se(errs[i])]; }), 'chart-err'));

    // điểm đang chọn
    if (x >= lo && x <= hi) {
      var xh2 = dequantize(quantize(x, p), p);
      parts.push(line(sx(x), top.y0, sx(x), bot.y1, 'chart-mark'));
      parts.push('<circle class="chart-dot" cx="' + sx(x).toFixed(2) + '" cy="' + sy(xh2).toFixed(2) + '" r="4"/>');
    }

    // nhãn
    parts.push(text(L - 4, top.y0 + 8, 'x̂', 'end'));
    parts.push(text(L - 4, bot.y0 + 8, 'x̂−x', 'end'));
    parts.push(text(sx(repLo), bot.y1 + 12, fmt(repLo, 2)));
    parts.push(text(sx(repHi), bot.y1 + 12, fmt(repHi, 2)));
    parts.push(text(W - R, top.y0 + 8, '±S/2 = ±' + fmt(p.S / 2, 4), 'end'));

    return svg(W, H, parts.join(''));
  }

  function updateAffine() {
    if (!$('afAlpha')) return;
    var alpha = Number($('afAlpha').value);
    var beta = Number($('afBeta').value);
    var bits = Number($('afBits').value);
    var mode = $('afMode').value;
    var x = Number($('afX').value);
    var note = $('afNote');

    if (!isFinite(alpha) || !isFinite(beta) || beta <= alpha) {
      note.innerHTML = '<b>Dải không hợp lệ:</b> cần β &gt; α.';
      ['afS', 'afZ', 'afRange', 'afQ', 'afXhat', 'afErr'].forEach(function (id) { setStat(id, '–'); });
      $('afChart').innerHTML = '';
      return;
    }

    var p = affineParams(alpha, beta, bits, mode);
    var q = quantize(x, p);
    var xh = dequantize(q, p);
    var repLo = dequantize(p.qmin, p);
    var repHi = dequantize(p.qmax, p);
    var clipped = rne(x / p.S) + p.Z !== q;

    setStat('afS', fmt(p.S));
    setStat('afZ', String(p.Z));
    setStat('afRange', '[' + p.qmin + ', ' + p.qmax + ']');
    setStat('afQ', String(q));
    setStat('afXhat', fmt(xh));
    setStat('afErr', fmt(xh - x), Math.abs(xh - x) > p.S / 2 + 1e-12 ? 'is-warn' : 'is-accent');

    var msgs = [];
    msgs.push('Dải thực sự biểu diễn được là <b>[' + fmt(repLo, 4) + '; ' + fmt(repHi, 4) + ']</b>.');
    if (mode !== 'symmetric') {
      var Zexact = p.qmin - Math.min(alpha, 0) / p.S;
      if (Math.abs(Zexact - p.Z) > 1e-9) {
        var up = p.Z > Zexact;
        msgs.push('Z được làm tròn ' + (up ? 'lên' : 'xuống') + ' từ ' + fmt(Zexact, 4) +
          ' thành ' + p.Z + ', nên toàn bộ lưới dịch ' + (up ? 'sang trái' : 'sang phải') + ' ' +
          fmt(Math.abs(p.Z - Zexact) * p.S, 4) + ' so với dải bạn nhập.');
      }
    }
    if (clipped) {
      msgs.push('<b>x = ' + fmt(x) + ' bị cắt.</b> Sai số ' + fmt(Math.abs(xh - x)) +
        ' lớn hơn hẳn mức làm tròn tối đa S/2 = ' + fmt(p.S / 2) + '.');
    } else if (Math.abs(xh - x) < 1e-12) {
      msgs.push('x nằm <b>đúng trên lưới</b> nên sai số bằng 0.');
    }
    if (mode === 'symmetric' && alpha < 0 && beta > 0 && Math.abs(alpha) !== Math.abs(beta)) {
      msgs.push('Lưới đối xứng dùng c = max(|α|, |β|) = ' +
        fmt(Math.max(Math.abs(alpha), Math.abs(beta))) + ', nên phía hẹp hơn bị phí mức — đúng hiện tượng ở Hình 5.');
    }
    note.innerHTML = msgs.join(' ');
    $('afChart').innerHTML = drawAffine(p, x);
  }

  /* =======================================  2. đánh đổi làm tròn – cắt */

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  var sampleCache = {};
  function samples(dist) {
    if (sampleCache[dist]) return sampleCache[dist];
    var rand = mulberry32(1207);
    var n = 50000;
    var a = new Float64Array(n);
    if (dist === 'laplace') {
      // Nghịch đảo hàm phân phối Laplace(0, 1) — dạng "nhọn đỉnh, đuôi dày"
      // giống trọng số của mạng đã huấn luyện.
      for (var i = 0; i < n; i++) {
        var u = rand() - 0.5;
        a[i] = -Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
      }
    } else {
      for (var j = 0; j < n; j++) {
        var u1 = Math.max(rand(), 1e-12), u2 = rand();
        var g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        a[j] = dist === 'relu' ? Math.max(0, g) : g;
      }
    }
    sampleCache[dist] = a;
    return a;
  }

  function mseAt(data, c, bits, oneSided) {
    var S, qlo, qhi, Z = 0;
    if (oneSided) { qlo = 0; qhi = Math.pow(2, bits) - 1; S = c / qhi; }
    else { qhi = Math.pow(2, bits - 1) - 1; qlo = -qhi; S = c / qhi; }
    if (!(S > 0)) return Infinity;
    var acc = 0;
    for (var i = 0; i < data.length; i++) {
      var x = data[i];
      var xh = S * (clamp(rne(x / S) + Z, qlo, qhi) - Z);
      var e = xh - x;
      acc += e * e;
    }
    return acc / data.length;
  }

  var clipState = { dist: null, bits: null, curve: null };

  function clipCompute(dist, bits) {
    if (clipState.dist === dist && clipState.bits === bits) return clipState;
    var data = samples(dist);
    var oneSided = dist === 'relu';
    var maxAbs = 0;
    for (var i = 0; i < data.length; i++) maxAbs = Math.max(maxAbs, Math.abs(data[i]));

    var pts = [];
    var steps = 70;
    for (var k = 1; k <= steps; k++) {
      var c = maxAbs * k / steps;
      pts.push([c, mseAt(data, c, bits, oneSided)]);
    }
    var bi = 0;
    for (var m = 1; m < pts.length; m++) if (pts[m][1] < pts[bi][1]) bi = m;

    // Quét mịn quanh cực tiểu thô, để c* không bị kẹt vào mắt lưới quét —
    // nếu không, MSE ở vị trí thanh trượt có thể thấp hơn "MSE nhỏ nhất".
    var lo = pts[Math.max(0, bi - 1)][0];
    var hi = pts[Math.min(pts.length - 1, bi + 1)][0];
    var best = pts[bi];
    for (var r = 0; r <= 60; r++) {
      var cc = lo + (hi - lo) * r / 60;
      var mm = mseAt(data, cc, bits, oneSided);
      if (mm < best[1]) best = [cc, mm];
    }

    clipState = {
      dist: dist, bits: bits, data: data, oneSided: oneSided,
      maxAbs: maxAbs, curve: pts, best: best,
      minmax: mseAt(data, maxAbs, bits, oneSided),
    };
    return clipState;
  }

  function drawClip(st, c) {
    var W = 560, H = 250, L = 52, R = 14, T = 14, B = 34;
    var xs = st.curve.map(function (p) { return p[0]; });
    var ys = st.curve.map(function (p) { return Math.log10(p[1]); });
    var xMin = 0, xMax = st.maxAbs * 1.03;
    var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
    var pad = (yMax - yMin) * 0.08 || 0.1;
    yMin -= pad; yMax += pad;

    var sx = function (v) { return L + (v - xMin) / (xMax - xMin) * (W - L - R); };
    var sy = function (v) { return H - B - (v - yMin) / (yMax - yMin) * (H - T - B); };

    var SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    var sup = function (n) {
      return String(n).split('').map(function (ch) { return SUP[ch] || ch; }).join('');
    };

    var parts = [];
    for (var g = Math.ceil(yMin); g <= yMax; g++) {
      parts.push(line(L, sy(g), W - R, sy(g), 'chart-grid'));
      parts.push(text(L - 6, sy(g) + 3, '10' + sup(g), 'end'));
    }
    parts.push(line(L, H - B, W - R, H - B, 'chart-axis'));
    parts.push(path(st.curve.map(function (p) { return [sx(p[0]), sy(Math.log10(p[1]))]; }), 'chart-line'));

    parts.push(line(sx(st.best[0]), T, sx(st.best[0]), H - B, 'chart-mark'));
    parts.push(text(sx(st.best[0]), T + 9, 'c* = ' + fmt(st.best[0], 2)));
    parts.push(line(sx(st.maxAbs), T, sx(st.maxAbs), H - B, 'chart-mark2'));
    parts.push(text(sx(st.maxAbs), H - B + 24, 'max|x|', 'end'));

    var mse = mseAt(st.data, c, st.bits, st.oneSided);
    parts.push('<circle class="chart-dot" cx="' + sx(c).toFixed(2) + '" cy="' +
      sy(Math.log10(mse)).toFixed(2) + '" r="4.5"/>');

    parts.push(text((L + W - R) / 2, H - 6, 'ngưỡng cắt c'));
    parts.push(text(L - 6, T + 6, 'MSE', 'end'));
    return svg(W, H, parts.join(''));
  }

  function updateClip() {
    if (!$('clDist')) return;
    var dist = $('clDist').value;
    var bits = Number($('clBits').value);
    var st = clipCompute(dist, bits);

    var slider = $('clC');
    slider.max = (Math.ceil(st.maxAbs * 10) / 10).toFixed(1);
    slider.min = (st.maxAbs / 70).toFixed(2);
    var c = clamp(Number(slider.value), Number(slider.min), Number(slider.max));
    slider.value = c;
    $('clCOut').textContent = fmt(c, 2);

    var mse = mseAt(st.data, c, bits, st.oneSided);
    setStat('clMse', fmt(mse), 'is-accent');
    setStat('clBest', fmt(st.best[1]));
    setStat('clMinmax', fmt(st.minmax), 'is-warn');
    setStat('clCstar', fmt(st.best[0], 2));
    setStat('clMaxAbs', fmt(st.maxAbs, 2));
    setStat('clRatio', fmt(st.minmax / st.best[1], 2) + '×', 'is-warn');

    var over = Math.max(1, mse / st.best[1]);
    var msg = 'Ở ngưỡng đang chọn, MSE lớn gấp <b>' + fmt(over, 2) +
      ' lần</b> mức tốt nhất. ';
    if (dist === 'laplace' && bits === 4) {
      msg = 'Hình 7 dùng 10⁶ mẫu nên max|x| lên tới 15,28 và min–max tệ hơn 6,4 lần; ' +
        'ở đây chỉ 50 000 mẫu nên đuôi phân phối ngắn hơn và tỉ lệ nhỏ hơn — ' +
        '<b>cơ chế thì y hệt</b>. ' + msg;
    }
    if (c > st.maxAbs * 0.95) {
      msg += 'Đây gần như chính là dải min–max: không cắt giá trị nào, nhưng phải trả bằng bước nhảy thô cho toàn bộ dữ liệu.';
    } else if (c < st.best[0] * 0.6) {
      msg += 'Cắt quá mạnh: bước nhảy đã rất mịn nhưng phần đuôi bị xén mất quá nhiều.';
    } else if (over < 1.05) {
      msg += 'Bạn đang ở rất gần điểm cân bằng tối ưu giữa sai số làm tròn và sai số cắt.';
    } else {
      msg += 'Kéo thanh trượt về phía c* để thấy MSE chạm đáy.';
    }
    $('clNote').innerHTML = msg;
    $('clChart').innerHTML = drawClip(st, c);
  }

  /* ==========================================  3. soi bit số thực */

  var FORMATS = {
    fp32: { E: 8, M: 23, name: 'FP32' },
    fp16: { E: 5, M: 10, name: 'FP16' },
    bf16: { E: 8, M: 7, name: 'BF16' },
    e4m3: { E: 4, M: 3, name: 'FP8 E4M3 (OCP)', ocp: true },
    e5m2: { E: 5, M: 2, name: 'FP8 E5M2' },
  };

  /** Mã hoá x theo định dạng (E bit exponent, M bit mantissa), làm tròn về số chẵn. */
  function encodeFloat(x, f) {
    var bias = Math.pow(2, f.E - 1) - 1;
    var eMin = 1 - bias;                                  // số mũ của vùng dưới chuẩn
    var eMaxField = f.ocp ? Math.pow(2, f.E) - 1 : Math.pow(2, f.E) - 2;
    var eMax = eMaxField - bias;
    var maxMan = f.ocp ? Math.pow(2, f.M) - 2 : Math.pow(2, f.M) - 1;
    var maxFinite = (1 + maxMan / Math.pow(2, f.M)) * Math.pow(2, eMax);

    var sign = x < 0 || Object.is(x, -0) ? 1 : 0;
    var a = Math.abs(x);
    var out = { sign: sign, bias: bias, maxFinite: maxFinite, eps: Math.pow(2, -f.M) };

    if (a === 0) {
      out.expField = 0; out.manField = 0; out.value = sign ? -0 : 0;
      out.e = null; out.spacing = Math.pow(2, eMin - f.M); out.subnormal = true;
      return out;
    }

    var e = Math.floor(Math.log2(a));
    if (a / Math.pow(2, e) >= 2) e++;                     // chống sai số của log2
    if (a / Math.pow(2, e) < 1) e--;
    if (e < eMin) e = eMin;

    var step = Math.pow(2, e - f.M);
    var m = rne(a / step);
    if (m >= Math.pow(2, f.M + 1)) { e += 1; step = Math.pow(2, e - f.M); m = rne(a / step); }

    var value = m * step;
    if (value > maxFinite) {
      out.overflow = true;
      out.value = f.ocp ? NaN : Infinity;
      out.expField = Math.pow(2, f.E) - 1;
      out.manField = f.ocp ? Math.pow(2, f.M) - 1 : 0;
      out.e = e; out.spacing = step;
      if (sign) out.value = -out.value;
      return out;
    }

    out.subnormal = e === eMin && m < Math.pow(2, f.M);
    out.expField = out.subnormal ? 0 : e + bias;
    out.manField = out.subnormal ? m : m - Math.pow(2, f.M);
    out.value = sign ? -value : value;
    out.e = out.subnormal ? eMin : e;
    out.spacing = step;
    return out;
  }

  function bitsHtml(enc, f) {
    var toBits = function (v, n) {
      var s = '';
      for (var i = n - 1; i >= 0; i--) s += (Math.floor(v / Math.pow(2, i)) % 2) ? '1' : '0';
      return s;
    };
    var group = function (label, bits, cls) {
      return '<div class="bit-group"><span class="bit-group-label">' + label + '</span>' +
        '<span class="bit-cells">' +
        bits.split('').map(function (b) { return '<span class="bit bit-' + cls + '">' + b + '</span>'; }).join('') +
        '</span></div>';
    };
    return group('dấu', String(enc.sign), 's') +
      group('exponent · ' + f.E + ' bit', toBits(enc.expField, f.E), 'e') +
      group('mantissa · ' + f.M + ' bit', toBits(enc.manField, f.M), 'm');
  }

  function updateFloat() {
    if (!$('flX')) return;
    var raw = $('flX').value.trim().replace(',', '.');
    var x = Number(raw);
    var f = FORMATS[$('flFmt').value];

    if (raw === '' || !isFinite(x)) {
      $('flBits').innerHTML = '';
      ['flStored', 'flErr', 'flRel', 'flExp', 'flSpacing', 'flEps'].forEach(function (id) { setStat(id, '–'); });
      $('flNote').innerHTML = 'Nhập một số hữu hạn để xem cách nó được lưu.';
      return;
    }

    var enc = encodeFloat(x, f);
    $('flBits').innerHTML = bitsHtml(enc, f);

    setStat('flStored', enc.overflow ? (f.ocp ? 'NaN' : (enc.value > 0 ? '+∞' : '−∞')) : fmt(enc.value),
      enc.overflow ? 'is-warn' : (enc.value === x ? 'is-accent' : ''));
    setStat('flErr', enc.overflow ? '–' : fmt(Math.abs(enc.value - x)));
    setStat('flRel', enc.overflow || x === 0 ? '–' : fmt(Math.abs(enc.value - x) / Math.abs(x) * 100, 4) + '%');
    setStat('flExp', enc.e === null ? '–' : String(enc.e));
    setStat('flSpacing', fmt(enc.spacing));
    setStat('flEps', fmt(enc.eps));

    var msgs = [];
    if (enc.overflow) {
      msgs.push('<b>Tràn số.</b> Giá trị lớn nhất mà ' + f.name + ' biểu diễn được là ' +
        fmt(enc.maxFinite) + '. ' +
        (f.ocp ? 'Biến thể OCP của E4M3 không có vô cực, nên kết quả là NaN.'
               : 'Mọi giá trị lớn hơn đều thành vô cực.'));
    } else if (enc.value === x) {
      msgs.push('Giá trị này <b>biểu diễn được chính xác</b>.');
    } else {
      var lo = enc.value <= x ? enc.value : enc.value - enc.spacing;
      var hi = lo + enc.spacing;
      msgs.push('Hai giá trị biểu diễn được kề nó là <b>' + fmt(lo) + '</b> và <b>' + fmt(hi) +
        '</b>, cách nhau ' + fmt(enc.spacing) + '. ');
      if (Math.abs(x - (lo + hi) / 2) < enc.spacing * 1e-9) {
        msgs.push('Nó rơi <b>đúng điểm giữa</b>, nên IEEE 754 làm tròn về phía có mantissa chẵn.');
      } else {
        msgs.push('Nó gần ' + fmt(enc.value) + ' hơn.');
      }
    }
    if (enc.subnormal && x !== 0) msgs.push(' Đây là một <b>số dưới chuẩn</b> (subnormal).');
    var p = f.M + 1;
    msgs.push(' Số nguyên dương nhỏ nhất mà ' + f.name + ' <i>không</i> biểu diễn chính xác được là 2<sup>' +
      p + '</sup> + 1 = ' + fmtInt(Math.pow(2, p) + 1) + '.');
    $('flNote').innerHTML = msgs.join('');
  }

  /* ===========================================  4. requantization */

  // Như Mục 6.4, nhưng bổ sung nhánh chuẩn hoá XUỐNG để mọi M > 0 đều dùng được.
  // Đoạn mã in trong giáo trình chỉ có nhánh đi lên nên giới hạn ở 0 < M <= 1.
  function quantizeMultiplier(M) {
    var shift = 0;
    while (M < 0.5 && shift < 62) { M *= 2; shift++; }
    while (M >= 1 && shift > -62) { M /= 2; shift--; }
    var M0 = rne(M * Math.pow(2, 31));
    if (M0 === Math.pow(2, 31)) { M0 /= 2; shift--; }
    return { M0: M0, shift: shift };
  }

  // Hai quy ước làm tròn của bước dịch phải. Chúng chỉ khác nhau ở đúng điểm
  // giữa của giá trị âm, lệch tối đa 1 LSB — xem ghi chú Mục 6.4.
  function shiftHalfUp(x, n) {              // như mã trong giáo trình (>> của Python)
    return Math.floor((x + Math.pow(2, n - 1)) / Math.pow(2, n));
  }
  function shiftHalfAwayFromZero(x, n) {    // như RoundingDivideByPOT của gemmlowp
    var d = Math.pow(2, n);
    var q = x / d;
    return q < 0 ? -Math.round(-q) : Math.round(q);
  }
  var roundingRightShift = shiftHalfUp;

  function updateRequant() {
    if (!$('rqSw')) return;
    var Sw = Number($('rqSw').value), Sx = Number($('rqSx').value), Sy = Number($('rqSy').value);
    var acc = Math.round(Number($('rqAcc').value));
    var note = $('rqNote');

    if (!(Sw > 0 && Sx > 0 && Sy > 0)) {
      note.innerHTML = '<b>Cả ba scale đều phải dương.</b>';
      return;
    }
    var M = Sw * Sx / Sy;
    var qm = quantizeMultiplier(M);
    var total = 31 + qm.shift;
    var prod = acc * qm.M0;
    var intResult = shiftHalfUp(prod, total);
    var gemmlowp = shiftHalfAwayFromZero(prod, total);
    var floatResult = rne(acc * M);

    setStat('rqM', fmt(M), M >= 1 ? 'is-warn' : '');
    setStat('rqM0', fmtInt(qm.M0));
    setStat('rqShift', String(total), total < 0 ? 'is-warn' : 'is-accent');
    setStat('rqInt', fmtInt(intResult), 'is-accent');
    setStat('rqFloat', fmtInt(floatResult));
    setStat('rqDiff', String(intResult - floatResult),
      intResult === floatResult ? 'is-accent' : 'is-warn');

    var shiftTxt = total >= 0 ? '>> ' + total : '<< ' + -total;
    $('rqFormula').textContent =
      'M  = S_w·S_x / S_y = ' + M.toPrecision(10) + '\n' +
      'M  = 2^-(' + qm.shift + ') · M₀ ,  M₀ = ' + (M * Math.pow(2, qm.shift)).toPrecision(10) +
      ' ∈ [0,5 ; 1)\n' +
      'M₀(int32) = round(M₀ · 2³¹) = ' + qm.M0 + '\n' +
      'kết quả = (acc · M₀) ' + shiftTxt + ' = (' + acc + ' · ' + qm.M0 + ') ' +
      shiftTxt + ' = ' + intResult;

    var msgs = [];
    msgs.push('Sai số của cách dùng dấu chấm tĩnh so với nhân số thực rồi làm tròn: <b>' +
      (intResult - floatResult) + ' LSB</b>' +
      (intResult === floatResult ? ' — trùng khớp.' : '.') +
      ' Giá trị thực chính xác là ' + fmt(acc * M, 4) + '.');

    if (M >= 1) {
      msgs.push('<b>M = ' + fmt(M) + ' ≥ 1.</b> Đoạn mã in ở Mục 6.4 chỉ chuẩn hoá <i>lên</i> ' +
        'nên sẽ tràn int32 ở đây; công cụ này thêm nhánh chuẩn hoá xuống, cho <code>shift = ' +
        qm.shift + '</code> âm — tức một phép <b>dịch trái</b>, đúng như gemmlowp/TFLite làm.');
    }
    if (intResult !== gemmlowp) {
      msgs.push('<b>Hai quy ước làm tròn lệch nhau ở đây:</b> cách của giáo trình (nửa về phía ' +
        '+∞) cho ' + fmtInt(intResult) + ', còn gemmlowp/TFLite (nửa ra xa số 0) cho ' +
        fmtInt(gemmlowp) + '. Chỉ xảy ra khi bộ cộng dồn âm và rơi đúng điểm giữa.');
    }
    msgs.push('Tích trung gian <code>acc · M₀</code> bằng ' + fmt(prod) +
      ', vượt xa int32 — đó là lý do bước này phải làm ở <b>số nguyên 64 bit</b>.');
    note.innerHTML = msgs.join(' ');
  }

  /* =============================================  5. tính dung lượng */

  var WEIGHT_FORMATS = {
    fp16: { bits: 16, why: 'Không lượng tử: 2 byte mỗi tham số.' },
    int8: { bits: 8 + 16 / 128, why: '8 + 16/128 — mỗi nhóm 128 phần tử lưu một scale FP16.' },
    q8_0: { bits: 8 + 16 / 32, why: '8 + 16/32 — khối 32 phần tử, mỗi khối một scale FP16.' },
    int4: { bits: 4 + 16 / 128, why: '4 + 16/128 — mỗi nhóm 128 phần tử lưu một scale FP16.' },
    q4_0: { bits: 4 + 16 / 32, why: '4 + 16/32 — khối 32 phần tử, mỗi khối một scale FP16.' },
    q4_k: {
      bits: (256 * 4 + 8 * 12 + 32) / 256,
      why: '(256·4 + 8·12 + 32)/256 — siêu khối 256 gồm 8 khối con ×32; scale và min của khối con lượng tử 6 bit; siêu khối có 2 hệ số FP16.',
    },
    nf4: {
      bits: 4 + 8 / 64 + 32 / (64 * 256),
      why: '4 + 8/64 + 32/(64·256) — khối 64, absmax lượng tử FP8 theo khối 256 (double quantization).',
    },
  };

  function gb(bytes) { return bytes / 1e9; }
  function gib(bytes) { return bytes / Math.pow(2, 30); }

  function updateSize() {
    if (!$('szParams')) return;
    var params = Number($('szParams').value) * 1e9;
    var fmtKey = $('szFmt').value;
    var spec = WEIGHT_FORMATS[fmtKey];
    var wBytes = params * spec.bits / 8;

    setStat('szBits', fmt(spec.bits, 3), 'is-accent');
    setStat('szSize', fmt(gb(wBytes), 2) + ' GB');
    $('szNote').innerHTML = '<code>' + spec.why + '</code> Tương đương ' +
      fmt(gib(wBytes), 2) + ' GiB.' +
      (spec.bits < 8
        ? ' File thật thường lớn hơn một chút vì embedding và lớp đầu ra được giữ ở độ chính xác cao hơn.'
        : '');

    var layers = Number($('kvLayers').value);
    var heads = Number($('kvHeads').value);
    var dim = Number($('kvDim').value);
    var tokens = Number($('kvTokens').value);
    var batch = Number($('kvBatch').value);
    var bytesPer = Number($('kvDtype').value);

    var perToken = 2 * layers * heads * dim * bytesPer;
    var kvBytes = perToken * tokens * batch;

    setStat('kvPerToken', fmt(perToken / Math.pow(2, 20), 3) + ' MiB');
    setStat('kvTotal', fmt(gb(kvBytes), 2) + ' GB', 'is-accent');
    $('kvNote').innerHTML = '<code>2 × ' + layers + ' × ' + heads + ' × ' + dim + ' × ' +
      bytesPer + ' = ' + fmtInt(perToken) + ' byte mỗi token</code> — hệ số 2 là vì phải lưu ' +
      '<b>cả key lẫn value</b>. Tổng tương đương ' + fmt(gib(kvBytes), 2) + ' GiB.';

    var ratio = kvBytes / wBytes;
    var totalBytes = kvBytes + wBytes;
    var verdict = $('szVerdict');
    var parts = ['Tổng cộng <b>' + fmt(gb(totalBytes), 2) + ' GB</b> (' +
      fmt(gib(totalBytes), 2) + ' GiB): trọng số ' + fmt(gb(wBytes), 2) + ' GB + KV cache ' +
      fmt(gb(kvBytes), 2) + ' GB. '];
    if (ratio >= 1) {
      parts.push('KV cache <b>lớn hơn trọng số ' + fmt(ratio, 2) + ' lần</b>. ' +
        'Lượng tử trọng số thôi là chưa đủ: hãy lượng tử KV cache, dùng GQA, ' +
        'giảm batch hoặc rút ngắn ngữ cảnh.');
    } else {
      parts.push('Trọng số vẫn là phần lớn hơn (KV cache chỉ bằng ' + fmt(ratio * 100, 1) +
        '% trọng số). Tăng số token hoặc batch để thấy cán cân đảo chiều.');
    }
    verdict.innerHTML = parts.join('');
  }

  /* ======================================================  kết nối */

  function on(ids, fn) {
    ids.forEach(function (id) {
      var el = $(id);
      if (!el) return;
      el.addEventListener('input', fn);
      el.addEventListener('change', fn);
    });
  }

  on(['afAlpha', 'afBeta', 'afBits', 'afMode', 'afX'], updateAffine);
  on(['clDist', 'clBits', 'clC'], updateClip);
  on(['flX', 'flFmt'], updateFloat);
  on(['rqSw', 'rqSx', 'rqSy', 'rqAcc'], updateRequant);
  on(['szParams', 'szFmt', 'kvLayers', 'kvHeads', 'kvDim', 'kvTokens', 'kvBatch', 'kvDtype'], updateSize);

  Array.prototype.forEach.call(document.querySelectorAll('[data-float-preset]'), function (btn) {
    btn.addEventListener('click', function () {
      $('flX').value = btn.dataset.floatPreset;
      updateFloat();
    });
  });

  updateAffine();
  updateClip();
  updateFloat();
  updateRequant();
  updateSize();

  // Vẽ lại các biểu đồ khi đổi giao diện sáng/tối (màu lấy từ biến CSS).
  var themeBtn = $('themeBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      setTimeout(function () { updateAffine(); updateClip(); }, 30);
    });
  }

  // Phần lõi số học được phơi ra để src/test.mjs kiểm chứng bằng Node.
  window.QZ_LAB = {
    rne: rne, affineParams: affineParams, quantize: quantize, dequantize: dequantize,
    encodeFloat: encodeFloat, FORMATS: FORMATS,
    quantizeMultiplier: quantizeMultiplier, roundingRightShift: roundingRightShift,
    shiftHalfUp: shiftHalfUp, shiftHalfAwayFromZero: shiftHalfAwayFromZero,
    samples: samples, mseAt: mseAt, clipCompute: clipCompute,
    WEIGHT_FORMATS: WEIGHT_FORMATS,
  };
};
window.QZ_INIT['playground']();
