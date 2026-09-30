/* Bốn công cụ của Phòng thí nghiệm MLOps.
   Mọi công thức lấy đúng từ giáo trình; hàm thống kê tự cài để không cần thư viện ngoài. */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['labMlops'] = function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  /* ==================================================== hàm thống kê */

  // Nghịch đảo hàm phân phối chuẩn (Acklam), sai số tương đối < 1,15e-9.
  function normInv(p) {
    if (p <= 0 || p >= 1) return NaN;
    var a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    var b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01];
    var c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    var d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
    var pl = 0.02425, q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    if (p > 1 - pl) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    q = p - 0.5; r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }

  function logGamma(x) {
    var g = [76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    var y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    var ser = 1.000000000190015;
    for (var j = 0; j < 6; j++) ser += g[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }

  // Hàm gamma không đầy đủ chuẩn hoá P(a,x) — chuỗi và phân số liên tục (Numerical Recipes).
  function gammaP(a, x) {
    if (x < 0 || a <= 0) return NaN;
    if (x === 0) return 0;
    if (x < a + 1) {                       // khai triển chuỗi
      var ap = a, sum = 1 / a, del = sum;
      for (var n = 0; n < 500; n++) {
        ap++; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-14) break;
      }
      return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    }
    // phân số liên tục cho Q(a,x) = 1 - P(a,x)
    var FPMIN = 1e-300, b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
    for (var i = 1; i <= 500; i++) {
      var an = -i * (i - a);
      b += 2; d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      var delt = d * c; h *= delt;
      if (Math.abs(delt - 1) < 1e-14) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  }

  var chi2cdf = function (x, k) { return gammaP(k / 2, x / 2); };
  var chi2sf = function (x, k) { return 1 - chi2cdf(x, k); };

  function chi2inv(p, k) {                 // nghịch đảo bằng chia đôi, đủ chính xác
    var lo = 0, hi = Math.max(10 * k, 100);
    while (chi2cdf(hi, k) < p) hi *= 2;
    for (var i = 0; i < 200; i++) {
      var mid = (lo + hi) / 2;
      if (chi2cdf(mid, k) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /* ==================================================== tiện ích chung */

  function fmt(v, d) {
    if (!isFinite(v)) return '–';
    var s = d === undefined
      ? (v !== 0 && (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e7) ? v.toExponential(3) : String(+v.toPrecision(5)))
      : v.toFixed(d);
    return s.replace('.', ',');
  }
  function fmtInt(v) { return isFinite(v) ? Math.round(v).toLocaleString('vi-VN') : '–'; }
  function setStat(id, text, cls) {
    var el = $(id); if (!el) return;
    el.textContent = text; el.className = 'stat-value' + (cls ? ' ' + cls : '');
  }
  function svg(w, h, inner) {
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" role="img" preserveAspectRatio="xMidYMid meet">' + inner + '</svg>';
  }
  function path(pts, cls) {
    if (!pts.length) return '';
    return '<path class="' + cls + '" d="M' +
      pts.map(function (p) { return p[0].toFixed(2) + ' ' + p[1].toFixed(2); }).join('L') + '"/>';
  }
  function line(x1, y1, x2, y2, cls) {
    return '<line class="' + cls + '" x1="' + x1.toFixed(2) + '" y1="' + y1.toFixed(2) +
      '" x2="' + x2.toFixed(2) + '" y2="' + y2.toFixed(2) + '"/>';
  }
  function text(x, y, s, anchor) {
    return '<text class="chart-label" x="' + x.toFixed(1) + '" y="' + y.toFixed(1) +
      '" text-anchor="' + (anchor || 'middle') + '">' + s + '</text>';
  }

  /* ============================================= 1. cỡ mẫu A/B test */

  function nPerArm(p1, relLift, alpha, power, varRed) {
    var p2 = p1 * (1 + relLift);
    if (p2 >= 1 || p2 <= 0 || relLift === 0) return Infinity;
    var za = normInv(1 - alpha / 2), zb = normInv(power);
    var v = p1 * (1 - p1) + p2 * (1 - p2);
    return Math.pow(za + zb, 2) * v * (1 - varRed) / Math.pow(p2 - p1, 2);
  }

  function updateAB() {
    if (!$('abBase')) return;
    var p1 = Number($('abBase').value) / 100;
    var lift = Number($('abLift').value) / 100;
    var alpha = Number($('abAlpha').value);
    var power = Number($('abPower').value);
    var traffic = Number($('abTraffic').value);
    var cuped = Math.min(0.8, Math.max(0, Number($('abCuped').value) / 100));

    if (!(p1 > 0 && p1 < 1 && lift > 0 && traffic > 0)) {
      ['abN','abDays','abAbs','abTotal','abWeeks','abSaved'].forEach(function (i) { setStat(i, '–'); });
      $('abNote').innerHTML = '<b>Tham số không hợp lệ.</b> Tỉ lệ nền phải trong (0; 100)% và mức cải thiện phải dương.';
      return;
    }

    var n = nPerArm(p1, lift, alpha, power, cuped);
    var n0 = nPerArm(p1, lift, alpha, power, 0);
    var days = n / traffic;

    setStat('abN', fmtInt(n), 'is-accent');
    setStat('abDays', fmt(days, 2) + ' ngày', days > 21 ? 'is-warn' : '');
    setStat('abAbs', fmt(p1 * lift * 100, 4) + ' điểm %');
    setStat('abTotal', fmtInt(2 * n));
    setStat('abWeeks', Math.ceil(days / 7) + ' tuần');
    setStat('abSaved', cuped > 0 ? fmtInt(n0 - n) + ' mẫu' : '—');

    var msgs = [];
    msgs.push('Chênh lệch tuyệt đối cần phát hiện chỉ là <b>' + fmt(p1 * lift * 100, 4) +
      ' điểm phần trăm</b>. Đây là đại lượng đi vào mẫu số của công thức, không phải mức cải thiện tương đối ' +
      fmt(lift * 100, 1) + '%.');
    if (days > 28) {
      msgs.push('<b>Hơn bốn tuần cho một ý tưởng.</b> Các cách rút ngắn: giảm phương sai bằng CUPED, ' +
        'tăng tỉ lệ lưu lượng dành cho thí nghiệm, hoặc chấp nhận chỉ phát hiện hiệu ứng lớn hơn; ' +
        'vì $n \\propto 1/\\Delta^2$, hiệu ứng cần phát hiện lớn hơn khoảng 1,41 lần thì cỡ mẫu giảm một nửa.'
          .replace('$n \\propto 1/\\Delta^2$', '<i>n</i> tỉ lệ nghịch với bình phương Δ'));
    }
    if (cuped > 0) {
      msgs.push('CUPED giảm phương sai ' + fmt(cuped * 100, 0) + '% nên cỡ mẫu giảm cùng tỉ lệ, ' +
        'từ ' + fmtInt(n0) + ' xuống ' + fmtInt(n) +
        ', tức tiết kiệm ' + fmt((n0 - n) / traffic, 1) + ' ngày.');
    }
    $('abNote').innerHTML = msgs.join(' ');

    // biểu đồ: số ngày theo mức cải thiện
    var W = 540, H = 210, L = 52, R = 14, T = 12, B = 34;
    var xs = [];
    for (var i = 0; i <= 60; i++) xs.push(0.002 + (0.25 - 0.002) * i / 60);
    var ys = xs.map(function (l) { return nPerArm(p1, l, alpha, power, cuped) / traffic; });
    var lg = ys.map(function (v) { return Math.log10(Math.max(v, 1e-3)); });
    var yMin = Math.min.apply(null, lg), yMax = Math.max.apply(null, lg);
    var sx = function (v) { return L + (v - 0.002) / (0.25 - 0.002) * (W - L - R); };
    var sy = function (v) { return H - B - (v - yMin) / (yMax - yMin || 1) * (H - T - B); };
    var parts = [];
    var SUP = {'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵'};
    for (var g = Math.ceil(yMin); g <= yMax; g++) {
      parts.push(line(L, sy(g), W - R, sy(g), 'chart-grid'));
      var lab = g === 0 ? '1' : (g === 1 ? '10' : (g === 2 ? '100' :
        '10' + String(g).split('').map(function (c) { return SUP[c] || c; }).join('')));
      parts.push(text(L - 6, sy(g) + 3, lab + ' ngày', 'end'));
    }
    parts.push(line(L, H - B, W - R, H - B, 'chart-axis'));
    parts.push(path(xs.map(function (v, i2) { return [sx(v), sy(lg[i2])]; }), 'chart-line'));
    if (lift >= 0.002 && lift <= 0.25) {
      parts.push('<circle class="chart-dot" cx="' + sx(lift).toFixed(2) + '" cy="' +
        sy(Math.log10(Math.max(days, 1e-3))).toFixed(2) + '" r="4.5"/>');
    }
    for (var p = 0.05; p <= 0.25; p += 0.05) parts.push(text(sx(p), H - B + 16, fmt(p * 100, 0) + '%'));
    parts.push(text((L + W - R) / 2, H - 4, 'mức cải thiện tương đối cần phát hiện'));
    $('abChart').innerHTML = svg(W, H, parts.join(''));
  }

  /* ============================================= 2. hiệu chuẩn PSI */

  function updatePSI() {
    if (!$('psiN')) return;
    var n = Number($('psiN').value), k = Number($('psiK').value);
    var feat = Number($('psiFeat').value), per = Number($('psiPeriod').value);
    var budget = Number($('psiAlarm').value);
    if (!(n > 1 && k > 1 && feat >= 1 && per >= 1 && budget > 0)) return;

    var df = k - 1;
    var mean = 2 * df / n;                          // E[PSI] khi không dịch chuyển
    var p99 = 2 / n * chi2inv(0.99, df);
    // ngưỡng sao cho so_luot_do * P(vuot) = budget
    var tests = feat * per;
    var pTarget = Math.min(0.5, budget / tests);
    var thresh = 2 / n * chi2inv(1 - pTarget, df);
    var fa25 = tests * chi2sf(0.25 * n / 2, df);
    var fa10 = tests * chi2sf(0.10 * n / 2, df);

    setStat('psiMean', fmt(mean, 4), mean > 0.1 ? 'is-warn' : '');
    setStat('psiP99', fmt(p99, 4));
    setStat('psiThresh', fmt(thresh, 4), 'is-accent');
    setStat('psiFa25', fmt(fa25, 3), fa25 > budget ? 'is-warn' : 'is-accent');
    setStat('psiFa10', fmt(fa10, 3), fa10 > budget ? 'is-warn' : 'is-accent');
    setStat('psiNeed', fmtInt(2 * df / 0.01));

    var msgs = [];
    msgs.push('Với <b>n = ' + fmtInt(n) + '</b> và <b>k = ' + k + ' bin</b>, PSI trung bình khi ' +
      '<b>không có dịch chuyển</b> đã là <b>' + fmt(mean, 4) + '</b>.');
    if (mean > 0.1) {
      msgs.push('<b>Con số này đã vượt ngưỡng 0,10.</b> Ở cấu hình này, quy tắc kinh nghiệm báo động cả khi dữ liệu không đổi; ' +
        'cần tăng n hoặc giảm số bin.');
    }
    if (fa25 > budget * 2) {
      msgs.push('Dùng ngưỡng 0,25 sẽ cho <b>' + fmt(fa25, 2) + ' báo động giả mỗi ngày</b> trên ' +
        fmtInt(tests) + ' lượt đo, vượt mức ' + fmt(budget, 2) + ' đã chọn.');
    } else if (fa25 < budget / 50 && mean < 0.02) {
      msgs.push('Ở cỡ mẫu này, ngưỡng 0,25 gần như <b>không bao giờ bị vượt</b> khi không có dịch chuyển (' + fmt(fa25, 4) +
        ' báo động/ngày), và cũng khó bị vượt khi có dịch chuyển nhỏ. Ngưỡng nên dùng là <b>' +
        fmt(thresh, 4) + '</b>.');
    }
    msgs.push('Ngưỡng <b>' + fmt(thresh, 4) + '</b> cho đúng ' + fmt(budget, 3) +
      ' báo động giả mỗi ngày trên ' + fmtInt(tests) + ' lượt đo.');
    $('psiNote').innerHTML = msgs.join(' ');

    // biểu đồ mật độ của PSI dưới giả thuyết không
    var W = 540, H = 200, L = 40, R = 14, T = 12, B = 32;
    var xMax = Math.max(thresh, 0.27, p99 * 1.3);
    var pts = [], maxD = 0, dens = [];
    for (var i = 0; i <= 160; i++) {
      var x = xMax * i / 160;
      var c = x * n / 2;                            // doi sang thang chi2
      var d = c <= 0 ? 0 : Math.exp((df / 2 - 1) * Math.log(c) - c / 2 -
              logGamma(df / 2) - (df / 2) * Math.LN2) * (n / 2);
      dens.push([x, d]); if (d > maxD) maxD = d;
    }
    var sx2 = function (v) { return L + v / xMax * (W - L - R); };
    var sy2 = function (v) { return H - B - v / (maxD || 1) * (H - T - B); };
    var parts2 = [];
    parts2.push(line(L, H - B, W - R, H - B, 'chart-axis'));
    dens.forEach(function (d) { pts.push([sx2(d[0]), sy2(d[1])]); });
    parts2.push(path(pts, 'chart-line'));
    parts2.push(line(sx2(thresh), T, sx2(thresh), H - B, 'chart-mark'));
    parts2.push(text(sx2(thresh), T + 9, 'ngưỡng đề nghị ' + fmt(thresh, 3)));
    if (0.25 <= xMax) {
      parts2.push(line(sx2(0.25), T, sx2(0.25), H - B, 'chart-mark2'));
      parts2.push(text(Math.min(sx2(0.25), W - R - 30), H - B + 22, 'quy tắc 0,25', 'end'));
    }
    for (var q = 0; q <= 4; q++) parts2.push(text(sx2(xMax * q / 4), H - B + 12, fmt(xMax * q / 4, 3)));
    parts2.push(text((L + W - R) / 2, H - 2, 'giá trị PSI khi hai mẫu có cùng phân phối'));
    $('psiChart').innerHTML = svg(W, H, parts2.join(''));
  }

  /* ========================================== 3. đuôi độ trễ toả nhánh */

  function updateTail() {
    if (!$('tlP50')) return;
    var p50 = Number($('tlP50').value), p99 = Number($('tlP99').value);
    var k = Math.round(Number($('tlK').value)), budget = Number($('tlBudget').value);
    if (!(p50 > 0 && p99 > p50 && k >= 1)) {
      $('tlNote').innerHTML = '<b>Tham số không hợp lệ:</b> cần p99 &gt; trung vị &gt; 0.';
      return;
    }
    var sigma = Math.log(p99 / p50) / normInv(0.99);
    var mu = Math.log(p50);
    var qOfMax = function (q, kk) { return Math.exp(mu + sigma * normInv(Math.pow(q, 1 / kk))); };

    var res = qOfMax(0.99, k);
    var slow = 1 - Math.pow(0.99, k);
    var need = Math.pow(0.99, 1 / k);
    var med = qOfMax(0.5, k);

    // so nhanh toi da con nam trong ngan sach
    var kmax = 0;
    for (var kk = 1; kk <= 1000; kk++) { if (qOfMax(0.99, kk) <= budget) kmax = kk; else break; }

    setStat('tlP99max', fmt(res, 1) + ' ms', res > budget ? 'is-warn' : 'is-accent');
    setStat('tlSlow', fmt(slow * 100, 1) + '%', slow > 0.1 ? 'is-warn' : '');
    setStat('tlNeed', 'p' + fmt(need * 100, 4));
    setStat('tlOne', '1 / ' + fmtInt(1 / (1 - need)));
    setStat('tlKmax', kmax >= 1 ? String(kmax) : 'không nhánh nào', kmax < k ? 'is-warn' : 'is-accent');
    setStat('tlMedian', fmt(med, 1) + ' ms');

    var msgs = [];
    msgs.push('Với <b>' + k + ' nhánh</b>, <b>' + fmt(slow * 100, 1) + '%</b> số yêu cầu gặp ít nhất ' +
      'một nhánh chậm hơn p99 của nhánh đó, dù mỗi nhánh chỉ chậm như vậy ở 1% số lần gọi.');
    msgs.push('Muốn p99 tổng thể bằng p99 của một nhánh thì mỗi nhánh phải đạt <b>p' +
      fmt(need * 100, 4) + '</b>, tức chỉ 1 trong ' + fmtInt(1 / (1 - need)) + ' yêu cầu được phép chậm.');
    if (res > budget) {
      msgs.push('<b>Vượt ngân sách ' + fmt(budget, 0) + ' ms.</b> Trong ngân sách này chỉ toả được tối đa ' +
        (kmax >= 1 ? '<b>' + kmax + ' nhánh</b>' : '<b>không nhánh nào</b>') +
        '. Ba cách xử lý: giảm số nhánh (gộp lô, lưu đệm), đặt thời hạn chờ và trả lời với kết quả một phần, ' +
        'hoặc gửi yêu cầu dự phòng.');
    }
    $('tlNote').innerHTML = msgs.join(' ');

    var W = 540, H = 200, L = 46, R = 14, T = 12, B = 34;
    var maxK = Math.max(100, k * 2);
    var ks = [], vs = [];
    for (var i = 1; i <= 80; i++) {
      var kv = Math.round(Math.pow(maxK, i / 80));
      if (ks[ks.length - 1] !== kv) { ks.push(kv); vs.push(qOfMax(0.99, kv)); }
    }
    var yMax = Math.max.apply(null, vs.concat([budget * 1.1]));
    var sx3 = function (v) { return L + Math.log(v) / Math.log(maxK) * (W - L - R); };
    var sy3 = function (v) { return H - B - v / yMax * (H - T - B); };
    var p3 = [];
    p3.push(line(L, H - B, W - R, H - B, 'chart-axis'));
    p3.push(line(L, sy3(budget), W - R, sy3(budget), 'chart-mark'));
    p3.push(text(W - R, sy3(budget) - 4, 'ngân sách ' + fmt(budget, 0) + ' ms', 'end'));
    p3.push(path(ks.map(function (kv, i2) { return [sx3(kv), sy3(vs[i2])]; }), 'chart-line'));
    p3.push('<circle class="chart-dot" cx="' + sx3(Math.max(k, 1)).toFixed(2) + '" cy="' +
      sy3(res).toFixed(2) + '" r="4.5"/>');
    [1, 10, 100].forEach(function (kv) {
      if (kv <= maxK) p3.push(text(sx3(kv), H - B + 13, String(kv)));
    });
    p3.push(text((L + W - R) / 2, H - 3, 'số nhánh gọi song song (thang log)'));
    p3.push(text(L - 6, T + 8, 'ms', 'end'));
    $('tlChart').innerHTML = svg(W, H, p3.join(''));
  }

  /* ============================================ 4. ML Test Score */

  var mts = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]];
  var MTS_NAMES = ['Dữ liệu', 'Mô hình', 'Hạ tầng', 'Giám sát'];

  function mtsBand(s) {
    if (s === 0) return 'Giống một dự án nghiên cứu hơn là một hệ thống sản xuất.';
    if (s <= 1) return 'Đã có kiểm thử, nhưng có thể còn những lỗ hổng nghiêm trọng về độ tin cậy.';
    if (s <= 2) return 'Đã có bước đầu đưa vào sản xuất, cần đầu tư thêm.';
    if (s <= 3) return 'Kiểm thử tương đối đủ, nhiều mục còn có thể tự động hoá.';
    if (s <= 5) return 'Kiểm thử và giám sát tự động ở mức cao, phù hợp với hệ thống trọng yếu.';
    return 'Kiểm thử và giám sát tự động ở mức rất cao.';
  }

  function updateMTS() {
    if (!$('mtsScore')) return;
    var sums = mts.map(function (g) { return g.reduce(function (a, b) { return a + b; }, 0); });
    sums.forEach(function (s, i) {
      var el = document.querySelector('[data-mts-score="' + i + '"]');
      if (el) el.textContent = fmt(s, 1);
    });
    var score = Math.min.apply(null, sums);
    var total = sums.reduce(function (a, b) { return a + b; }, 0);
    var weakIdx = sums.indexOf(score);

    setStat('mtsScore', fmt(score, 1), score >= 3 ? 'is-accent' : (score < 1 ? 'is-warn' : ''));
    setStat('mtsWeak', MTS_NAMES[weakIdx] + ' (' + fmt(score, 1) + ')');
    setStat('mtsSum', fmt(total, 1) + ' / 28');

    var msg = '<b>' + mtsBand(score) + '</b> ';
    var zeros = sums.map(function (s, i) { return s === score ? MTS_NAMES[i] : null; }).filter(Boolean);
    if (score < total / 4) {
      msg += 'Vì điểm là <b>giá trị nhỏ nhất</b> của bốn nhóm, đầu tư thêm vào nhóm mạnh ' +
        '<b>không làm tăng điểm</b>. Cần nâng ' +
        (zeros.length > 1 ? 'cả ' + zeros.length + ' nhóm đang thấp nhất (' + zeros.join(', ') + ')'
                          : 'nhóm ' + zeros[0]) + ' trước.';
    } else {
      msg += 'Bốn nhóm có điểm tương đối cân bằng; quy tắc lấy giá trị nhỏ nhất ưu tiên sự cân bằng này.';
    }
    $('mtsVerdict').innerHTML = msg;
  }

  document.addEventListener('click', function (e) {
    var b = e.target.closest('.mts-opt');
    if (b) {
      var g = +b.dataset.g, i = +b.dataset.i;
      mts[g][i] = Number(b.dataset.v);
      var row = b.closest('.mts-item');
      Array.prototype.forEach.call(row.querySelectorAll('.mts-opt'), function (o) {
        o.classList.toggle('is-on', o === b);
      });
      updateMTS();
      return;
    }
    if (e.target.id === 'mtsReset') {
      mts = mts.map(function (g) { return g.map(function () { return 0; }); });
      Array.prototype.forEach.call(document.querySelectorAll('.mts-item'), function (row) {
        Array.prototype.forEach.call(row.querySelectorAll('.mts-opt'), function (o, j) {
          o.classList.toggle('is-on', j === 0);
        });
      });
      updateMTS();
    }
  });

  /* ==================================================== kết nối */

  function on(ids, fn) {
    ids.forEach(function (id) {
      var el = $(id); if (!el) return;
      el.addEventListener('input', fn); el.addEventListener('change', fn);
    });
  }
  on(['abBase', 'abLift', 'abPower', 'abAlpha', 'abTraffic', 'abCuped'], updateAB);
  on(['psiN', 'psiK', 'psiFeat', 'psiPeriod', 'psiAlarm'], updatePSI);
  on(['tlP50', 'tlP99', 'tlK', 'tlBudget'], updateTail);

  updateAB(); updatePSI(); updateTail(); updateMTS();

  var themeBtn = $('themeBtn');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    setTimeout(function () { updateAB(); updatePSI(); updateTail(); }, 30);
  });

  // Phơi phần lõi số học ra để src/test.mjs kiểm chứng bằng Node.
  window.QZ_MLOPS = {
    normInv: normInv, chi2cdf: chi2cdf, chi2sf: chi2sf, chi2inv: chi2inv,
    nPerArm: nPerArm,
    psiMeanNull: function (n, k) { return 2 * (k - 1) / n; },
    psiQuantile: function (q, n, k) { return 2 / n * chi2inv(q, k - 1); },
    psiFalseAlarmRate: function (thresh, n, k) { return chi2sf(thresh * n / 2, k - 1); },
    tailQuantile: function (q, k, p50, p99) {
      var sigma = Math.log(p99 / p50) / normInv(0.99);
      return Math.exp(Math.log(p50) + sigma * normInv(Math.pow(q, 1 / k)));
    },
    mtsScore: function (sections) { return Math.min.apply(null, sections); },
  };
};
window.QZ_INIT['labMlops']();
