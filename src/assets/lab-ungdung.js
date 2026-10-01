/* Phòng thí nghiệm của giáo trình Ứng dụng LLM: bốn công cụ.
   Phần lõi số học nằm trong window.QZ_UNGDUNG để src/test.mjs kiểm chứng bằng chính
   các con số đã in trong giáo trình (Mục 2.2, 2.5, 9.7, 12.4). */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['labUngdung'] = function () {
  'use strict';

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

  // Hàm phân phối chuẩn qua erfc (Numerical Recipes), sai số tương đối < 1,2e-7.
  function normCdf(x) {
    var z = Math.abs(x) / Math.SQRT2;
    var t = 1 / (1 + 0.5 * z);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r / 2 : r / 2;
  }

  // log(k!) cho k = 0..n, tính một lần rồi dùng lại.
  var LOGF = [0];
  function logGiaiThua(n) {
    for (var k = LOGF.length; k <= n; k++) LOGF.push(LOGF[k - 1] + Math.log(k));
    return LOGF;
  }
  function logNhiThuc(x, n, p) {
    var lf = logGiaiThua(n);
    if (p <= 0) return x === 0 ? 0 : -Infinity;
    if (p >= 1) return x === n ? 0 : -Infinity;
    return lf[n] - lf[x] - lf[n - x] + x * Math.log(p) + (n - x) * Math.log(1 - p);
  }

  /* ==================================================== 1 · chi phí */

  /** Chi phí một yêu cầu theo cách tính ở Mục 2.5. Đơn giá tính cho mỗi triệu token. */
  function chiPhi(o) {
    var truyXuat = o.soDoan * o.doan;
    var tongVao = o.heThong + truyXuat + o.lichSu + o.cauHoi;
    var heSoCache = o.cache ? o.giaCache : 1;
    var phan = {
      heThong: o.heThong * o.giaVao * heSoCache / 1e6,
      truyXuat: truyXuat * o.giaVao / 1e6,
      lichSu: o.lichSu * o.giaVao / 1e6,
      cauHoi: o.cauHoi * o.giaVao / 1e6,
      ra: o.ra * o.giaRa / 1e6,
    };
    var moiYC = phan.heThong + phan.truyXuat + phan.lichSu + phan.cauHoi + phan.ra;
    var khongCache = (tongVao * o.giaVao + o.ra * o.giaRa) / 1e6;
    var tietKiem = khongCache > 0 ? o.heThong * o.giaVao * (1 - o.giaCache) / 1e6 / khongCache : 0;
    return {
      tongVao: tongVao, phan: phan, moiYC: moiYC, khongCache: khongCache,
      tiLeRa: moiYC > 0 ? phan.ra / moiYC : 0,
      tiLeTokenRa: (tongVao + o.ra) > 0 ? o.ra / (tongVao + o.ra) : 0,
      tiLeHoi: tongVao > 0 ? o.cauHoi / tongVao : 0,
      tietKiem: tietKiem,
      thang: moiYC * o.soYeuCau,
    };
  }

  /* ==================================================== 2 · KV cache */

  /** Số byte KV cache: 2 (key và value) × số lớp × số đầu KV × số chiều mỗi đầu × số byte. */
  function kvMoiToken(L, nkv, dHead, soByte) { return 2 * L * nkv * dHead * soByte; }

  /* ==================================================== 3 · agent */

  /** Mục 9.7: một bước thành công nếu đúng ở một lần thử trước khi hết lượt;
      lần sai chỉ được thử lại khi bị phát hiện. */
  function agent(p, n, c, r) {
    var x = (1 - p) * c, q = 0, goi = 0;
    for (var j = 0; j <= r; j++) { q += p * Math.pow(x, j); goi += Math.pow(x, j); }
    return { khong: Math.pow(p, n), q: q, co: Math.pow(q, n), goiMoiBuoc: goi };
  }

  /* ==================================================== 4 · cỡ bộ đánh giá */

  /** Nửa độ rộng khoảng tin cậy 95% của một tỉ lệ (xấp xỉ chuẩn, như Mục 12.4). */
  function ktc(p, n) { return 1.96 * Math.sqrt(p * (1 - p) / n); }

  /** Lực của kiểm định McNemar chính xác hai phía, chỉ tính khi B thắng nhiều hơn.
      p01 = P(A sai, B đúng), p10 = P(A đúng, B sai). Tính đúng bằng tổng hai lớp nhị thức:
      số cặp bất đồng D ~ Bin(n, p01 + p10), và trong D cặp, số cặp B thắng ~ Bin(D, p01/(p01+p10)). */
  function lucGhepCap(n, p01, p10, alpha) {
    alpha = alpha || 0.05;
    var s = p01 + p10;
    if (s <= 0 || n <= 0) return 0;
    var rr = p01 / s;
    var tb = n * s, lech = Math.sqrt(n * s * (1 - s));
    var dLo = Math.max(1, Math.floor(tb - 9 * lech) - 1);
    var dHi = Math.min(n, Math.ceil(tb + 9 * lech) + 1);
    var luc = 0;
    for (var d = dLo; d <= dHi; d++) {
      var pd = Math.exp(logNhiThuc(d, n, s));
      if (pd < 1e-14) continue;
      // giá trị tới hạn: x nhỏ nhất (x > d/2) mà 2·P(X ≥ x | p = 0,5) < alpha
      var duoi = 0, tiHan = -1;
      for (var x = d; x > d / 2; x--) {
        duoi += Math.exp(logNhiThuc(x, d, 0.5));
        if (2 * duoi < alpha) tiHan = x; else break;
      }
      if (tiHan < 0) continue;
      var pr = 0;
      for (var y = tiHan; y <= d; y++) pr += Math.exp(logNhiThuc(y, d, rr));
      luc += pd * pr;
    }
    return luc;
  }

  /** Lực của kiểm định hai tỉ lệ trên hai bộ câu hỏi độc lập (xấp xỉ chuẩn, phương sai gộp dưới H0). */
  function lucDocLap(n, pA, pB, alpha) {
    var z = normInv(1 - (alpha || 0.05) / 2);
    var pb = (pA + pB) / 2;
    var se0 = Math.sqrt(2 * pb * (1 - pb) / n);
    var se1 = Math.sqrt((pA * (1 - pA) + pB * (1 - pB)) / n);
    return normCdf((pB - pA - z * se0) / se1);
  }

  /** Số câu cần cho lực 1 - beta. Ghép cặp: công thức xấp xỉ chuẩn cho kiểm định McNemar. */
  function nGhepCap(p01, p10, alpha, power) {
    var za = normInv(1 - alpha / 2), zb = normInv(power);
    var s = p01 + p10, dl = p01 - p10;
    if (dl <= 0) return Infinity;
    var t = za * Math.sqrt(s) + zb * Math.sqrt(s - dl * dl);
    return t * t / (dl * dl);
  }
  function nDocLap(pA, pB, alpha, power) {
    var za = normInv(1 - alpha / 2), zb = normInv(power);
    var pb = (pA + pB) / 2, dl = pB - pA;
    if (dl <= 0) return Infinity;
    var t = za * Math.sqrt(2 * pb * (1 - pb)) + zb * Math.sqrt(pA * (1 - pA) + pB * (1 - pB));
    return t * t / (dl * dl);
  }

  // Phơi phần lõi số học ra trước khi động tới DOM, để Node kiểm chứng được.
  window.QZ_UNGDUNG = {
    normInv: normInv, normCdf: normCdf, chiPhi: chiPhi, kvMoiToken: kvMoiToken,
    agent: agent, ktc: ktc, lucGhepCap: lucGhepCap, lucDocLap: lucDocLap,
    nGhepCap: nGhepCap, nDocLap: nDocLap,
  };

  /* ==================================================== giao diện */

  var V = window.QZ_VIZ;
  if (!V || !document.querySelector('[data-lab-ungdung]')) return;
  var $ = function (id) { return document.getElementById(id); };
  var C = V.bang();
  function so(id, v) { var e = $(id); if (e) e.textContent = v; }
  function html(id, v) { var e = $(id); if (e) e.innerHTML = v; }
  function fmt(v, n) {
    return (v == null || !isFinite(v)) ? '–' : v.toFixed(n == null ? 4 : n).replace('.', ',');
  }
  function nhom(v) { return isFinite(v) ? Math.round(v).toLocaleString('vi-VN') : '–'; }
  function soDuong(id, mac) { var v = parseFloat($(id).value); return isFinite(v) && v >= 0 ? v : mac; }
  function nghe(ids, fn) {
    ids.forEach(function (i) {
      var e = $(i);
      if (!e) return;
      e.addEventListener('input', fn);
      e.addEventListener('change', fn);
    });
    fn();
  }
  function bytes(b) {
    var KiB = 1024, MiB = KiB * 1024, GiB = MiB * 1024, TiB = GiB * 1024;
    if (b >= TiB) return fmt(b / TiB, 2) + ' TiB';
    if (b >= GiB) return fmt(b / GiB, 2) + ' GiB';
    if (b >= MiB) return fmt(b / MiB, 1) + ' MiB';
    return fmt(b / KiB, 1) + ' KiB';
  }

  /* ---------------------------------------------- 1 · chi phí theo token */
  (function () {
    var p = $('cpPlot') ? new V.Plot($('cpPlot'), { pad: { t: 14, r: 10, b: 34, l: 46 } }) : null;
    function chay() {
      var o = {
        heThong: soDuong('cpHeThong', 0), soDoan: soDuong('cpSoDoan', 0), doan: soDuong('cpDoan', 0),
        lichSu: soDuong('cpLichSu', 0), cauHoi: soDuong('cpCauHoi', 0), ra: soDuong('cpRa', 0),
        giaVao: soDuong('cpGiaVao', 0), giaRa: soDuong('cpGiaRa', 0),
        cache: $('cpCache').checked, giaCache: soDuong('cpGiaCache', 10) / 100,
        soYeuCau: soDuong('cpSoYeuCau', 0),
      };
      var k = chiPhi(o);
      so('cpTongVao', nhom(k.tongVao) + ' token');
      so('cpMoiYC', fmt(k.moiYC, 5) + ' đơn vị');
      so('cpTiLeRa', fmt(k.tiLeRa * 100, 1) + '%');
      so('cpTiLeHoi', fmt(k.tiLeHoi * 100, 1) + '%');
      so('cpTietKiem', fmt(k.tietKiem * 100, 1) + '%');
      so('cpThang', nhom(k.thang) + ' đơn vị');

      if (p) {
        var ten = ['chỉ dẫn', 'truy xuất', 'lịch sử', 'câu hỏi', 'đầu ra'];
        var v = [k.phan.heThong, k.phan.truyXuat, k.phan.lichSu, k.phan.cauHoi, k.phan.ra].map(function (x) {
          return x * 1e3;   // nghìn phần đơn vị tiền, cho dễ đọc
        });
        var mx = Math.max.apply(null, v.concat([1e-6]));
        p.xlim = [0, 5]; p.ylim = [0, mx * 1.15];
        p.xoa();
        p.truc({ xticks: [], ylabel: '× 10⁻³ đơn vị' });
        p.cot(v, { mau: function (i) { return i === 4 ? C.canh : C.nhan; } });
        for (var i = 0; i < 5; i++) {
          p.chu(i + 0.5, 0, ten[i], { canh: 'center', doc: 'top', dy: 6, co: 10 });
        }
      }
      html('cpNote', 'Đầu ra chiếm <b>' + fmt(k.tiLeTokenRa * 100, 1) + '%</b> số token nhưng <b>' +
        fmt(k.tiLeRa * 100, 1) + '%</b> chi phí; câu hỏi của người dùng chỉ chiếm ' +
        fmt(k.tiLeHoi * 100, 1) + '% đầu vào. Phần lớn chi phí nằm ở ngữ cảnh do ứng dụng thêm vào.' +
        (o.cache ? ' Đang tính chỉ dẫn hệ thống theo giá bộ đệm.' : ''));
    }
    nghe(['cpHeThong', 'cpSoDoan', 'cpDoan', 'cpLichSu', 'cpCauHoi', 'cpRa', 'cpGiaVao', 'cpGiaRa',
      'cpCache', 'cpGiaCache', 'cpSoYeuCau'], chay);
  })();

  /* ---------------------------------------------- 2 · KV cache */
  (function () {
    var p = $('kvPlot') ? new V.Plot($('kvPlot'), { pad: { t: 12, r: 12, b: 32, l: 46 } }) : null;
    function chay() {
      var L = soDuong('kvL', 1), nq = soDuong('kvQ', 1), nkv = soDuong('kvKV', 1), dh = soDuong('kvD', 1);
      var P = soDuong('kvP', 0), sb = parseFloat($('kvByte').value);
      var T = soDuong('kvT', 1), B = soDuong('kvB', 1);
      var moi = kvMoiToken(L, nkv, dh, sb);
      var chuoi = moi * T, tong = chuoi * B;
      var trongSo = P * 1e9 * 2;
      so('kvMoiToken', bytes(moi));
      so('kvMoiChuoi', bytes(chuoi));
      so('kvTong', bytes(tong));
      so('kvTrongSo', fmt(trongSo / 1e9, 1) + ' GB');
      so('kvMHA', bytes(kvMoiToken(L, nq, dh, sb) * T * B));
      so('kvTiLe', fmt(trongSo > 0 ? tong / trongSo : NaN, 2) + ' lần');

      if (p) {
        // trục hoành: log2 của độ dài ngữ cảnh, từ 1 nghìn tới 1 triệu token
        var lo = 10, hi = 20, GiB = Math.pow(2, 30);
        var pts = [], ptsM = [];
        for (var e = lo; e <= hi; e += 0.25) {
          pts.push([e, moi * Math.pow(2, e) * B / GiB]);
          ptsM.push([e, kvMoiToken(L, nq, dh, sb) * Math.pow(2, e) * B / GiB]);
        }
        var ymax = Math.max(trongSo / GiB * 1.3, moi * Math.pow(2, hi) * B / GiB * 1.05);
        p.xlim = [lo, hi]; p.ylim = [0, ymax];
        p.xoa();
        p.truc({ xticks: [10, 12, 14, 16, 18, 20], xlabel: 'độ dài ngữ cảnh (token, thang log)',
          fx: function (v) { var n = Math.pow(2, v); return n >= 1048576 ? (n / 1048576) + 'M' : (n / 1024) + 'K'; },
          ylabel: 'GiB' });
        // đường MHA lớn hơn nhiều lần: chỉ vẽ phần nằm trong khung
        ptsM = ptsM.filter(function (q) { return q[1] <= ymax; });
        if (nq !== nkv && ptsM.length > 1) p.duong(ptsM, { mau: C.mo, day: 1.2, net: [4, 3] });
        p.duong(pts, { mau: C.nhan, day: 2 });
        p.duong([[lo, trongSo / GiB], [hi, trongSo / GiB]], { mau: C.canh, day: 1.2, net: [6, 4] });
        p.chu(lo + 0.1, trongSo / GiB, 'trọng số', { mau: C.canh, doc: 'bottom', dy: -3 });
        var xT = Math.log2(Math.max(T, 1));
        if (xT >= lo && xT <= hi) p.diem([[xT, chuoi * B / GiB]], { mau: C.muc, r: 4 });
      }
      html('kvNote', 'Mỗi token cần <b>' + bytes(moi) + '</b>. Với ' + nhom(T) + ' token và ' + nhom(B) +
        ' chuỗi, KV cache là <b>' + bytes(tong) + '</b>, bằng ' + fmt(trongSo > 0 ? tong / trongSo : NaN, 2) +
        ' lần trọng số. ' + (nq !== nkv ? 'Đường đứt là trường hợp mỗi đầu query có một đầu key/value riêng (không dùng GQA), lớn gấp ' +
        fmt(nq / nkv, 0) + ' lần.' : ''));
    }
    nghe(['kvL', 'kvQ', 'kvKV', 'kvD', 'kvP', 'kvByte', 'kvT', 'kvB'], chay);
    var nut = $('kvPreset');
    if (nut) nut.addEventListener('change', function () {
      var v = nut.value.split(',');
      ['kvL', 'kvQ', 'kvKV', 'kvD', 'kvP'].forEach(function (id, i) { $(id).value = v[i]; });
      chay();
    });
  })();

  /* ---------------------------------------------- 3 · agent nhiều bước */
  (function () {
    var p = $('agPlot') ? new V.Plot($('agPlot'), { pad: { t: 12, r: 12, b: 32, l: 40 } }) : null;
    function chay() {
      var pp = +$('agP').value / 1000, n = +$('agN').value, c = +$('agC').value / 100, r = +$('agR').value;
      so('agPOut', fmt(pp, 3)); so('agNOut', n); so('agCOut', fmt(c, 2)); so('agROut', r);
      var k = agent(pp, n, c, r);
      so('agKhong', fmt(k.khong * 100, 1) + '%');
      so('agCo', fmt(k.co * 100, 1) + '%');
      so('agGoi', fmt(k.goiMoiBuoc, 3));
      if (p) {
        var nMax = Math.max(50, n), a = [], b = [];
        for (var i = 1; i <= nMax; i++) { a.push([i, Math.pow(pp, i)]); b.push([i, Math.pow(k.q, i)]); }
        p.xlim = [1, nMax]; p.ylim = [0, 1.02];
        p.xoa();
        p.truc({ xlabel: 'số bước', ylabel: 'xác suất hoàn thành',
          fx: function (v) { return String(Math.round(v)); } });
        p.duong(a, { mau: C.canh, day: 1.8 });
        p.duong(b, { mau: C.nhan, day: 2.2 });
        p.diem([[n, k.khong], [n, k.co]], { mau: function (i) { return i ? C.nhan : C.canh; }, r: 4 });
        p.chu(nMax * 0.98, 0.97, 'có kiểm tra và thử lại', { canh: 'right', mau: C.nhan });
        p.chu(nMax * 0.98, 0.89, 'không kiểm tra', { canh: 'right', mau: C.canh });
      }
      html('agNote', 'Với ' + n + ' bước, mỗi bước đúng ' + fmt(pp * 100, 1) + '%, agent hoàn thành <b>' +
        fmt(k.khong * 100, 1) + '%</b> số nhiệm vụ nếu không kiểm tra, và <b>' + fmt(k.co * 100, 1) +
        '%</b> nếu kiểm tra phát hiện ' + fmt(c * 100, 0) + '% lỗi và cho thử lại tối đa ' + r +
        ' lần; số lời gọi mỗi bước tăng ' + fmt((k.goiMoiBuoc - 1) * 100, 1) + '%. ' +
        (c === 0 || r === 0 ? 'Khi c = 0 hoặc r = 0, kiểm tra không có tác dụng.' :
          'Tăng c thường có lợi hơn tăng r, vì lỗi không bị phát hiện thì không được thử lại.'));
    }
    nghe(['agP', 'agN', 'agC', 'agR'], chay);
  })();

  /* ---------------------------------------------- 4 · cỡ bộ đánh giá */
  (function () {
    var p = $('dgPlot') ? new V.Plot($('dgPlot'), { pad: { t: 12, r: 12, b: 32, l: 40 } }) : null;
    var NS = [];
    for (var e = 1; e <= 3.7; e += 0.1) NS.push(Math.round(Math.pow(10, e)));
    function chay() {
      var pA = Math.min(Math.max(soDuong('dgA', 80) / 100, 0.01), 0.99);
      var giu = Math.min(soDuong('dgGiu', 97) / 100, 1), sua = Math.min(soDuong('dgSua', 27) / 100, 1);
      var n = Math.min(Math.max(Math.round(soDuong('dgN', 1000)), 10), 5000);
      var p10 = pA * (1 - giu), p01 = (1 - pA) * sua, pB = pA - p10 + p01;
      var lc = lucGhepCap(n, p01, p10, 0.05), lr = lucDocLap(n, pA, pB, 0.05);
      var nc = nGhepCap(p01, p10, 0.05, 0.8), nr = nDocLap(pA, pB, 0.05, 0.8);
      so('dgB', fmt(pB * 100, 1) + '%');
      so('dgKTC', '± ' + fmt(ktc(pA, n) * 100, 1));
      so('dgLucCap', pB > pA ? fmt(lc, 3) : '–');
      so('dgLucRieng', pB > pA ? fmt(lr, 3) : '–');
      so('dgNCap', isFinite(nc) ? nhom(Math.ceil(nc)) : 'không đạt');
      so('dgNRieng', isFinite(nr) ? nhom(Math.ceil(nr)) : 'không đạt');
      if (p) {
        var a = [], b = [];
        NS.forEach(function (m) {
          a.push([Math.log10(m), pB > pA ? lucGhepCap(m, p01, p10, 0.05) : 0]);
          b.push([Math.log10(m), pB > pA ? lucDocLap(m, pA, pB, 0.05) : 0]);
        });
        p.xlim = [1, 3.7]; p.ylim = [0, 1.02];
        p.xoa();
        p.truc({ xticks: [1, 2, 3, Math.log10(5000)], xlabel: 'số câu hỏi (thang log)',
          fx: function (v) { return nhom(Math.pow(10, v)); }, ylabel: 'xác suất phát hiện' });
        p.duong([[1, 0.8], [3.7, 0.8]], { mau: C.duong, day: 1, net: [4, 3] });
        p.duong(b, { mau: C.mo, day: 1.8 });
        p.duong(a, { mau: C.nhan, day: 2.2 });
        p.diem([[Math.log10(n), pB > pA ? lc : 0]], { mau: C.muc, r: 4 });
        p.chu(1.05, 0.95, 'ghép cặp', { mau: C.nhan });
        p.chu(1.05, 0.86, 'hai bộ độc lập', { mau: C.mo });
      }
      html('dgNote', pB <= pA
        ? 'Với các tham số này B không tốt hơn A, nên không có gì để phát hiện.'
        : 'B đúng ' + fmt(pB * 100, 1) + '%, hơn A ' + fmt((pB - pA) * 100, 1) + ' điểm phần trăm. Với ' + nhom(n) +
          ' câu, so sánh ghép cặp phát hiện được ở <b>' + fmt(lc * 100, 0) + '%</b> số lần, hai bộ độc lập ở <b>' +
          fmt(lr * 100, 0) + '%</b>. Ghép cặp chỉ dùng các câu hai phiên bản trả lời khác nhau, nên loại được phần biến động do độ khó của câu hỏi.');
    }
    nghe(['dgA', 'dgGiu', 'dgSua', 'dgN'], chay);
  })();

  var themeBtn = $('themeBtn');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    setTimeout(function () {
      C = V.bang();
      ['cpGiaVao', 'kvT', 'agN', 'dgN'].forEach(function (id) {
        var e = $(id); if (e) e.dispatchEvent(new Event('input'));
      });
    }, 30);
  });
};
window.QZ_INIT['labUngdung']();
