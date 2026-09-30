/* Phòng thí nghiệm của giáo trình Biểu diễn, Sinh và Căn chỉnh: ba công cụ. */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['labBieudien'] = function () {
  'use strict';
  var V = window.QZ_VIZ;
  if (!V || !document.querySelector('[data-lab-bieudien]')) return;

  var $ = function (id) { return document.getElementById(id); };
  var C = V.bang();
  function so(id, v) { var e = $(id); if (e) e.textContent = v; }
  function fmt(v, n) {
    return (v == null || !isFinite(v)) ? '–' : v.toFixed(n == null ? 4 : n).replace('.', ',');
  }
  function nhom(v) { return Math.round(v).toLocaleString('vi-VN'); }
  function nghe(ids, fn) {
    ids.forEach(function (i) {
      var e = $(i);
      if (!e) return;
      e.addEventListener('input', fn);
      e.addEventListener('change', fn);
    });
    fn();
  }

  /* ================================================================
     1 · RLHF: beta là vị trí trên một đường đánh đổi
     ================================================================ */
  (function () {
    var p1 = $('rlBar') ? new V.Plot($('rlBar'), { pad: { t: 12, r: 10, b: 30, l: 42 } }) : null;
    var p2 = $('rlTrade') ? new V.Plot($('rlTrade'), { pad: { t: 12, r: 12, b: 32, l: 46 } }) : null;
    if (!p1 || !p2) return;

    var N = 8;
    var piRef = V.softmax([0.4, -1.1, 0.9, 0.2, -0.3, 1.4, -0.8, 0.1]);
    var thuong = [0.6, -1.3, 1.9, 0.1, -0.7, 0.4, 1.2, -0.2];

    function piSao(beta) {
      return V.softmax(piRef.map(function (p, i) {
        return Math.log(p) + thuong[i] / beta;
      }));
    }
    function kl(pi) {
      var s = 0;
      for (var i = 0; i < N; i++) if (pi[i] > 1e-12) s += pi[i] * (Math.log(pi[i]) - Math.log(piRef[i]));
      return s;
    }
    function er(pi) {
      var s = 0;
      for (var i = 0; i < N; i++) s += pi[i] * thuong[i];
      return s;
    }

    function chay() {
      var beta = Math.pow(10, +$('rlBeta').value / 20);
      so('rlBetaOut', V.dinhDang(beta));
      var pi = piSao(beta);

      p1.xlim = [0, N]; p1.ylim = [0, 1];
      p1.xoa();
      p1.truc({ xticks: [], yticks: [0, .25, .5, .75, 1],
        xlabel: 'tám câu trả lời khả dĩ' });
      // π_ref mờ ở nền, π* đậm chồng lên
      var g = p1.ctx, i, pl = p1.pad.l, pr = p1.w - p1.pad.r, bw = (pr - pl) / N;
      for (i = 0; i < N; i++) {
        var a = p1.px(0, piRef[i]), b = p1.px(0, 0);
        g.fillStyle = C.mo; g.globalAlpha = .35;
        g.fillRect(pl + i * bw + bw * .10, a[1], bw * .80, b[1] - a[1]);
        var c = p1.px(0, pi[i]);
        g.fillStyle = thuong[i] > 0 ? C.nhan : C.canh; g.globalAlpha = .92;
        g.fillRect(pl + i * bw + bw * .26, c[1], bw * .48, b[1] - c[1]);
      }
      g.globalAlpha = 1;
      for (i = 0; i < N; i++) {
        p1.chu(i + .5, 0, fmt(thuong[i], 1), { canh: 'center', dy: 13, co: 9 });
      }
      p1.chu(0.1, 0.95, 'xám rộng = π_ref · đậm hẹp = π*  (số dưới là phần thưởng)',
        { co: 9.5, mau: C.mo });

      // mặt đánh đổi
      var bs = [], pts = [];
      for (i = 0; i <= 48; i++) {
        var b2 = Math.pow(10, -1.5 + 3 * i / 48);
        var q = piSao(b2);
        bs.push(b2); pts.push([kl(q), er(q)]);
      }
      var maxKL = Math.max.apply(null, pts.map(function (q) { return q[0]; }));
      p2.xlim = [0, maxKL * 1.05];
      p2.ylim = [Math.min.apply(null, pts.map(function (q) { return q[1]; })) - .15,
        Math.max.apply(null, pts.map(function (q) { return q[1]; })) + .15];
      p2.xoa();
      p2.truc({ xlabel: 'KL(π* ‖ π_ref)', ylabel: 'E[r] đạt được' });
      p2.duong(pts, { mau: C.nhan, day: 2 });
      p2.diem([[kl(pi), er(pi)]], { mau: C.canh, r: 5.5, vien: '#fff', vienDay: 1.4 });

      so('rlKL', fmt(kl(pi)));
      so('rlEr', fmt(er(pi)));
      so('rlMax', fmt(Math.max.apply(null, pi)));
      so('rlNote', beta < 0.1
        ? 'β rất nhỏ: gần như bỏ qua π_ref, dồn xác suất vào câu trả lời có phần thưởng cao nhất. Đây là vùng dễ bị lách phần thưởng nhất.'
        : beta > 5
          ? 'β rất lớn: π* gần trùng π_ref, mô hình hầu như không học được gì từ phần thưởng.'
          : 'KL và phần thưởng tăng giảm cùng nhau: muốn phần thưởng cao hơn thì phải chấp nhận lệch xa π_ref hơn. β chọn một vị trí trên đường đánh đổi bên phải.');
    }
    nghe(['rlBeta'], chay);
  })();

  /* ================================================================
     2 · Lịch nhiễu của mô hình khuếch tán
     ================================================================ */
  (function () {
    var p1 = $('dfSig') ? new V.Plot($('dfSig'), { pad: { t: 12, r: 12, b: 30, l: 46 } }) : null;
    var p2 = $('dfX') ? new V.Plot($('dfX'), { pad: { t: 12, r: 10, b: 28, l: 34 } }) : null;
    if (!p1 || !p2) return;

    var T = 1000, beta = [], alphaN = [], acc = 1, i;
    for (i = 0; i < T; i++) beta.push(1e-4 + (0.02 - 1e-4) * i / (T - 1));
    for (i = 0; i < T; i++) { acc *= (1 - beta[i]); alphaN.push(acc); }

    // một "ảnh" một chiều để nhìn tín hiệu tan dần
    var x0 = [], r0 = V.rng(5);
    for (i = 0; i < 90; i++) x0.push(1.6 * Math.sin(i / 9) + 0.5 * Math.sin(i / 2.5));

    function chay() {
      var t = +$('dfT').value;
      so('dfTOut', t);
      var an = alphaN[t], snr = an / (1 - an);

      p1.xlim = [0, T - 1]; p1.ylim = [-5, 4.2];
      p1.xoa();
      p1.truc({ xlabel: 'bước t', ylabel: 'log₁₀',
        fy: function (v) { return v === 0 ? '1' : '10^' + v; } });
      p1.duong(alphaN.map(function (v, k) { return [k, Math.log10(Math.max(v, 1e-6))]; }),
        { mau: C.nhan, day: 1.9 });
      p1.duong(alphaN.map(function (v, k) {
        return [k, Math.log10(Math.max(v / (1 - v), 1e-6))];
      }), { mau: C.canh, day: 1.9 });
      p1.duong([[0, 0], [T - 1, 0]], { mau: C.duong, day: 1, net: [4, 3] });
      p1.duong([[t, -5], [t, 4.2]], { mau: C.mo, day: 1.3 });
      p1.chu(20, 3.8, 'xanh = ᾱ_t · đỏ = SNR', { co: 9.5, mau: C.mo });

      var r = V.rng(11);
      var xt = x0.map(function (v) {
        return Math.sqrt(an) * v + Math.sqrt(1 - an) * r.n();
      });
      p2.xlim = [0, x0.length - 1]; p2.ylim = [-4, 4];
      p2.xoa();
      p2.truc({ xticks: [], yticks: [-3, 0, 3], xlabel: 'x_t — tín hiệu tan dần theo t' });
      p2.duong(x0.map(function (v, k) { return [k, v]; }), { mau: C.mo, day: 1.2, mo: .55 });
      p2.duong(xt.map(function (v, k) { return [k, v]; }), { mau: C.nhan, day: 1.6 });

      so('dfAn', an < 1e-3 ? an.toExponential(2).replace('.', ',') : fmt(an, 5));
      so('dfSnr', snr < 1e-2 ? snr.toExponential(2).replace('.', ',') : fmt(snr, 4));
      so('dfBd', fmt(Math.sqrt(an) * 100, 2) + '%');
      so('dfNote', t < 60
        ? 'SNR rất cao: tín hiệu gần như nguyên vẹn. Ở các bước này, việc khử nhiễu liên quan tới chi tiết nhỏ: phân biệt nhiễu với kết cấu thật.'
        : t > 700
          ? 'SNR rất thấp: chi tiết đã mất, chỉ còn các thành phần biên độ lớn. Ở các bước này, việc khử nhiễu liên quan tới bố cục tổng thể.'
          : 'Vùng giữa: SNR quanh 1. Kéo về hai đầu để thấy việc khử nhiễu ở hai đầu lịch nhiễu khác nhau thế nào.');
    }
    nghe(['dfT'], chay);
  })();

  /* ================================================================
     3 · LoRA: bộ nhớ huấn luyện
     ================================================================ */
  (function () {
    if (!$('loD')) return;
    function chay() {
      var d = +$('loD').value, L = +$('loL').value, r = +$('loR').value;
      var N = +$('loN').value * 1e9;
      var qv = $('loQV').checked;

      var moiLop = (qv ? 2 : 4) * 2 * d * r;
      var tong = L * moiLop;
      var motLop = d * d;

      var adamLo = tong * 2 * 4;
      var adamFull = N * 2 * 4;
      var trongSo = N * 4;
      var tongFull = trongSo * 2 + adamFull;

      so('loTiLeLop', fmt(2 * r / d * 100, 3) + '%');
      so('loTong', nhom(tong));
      so('loPct', fmt(tong / N * 100, 4) + '%');
      so('loAdam', adamLo / Math.pow(2, 20) < 1024
        ? fmt(adamLo / Math.pow(2, 20), 1) + ' MiB'
        : fmt(adamLo / Math.pow(2, 30), 2) + ' GiB');
      so('loAdamFull', fmt(adamFull / Math.pow(2, 30), 1) + ' GiB');
      so('loLan', nhom(adamFull / adamLo) + ' lần');
      so('loFull', fmt(tongFull / Math.pow(2, 30), 1) + ' GiB');
      so('loVua', tongFull / Math.pow(2, 30) > 80 ? 'KHÔNG vừa A100 80 GB' : 'vừa A100 80 GB');
      so('loNote', 2 * r / d < 0.005
        ? 'Tỉ lệ 2r/d giảm khi d tăng: mô hình càng lớn, LoRA càng tiết kiệm.'
        : 'Tăng d lên 4096 hoặc 8192 để thấy tỉ lệ 2r/d giảm, lý do LoRA đặc biệt có lợi với mô hình lớn.');
    }
    nghe(['loD', 'loL', 'loR', 'loN', 'loQV'], chay);
  })();
};
window.QZ_INIT['labBieudien']();
