/* Phòng thí nghiệm của giáo trình Nền tảng: sáu công cụ tương tác.
   Mọi công thức cài đúng theo giáo trình; chỗ nào có số liệu đo được ở chương
   tương ứng thì công cụ phải cho ra cùng con số. */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['labNentang'] = function () {
  'use strict';
  var V = window.QZ_VIZ;
  if (!V || !document.querySelector('[data-lab-nentang]')) return;

  var $ = function (id) { return document.getElementById(id); };
  var C = V.bang();

  function ve(id, opt) {
    var el = $(id);
    if (!el) return null;
    return new V.Plot(el, opt);
  }
  function so(id, v) { var e = $(id); if (e) e.textContent = v; }
  function nghe(ids, fn) {
    ids.forEach(function (i) {
      var e = $(i);
      if (e) e.addEventListener('input', fn);
    });
    fn();
  }
  function fmt(v, n) {
    return (v == null || !isFinite(v)) ? '–' : v.toFixed(n == null ? 4 : n).replace('.', ',');
  }

  /* ================================================================
     1 · Thiên lệch – phương sai
     Khớp đa thức bậc d trên nhiều tập huấn luyện, tách sai số làm ba phần.
     ================================================================ */
  (function () {
    var p1 = ve('bvFit', { pad: { t: 10, r: 10, b: 28, l: 38 } });
    var p2 = ve('bvBar', { pad: { t: 10, r: 10, b: 28, l: 44 }, xlim: [0, 3] });
    if (!p1 || !p2) return;

    var f = function (x) { return Math.sin(2.2 * x) + 0.35 * x; };
    var NOISE = 0.35, NTR = 40, NSET = 150;
    var xte = [], i;
    for (i = 0; i < 120; i++) xte.push(-3 + 6 * i / 119);

    function khop(xs, ys, d) {
      // Bình phương tối thiểu trên cơ sở đa thức, miền chuẩn hoá về [-1,1]
      // để ma trận đỡ mất điều kiện — đúng lý do nêu ở Mục 2.3 của giáo trình.
      var n = d + 1, A = [], b = [], j, k, r;
      for (j = 0; j < n; j++) { A.push(new Array(n).fill(0)); b.push(0); }
      for (r = 0; r < xs.length; r++) {
        var t = xs[r] / 3, pw = [1];
        for (k = 1; k < n; k++) pw.push(pw[k - 1] * t);
        for (j = 0; j < n; j++) {
          b[j] += pw[j] * ys[r];
          for (k = 0; k < n; k++) A[j][k] += pw[j] * pw[k];
        }
      }
      for (j = 0; j < n; j++) A[j][j] += 1e-9;
      return V.giaiTuyenTinh(A, b);
    }
    function danhGia(w, x) {
      var t = x / 3, s = 0, pw = 1;
      for (var k = 0; k < w.length; k++) { s += w[k] * pw; pw *= t; }
      return s;
    }

    function chay() {
      var d = +$('bvDeg').value;
      so('bvDegOut', d);
      var preds = [], mau = [];
      for (var s = 0; s < NSET; s++) {
        var r = V.rng(1000 + s), xs = [], ys = [];
        for (var i2 = 0; i2 < NTR; i2++) {
          var x = -3 + 6 * r.u();
          xs.push(x); ys.push(f(x) + NOISE * r.n());
        }
        var w = khop(xs, ys, d);
        if (!w) continue;
        var row = xte.map(function (x) { return danhGia(w, x); });
        preds.push(row);
        if (s < 14) mau.push(row);
      }
      // Phân rã: thiên lệch² = (trung bình dự đoán − sự thật)², phương sai = dao động
      // giữa các TẬP HUẤN LUYỆN, đúng định nghĩa ở Mục 2.2.
      var b2 = 0, va = 0;
      for (var j = 0; j < xte.length; j++) {
        var m = 0;
        for (i = 0; i < preds.length; i++) m += preds[i][j];
        m /= preds.length;
        b2 += Math.pow(m - f(xte[j]), 2);
        var v = 0;
        for (i = 0; i < preds.length; i++) v += Math.pow(preds[i][j] - m, 2);
        va += v / preds.length;
      }
      b2 /= xte.length; va /= xte.length;
      var nz = NOISE * NOISE;

      var lo = -3, hi = 3;
      p1.ylim = [-3.2, 3.2]; p1.xlim = [lo, hi];
      p1.xoa();
      p1.truc({ xlabel: 'x' });
      mau.forEach(function (row) {
        p1.duong(xte.map(function (x, k) { return [x, Math.max(-9, Math.min(9, row[k]))]; }),
          { mau: C.canh, day: 1, mo: .28 });
      });
      p1.duong(xte.map(function (x) { return [x, f(x)]; }), { mau: C.muc, day: 2.2 });
      p1.chu(lo + .15, 3.0, 'đen: hàm thật · đỏ: 14 lần khớp trên 14 tập huấn luyện',
        { co: 9.5, mau: C.mo });

      var tong = b2 + va + nz;
      p2.ylim = [0, Math.max(tong * 1.15, 0.6)];
      p2.xoa();
      p2.truc({ xticks: [], fy: function (v) { return V.dinhDang(v); } });
      p2.cot([b2, va, nz], {
        mau: function (k) { return [C.lam, C.canh, C.mo][k]; },
      });
      ['độ chệch²', 'phương sai', 'nhiễu²'].forEach(function (t, k) {
        p2.chu(0.5 + k, 0, t, { canh: 'center', dy: 14, co: 9.5 });
      });
      so('bvB2', fmt(b2)); so('bvVar', fmt(va)); so('bvTot', fmt(tong));
      so('bvNote',
        b2 > va * 3 ? 'Độ chệch chiếm phần lớn sai số: thêm dữ liệu không giúp được, cần mô hình linh hoạt hơn.'
          : va > b2 * 3 ? 'Phương sai chiếm phần lớn sai số: thêm dữ liệu hoặc regularization sẽ giúp, vì phương sai giảm theo cỡ mẫu.'
            : 'Hai thành phần xấp xỉ nhau: đây là vùng quanh điểm có tổng sai số nhỏ nhất.');
    }
    nghe(['bvDeg'], chay);
  })();

  /* ================================================================
     2 · Ridge và Lasso
     ================================================================ */
  (function () {
    var p = ve('rlPath', { pad: { t: 12, r: 12, b: 30, l: 44 } });
    if (!p) return;

    var NP = 12, NN = 80;
    var wThat = [3, -2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    var X = [], y = [];
    (function () {
      var r = V.rng(55);
      for (var i = 0; i < NN; i++) {
        var row = [], s = 0;
        for (var j = 0; j < NP; j++) { var v = r.n(); row.push(v); s += v * wThat[j]; }
        X.push(row); y.push(s + r.n());
      }
    })();

    var XtX = [], Xty = new Array(NP).fill(0);
    (function () {
      for (var j = 0; j < NP; j++) XtX.push(new Array(NP).fill(0));
      for (var i = 0; i < NN; i++) {
        for (var j2 = 0; j2 < NP; j2++) {
          Xty[j2] += X[i][j2] * y[i];
          for (var k = 0; k < NP; k++) XtX[j2][k] += X[i][j2] * X[i][k];
        }
      }
    })();

    function ridge(lam) {
      var A = XtX.map(function (r, j) {
        return r.map(function (v, k) { return v + (j === k ? lam : 0); });
      });
      return V.giaiTuyenTinh(A, Xty.slice());
    }
    function lasso(lam) {
      // Xuống dốc theo toạ độ với toán tử ngưỡng mềm — cách chuẩn cho lasso.
      var w = new Array(NP).fill(0), it, j, k;
      for (it = 0; it < 400; it++) {
        for (j = 0; j < NP; j++) {
          var rho = Xty[j];
          for (k = 0; k < NP; k++) if (k !== j) rho -= XtX[j][k] * w[k];
          var z = XtX[j][j];
          w[j] = Math.sign(rho) * Math.max(Math.abs(rho) - lam / 2, 0) / z;
        }
      }
      return w;
    }

    var lams = [];
    for (var i = 0; i <= 60; i++) lams.push(Math.pow(10, -2 + 4.5 * i / 60));
    var duongR = [], duongL = [];
    lams.forEach(function (l) { duongR.push(ridge(l)); duongL.push(lasso(l)); });

    function chay() {
      var idx = +$('rlLam').value;
      var lam = lams[idx];
      so('rlLamOut', V.dinhDang(lam));
      var wr = duongR[idx], wl = duongL[idx];
      var zr = wr.filter(function (v) { return Math.abs(v) < 1e-10; }).length;
      var zl = wl.filter(function (v) { return Math.abs(v) < 1e-10; }).length;

      p.xlim = [-2, 2.5]; p.ylim = [-2.6, 3.6];
      p.xoa();
      p.truc({ xlabel: 'log₁₀ λ', fx: function (v) { return V.dinhDang(v); } });
      for (var j = 0; j < NP; j++) {
        var mauJ = wThat[j] !== 0 ? C.nhan : C.mo;
        p.duong(lams.map(function (l, k) { return [Math.log10(l), duongR[k][j]]; }),
          { mau: mauJ, day: 1.1, mo: .5 });
        p.duong(lams.map(function (l, k) { return [Math.log10(l), duongL[k][j]]; }),
          { mau: mauJ, day: 1.6, net: [4, 3] });
      }
      p.duong([[Math.log10(lam), -2.6], [Math.log10(lam), 3.6]], { mau: C.canh, day: 1.4 });
      p.duong([[-2, 0], [2.5, 0]], { mau: C.duong, day: 1 });
      p.chu(-1.9, 3.35, 'liền: ridge · đứt: lasso · xanh: hệ số thật khác 0',
        { co: 9.5, mau: C.mo });

      so('rlZR', zr); so('rlZL', zl);
      so('rlNR', fmt(Math.sqrt(wr.reduce(function (a, v) { return a + v * v; }, 0)), 3));
      so('rlNL', fmt(Math.sqrt(wl.reduce(function (a, v) { return a + v * v; }, 0)), 3));
      so('rlNote', zl >= 9
        ? 'Lasso đưa đúng 9 hệ số về 0, trùng với số hệ số bằng 0 của mô hình sinh dữ liệu.'
        : 'Ridge không đưa hệ số nào về đúng 0 ở bất kỳ giá trị λ nào; nó chỉ co các hệ số lại.');
    }
    nghe(['rlLam'], chay);
  })();

  /* ================================================================
     3 · Gradient descent: số điều kiện quyết định tốc độ
     ================================================================ */
  (function () {
    var p = ve('gdPath', { pad: { t: 12, r: 12, b: 30, l: 40 } });
    if (!p) return;

    function chay() {
      var kap = Math.pow(10, +$('gdKap').value / 10);
      var etaTiLe = +$('gdEta').value / 100;
      var quanTinh = $('gdMom').checked;
      var L = kap, m = 1;
      var etaOpt = 2 / (L + m);
      var eta = etaTiLe * (2 / L);           // 2/L là ngưỡng phân kỳ
      var beta = quanTinh
        ? Math.pow((Math.sqrt(L) - 1) / (Math.sqrt(L) + 1), 2) : 0;
      if (quanTinh) eta = etaTiLe * 4 / Math.pow(Math.sqrt(L) + 1, 2);

      so('gdKapOut', V.dinhDang(kap));
      so('gdEtaOut', Math.round(etaTiLe * 100) + '%');
      so('gdEta2', fmt(eta, 5));
      so('gdOpt', fmt(etaOpt, 5));

      var x = [1, 1], v = [0, 0], duong = [[1, 1]], n = 0, phanKy = false;
      for (n = 1; n <= 20000; n++) {
        var g = [x[0], kap * x[1]];
        v = [beta * v[0] - eta * g[0], beta * v[1] - eta * g[1]];
        x = [x[0] + v[0], x[1] + v[1]];
        if (!isFinite(x[0]) || Math.hypot(x[0], x[1]) > 1e6) { phanKy = true; break; }
        if (duong.length < 400) duong.push([x[0], x[1]]);
        if (Math.hypot(x[0], x[1]) / Math.SQRT2 < 1e-8) break;
      }

      p.xlim = [-1.4, 1.4]; p.ylim = [-1.4, 1.4]; p.giuTiLe = true;
      p.xoa();
      // Đường đồng mức của f(x) = ½(x₀² + κx₁²)
      for (var c = 1; c <= 5; c++) {
        var lv = Math.pow(c / 5, 2) * 1.2, pts = [];
        for (var t = 0; t <= 64; t++) {
          var a = 6.2832 * t / 64;
          pts.push([Math.sqrt(2 * lv) * Math.cos(a), Math.sqrt(2 * lv / kap) * Math.sin(a)]);
        }
        p.duong(pts, { mau: C.duong, day: 1, mo: .8 });
      }
      p.truc({ xticks: [-1, 0, 1], yticks: [-1, 0, 1] });
      p.duong(duong, { mau: phanKy ? C.canh : C.nhan, day: 1.5 });
      p.diem(duong.slice(0, 40), { mau: phanKy ? C.canh : C.nhan, r: 2 });
      p.diem([[0, 0]], { mau: C.vang, r: 4, vien: '#fff' });

      so('gdIter', phanKy ? 'phân kỳ' : (n >= 20000 ? '> 20 000' : n.toLocaleString('vi-VN')));
      so('gdNote', phanKy
        ? 'Tốc độ học vượt ngưỡng 2/λ_max nên thuật toán phân kỳ. Với hàm bậc hai, ngưỡng này là chính xác.'
        : quanTinh
          ? 'Momentum giảm số vòng lặp từ bậc κ xuống bậc √κ. Tắt momentum để so sánh.'
          : 'Số vòng lặp tăng tuyến tính theo κ. Bật momentum để thấy nó giảm xuống bậc √κ.');
    }
    ['gdKap', 'gdEta'].forEach(function (i) { var e = $(i); if (e) e.addEventListener('input', chay); });
    var mm = $('gdMom'); if (mm) mm.addEventListener('change', chay);
    chay();
  })();

  /* ================================================================
     4 · SVM lề mềm: C chia các điểm làm ba nhóm theo KKT
     ================================================================ */
  (function () {
    var p = ve('svmPlot', { pad: { t: 10, r: 10, b: 28, l: 34 } });
    if (!p) return;

    var X = [], yv = [];
    (function () {
      var r = V.rng(99);
      for (var i = 0; i < 60; i++) {
        X.push([-0.9 + 1.15 * r.n(), -0.6 + 1.15 * r.n()]); yv.push(-1);
        X.push([0.9 + 1.15 * r.n(), 0.7 + 1.15 * r.n()]); yv.push(1);
      }
    })();

    function huanLuyen(Cv) {
      // Xuống dốc trên bài toán gốc lề mềm: ½‖w‖² + C·Σ max(0, 1 − yᵢ(w·xᵢ+b)).
      // Dùng bài toán gốc thay vì đối ngẫu cho gọn; phân loại ba nhóm theo
      // giá trị của biên, tương đương cách chia theo α ở Mục 13.4.
      var w = [0, 0], b = 0, n = X.length;
      for (var it = 0; it < 4000; it++) {
        var eta = 0.5 / (1 + it * 0.02);
        var gw = [w[0], w[1]], gb = 0;
        for (var i = 0; i < n; i++) {
          var m = yv[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b);
          if (m < 1) {
            gw[0] -= Cv * yv[i] * X[i][0];
            gw[1] -= Cv * yv[i] * X[i][1];
            gb -= Cv * yv[i];
          }
        }
        w = [w[0] - eta * gw[0] / n, w[1] - eta * gw[1] / n];
        b -= eta * gb / n;
      }
      return { w: w, b: b };
    }

    function chay() {
      var Cv = Math.pow(10, +$('svmC').value / 10);
      so('svmCOut', V.dinhDang(Cv));
      var m = huanLuyen(Cv);
      var nw = Math.hypot(m.w[0], m.w[1]) || 1e-9;
      var le = 2 / nw;

      var ngoai = 0, tren = 0, viPham = 0, sai = 0;
      var nhom = X.map(function (x, i) {
        var mm = yv[i] * (m.w[0] * x[0] + m.w[1] * x[1] + m.b);
        if (mm > 1.02) { ngoai++; return 0; }
        if (mm > 0.98) { tren++; return 1; }
        viPham++;
        if (mm < 0) sai++;
        return 2;
      });

      p.xlim = [-4.5, 4.5]; p.ylim = [-4.2, 4.2]; p.giuTiLe = true;
      p.xoa();
      p.vung(function (a, b2) {
        return (m.w[0] * a + m.w[1] * b2 + m.b) > 0 ? C.canh : C.lam;
      }, { mo: .09, buoc: 6 });
      p.truc({ xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4] });

      function ke(c, mauD, net) {
        // w·x + b = c  ->  đường thẳng trong mặt phẳng
        var pts = [];
        for (var t = -5; t <= 5; t += 10) {
          if (Math.abs(m.w[1]) > 1e-6) pts.push([t, (c - m.b - m.w[0] * t) / m.w[1]]);
        }
        if (pts.length === 2) p.duong(pts, { mau: mauD, day: net ? 1 : 1.8, net: net });
      }
      ke(-1, C.mo, [5, 4]); ke(1, C.mo, [5, 4]); ke(0, C.muc);

      p.diem(X, {
        mau: function (i) { return yv[i] > 0 ? C.canh : C.lam; },
        r: function (i) { return nhom[i] === 0 ? 2.6 : 4.2; },
        vien: '#ffffff', vienDay: 1,
      });

      so('svmLe', fmt(le, 3));
      so('svmNgoai', ngoai); so('svmTren', tren); so('svmViPham', viPham);
      so('svmSai', fmt(sai / X.length, 4));
      so('svmNote', Cv < 0.05
        ? 'C nhỏ: lề rộng, phần lớn vector hỗ trợ là điểm vi phạm lề; regularization mạnh.'
        : Cv > 30
          ? 'C lớn: lề hẹp, mô hình phạt nặng từng vi phạm và bám sát dữ liệu hơn.'
          : 'Ba con số ở trên ứng với ba nhóm của điều kiện KKT: ngoài lề, trên lề, vi phạm lề.');
    }
    nghe(['svmC'], chay);
  })();

  /* ================================================================
     5 · PCA so với LDA — KÉO THẢ điểm để thấy hai trục đổi theo
     ================================================================ */
  (function () {
    var p = ve('pcaPlot', { pad: { t: 10, r: 10, b: 28, l: 34 } });
    if (!p) return;

    var pts = [], lab = [];
    function datLai() {
      pts = []; lab = [];
      var r = V.rng(7);
      for (var i = 0; i < 26; i++) {
        pts.push([r.n() * 2.6, -1.1 + r.n() * 0.5]); lab.push(0);
        pts.push([r.n() * 2.6, 1.1 + r.n() * 0.5]); lab.push(1);
      }
    }
    datLai();

    function tinh() {
      var n = pts.length, i;
      var mx = 0, my = 0;
      for (i = 0; i < n; i++) { mx += pts[i][0]; my += pts[i][1]; }
      mx /= n; my /= n;
      var sxx = 0, sxy = 0, syy = 0;
      for (i = 0; i < n; i++) {
        var a = pts[i][0] - mx, b = pts[i][1] - my;
        sxx += a * a; sxy += a * b; syy += b * b;
      }
      sxx /= n - 1; sxy /= n - 1; syy /= n - 1;
      // Vector riêng lớn nhất của ma trận 2x2 đối xứng, dạng đóng.
      var tr = sxx + syy, det = sxx * syy - sxy * sxy;
      var l1 = tr / 2 + Math.sqrt(Math.max(tr * tr / 4 - det, 0));
      var l2 = tr - l1;
      var pc = Math.abs(sxy) > 1e-9 ? [l1 - syy, sxy] : (sxx >= syy ? [1, 0] : [0, 1]);
      var np_ = Math.hypot(pc[0], pc[1]) || 1; pc = [pc[0] / np_, pc[1] / np_];

      // LDA: w ∝ S_W⁻¹(μ₁ − μ₀)
      var m0 = [0, 0], m1 = [0, 0], n0 = 0, n1 = 0;
      for (i = 0; i < n; i++) {
        if (lab[i]) { m1[0] += pts[i][0]; m1[1] += pts[i][1]; n1++; }
        else { m0[0] += pts[i][0]; m0[1] += pts[i][1]; n0++; }
      }
      m0 = [m0[0] / n0, m0[1] / n0]; m1 = [m1[0] / n1, m1[1] / n1];
      var w00 = 0, w01 = 0, w11 = 0;
      for (i = 0; i < n; i++) {
        var mu = lab[i] ? m1 : m0;
        var da = pts[i][0] - mu[0], db = pts[i][1] - mu[1];
        w00 += da * da; w01 += da * db; w11 += db * db;
      }
      w00 += 1e-6; w11 += 1e-6;
      var dm = [m1[0] - m0[0], m1[1] - m0[1]];
      var dt = w00 * w11 - w01 * w01;
      var ld = Math.abs(dt) < 1e-12 ? dm
        : [(w11 * dm[0] - w01 * dm[1]) / dt, (w00 * dm[1] - w01 * dm[0]) / dt];
      var nl = Math.hypot(ld[0], ld[1]) || 1; ld = [ld[0] / nl, ld[1] / nl];

      function auc(v) {
        var s = pts.map(function (q, k) { return [q[0] * v[0] + q[1] * v[1], lab[k]]; });
        s.sort(function (a, b) { return a[0] - b[0]; });
        var P = 0, N = 0, tot = 0, cum = 0;
        for (i = 0; i < s.length; i++) { if (s[i][1]) P++; else N++; }
        for (i = 0; i < s.length; i++) {
          if (s[i][1]) cum++; else tot += cum;
        }
        var a2 = (P && N) ? tot / (P * N) : 0.5;
        return Math.max(a2, 1 - a2);
      }
      return { mx: mx, my: my, pc: pc, ld: ld, l1: l1, l2: l2,
        aucPC: auc(pc), aucLD: auc(ld) };
    }

    function chay() {
      var r = tinh();
      p.xlim = [-7, 7]; p.ylim = [-5, 5]; p.giuTiLe = true;
      p.xoa();
      p.truc({ xticks: [-6, -3, 0, 3, 6], yticks: [-4, -2, 0, 2, 4] });
      function truc(v, mauD, ten) {
        var L = 6.5;
        p.duong([[r.mx - v[0] * L, r.my - v[1] * L], [r.mx + v[0] * L, r.my + v[1] * L]],
          { mau: mauD, day: 2 });
        p.chu(r.mx + v[0] * L * .82, r.my + v[1] * L * .82, ten,
          { mau: mauD, co: 10.5, canh: 'center' });
      }
      truc(r.pc, C.mo, 'PCA');
      truc(r.ld, C.nhan, 'LDA');
      p.diem(pts, {
        mau: function (i) { return lab[i] ? C.canh : C.lam; },
        r: 4.4, vien: '#ffffff', vienDay: 1.1,
      });
      so('pcaVar', fmt(r.l1 / (r.l1 + r.l2) * 100, 1) + '%');
      so('pcaAucPC', fmt(r.aucPC));
      so('pcaAucLD', fmt(r.aucLD));
      so('pcaNote', r.aucPC < 0.62
        ? 'Thành phần chính thứ nhất giữ gần hết phương sai nhưng AUC gần 0,5: PCA không dùng nhãn nên không biết hướng nào phân biệt được hai lớp.'
        : 'Hai hướng đang gần trùng nhau. Kéo các điểm để dữ liệu trải rộng theo một hướng không chứa thông tin về lớp.');
    }

    p.keoTha(function () { return pts; }, function (i, x, y) {
      pts[i] = [Math.max(-7, Math.min(7, x)), Math.max(-5, Math.min(5, y))];
      chay();
    });
    var btn = $('pcaReset');
    if (btn) btn.addEventListener('click', function () { datLai(); chay(); });
    chay();
  })();

  /* ================================================================
     6 · K-means: bấm từng bước để thấy nó hội tụ, và kẹt ở đâu
     ================================================================ */
  (function () {
    var p = ve('kmPlot', { pad: { t: 10, r: 10, b: 28, l: 34 } });
    if (!p) return;

    var X = [], tam = [], gan = [], buoc = 0, xong = false;
    var tamThat = [[0, 0], [4.6, .4], [2.2, 3.9], [6.6, 3.4]];

    function sinh() {
      X = [];
      var r = V.rng(11);
      tamThat.forEach(function (c) {
        for (var i = 0; i < 55; i++) X.push([c[0] + r.n() * .72, c[1] + r.n() * .72]);
      });
    }
    function khoiTao(kieu) {
      var k = +$('kmK').value, r = V.rng(Date.now() & 0xffff);
      tam = [];
      if (kieu === 'xau') {
        // Cố tình khởi tạo dồn một chỗ để thấy nó kẹt.
        for (var i = 0; i < k; i++) tam.push([X[0][0] + r.n() * .25, X[0][1] + r.n() * .25]);
      } else {
        var dung = {};
        while (tam.length < k) {
          var j = Math.floor(r.u() * X.length);
          if (!dung[j]) { dung[j] = 1; tam.push([X[j][0], X[j][1]]); }
        }
      }
      buoc = 0; xong = false;
      ganCum();
    }
    function ganCum() {
      gan = X.map(function (x) {
        var tot = 0, min = Infinity;
        for (var j = 0; j < tam.length; j++) {
          var d = Math.pow(x[0] - tam[j][0], 2) + Math.pow(x[1] - tam[j][1], 2);
          if (d < min) { min = d; tot = j; }
        }
        return tot;
      });
    }
    function capNhat() {
      var s = tam.map(function () { return [0, 0, 0]; });
      X.forEach(function (x, i) {
        s[gan[i]][0] += x[0]; s[gan[i]][1] += x[1]; s[gan[i]][2]++;
      });
      var doi = 0;
      tam = tam.map(function (c, j) {
        if (!s[j][2]) return c;
        var nc = [s[j][0] / s[j][2], s[j][1] / s[j][2]];
        doi += Math.hypot(nc[0] - c[0], nc[1] - c[1]);
        return nc;
      });
      ganCum();
      buoc++;
      if (doi < 1e-7) xong = true;
    }
    function quanTinh() {
      var s = 0;
      X.forEach(function (x, i) {
        s += Math.pow(x[0] - tam[gan[i]][0], 2) + Math.pow(x[1] - tam[gan[i]][1], 2);
      });
      return s;
    }
    function chay() {
      p.xlim = [-2.5, 9]; p.ylim = [-2.5, 6.5]; p.giuTiLe = true;
      p.xoa();
      p.truc({ xticks: [-2, 0, 2, 4, 6, 8], yticks: [-2, 0, 2, 4, 6] });
      var mauCum = [C.nhan, C.canh, C.vang, C.tim, C.lam, C.mo];
      p.diem(X, { mau: function (i) { return mauCum[gan[i] % 6]; }, r: 2.8, mo: .78 });
      tam.forEach(function (c, j) {
        p.diem([c], { mau: mauCum[j % 6], r: 7, vien: '#ffffff', vienDay: 2 });
      });
      so('kmBuoc', buoc);
      so('kmQT', fmt(quanTinh(), 2));
      so('kmNote', xong
        ? 'Đã hội tụ. Bấm "Khởi tạo dồn một chỗ" vài lần để thấy thuật toán dừng ở các nghiệm tồi khác nhau.'
        : 'Mỗi bước gồm hai việc: gán mỗi điểm cho tâm gần nhất, rồi dời tâm về trung bình cụm.');
    }

    sinh(); khoiTao('tot'); chay();
    var b1 = $('kmStep'), b2 = $('kmRun'), b3 = $('kmNew'), b4 = $('kmBad'), b5 = $('kmK');
    if (b1) b1.addEventListener('click', function () { if (!xong) capNhat(); chay(); });
    if (b2) b2.addEventListener('click', function () {
      for (var i = 0; i < 60 && !xong; i++) capNhat();
      chay();
    });
    if (b3) b3.addEventListener('click', function () { khoiTao('tot'); chay(); });
    if (b4) b4.addEventListener('click', function () { khoiTao('xau'); chay(); });
    if (b5) b5.addEventListener('input', function () {
      so('kmKOut', $('kmK').value); khoiTao('tot'); chay();
    });
    so('kmKOut', $('kmK') ? $('kmK').value : '');
  })();
};
window.QZ_INIT['labNentang']();
