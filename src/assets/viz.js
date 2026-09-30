/* Bộ vẽ dùng chung cho các phòng thí nghiệm.

   Cố ý KHÔNG dùng thư viện đồ thị ngoài. Lý do: site này chạy được cả khi mở
   bằng file:// và không cần mạng, nên mọi thứ phải nằm trong repo; và các hình
   ở đây cần kéo thả điểm, vẽ đường đồng mức, tô vùng quyết định — những việc mà
   một thư viện đồ thị thông thường không làm sẵn, nên cuối cùng vẫn phải vẽ tay.

   Khoảng 300 dòng này đủ cho toàn bộ 12 công cụ, và nó đọc được. */
window.QZ_VIZ = (function () {
  'use strict';

  function mau(ten, duPhong) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(ten);
    return (v && v.trim()) || duPhong;
  }

  /** Bảng màu đọc từ CSS nên tự đổi theo giao diện sáng/tối. */
  function bang() {
    return {
      muc: mau('--text', '#1a1a1a'),
      mo: mau('--text-3', '#8a857c'),
      duong: mau('--line', '#cfd8d6'),
      nen: mau('--surface', '#ffffff'),
      nhan: mau('--accent', '#1f6f68'),
      canh: '#b0413e',
      vang: '#b8860b',
      tim: '#8a5fa8',
      lam: '#2f6f9f',
    };
  }

  function Plot(canvas, opt) {
    opt = opt || {};
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.pad = Object.assign({ t: 12, r: 12, b: 30, l: 42 }, opt.pad || {});
    this.xlim = opt.xlim || [0, 1];
    this.ylim = opt.ylim || [0, 1];
    this.giuTiLe = !!opt.giuTiLe;
    this.doPhanGiai();
  }

  Plot.prototype.doPhanGiai = function () {
    var r = window.devicePixelRatio || 1;
    var w = this.c.clientWidth || 360;
    var h = this.c.clientHeight || 220;
    if (this.c.width !== Math.round(w * r) || this.c.height !== Math.round(h * r)) {
      this.c.width = Math.round(w * r);
      this.c.height = Math.round(h * r);
    }
    this.w = w;
    this.h = h;
    this.ctx.setTransform(r, 0, 0, r, 0, 0);
  };

  /** Toạ độ dữ liệu -> toạ độ pixel. */
  Plot.prototype.px = function (x, y) {
    var pl = this.pad.l, pr = this.w - this.pad.r;
    var pt = this.pad.t, pb = this.h - this.pad.b;
    var xl = this.xlim, yl = this.ylim;
    var sx = (pr - pl) / (xl[1] - xl[0]);
    var sy = (pb - pt) / (yl[1] - yl[0]);
    if (this.giuTiLe) {
      var s = Math.min(sx, sy);
      var cx = (pl + pr) / 2, cy = (pt + pb) / 2;
      var mx = (xl[0] + xl[1]) / 2, my = (yl[0] + yl[1]) / 2;
      return [cx + (x - mx) * s, cy - (y - my) * s];
    }
    return [pl + (x - xl[0]) * sx, pb - (y - yl[0]) * sy];
  };

  /** Toạ độ pixel -> toạ độ dữ liệu (cho kéo thả). */
  Plot.prototype.data = function (px, py) {
    var pl = this.pad.l, pr = this.w - this.pad.r;
    var pt = this.pad.t, pb = this.h - this.pad.b;
    var xl = this.xlim, yl = this.ylim;
    var sx = (pr - pl) / (xl[1] - xl[0]);
    var sy = (pb - pt) / (yl[1] - yl[0]);
    if (this.giuTiLe) {
      var s = Math.min(sx, sy);
      var cx = (pl + pr) / 2, cy = (pt + pb) / 2;
      var mx = (xl[0] + xl[1]) / 2, my = (yl[0] + yl[1]) / 2;
      return [mx + (px - cx) / s, my - (py - cy) / s];
    }
    return [xl[0] + (px - pl) / sx, yl[0] + (pb - py) / sy];
  };

  Plot.prototype.xoa = function () {
    this.doPhanGiai();
    this.ctx.clearRect(0, 0, this.w, this.h);
  };

  Plot.prototype.truc = function (o) {
    o = o || {};
    var g = this.ctx, C = bang();
    var x0 = this.px(this.xlim[0], this.ylim[0]);
    var x1 = this.px(this.xlim[1], this.ylim[0]);
    var y1 = this.px(this.xlim[0], this.ylim[1]);
    g.save();
    g.strokeStyle = C.duong; g.lineWidth = 1; g.fillStyle = C.mo;
    g.font = '10px system-ui, sans-serif';
    var i, t, p;
    var tx = o.xticks || moc(this.xlim[0], this.xlim[1], 5);
    var ty = o.yticks || moc(this.ylim[0], this.ylim[1], 4);
    g.globalAlpha = .45;
    for (i = 0; i < tx.length; i++) {
      p = this.px(tx[i], this.ylim[0]);
      var pTop = this.px(tx[i], this.ylim[1]);
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(pTop[0], pTop[1]); g.stroke();
    }
    for (i = 0; i < ty.length; i++) {
      p = this.px(this.xlim[0], ty[i]);
      var pR = this.px(this.xlim[1], ty[i]);
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(pR[0], pR[1]); g.stroke();
    }
    g.globalAlpha = 1;
    g.beginPath(); g.moveTo(x0[0], x0[1]); g.lineTo(x1[0], x1[1]); g.stroke();
    g.beginPath(); g.moveTo(x0[0], x0[1]); g.lineTo(y1[0], y1[1]); g.stroke();
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (i = 0; i < tx.length; i++) {
      p = this.px(tx[i], this.ylim[0]);
      t = o.fx ? o.fx(tx[i]) : dinhDang(tx[i]);
      g.fillText(t, p[0], p[1] + 5);
    }
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (i = 0; i < ty.length; i++) {
      p = this.px(this.xlim[0], ty[i]);
      t = o.fy ? o.fy(ty[i]) : dinhDang(ty[i]);
      g.fillText(t, p[0] - 6, p[1]);
    }
    if (o.xlabel) {
      g.textAlign = 'center'; g.textBaseline = 'bottom';
      g.fillText(o.xlabel, (x0[0] + x1[0]) / 2, this.h - 2);
    }
    if (o.ylabel) {
      g.save();
      g.translate(10, (x0[1] + y1[1]) / 2);
      g.rotate(-Math.PI / 2);
      g.textAlign = 'center'; g.textBaseline = 'top';
      g.fillText(o.ylabel, 0, 0);
      g.restore();
    }
    g.restore();
  };

  Plot.prototype.duong = function (pts, o) {
    o = o || {};
    if (pts.length < 2) return;
    var g = this.ctx;
    g.save();
    g.strokeStyle = o.mau || bang().nhan;
    g.lineWidth = o.day || 1.8;
    if (o.net) g.setLineDash(o.net);
    g.globalAlpha = o.mo == null ? 1 : o.mo;
    g.beginPath();
    for (var i = 0; i < pts.length; i++) {
      var p = this.px(pts[i][0], pts[i][1]);
      if (i === 0) g.moveTo(p[0], p[1]); else g.lineTo(p[0], p[1]);
    }
    g.stroke();
    g.restore();
  };

  Plot.prototype.diem = function (pts, o) {
    o = o || {};
    var g = this.ctx;
    g.save();
    g.globalAlpha = o.mo == null ? 1 : o.mo;
    for (var i = 0; i < pts.length; i++) {
      var p = this.px(pts[i][0], pts[i][1]);
      var r = (typeof o.r === 'function' ? o.r(i) : o.r) || 3.2;
      g.fillStyle = typeof o.mau === 'function' ? o.mau(i) : (o.mau || bang().nhan);
      g.beginPath(); g.arc(p[0], p[1], r, 0, 6.2832); g.fill();
      if (o.vien) {
        g.strokeStyle = o.vien; g.lineWidth = o.vienDay || 1.2; g.stroke();
      }
    }
    g.restore();
  };

  Plot.prototype.cot = function (vals, o) {
    o = o || {};
    var g = this.ctx, n = vals.length;
    var pl = this.pad.l, pr = this.w - this.pad.r;
    var bw = (pr - pl) / n;
    g.save();
    for (var i = 0; i < n; i++) {
      var p0 = this.px(this.xlim[0], Math.max(vals[i], 0));
      var p1 = this.px(this.xlim[0], Math.min(vals[i], 0));
      var x = pl + i * bw + bw * .16;
      g.fillStyle = typeof o.mau === 'function' ? o.mau(i) : (o.mau || bang().nhan);
      g.globalAlpha = o.mo == null ? .9 : o.mo;
      g.fillRect(x, p0[1], bw * .68, Math.max(1, p1[1] - p0[1]));
    }
    g.restore();
  };

  /** Tô nền theo một hàm quyết định: dùng cho SVM, K-means, phân loại. */
  Plot.prototype.vung = function (fn, o) {
    o = o || {};
    var g = this.ctx;
    var buoc = o.buoc || 5;
    var pl = this.pad.l, pr = this.w - this.pad.r;
    var pt = this.pad.t, pb = this.h - this.pad.b;
    g.save();
    g.globalAlpha = o.mo == null ? .13 : o.mo;
    for (var px = pl; px < pr; px += buoc) {
      for (var py = pt; py < pb; py += buoc) {
        var d = this.data(px + buoc / 2, py + buoc / 2);
        var c = fn(d[0], d[1]);
        if (!c) continue;
        g.fillStyle = c;
        g.fillRect(px, py, buoc, buoc);
      }
    }
    g.restore();
  };

  Plot.prototype.chu = function (x, y, s, o) {
    o = o || {};
    var g = this.ctx, p = this.px(x, y);
    g.save();
    g.fillStyle = o.mau || bang().mo;
    g.font = (o.co || 10) + 'px system-ui, sans-serif';
    g.textAlign = o.canh || 'left';
    g.textBaseline = o.doc || 'middle';
    g.fillText(s, p[0] + (o.dx || 0), p[1] + (o.dy || 0));
    g.restore();
  };

  /** Gắn kéo thả: gọi lại với chỉ số điểm và toạ độ dữ liệu mới. */
  Plot.prototype.keoTha = function (layDiem, khiKeo) {
    var self = this;
    var dangKeo = -1;

    function viTri(e) {
      var r = self.c.getBoundingClientRect();
      var t = e.touches ? e.touches[0] : e;
      return [t.clientX - r.left, t.clientY - r.top];
    }
    function gan(v) {
      var pts = layDiem(), tot = -1, min = 14;
      for (var i = 0; i < pts.length; i++) {
        var p = self.px(pts[i][0], pts[i][1]);
        var d = Math.hypot(p[0] - v[0], p[1] - v[1]);
        if (d < min) { min = d; tot = i; }
      }
      return tot;
    }
    function bat(e) {
      var v = viTri(e);
      dangKeo = gan(v);
      if (dangKeo >= 0) { e.preventDefault(); self.c.style.cursor = 'grabbing'; }
    }
    function di(e) {
      var v = viTri(e);
      if (dangKeo < 0) {
        self.c.style.cursor = gan(v) >= 0 ? 'grab' : 'default';
        return;
      }
      e.preventDefault();
      var d = self.data(v[0], v[1]);
      khiKeo(dangKeo, d[0], d[1]);
    }
    function tha() { dangKeo = -1; self.c.style.cursor = 'default'; }

    self.c.addEventListener('mousedown', bat);
    self.c.addEventListener('mousemove', di);
    window.addEventListener('mouseup', tha);
    self.c.addEventListener('touchstart', bat, { passive: false });
    self.c.addEventListener('touchmove', di, { passive: false });
    window.addEventListener('touchend', tha);
  };

  function moc(a, b, n) {
    var buoc = (b - a) / n;
    var mu = Math.pow(10, Math.floor(Math.log10(Math.abs(buoc) || 1)));
    var d = buoc / mu;
    d = d < 1.5 ? 1 : d < 3 ? 2 : d < 7 ? 5 : 10;
    buoc = d * mu;
    var out = [], v = Math.ceil(a / buoc) * buoc;
    for (; v <= b + 1e-9; v += buoc) out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
  }

  function dinhDang(v) {
    if (v === 0) return '0';
    var a = Math.abs(v);
    if (a >= 1e4 || a < 1e-3) return v.toExponential(0);
    var s = (a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : v.toFixed(2));
    return s.replace('.', ',');
  }

  /* ------------------------------------------------- vài hàm số dùng chung */

  function giaiTuyenTinh(A, b) {
    var n = b.length, i, j, k, M = A.map(function (r, idx) { return r.concat([b[idx]]); });
    for (i = 0; i < n; i++) {
      var tot = i;
      for (j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[tot][i])) tot = j;
      var tmp = M[i]; M[i] = M[tot]; M[tot] = tmp;
      if (Math.abs(M[i][i]) < 1e-12) return null;
      for (j = i + 1; j < n; j++) {
        var f = M[j][i] / M[i][i];
        for (k = i; k <= n; k++) M[j][k] -= f * M[i][k];
      }
    }
    var x = new Array(n);
    for (i = n - 1; i >= 0; i--) {
      var s = M[i][n];
      for (j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  /** Nhiễu Gauss lặp lại được: cùng hạt giống cho cùng dãy số. */
  function rng(hat) {
    var s = hat >>> 0 || 1;
    function u() {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    }
    return {
      u: u,
      n: function () {
        var a = Math.max(u(), 1e-12), b = u();
        return Math.sqrt(-2 * Math.log(a)) * Math.cos(6.283185307 * b);
      },
    };
  }

  /**
   * Độ lớn tương đối của ∂L/∂a sau khi truyền ngược qua `sau` lớp.
   *
   * Hệ số mỗi lớp là gain/√2: khởi tạo He đặt Var(W) = gain²/n_in và ReLU vứt
   * một nửa, nên ở thang ĐỘ LỚN (không phải phương sai) hệ số là gain/√2 — với
   * gain = √2 thì nó bằng đúng 1.
   *
   * Ba số mũ dưới đây là cách TÓM TẮT hành vi, không phải mô phỏng lại phép
   * truyền ngược đầy đủ. Chúng được chỉnh cho khớp sáu số đo ở Mục 6.3 của giáo
   * trình Mô hình & Kiến trúc, và src/test.mjs kiểm lại sự khớp ấy — nên nếu ai
   * sửa chúng mà làm lệch khỏi số đã in thì bộ kiểm tra báo ngay.
   */
  function doLonGradient(gain, sau, chuanHoa, tat) {
    var v = 1, f = gain / Math.SQRT2;
    for (var i = 0; i < sau; i++) {
      var he;
      if (tat && chuanHoa) he = Math.pow(1 + f, 0.03);   // chuẩn hoá ghìm phần cộng dồn
      else if (tat) he = Math.pow(1 + f, 0.70);          // cộng dồn không ai ghìm
      else if (chuanHoa) he = Math.pow(f, 0.12);         // kéo mạnh về 1
      else he = f;
      v *= he;
    }
    return v;
  }

  function softmax(z) {
    var m = Math.max.apply(null, z);
    var e = z.map(function (v) { return Math.exp(v - m); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }

  return {
    Plot: Plot, bang: bang, moc: moc, dinhDang: dinhDang,
    giaiTuyenTinh: giaiTuyenTinh, rng: rng, softmax: softmax,
    doLonGradient: doLonGradient,
  };
})();
