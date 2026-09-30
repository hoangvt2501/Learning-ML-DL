/* Phòng thí nghiệm của giáo trình Học sâu: ba công cụ. */
window.QZ_INIT = window.QZ_INIT || {};
window.QZ_INIT['labModels'] = function () {
  'use strict';
  var V = window.QZ_VIZ;
  if (!V || !document.querySelector('[data-lab-models]')) return;

  var $ = function (id) { return document.getElementById(id); };
  var C = V.bang();
  function so(id, v) { var e = $(id); if (e) e.textContent = v; }
  function fmt(v, n) {
    return (v == null || !isFinite(v)) ? '–' : v.toFixed(n == null ? 4 : n).replace('.', ',');
  }
  function nhom(v) {
    return Math.round(v).toLocaleString('vi-VN');
  }
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
     1 · Vì sao phải chia cho căn d_k
     ================================================================ */
  (function () {
    var p1 = $('atW') ? new V.Plot($('atW'), { pad: { t: 12, r: 10, b: 26, l: 40 } }) : null;
    var p2 = $('atE') ? new V.Plot($('atE'), { pad: { t: 12, r: 10, b: 30, l: 42 } }) : null;
    if (!p1 || !p2) return;
    var T = 64;

    function diem(dk, chia, hat) {
      // q·k với các thành phần độc lập, trung bình 0, phương sai 1
      // có phương sai đúng bằng d_k — đó là toàn bộ lý do của phép chia.
      var r = V.rng(hat), s = [], i, j;
      for (j = 0; j < T; j++) {
        var acc = 0;
        for (i = 0; i < dk; i++) acc += r.n() * r.n();
        s.push(chia ? acc / Math.sqrt(dk) : acc);
      }
      return s;
    }
    function entropy(p) {
      var h = 0;
      for (var i = 0; i < p.length; i++) if (p[i] > 1e-12) h -= p[i] * Math.log(p[i]);
      return h;
    }

    var DKS = [4, 16, 64, 256, 1024];
    function chay() {
      var idx = +$('atDk').value;
      var dk = DKS[idx];
      var chia = $('atScale').checked;
      so('atDkOut', dk);

      var s = diem(dk, chia, 3);
      var w = V.softmax(s);
      var maxW = Math.max.apply(null, w);
      // Trung bình trên 24 lần rút: một lần rút với 64 khoá cho ước lượng quá nhiễu
      // để so với con số lý thuyết.
      var H = 0, va = 0, lan;
      for (lan = 0; lan < 24; lan++) {
        var ss = diem(dk, chia, 3 + lan * 37);
        H += entropy(V.softmax(ss));
        var m = ss.reduce(function (a, b) { return a + b; }, 0) / T;
        va += ss.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / (T - 1);
      }
      H /= 24; va /= 24;

      p1.xlim = [0, T]; p1.ylim = [0, Math.max(maxW * 1.12, 0.06)];
      p1.xoa();
      p1.truc({ xticks: [], xlabel: 'trọng số attention trên 64 khoá' });
      p1.cot(w, { mau: C.nhan });
      p1.chu(1, p1.ylim[1] * 0.93, 'mức đều = 1/64 = 0,0156', { co: 9.5, mau: C.mo });
      p1.duong([[0, 1 / T], [T, 1 / T]], { mau: C.canh, day: 1, net: [4, 3] });

      // đường entropy theo d_k, cả hai trường hợp
      var eC = [], eK = [];
      DKS.forEach(function (d, k) {
        var hc = 0, hk = 0;
        for (var q = 0; q < 12; q++) {
          hc += entropy(V.softmax(diem(d, true, 3 + q * 37)));
          hk += entropy(V.softmax(diem(d, false, 3 + q * 37)));
        }
        eC.push([k, hc / 12]);
        eK.push([k, hk / 12]);
      });
      p2.xlim = [-0.3, DKS.length - 0.7]; p2.ylim = [0, 4.5];
      p2.xoa();
      p2.truc({ xticks: [0, 1, 2, 3, 4], fx: function (v) { return DKS[v] || ''; },
        xlabel: 'd_k', ylabel: 'entropy (nat)' });
      p2.duong([[-0.3, Math.log(T)], [DKS.length - 0.7, Math.log(T)]],
        { mau: C.mo, day: 1, net: [4, 3] });
      p2.chu(-0.2, Math.log(T) + 0.18, 'ln 64 = 4,159 — mức đều', { co: 9.5, mau: C.mo });
      p2.duong(eC, { mau: C.nhan, day: 2 });
      p2.duong(eK, { mau: C.canh, day: 2 });
      p2.diem([[idx, chia ? eC[idx][1] : eK[idx][1]]],
        { mau: chia ? C.nhan : C.canh, r: 5, vien: '#fff' });
      p2.chu(DKS.length - 0.75, eC[DKS.length - 1][1] + 0.25, 'có chia',
        { co: 9.5, mau: C.nhan, canh: 'right' });
      p2.chu(DKS.length - 0.75, eK[DKS.length - 1][1] + 0.25, 'không chia',
        { co: 9.5, mau: C.canh, canh: 'right' });

      so('atVar', fmt(va, 2));
      so('atVarLt', chia ? '1' : nhom(dk));
      so('atH', fmt(H, 4));
      so('atMax', fmt(maxW, 4));
      so('atNote', chia
        ? 'Có chia: phương sai điểm số về 1 với MỌI d_k, nên entropy đứng yên quanh 3,68. Đó chính là điều phép chia được thiết kế để làm.'
        : (dk >= 256
          ? 'Không chia: một khoá duy nhất chiếm gần hết trọng số. Softmax bão hoà, và đạo hàm p(δ−p) tắt theo — gradient gần như bằng 0.'
          : 'Không chia: phương sai điểm số bằng đúng d_k. Tăng d_k lên 256 rồi 1024 để thấy softmax bão hoà.'));
    }
    nghe(['atDk', 'atScale'], chay);
  })();

  /* ================================================================
     2 · Máy tính tham số, FLOP và bộ nhớ
     ================================================================ */
  (function () {
    if (!$('pcL')) return;
    function chay() {
      var L = +$('pcL').value, d = +$('pcD').value, Vv = +$('pcV').value;
      var T = +$('pcT').value, B = +$('pcB').value, nkv = +$('pcKV').value;
      var h = +$('pcH').value;
      var dhead = d / Math.max(h, 1);

      var attn = L * 4 * d * d;
      var ffn = L * 8 * d * d;
      var phiEmb = attn + ffn;                 // = 12 L d²
      var emb = Vv * d;
      var tong = phiEmb + emb;

      var D = +$('pcTok').value * 1e9;
      var flop = 6 * phiEmb * D;
      var gioA100 = flop / (312e12 * 0.4) / 3600;

      var kv = 2 * L * nkv * dhead * T * B * 2;   // FP16
      var tiLe = T / (6 * d);

      so('pcPhiEmb', nhom(phiEmb));
      so('pcCheck', nhom(12 * L * d * d));
      so('pcTong', nhom(tong));
      so('pcEmbPct', fmt(emb / tong * 100, 1) + '%');
      so('pcFfnPct', fmt(ffn / phiEmb * 100, 1) + '%');
      so('pcFlop', flop.toExponential(3).replace('.', ',').replace('e+', ' × 10^'));
      so('pcGio', nhom(gioA100));
      so('pcKV', fmt(kv / Math.pow(2, 30), 2) + ' GiB');
      so('pcKVTok', fmt(2 * L * nkv * dhead * 2 / Math.pow(2, 20), 4) + ' MiB');
      so('pcTiLe', fmt(tiLe, 3));
      so('pcNguong', nhom(6 * d));
      so('pcNote', tiLe > 1
        ? 'T > 6d: chi phí bậc hai của attention ĐÃ chi phối. Đây là lúc attention thưa bắt đầu đáng cân nhắc.'
        : 'T < 6d: attention chỉ chiếm ' + fmt(tiLe / (1 + tiLe) * 100, 0) +
          '% chi phí. Phần còn lại của khối mới là chỗ tốn — đó là lý do phần lớn hệ thống chưa cần attention thưa.');
    }
    nghe(['pcL', 'pcD', 'pcV', 'pcT', 'pcB', 'pcKV', 'pcH', 'pcTok'], chay);
    var nut = $('pcPreset');
    if (nut) nut.addEventListener('change', function () {
      var v = nut.value.split(',');
      if (v.length < 7) return;
      ['pcL', 'pcD', 'pcV', 'pcH', 'pcKV', 'pcT', 'pcTok'].forEach(function (id, i) {
        $(id).value = v[i];
      });
      chay();
    });
  })();

  /* ================================================================
     3 · Gradient qua 40 lớp: ba cách chữa, và chúng cộng dồn
     ================================================================ */
  (function () {
    var p = $('ggPlot') ? new V.Plot($('ggPlot'), { pad: { t: 12, r: 12, b: 30, l: 46 } }) : null;
    if (!p) return;

    // Dùng chung với src/test.mjs: hàm nằm trong viz.js nên bộ kiểm tra soi được
    // đúng mã đang chạy chứ không phải một bản chép.
    function hoSo(gain, sau, chuanHoa, tat) {
      var out = [1];
      for (var i = 1; i <= sau; i++) out.push(V.doLonGradient(gain, i, chuanHoa, tat));
      return out;
    }
    function chay() {
      var gain = +$('ggGain').value / 100;   // 141 -> 1,41 ≈ √2
      var sau = +$('ggDepth').value;
      var norm = $('ggNorm').checked, res = $('ggRes').checked;
      so('ggGainOut', fmt(gain, 2));
      so('ggDepthOut', sau);

      var cur = hoSo(gain, sau, norm, res);
      var he = Math.sqrt(2);
      var duongs = [
        { v: hoSo(0.5, sau, false, false), mau: C.lam, ten: 'gain 0,5' },
        { v: hoSo(he, sau, false, false), mau: C.nhan, ten: 'He (√2)' },
        { v: hoSo(2.0, sau, false, false), mau: C.canh, ten: 'gain 2,0' },
        { v: hoSo(he, sau, false, true), mau: C.vang, ten: 'He + tắt' },
        { v: hoSo(he, sau, true, true), mau: C.tim, ten: 'He + chuẩn hoá + tắt' },
      ];
      var lo = 1e-20, hi = 1e20;
      p.xlim = [0, sau]; p.ylim = [Math.log10(lo), Math.log10(hi)];
      p.xoa();
      p.truc({ xlabel: 'chỉ số lớp (0 = gần đầu vào)',
        fy: function (v) { return v === 0 ? '1' : '10^' + v; } });
      duongs.forEach(function (o) {
        p.duong(o.v.map(function (v, i) {
          return [sau - i, Math.log10(Math.max(Math.min(v, hi), lo))];
        }), { mau: o.mau, day: 1.3, mo: .45 });
      });
      p.duong(cur.map(function (v, i) {
        return [sau - i, Math.log10(Math.max(Math.min(v, hi), lo))];
      }), { mau: C.muc, day: 2.4 });
      p.duong([[0, 0], [sau, 0]], { mau: C.duong, day: 1 });

      var tiLe = cur[cur.length - 1];
      so('ggTiLe', tiLe.toExponential(2).replace('.', ','));
      so('ggKl', tiLe < 1e-6 ? 'TIÊU BIẾN' : tiLe > 1e6 ? 'BÙNG NỔ' : 'ổn định');
      so('ggNote', (!norm && res)
        ? 'Kết nối tắt MỘT MÌNH làm BÙNG NỔ: mỗi lớp cộng thêm vào tín hiệu nên nó tích luỹ. Đây là chỗ hay bị nói sai nhất — bật thêm chuẩn hoá để thấy nó ổn định lại.'
        : (norm && res)
          ? 'Chuẩn hoá cộng kết nối tắt cho tỉ lệ gần 1 — đúng cấu hình pre-LN của Transformer hiện đại.'
          : Math.abs(gain - he) < 0.06
            ? 'Khởi tạo He: hệ số 2 bù cho việc ReLU vứt một nửa phương sai. Thử kéo gain sang 0,5 rồi 2,0.'
            : 'Chỉ đổi hệ số khởi tạo thôi mà tỉ lệ trải qua hàng chục bậc độ lớn — đó là Mục 6.3.');
    }
    ['ggGain', 'ggDepth'].forEach(function (i) {
      var e = $(i); if (e) e.addEventListener('input', chay);
    });
    ['ggNorm', 'ggRes'].forEach(function (i) {
      var e = $(i); if (e) e.addEventListener('change', chay);
    });
    chay();
  })();
};
window.QZ_INIT['labModels']();
