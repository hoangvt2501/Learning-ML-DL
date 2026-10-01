// Kiểm chứng phần lõi số học của Phòng thí nghiệm bằng chính các con số
// đã in trong giáo trình. Chạy: npm test

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Một DOM giả tối thiểu: mọi truy vấn đều rỗng nên các hàm update* thoát sớm.
const stubDoc = {
  getElementById: () => null,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: () => {},
};
const sandbox = {
  window: {}, document: stubDoc, console,
  Math, Number, Object, Array, String, JSON, Float64Array,
  isFinite, parseFloat, parseInt, Infinity, NaN,
};
sandbox.window.document = stubDoc;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'src/assets/playground.js'), 'utf8'), sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'src/assets/lab-mlops.js'), 'utf8'), sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'src/assets/viz.js'), 'utf8'), sandbox);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'src/assets/lab-ungdung.js'), 'utf8'), sandbox);

const L = sandbox.window.QZ_LAB;
const M = sandbox.window.QZ_MLOPS;
const Z = sandbox.window.QZ_VIZ;
const U = sandbox.window.QZ_UNGDUNG;

let pass = 0;
let fail = 0;
function check(name, actual, expected, tol) {
  const ok =
    tol === undefined
      ? Object.is(actual, expected) || actual === expected
      : Math.abs(actual - expected) <= tol;
  if (ok) {
    pass++;
  } else {
    fail++;
    console.error('  ✗ ' + name + '\n      nhận:  ' + actual + '\n      chờ:   ' + expected);
  }
}
function group(name) { console.log('\n' + name); }

/* ---------------------------------------------- làm tròn nửa về số chẵn */
group('Làm tròn nửa về số chẵn (giống NumPy)');
check('rne(42.5) = 42', L.rne(42.5), 42);
check('rne(43.5) = 44', L.rne(43.5), 44);
check('rne(-42.5) = -42', L.rne(-42.5), -42);
check('rne(1.75) = 2', L.rne(1.75), 2);
check('rne(2.4) = 2', L.rne(2.4), 2);
check('rne(-2.6) = -3', L.rne(-2.6), -3);

/* ------------------------------------------------- Hình 4 của giáo trình */
group('Hình 4 — lượng tử 3 bit không dấu trên [-1, 3]');
{
  const p = L.affineParams(-1, 3, 3, 'unsigned');
  check('S = 4/7', p.S, 4 / 7, 1e-12);
  check('Z = 2', p.Z, 2);
  check('cận dưới biểu diễn được = -1,143', L.dequantize(p.qmin, p), -1.142857, 1e-5);
  check('cận trên biểu diễn được = 2,857', L.dequantize(p.qmax, p), 2.857143, 1e-5);
}

/* ------------------------------------------------- Mục 3.3 và Mục 3.6 */
group('Mục 3.3 — lượng tử [-1, 3] sang uint8');
{
  const p = L.affineParams(-1, 3, 8, 'unsigned');
  check('S = 4/255', p.S, 4 / 255, 1e-12);
  check('Z = 64', p.Z, 64);
  check('q(1,5) = 160', L.quantize(1.5, p), 160);
  check('x̂(1,5) = 1,5059', L.dequantize(L.quantize(1.5, p), p), 1.505882, 1e-5);
  check('q(0) = Z', L.quantize(0, p), p.Z);
  check('x̂(0) = 0 chính xác', L.dequantize(L.quantize(0, p), p), 0);
  check('q(5) bị cắt về 255', L.quantize(5, p), 255);
  check('q(-1) = 0', L.quantize(-1, p), 0);
}

/* ---------------------------------------------------------- Bài 1 */
group('Bài 1 — lượng tử [-0,5; 2,5] sang uint8');
{
  const p = L.affineParams(-0.5, 2.5, 8, 'unsigned');
  check('S = 3/255', p.S, 3 / 255, 1e-12);
  check('Z = 42 (nửa về số chẵn, không phải 43)', p.Z, 42);
  check('q(-0,5) = 0', L.quantize(-0.5, p), 0);
  check('x̂(-0,5) = -0,494118', L.dequantize(L.quantize(-0.5, p), p), -0.494118, 1e-5);
  check('q(0) = 42', L.quantize(0, p), 42);
  check('q(1) = 127', L.quantize(1, p), 127);
  check('x̂(1) = 1 chính xác', L.dequantize(L.quantize(1, p), p), 1, 1e-12);
  check('q(3) bị cắt về 255', L.quantize(3, p), 255);
  check('x̂(3) = 2,505882', L.dequantize(L.quantize(3, p), p), 2.505882, 1e-5);
}

/* ------------------------------------------------ Mục 2.2 — soi bit FP32 */
group('Mục 2.2 — ba trường của số FP32');
{
  const f = L.FORMATS.fp32;
  const a = L.encodeFloat(1993, f);
  check('1993: dấu = 0', a.sign, 0);
  check('1993: số mũ thật = 10', a.e, 10);
  check('1993: exponent lưu = 137', a.expField, 137);
  check('1993: lưu lại đúng 1993', a.value, 1993);

  const b = L.encodeFloat(0.1, f);
  check('0,1: số mũ thật = -4', b.e, -4);
  check('0,1: FP32 chỉ lưu được xấp xỉ', b.value !== 0.1, true);
  check('0,1: khớp Math.fround', b.value, Math.fround(0.1));

  const c = L.encodeFloat(-6.5, f);
  check('-6,5: dấu = 1', c.sign, 1);
  check('-6,5: số mũ thật = 2', c.e, 2);
  check('-6,5: lưu lại đúng -6,5', c.value, -6.5);
}

/* ------------------------------------------------ Mục 2.3 — FP16 và FP8 */
group('Mục 2.3 — FP16, BF16, FP8');
{
  const h = L.FORMATS.fp16;
  check('np.float16(1993) = 1993', L.encodeFloat(1993, h).value, 1993);
  check('np.float16(1993,5) = 1994', L.encodeFloat(1993.5, h).value, 1994);
  check('np.float16(2049) = 2048', L.encodeFloat(2049, h).value, 2048);
  check('np.float16(70000) = vô cực', L.encodeFloat(70000, h).value, Infinity);
  check('FP16 max = 65504', L.encodeFloat(65504, h).maxFinite, 65504);
  check('FP16 ε = 2^-10', L.encodeFloat(1, h).eps, Math.pow(2, -10), 1e-15);

  check('BF16 max ≈ 3,39e38', L.encodeFloat(1, L.FORMATS.bf16).maxFinite, 3.3895e38, 1e35);
  check('BF16 ε = 2^-7', L.encodeFloat(1, L.FORMATS.bf16).eps, Math.pow(2, -7), 1e-15);

  const e4 = L.FORMATS.e4m3;
  check('FP8 E4M3 max = 448', L.encodeFloat(448, e4).maxFinite, 448);
  check('FP8 E4M3 lưu được 448', L.encodeFloat(448, e4).value, 448);
  check('FP8 E4M3 vượt 448 -> NaN', Number.isNaN(L.encodeFloat(500, e4).value), true);
  check('FP8 E4M3 số chuẩn nhỏ nhất = 2^-6', Math.pow(2, 1 - 7), 0.015625, 1e-12);

  const e5 = L.FORMATS.e5m2;
  check('FP8 E5M2 max = 57344', L.encodeFloat(57344, e5).maxFinite, 57344);
  check('FP8 E5M2 ε = 2^-2', L.encodeFloat(1, e5).eps, 0.25, 1e-15);
}

group('Bài 2 — số nguyên nhỏ nhất không biểu diễn chính xác được');
for (const [key, expected] of [['fp32', 16777217], ['fp16', 2049], ['bf16', 257], ['e4m3', 17]]) {
  const f = L.FORMATS[key];
  let n = 1;
  while (n < expected + 4096 && L.encodeFloat(n, f).value === n) n++;
  check(key + ' -> ' + expected, n, expected);
}

/* ------------------------------------------- Mục 6.4/6.5 — requantization */
group('Mục 6.5 — requantization bằng dấu chấm tĩnh');
{
  // Giáo trình in ra M = 0,0020223740 và M₀ = 1111811840 với n = 8.
  // M in ra chỉ có 10 chữ số nên phải dựng lại M đúng từ chính M₀.
  const Mexact = 1111811840 * Math.pow(2, -39);
  const qm = L.quantizeMultiplier(Mexact);
  check('M in trong sách = 0,0020223740', Number(Mexact.toFixed(10)), 0.002022374);
  check('dịch thêm n = 8', qm.shift, 8);
  check('M₀ = 1111811840', qm.M0, 1111811840);
  check('khôi phục lại đúng M', qm.M0 * Math.pow(2, -(31 + qm.shift)), Mexact, 1e-18);
  check('M₀ luôn nằm trong [2^30, 2^31)',
    qm.M0 >= Math.pow(2, 30) && qm.M0 < Math.pow(2, 31), true);
  check('dịch phải có làm tròn: 100 >> 2 = 25', L.roundingRightShift(100, 2), 25);
  check('dịch phải có làm tròn: 6 >> 2 = 2', L.roundingRightShift(6, 2), 2);
}

/* ------------------- Ghi chú Mục 6.4 — phạm vi M và quy ước làm tròn */
group('Ghi chú Mục 6.4 — M ≥ 1 và cách làm tròn số âm');
{
  // Đoạn mã in trong sách chỉ chuẩn hoá lên nên chỉ đúng với 0 < M <= 1.
  // Bản dùng trong công cụ có thêm nhánh xuống, nên mọi M > 0 đều hợp lệ.
  for (const M of [0.002, 0.75, 1, 2.5, 7.9]) {
    const qm = L.quantizeMultiplier(M);
    check('M = ' + M + ': M₀ nằm trong [2³⁰, 2³¹)',
      qm.M0 >= Math.pow(2, 30) && qm.M0 < Math.pow(2, 31), true);
    check('M = ' + M + ': khôi phục lại đúng M',
      qm.M0 * Math.pow(2, -(31 + qm.shift)), M, Math.abs(M) * 1e-9);
  }
  check('M = 2,5 cần dịch TRÁI (shift âm)', L.quantizeMultiplier(2.5).shift < 0, true);

  // Hai quy ước chỉ khác nhau ở đúng điểm giữa của giá trị âm.
  check('nửa về +∞:   -6 >> 2 = -1', L.shiftHalfUp(-6, 2), -1);
  check('xa số 0:     -6 >> 2 = -2', L.shiftHalfAwayFromZero(-6, 2), -2);
  check('nửa về +∞:   -2 >> 2 = 0', L.shiftHalfUp(-2, 2), 0);
  check('xa số 0:     -2 >> 2 = -1', L.shiftHalfAwayFromZero(-2, 2), -1);
  check('số dương thì hai quy ước trùng nhau (+6)',
    L.shiftHalfUp(6, 2) === L.shiftHalfAwayFromZero(6, 2), true);
  check('số dương thì hai quy ước trùng nhau (+10)',
    L.shiftHalfUp(10, 2) === L.shiftHalfAwayFromZero(10, 2), true);
  check('không rơi điểm giữa thì cũng trùng nhau (-7)',
    L.shiftHalfUp(-7, 2) === L.shiftHalfAwayFromZero(-7, 2), true);
}

/* --------------------------------------------- Mục 11.8 — số bit thực tế */
group('Mục 11.8 — số bit thực tế mỗi trọng số');
check('INT4 nhóm 128 = 4,125', L.WEIGHT_FORMATS.int4.bits, 4.125, 1e-12);
check('GGUF Q4_0 = 4,5', L.WEIGHT_FORMATS.q4_0.bits, 4.5, 1e-12);
check('GGUF Q8_0 = 8,5', L.WEIGHT_FORMATS.q8_0.bits, 8.5, 1e-12);
check('GGUF Q4_K = 4,5', L.WEIGHT_FORMATS.q4_k.bits, 4.5, 1e-12);
check('NF4 + double quant ≈ 4,127', L.WEIGHT_FORMATS.nf4.bits, 4.127, 5e-4);

/* --------------------------------------------------- Bài 9 — dung lượng */
group('Bài 9 — mô hình 13 tỉ tham số');
{
  const weightBytes = 13e9 * L.WEIGHT_FORMATS.q4_k.bits / 8;
  check('trọng số Q4_K = 7,3125 GB', weightBytes / 1e9, 7.3125, 1e-9);
  const perToken = 2 * 40 * 40 * 128 * 2;
  check('KV mỗi token = 819200 byte', perToken, 819200);
  check('KV mỗi token = 0,78125 MiB', perToken / Math.pow(2, 20), 0.78125, 1e-12);
  const total = perToken * 8192 * 4;
  check('KV tổng = 26,84 GB', total / 1e9, 26.8435456, 1e-6);
  check('KV tổng = 25 GiB', total / Math.pow(2, 30), 25, 1e-9);
  check('KV lớn hơn trọng số 3,67 lần', total / weightBytes, 3.6709, 1e-3);
}

/* ------------------------------ Mục 4.3 — đánh đổi giữa làm tròn và cắt */
group('Mục 4.3 — ngưỡng cắt tối ưu trên dữ liệu Laplace, INT4');
{
  const st = L.clipCompute('laplace', 4);
  check('ngưỡng tối ưu nằm quanh 4,8 (Hình 7: 4,83)', st.best[0], 4.83, 1.2);
  check('min–max tệ hơn ít nhất 3 lần', st.minmax / st.best[1] > 3, true);
  check('max|x| của mẫu nằm trong khoảng hợp lý', st.maxAbs > 8 && st.maxAbs < 20, true);
  // Quy tắc 6 dB: thêm một bit phải giảm MSE khoảng 4 lần ở cùng tỉ lệ ngưỡng.
  const a = L.clipCompute('laplace', 4).best[1];
  const b = L.clipCompute('laplace', 6).best[1];
  check('mỗi bit giảm MSE ~4 lần (2 bit -> ~16 lần)', a / b > 8 && a / b < 32, true);
}

/* =====================================================================
   Giáo trình 2 — MLOps: đối chiếu với kết quả của code/mlops/experiments.py
   ===================================================================== */

group('MLOps — hàm thống kê nền');
check('normInv(0,975) = 1,959964', M.normInv(0.975), 1.959964, 1e-6);
check('normInv(0,80) = 0,841621', M.normInv(0.80), 0.841621, 1e-6);
check('normInv(0,99) = 2,326348', M.normInv(0.99), 2.326348, 1e-6);
check('median của χ²(9) ≈ 8,343', M.chi2inv(0.5, 9), 8.3428, 1e-3);
check('chi2sf(25; 9) = 0,002964', M.chi2sf(25, 9), 0.002964, 1e-5);
check('chi2inv là nghịch đảo của chi2cdf', M.chi2cdf(M.chi2inv(0.9, 7), 7), 0.9, 1e-8);

group('MLOps Mục 8.4 — cỡ mẫu A/B');
// Các con số dưới đây in ra từ code/mlops/experiments.py phần (C).
check('nền 5%, lift 1% -> 2.996.694', Math.round(M.nPerArm(0.05, 0.01, 0.05, 0.8, 0)), 2996694, 2);
check('nền 5%, lift 2% -> 752.700', Math.round(M.nPerArm(0.05, 0.02, 0.05, 0.8, 0)), 752700, 2);
check('nền 5%, lift 5% -> 122.121', Math.round(M.nPerArm(0.05, 0.05, 0.05, 0.8, 0)), 122121, 2);
check('nền 5%, lift 20% -> 8.155', Math.round(M.nPerArm(0.05, 0.20, 0.05, 0.8, 0)), 8155, 2);
check('Bài 2: nền 2%, lift 5% -> 315.203', Math.round(M.nPerArm(0.02, 0.05, 0.05, 0.8, 0)), 315203, 2);
check('CUPED giảm 40% phương sai thì cỡ mẫu giảm đúng 40%',
  M.nPerArm(0.02, 0.05, 0.05, 0.8, 0.4) / M.nPerArm(0.02, 0.05, 0.05, 0.8, 0), 0.6, 1e-12);
check('gấp đôi mức lift thì cỡ mẫu giảm khoảng 4 lần',
  M.nPerArm(0.05, 0.05, 0.05, 0.8, 0) / M.nPerArm(0.05, 0.10, 0.05, 0.8, 0), 3.911, 0.02);

group('MLOps Mục 9.5 — PSI dưới giả thuyết không');
check('E[PSI] với n=500, k=10 là 0,036', M.psiMeanNull(500, 10), 0.036, 1e-12);
check('E[PSI] với n=200, k=10 là 0,090', M.psiMeanNull(200, 10), 0.09, 1e-12);
check('E[PSI] với n=200, k=20 là 0,190', M.psiMeanNull(200, 20), 0.19, 1e-12);
check('phân vị 99% của PSI (n=500, k=10) = 0,0867', M.psiQuantile(0.99, 500, 10), 0.0867, 5e-4);
check('P(PSI > 0,10) khi n=500, k=10 = 0,00297', M.psiFalseAlarmRate(0.10, 500, 10), 0.00297, 1e-5);
check('Bài 1: 200 đặc trưng -> 0,59 báo động giả mỗi ngày',
  200 * M.psiFalseAlarmRate(0.10, 500, 10), 0.593, 5e-3);
check('n để E[PSI] < 0,01 với k=10 là 1800', 2 * 9 / 0.01, 1800, 1e-9);
check('ngưỡng 0,25 ở n=20.000 gần như không bao giờ chạm',
  M.psiFalseAlarmRate(0.25, 20000, 10) < 1e-12, true);

group('MLOps Mục 7.3 — đuôi độ trễ khi toả nhánh');
// Một dịch vụ có trung vị 20 ms và p99 = 100 ms.
check('k=1 thì trả về đúng p99 = 100 ms', M.tailQuantile(0.99, 1, 20, 100), 100, 1e-9);
check('k=10 -> 169,5 ms', M.tailQuantile(0.99, 10, 20, 100), 169.5, 0.1);
check('k=50 -> 231,4 ms', M.tailQuantile(0.99, 50, 20, 100), 231.4, 0.1);
check('k=100 -> 261,9 ms', M.tailQuantile(0.99, 100, 20, 100), 261.9, 0.1);
check('Bài 3: k=25 thì 22,22% yêu cầu chạm nhánh chậm', 1 - Math.pow(0.99, 25), 0.2222, 1e-4);
check('Bài 3: mỗi nhánh phải đạt phân vị 99,9598%', Math.pow(0.99, 1 / 25) * 100, 99.9598, 1e-4);
check('k=100 thì 63,4% yêu cầu chạm nhánh chậm', 1 - Math.pow(0.99, 100), 0.634, 1e-3);

group('MLOps Mục 6.5 — quy tắc lấy nhỏ nhất của ML Test Score');
check('Bài 6: min(2; 0; 4; 0) = 0', M.mtsScore([2, 0, 4, 0]), 0);
check('hạ tầng hoàn hảo mà không giám sát vẫn bằng 0', M.mtsScore([7, 7, 7, 0]), 0);
check('bốn nhóm cân nhau thì điểm bằng chính mức đó', M.mtsScore([3, 3, 3, 3]), 3);

// ---------------------------------------------------------------------------
// Giáo trình 3 — các công thức đếm ở Chương 12.
// Đây là những khẳng định kiểm chứng được: chúng phải khớp với số đã công bố.
// ---------------------------------------------------------------------------

// Tham số của một Transformer kiểu GPT-2: post-LN, học embedding vị trí,
// FFN hai ma trận với d_ff = 4d, lớp ra dùng chung trọng số với embedding.
function gpt2Params(nLayer, d, vocab, nCtx) {
  const attn = nLayer * 4 * d * d + nLayer * 4 * d;        // W_Q,K,V,O và độ lệch
  const ffn = nLayer * 8 * d * d + nLayer * 5 * d;         // hai ma trận và độ lệch
  const norms = nLayer * 4 * d + 2 * d;                    // hai LayerNorm mỗi lớp, một cuối
  const emb = vocab * d + nCtx * d;
  return { nonEmb: nLayer * 12 * d * d, total: attn + ffn + norms + emb };
}

// Tham số của một Transformer kiểu Llama: RoPE (không có embedding vị trí),
// RMSNorm (chỉ γ), không độ lệch, FFN SwiGLU ba ma trận, lớp ra riêng.
function llamaParams(nLayer, d, dFF, vocab, nKV, nHeads) {
  const dHead = d / nHeads;
  const attn = nLayer * (2 * d * d + 2 * d * nKV * dHead);
  const ffn = nLayer * 3 * d * dFF;
  const norms = nLayer * 2 * d + d;
  return { nonEmb: attn + ffn, total: attn + ffn + norms + 2 * vocab * d };
}

group('Mô hình Mục 12.2–12.3 — đếm tham số khớp số đã công bố');
const g2s = gpt2Params(12, 768, 50257, 1024);
check('GPT-2 small = 124.439.808 tham số', g2s.total, 124439808);
check('GPT-2 small phi-embedding = 12·12·768² = 84.934.656', g2s.nonEmb, 84934656);
check('embedding từ vựng chiếm 31,0% GPT-2 small', 50257 * 768 / g2s.total, 0.310, 5e-4);
check('kể thêm embedding vị trí thì thành 31,6%',
  (50257 * 768 + 1024 * 768) / g2s.total, 0.3165, 5e-4);
check('GPT-2 medium = 354.823.168 tham số', gpt2Params(24, 1024, 50257, 1024).total, 354823168);
check('GPT-2 large = 774.030.080 tham số', gpt2Params(36, 1280, 50257, 1024).total, 774030080);
check('công thức 12Ld² cho GPT-2 large', gpt2Params(36, 1280, 50257, 1024).nonEmb,
  12 * 36 * 1280 * 1280);
check('FFN chiếm 2/3 khối khi d_ff = 4d', 8 / 12, 2 / 3, 1e-12);

group('Mô hình Mục 12.3 — kiến trúc kiểu Llama');
const l7 = llamaParams(32, 4096, 11008, 32000, 32, 32);
const l13 = llamaParams(40, 5120, 13824, 32000, 40, 40);
check('Llama-2 7B = 6.738.415.616 tham số', l7.total, 6738415616);
check('Llama-2 13B = 13.015.864.320 tham số', l13.total, 13015864320);
check('Llama-2 7B phi-embedding = 6.476.005.376', l7.nonEmb, 6476005376);
check('Bài 1: FFN SwiGLU của 7B = 4.328.521.728', 32 * 3 * 4096 * 11008, 4328521728);
check('d_ff của 7B là 8/3·d làm tròn lên bội 256',
  Math.ceil((8 / 3) * 4096 / 256) * 256, 11008);
check('SwiGLU giảm bề rộng còn 2/3 để khớp tham số', (2 / 3) * 4 * 4096, 10922.67, 0.01);

group('Mô hình Mục 12.4 — quy tắc 6ND và ngân sách giờ-GPU');
const C7 = 6 * l7.nonEmb * 2e12;
check('C = 6ND = 7,7712e22 FLOP', C7 / 1e22, 7.7712, 1e-4);
check('lượt xuôi 2N, lượt ngược 4N, tổng 6N', 2 + 4, 6);
const hours = C7 / (312e12 * 0.376) / 3600;
check('ở 37,6% MFU cho 184.011 giờ-GPU', hours, 184011, 1);
check('lệch dưới 0,2% so với 184.320 giờ đã công bố',
  Math.abs(hours - 184320) / 184320 < 0.002, true);

group('Mô hình Mục 12.5 — KV cache và ngưỡng T > 6d');
const kvBytes = (L, nKV, dHead, T, B) => 2 * L * nKV * dHead * T * B * 2;
const mha = kvBytes(80, 64, 128, 4096, 8);
const gqa = kvBytes(80, 8, 128, 4096, 8);
check('Bài 8: MHA cần 80,00 GiB', mha / 2 ** 30, 80, 1e-9);
check('Bài 8: GQA 8 nhóm cần 10,00 GiB', gqa / 2 ** 30, 10, 1e-9);
check('GQA giảm đúng bằng tỉ lệ nhóm: 8 lần', mha / gqa, 8, 1e-12);
check('mỗi token mỗi chuỗi với GQA là 0,3125 MiB', kvBytes(80, 8, 128, 1, 1) / 2 ** 20, 0.3125, 1e-12);
check('trạng thái Adam cho 7B ở FP32 là 56 GB', 7e9 * 2 * 4 / 1e9, 56, 1e-9);
check('ngưỡng attention chi phối với d=4096 là 24.576 token', 6 * 4096, 24576);
check('ở T=4096, d=4096 attention chỉ chiếm 17%', 4096 / (6 * 4096), 0.1667, 1e-4);
check('ở T=131072, d=4096 attention chi phối', 131072 / (6 * 4096) > 1, true);

group('Mô hình Chương 2–9 — các con số suy ra được');
check('Var(q·k) = d_k khi các thành phần có phương sai 1', 1024, 1024);
check('entropy tối đa trên 64 khoá là ln 64 = 4,1589', Math.log(64), 4.1589, 1e-4);
check('Bài 4a: 112 lớp 3×3 để phủ 224 (r = 2L+1)', Math.ceil((224 - 1) / 2), 112);
check('Bài 4b: 10 lớp có bước nhảy xen kẽ cho r = 125', (() => {
  let r = 1, P = 1;
  const s = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
  for (const si of s) { r += 2 * P; P *= si; }
  return r;
})(), 125);
check('Bài 7a: RNN ρ=0,9 sau 100 bước còn 2,66e-5', Math.pow(0.9, 100), 2.656e-5, 1e-8);
check('Bài 7a: RNN ρ=1,1 sau 100 bước thành 1,38e4', Math.pow(1.1, 100), 1.378e4, 10);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
check('σ(1) = 0,7311', sigmoid(1), 0.7311, 1e-4);
check('σ(4) = 0,9820', sigmoid(4), 0.9820, 1e-4);
check('Bài 7b: LSTM độ lệch cổng quên 1 -> sau 100 bước còn 2,48e-14',
  Math.pow(sigmoid(1), 100), 2.484e-14, 1e-17);
check('Bài 7b: LSTM độ lệch cổng quên 4 -> sau 100 bước còn 0,163',
  Math.pow(sigmoid(4), 100), 0.1628, 1e-4);
check('chênh gần 13 bậc độ lớn chỉ do một giá trị khởi tạo',
  Math.log10(Math.pow(sigmoid(4), 100) / Math.pow(sigmoid(1), 100)), 12.82, 0.01);
check('Bài 6: bề rộng cần cho sóng răng cưa 2^7 là 128', Math.pow(2, 7), 128);
check('Bài 6: mạng sâu k=7 dùng 6k=42 tham số so với 3·2^7+2=386',
  (3 * 128 + 2) / 42, 9.2, 0.05);
check('bagging: phương sai trung bình dừng ở ρσ² khi B→∞',
  0.3 * 1 + (1 - 0.3) / 1e9, 0.3, 1e-8);

// ---------------------------------------------------------------------------
// Giáo trình 1 — Nền tảng. Các con số tính tay ở Chương 17 và các hằng số
// mà phần chữ dựa vào. Mỗi phép kiểm ở đây tương ứng một khẳng định trong bài.
// ---------------------------------------------------------------------------

// Giải hệ 2x2 bằng công thức Cramer, đủ cho mọi bài tính tay của giáo trình.
function giai2x2(A, b) {
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  return [(b[0] * A[1][1] - A[0][1] * b[1]) / det,
          (A[0][0] * b[1] - b[0] * A[1][0]) / det];
}

group('Nền tảng Bài 1 — phương trình chuẩn tắc tính tay');
const Xb1 = [[1, 1], [1, 2], [1, 3], [1, 4]];
const yb1 = [2, 3, 5, 6];
const XtX = [[0, 0], [0, 0]];
const Xty = [0, 0];
for (let i = 0; i < 4; i++) {
  for (let r = 0; r < 2; r++) {
    Xty[r] += Xb1[i][r] * yb1[i];
    for (let c = 0; c < 2; c++) XtX[r][c] += Xb1[i][r] * Xb1[i][c];
  }
}
check('X^T X = [[4,10],[10,30]]', JSON.stringify(XtX), JSON.stringify([[4, 10], [10, 30]]));
check('X^T y = [16, 47]', JSON.stringify(Xty), JSON.stringify([16, 47]));
const wb1 = giai2x2(XtX, Xty);
check('w0 = 0,5', wb1[0], 0.5, 1e-12);
check('w1 = 1,4', wb1[1], 1.4, 1e-12);
const du = yb1.map((y, i) => y - (wb1[0] + wb1[1] * Xb1[i][1]));
check('tổng phần dư bằng 0 vì có cột hằng số',
  Math.abs(du.reduce((a, b) => a + b, 0)) < 1e-12, true);
check('RSS = 0,20', du.reduce((a, b) => a + b * b, 0), 0.20, 1e-12);
const wRidge = giai2x2([[XtX[0][0] + 1, XtX[0][1]], [XtX[1][0], XtX[1][1] + 1]], Xty);
check('ridge lam=1 cho w0 = 0,4727', wRidge[0], 0.4727, 5e-5);
check('ridge lam=1 cho w1 = 1,3636', wRidge[1], 1.3636, 5e-5);
check('ridge co cả hai hệ số về phía 0',
  Math.abs(wRidge[0]) < Math.abs(wb1[0]) && Math.abs(wRidge[1]) < Math.abs(wb1[1]), true);

group('Nền tảng Mục 5.5 — điểm tối ưu của epsilon khi kiểm tra đạo hàm');
const uMay = Number.EPSILON;
check('epsilon máy = 2,22e-16', uMay, 2.220446049250313e-16, 1e-30);
check('sai phân trung tâm: u^(1/3) = 6,06e-6', Math.cbrt(uMay), 6.06e-6, 1e-8);
check('sai phân tiến: u^(1/2) = 1,49e-8', Math.sqrt(uMay), 1.49e-8, 1e-10);
check('giá trị đo được 5,62e-6 nằm trong một bước lưới của lý thuyết',
  Math.abs(Math.log10(5.62e-6 / Math.cbrt(uMay))) < Math.log10(1.8), true);
check('giá trị đo được 1,78e-8 nằm trong một bước lưới của lý thuyết',
  Math.abs(Math.log10(1.78e-8 / Math.sqrt(uMay))) < Math.log10(1.8), true);
check('sai số tốt nhất của sai phân trung tâm cỡ u^(2/3)',
  Math.pow(uMay, 2 / 3), 3.7e-11, 5e-12);

group('Nền tảng Mục 5.3 — số vòng lặp theo số điều kiện');
// Số liệu đo được: GD tuyến tính theo kappa, heavy ball theo căn kappa.
const gdDo = { 10: 92, 100: 922, 1000: 9211, 10000: 92104 };
const hbDo = { 10: 34, 100: 117, 1000: 391, 10000: 1297 };
check('GD: 10 lần kappa thì 10 lần số vòng (100 so với 10)', gdDo[100] / gdDo[10], 10, 0.05);
check('GD: 10 lần kappa thì 10 lần số vòng (10000 so với 1000)',
  gdDo[10000] / gdDo[1000], 10, 0.05);
check('heavy ball: 10 lần kappa thì ~3,16 lần số vòng', hbDo[10000] / hbDo[1000], 3.16, 0.2);
check('tỉ lệ GD/HB ở kappa=10000 là 71', gdDo[10000] / hbDo[10000], 71, 1);
check('Bài 3: ước lượng GD ở kappa=40000 là ~368.000', gdDo[10000] * 4, 368416, 1);
check('Bài 3: ước lượng HB ở kappa=40000 là ~2.600', hbDo[10000] * 2, 2594, 1);

group('Nền tảng Bài 5 — chặn Novikoff');
const novikoff = (R, g) => (R / g) ** 2;
check('R=2, gamma=0,1 cho chặn 400', novikoff(2, 0.1), 400, 1e-9);
check('chặn không phụ thuộc số điểm hay số chiều', novikoff(2, 0.1), novikoff(2, 0.1));
check('nhân mọi x với 10 thì chặn không đổi', novikoff(20, 1), novikoff(2, 0.1), 1e-9);
check('lề co một nửa thì chặn gấp bốn', novikoff(2, 0.05) / novikoff(2, 0.1), 4, 1e-9);

group('Nền tảng Bài 6 — các chỉ số trên dữ liệu mất cân bằng');
const TP = 102, FP = 105, FN = 94, TN = 19699;
const nTot = TP + FP + FN + TN;
const prec = TP / (TP + FP), rec = TP / (TP + FN);
check('tổng số giao dịch = 20.000', nTot, 20000);
check('độ chính xác = 0,9900', (TP + TN) / nTot, 0.9900, 5e-5);
check('precision = 0,4928', prec, 0.4928, 5e-5);
check('recall = 0,5204', rec, 0.5204, 5e-5);
check('F1 = 0,5062', 2 * prec * rec / (prec + rec), 0.5062, 5e-5);
check('đoán TẤT CẢ là âm cho độ chính xác 0,9902', (TN + FP) / nTot, 0.9902, 5e-5);
check('bộ phân loại vô dụng có độ chính xác CAO HƠN bộ dùng được',
  (TN + FP) / nTot > (TP + TN) / nTot, true);
check('PR-AUC của bộ đoán ngẫu nhiên bằng tỉ lệ lớp dương',
  (TP + FN) / nTot, 0.0098, 5e-5);

group('Nền tảng Bài 7 — SVM lề cứng trên hai điểm');
// x1 = (1,1) nhãn +1; x2 = (-1,-1) nhãn -1. Theo đối xứng thì b = 0, w = (a,a).
const aSVM = 0.5;
const wSVM = [aSVM, aSVM];
const normW = Math.hypot(wSVM[0], wSVM[1]);
check('w = (0,5; 0,5)', wSVM[0], 0.5, 1e-12);
check('ràng buộc y_i(w.x_i) = 1 tại cả hai điểm', wSVM[0] * 1 + wSVM[1] * 1, 1, 1e-12);
check('lề 2/||w|| = 2,8284', 2 / normW, 2.8284271247461903, 1e-12);
check('lề bằng đúng khoảng cách giữa hai điểm', 2 / normW, Math.hypot(2, 2), 1e-12);
const alpha = 0.25;
// Dung lai w tu nghiem doi ngau: w = a1*(+1)*(1,1) + a2*(-1)*(-1,-1) = (a1+a2)*(1,1)
const wTuAlpha = [alpha * 1 * 1 + alpha * (-1) * (-1), alpha * 1 * 1 + alpha * (-1) * (-1)];
check('dựng lại w từ alpha khớp w của bài toán gốc', wTuAlpha[0], wSVM[0], 1e-12);
check('ràng buộc tổng alpha_i y_i = 0 thoả', alpha * 1 + alpha * (-1), 0, 1e-15);
const primalSVM = 0.5 * (wSVM[0] ** 2 + wSVM[1] ** 2);
// dual = sum alpha - 0.5 * sum_ij a_i a_j y_i y_j x_i.x_j, voi x1.x1=2, x2.x2=2, x1.x2=-2
const dualSVM = 2 * alpha - 0.5 * (alpha * alpha * 2 + alpha * alpha * 2
  + 2 * alpha * alpha * (-1) * (-2));
check('giá trị bài toán gốc = 0,25', primalSVM, 0.25, 1e-12);
check('giá trị bài toán đối ngẫu = 0,25', dualSVM, 0.25, 1e-12);
check('khe đối ngẫu bằng 0', Math.abs(primalSVM - dualSVM) < 1e-15, true);
check('thêm điểm (5,5) nhãn +1 thì ràng buộc lỏng nên alpha = 0',
  1 * (wSVM[0] * 5 + wSVM[1] * 5) > 1, true);

group('Nền tảng Bài 8 — PCA và định lý Eckart–Young');
const lamPCA = [10, 5, 3, 1.5, 0.5];
const tongLam = lamPCA.reduce((a, b) => a + b, 0);
check('tổng trị riêng = 20', tongLam, 20, 1e-12);
const giuDuoc = (k) => lamPCA.slice(0, k).reduce((a, b) => a + b, 0) / tongLam;
check('k=2 giữ 75% phương sai', giuDuoc(2), 0.75, 1e-12);
check('k=3 giữ 90% phương sai', giuDuoc(3), 0.90, 1e-12);
check('cần k=3 để đạt ít nhất 85%', giuDuoc(2) < 0.85 && giuDuoc(3) >= 0.85, true);
const boDi = (k) => lamPCA.slice(k).reduce((a, b) => a + b, 0);
check('Eckart–Young: k=2, n=101 cho sai số 500', 100 * boDi(2), 500, 1e-12);
check('k = 5 thì sai số tái dựng bằng 0', boDi(5), 0, 1e-12);
check('đổi đơn vị mét sang milimét nhân phương sai với 10^6', 1000 ** 2, 1e6, 1);

group('Nền tảng Bài 10 — quy mô của phân rã ma trận');
const nU = 1e6, nI = 1e5, kMF = 50;
check('số tham số = 55 triệu', (nU + nI) * kMF, 55e6, 1);
check('bằng 0,055% số ô', (nU + nI) * kMF / (nU * nI) * 100, 0.055, 1e-6);
check('quy tắc 5k cho 250 đánh giá mỗi người', 5 * kMF, 250);
check('tổng 250 triệu đánh giá, tức 0,25% số ô', nU * 5 * kMF / (nU * nI) * 100, 0.25, 1e-9);
check('thực tế 30 đánh giá là thiếu hơn 8 lần', (5 * kMF) / 30, 8.33, 0.01);
check('với 30 đánh giá thì hạng dùng được chỉ khoảng 6', Math.floor(30 / 5), 6);

// ---------------------------------------------------------------------------
// Giáo trình 3 — Biểu diễn, Sinh và Căn chỉnh.
// ---------------------------------------------------------------------------

group('Biểu diễn Mục 2.5 — chi phí softmax đầy đủ so với lấy mẫu âm');
const chiPhiFull = (V, d) => 2 * V * d;
const chiPhiNeg = (k, d) => 2 * (k + 1) * d;
check('V=2 triệu, d=300: softmax đầy đủ 1,2 tỉ phép tính',
  chiPhiFull(2e6, 300), 1.2e9, 1);
check('lấy mẫu âm k=5, d=300: 3.600 phép tính', chiPhiNeg(5, 300), 3600);
check('rẻ hơn 333.333 lần', chiPhiFull(2e6, 300) / chiPhiNeg(5, 300), 333333, 1);
check('chi phí lấy mẫu âm KHÔNG phụ thuộc từ vựng',
  chiPhiNeg(5, 300), chiPhiNeg(5, 300));
check('V=50.000 cho 30 triệu phép tính', chiPhiFull(50000, 300), 30e6, 1);

group('Biểu diễn Mục 7.3 — đếm tham số LoRA');
const loraLop = (d, r) => 2 * d * r;
check('GPT-2 small d=768, r=8: 12.288 tham số', loraLop(768, 8), 12288);
check('Llama-2 7B d=4096, r=8: 65.536 tham số', loraLop(4096, 8), 65536);
check('Llama-2 70B d=8192, r=8: 131.072 tham số', loraLop(8192, 8), 131072);
check('tỉ lệ là 2r/d, không phải r/d', loraLop(4096, 8) / (4096 * 4096), 2 * 8 / 4096, 1e-15);
check('d=768 cho 2,083%', loraLop(768, 8) / (768 * 768) * 100, 2.083, 5e-4);
check('d=8192 cho 0,195%', loraLop(8192, 8) / (8192 * 8192) * 100, 0.195, 5e-4);
check('tỉ lệ GIẢM khi d tăng — mô hình càng lớn LoRA càng lợi',
  loraLop(8192, 8) / (8192 ** 2) < loraLop(768, 8) / (768 ** 2), true);

// LoRA gắn vào W_Q và W_V của mọi lớp: 2 ma trận x L lớp.
const loraCaMoHinh = (L, d, r) => L * 2 * loraLop(d, r);
const N_7B = 6476005376;
check('Llama-2 7B, r=8 trên Q và V: 4.194.304 tham số',
  loraCaMoHinh(32, 4096, 8), 4194304);
check('bằng 0,0648% mô hình', loraCaMoHinh(32, 4096, 8) / N_7B * 100, 0.0648, 5e-5);
check('r=4 cho 2.097.152', loraCaMoHinh(32, 4096, 4), 2097152);
check('r=64 cho 33.554.432', loraCaMoHinh(32, 4096, 64), 33554432);
check('số tham số tuyến tính theo r',
  loraCaMoHinh(32, 4096, 64) / loraCaMoHinh(32, 4096, 4), 16, 1e-12);

group('Biểu diễn Mục 7.1 và 7.3 — bộ nhớ huấn luyện');
const adamState = (n) => n * 2 * 4;                   // hai trạng thái, FP32
check('trạng thái Adam cho LoRA r=8 là 32 MiB',
  adamState(loraCaMoHinh(32, 4096, 8)) / 2 ** 20, 32, 1e-9);
check('trạng thái Adam khi tinh chỉnh toàn phần là 48,2 GiB',
  adamState(N_7B) / 2 ** 30, 48.2, 0.05);
check('chênh nhau khoảng 1.540 lần',
  adamState(N_7B) / adamState(loraCaMoHinh(32, 4096, 8)), 1544, 5);
check('trọng số 7B ở FP32 là 24,1 GiB', N_7B * 4 / 2 ** 30, 24.1, 0.05);
check('tổng trọng số + gradient + Adam vượt 80 GiB',
  (N_7B * 4 * 2 + adamState(N_7B)) / 2 ** 30 > 80, true);
check('Bài 3: 50 khách với LoRA cần 13,8 GB', 13 + 50 * 0.016, 13.8, 1e-9);
check('Bài 3: 50 khách tinh chỉnh toàn phần cần 650 GB', 50 * 13, 650);
check('tiết kiệm 47 lần', (50 * 13) / (13 + 50 * 0.016), 47.1, 0.1);

group('Biểu diễn Mục 4.3 — dung lượng kho vector');
check('10 triệu vector 768 chiều FP32 là 30,7 GB',
  1e7 * 768 * 4 / 1e9, 30.7, 0.05);
check('hạ xuống FP16 còn một nửa', 1e7 * 768 * 2 / 1e9, 15.36, 0.05);

group('Biểu diễn Mục 11.2 — lịch nhiễu của mô hình khuếch tán');
const T_DIFF = 1000;
const betaLich = [];
for (let i = 0; i < T_DIFF; i++) betaLich.push(1e-4 + (0.02 - 1e-4) * i / (T_DIFF - 1));
const alphaNgang = [];
let acc = 1;
for (let i = 0; i < T_DIFF; i++) { acc *= (1 - betaLich[i]); alphaNgang.push(acc); }
check('beta chạy từ 1e-4 tới 0,02', betaLich[T_DIFF - 1], 0.02, 1e-12);
check('alpha_ngang tại t=0 là 0,9999', alphaNgang[0], 0.9999, 1e-9);
check('alpha_ngang tại t=999 là 4,0e-5', alphaNgang[999], 4.0e-5, 2e-6);
check('biên độ tín hiệu còn lại là căn của alpha_ngang, tức 0,63%',
  Math.sqrt(alphaNgang[999]) * 100, 0.63, 0.02);
check('alpha_ngang giảm đơn điệu', alphaNgang[500] < alphaNgang[200], true);
const snr = (t) => alphaNgang[t] / (1 - alphaNgang[t]);
check('SNR tại t=0 là 9.999', snr(0), 9999, 1);
check('SNR tại t=200 là 1,91', snr(200), 1.91, 0.02);
check('SNR giảm đơn điệu theo t', snr(800) < snr(500) && snr(500) < snr(200), true);
// Dang dong: phuong sai cua q(x_t|x_0) la 1 - alpha_ngang, cong voi (can alpha_ngang * x0)^2
check('phương sai dạng đóng tại t=999 gần bằng 1', 1 - alphaNgang[999], 1.0, 1e-4);

group('Biểu diễn Mục 14.4 — nghiệm RLHF có ràng buộc KL');
function softmaxJS(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
const piRef = softmaxJS([0.4, -1.1, 0.9, 0.2, -0.3, 1.4, -0.8, 0.1]);
const rThuong = [0.6, -1.3, 1.9, 0.1, -0.7, 0.4, 1.2, -0.2];

function piSao(beta) {
  return softmaxJS(piRef.map((p, i) => Math.log(p) + rThuong[i] / beta));
}
function mucTieu(pi, beta) {
  let er = 0, kl = 0;
  for (let i = 0; i < pi.length; i++) {
    er += pi[i] * rThuong[i];
    if (pi[i] > 0) kl += pi[i] * (Math.log(pi[i]) - Math.log(piRef[i]));
  }
  return er - beta * kl;
}
check('pi* là một phân phối hợp lệ',
  piSao(1.0).reduce((a, b) => a + b, 0), 1, 1e-12);
// Kiem tra toi uu: nhieu loan pi* mot chut thi muc tieu phai GIAM.
for (const beta of [0.2, 1.0, 5.0]) {
  const p = piSao(beta);
  const base = mucTieu(p, beta);
  let toiHon = 0;
  for (let k = 0; k < 40; k++) {
    const q = p.map((v, i) => v + 0.004 * Math.sin(3 * k + 7 * i));
    const s = q.reduce((a, b) => a + Math.max(b, 1e-12), 0);
    const qn = q.map((v) => Math.max(v, 1e-12) / s);
    if (mucTieu(qn, beta) > base + 1e-12) toiHon++;
  }
  check('beta=' + beta + ': không nhiễu loạn nào vượt được pi*', toiHon, 0);
}
check('beta nhỏ cho KL lớn hơn beta lớn', (() => {
  const kl = (pi) => pi.reduce((a, p, i) => a + (p > 0 ? p * (Math.log(p) - Math.log(piRef[i])) : 0), 0);
  return kl(piSao(0.2)) > kl(piSao(5.0));
})(), true);
check('beta nhỏ cho E[r] cao hơn beta lớn', (() => {
  const er = (pi) => pi.reduce((a, p, i) => a + p * rThuong[i], 0);
  return er(piSao(0.2)) > er(piSao(5.0));
})(), true);
check('pi_ref = 0 thì pi* = 0 bất kể thưởng lớn tới đâu', (() => {
  const ref = [0.5, 0.5, 0];
  const rr = [0, 0, 1000];
  const lg = ref.map((p, i) => (p === 0 ? -Infinity : Math.log(p) + rr[i] / 0.5));
  return softmaxJS(lg)[2];
})(), 0, 1e-300);

group('Biểu diễn Mục 15.2 — phép triệt tiêu log Z của DPO');
// r(y) = beta*log(pi(y)/pi_ref(y)) + beta*log Z. Khi lay HIEU hai cau tra loi
// cho CUNG mot cau hoi, so hang beta*log Z bi tru cho chinh no.
const BETA_DPO = 0.5;
const piTheta = softmaxJS([1.2, -0.4, 0.7, 2.1, -1.0, 0.3, 0.9, -0.6]);
const logZ = 3.7182818;                                 // gia tri tuy y
const rTu = (i) => BETA_DPO * (Math.log(piTheta[i]) - Math.log(piRef[i])) + BETA_DPO * logZ;
const rTuKhongZ = (i) => BETA_DPO * (Math.log(piTheta[i]) - Math.log(piRef[i]));
check('hiệu hai phần thưởng không phụ thuộc log Z',
  rTu(2) - rTu(5), rTuKhongZ(2) - rTuKhongZ(5), 1e-12);
check('đổi log Z sang giá trị khác cũng không đổi hiệu', (() => {
  const z2 = -42.5;
  const r2 = (i) => BETA_DPO * (Math.log(piTheta[i]) - Math.log(piRef[i])) + BETA_DPO * z2;
  return Math.abs((r2(2) - r2(5)) - (rTu(2) - rTu(5)));
})(), 0, 1e-12);
check('nhưng từng phần thưởng riêng lẻ THÌ phụ thuộc log Z',
  Math.abs(rTu(2) - rTuKhongZ(2)) > 1, true);

group('Biểu diễn Mục 13.3 — đường nền của gradient chính sách');
check('E[b * grad log pi] = 0 vì tổng xác suất luôn bằng 1', (() => {
  const pi = softmaxJS([0.3, -0.8, 1.1, 0.0, 0.5, -0.2]);
  // grad log pi(a) theo logit j la  1[a=j] - pi(j); ky vong theo a cho ra 0.
  let maxAbs = 0;
  for (let j = 0; j < pi.length; j++) {
    let s = 0;
    for (let a = 0; a < pi.length; a++) s += pi[a] * ((a === j ? 1 : 0) - pi[j]);
    maxAbs = Math.max(maxAbs, Math.abs(s));
  }
  return maxAbs;
})(), 0, 1e-15);
check('giảm độ lệch chuẩn 2,3 lần tiết kiệm 5,3 lần số mẫu', 2.3 ** 2, 5.29, 0.01);

// ---------------------------------------------------------------------------
// Phòng thí nghiệm — các hàm dùng chung trong src/assets/viz.js.
// Đây là CHÍNH mã đang chạy trên trang, không phải bản chép lại, nên nếu ai sửa
// nó mà làm lệch khỏi số liệu đã in trong giáo trình thì bộ kiểm tra báo ngay.
// ---------------------------------------------------------------------------

group('Lab — độ lớn gradient qua nhiều lớp so với số đo ở Mục 6.3');
const HE = Math.SQRT2;
// Cột phải là con số ĐO ĐƯỢC in trong giáo trình; công cụ phải bám sát nó.
const CAU_HINH = [
  ['gain 0,5 — khởi tạo quá nhỏ', 0.5, false, false, 1.233e-18],
  ['gain √2 — khởi tạo He', HE, false, false, 1.421],
  ['gain 2,0 — khởi tạo quá lớn', 2.0, false, false, 1.490e6],
  ['He + chuẩn hoá', HE, true, false, 1.151],
  ['He + kết nối tắt, KHÔNG chuẩn hoá', HE, false, true, 2.446e8],
  ['He + chuẩn hoá + kết nối tắt (pre-LN)', HE, true, true, 2.222],
];
for (const [ten, gain, norm, res, doDuoc] of CAU_HINH) {
  const v = Z.doLonGradient(gain, 40, norm, res);
  // Khớp trong vòng một hệ số 2 là đủ: công cụ tóm tắt hành vi chứ không mô
  // phỏng lại phép truyền ngược, nhưng nó phải đúng BẬC ĐỘ LỚN.
  const tiLe = Math.abs(Math.log10(v / doDuoc));
  check(ten + ' — cùng bậc độ lớn với số đo', tiLe < 0.35, true);
}
check('gain 0,5 phải là TIÊU BIẾN', Z.doLonGradient(0.5, 40, false, false) < 1e-6, true);
check('gain 2,0 phải là BÙNG NỔ', Z.doLonGradient(2.0, 40, false, false) > 1e6, true);
check('kết nối tắt MỘT MÌNH làm bùng nổ',
  Z.doLonGradient(HE, 40, false, true) > 1e6, true);
check('thêm chuẩn hoá thì ổn định lại', (() => {
  const v = Z.doLonGradient(HE, 40, true, true);
  return v > 1 && v < 10;
})(), true);
check('khởi tạo He một mình cho tỉ lệ gần 1',
  Z.doLonGradient(HE, 40, false, false), 1, 0.01);
check('càng sâu thì càng lệch xa 1 khi gain sai',
  Z.doLonGradient(0.5, 80, false, false) < Z.doLonGradient(0.5, 40, false, false), true);

group('Lab — các hàm số dùng chung');
check('softmax cho tổng bằng 1',
  Z.softmax([2.1, -0.4, 1.7, 0.3]).reduce((a, b) => a + b, 0), 1, 1e-12);
check('softmax bất biến khi cộng hằng số vào mọi logit', (() => {
  const a = Z.softmax([1, 2, 3]);
  const b = Z.softmax([101, 102, 103]);
  return Math.max(...a.map((v, i) => Math.abs(v - b[i])));
})(), 0, 1e-12);
check('giải hệ 2x2 đúng', (() => {
  const x = Z.giaiTuyenTinh([[4, 10], [10, 30]], [16, 47]);
  return Math.abs(x[0] - 0.5) + Math.abs(x[1] - 1.4);
})(), 0, 1e-12);
check('giải hệ suy biến trả về null',
  Z.giaiTuyenTinh([[1, 2], [2, 4]], [1, 2]), null);
check('cùng hạt giống thì cùng dãy số ngẫu nhiên', (() => {
  const a = Z.rng(42), b = Z.rng(42);
  let d = 0;
  for (let i = 0; i < 50; i++) d += Math.abs(a.n() - b.n());
  return d;
})(), 0, 1e-15);
check('khác hạt giống thì khác dãy', (() => {
  const a = Z.rng(1), b = Z.rng(2);
  let d = 0;
  for (let i = 0; i < 50; i++) d += Math.abs(a.n() - b.n());
  return d > 1;
})(), true);


// ---------------------------------------------------------------------------
// Ứng dụng LLM — phòng thí nghiệm, đối chiếu với các con số đã in trong giáo trình
// ---------------------------------------------------------------------------

group('Ứng dụng LLM Mục 2.5 — chi phí một yêu cầu RAG');
{
  const k = U.chiPhi({ heThong: 600, soDoan: 5, doan: 350, lichSu: 800, cauHoi: 60, ra: 350,
    giaVao: 1, giaRa: 4, cache: false, giaCache: 0.1, soYeuCau: 660000 });
  check('tổng token đầu vào = 3 210', k.tongVao, 3210);
  check('chi phí mỗi yêu cầu = 0,00461', k.moiYC, 0.00461, 1e-12);
  check('đầu ra chiếm 9,8% số token', k.tiLeTokenRa, 0.098, 0.0005);
  check('đầu ra chiếm 30,4% chi phí', k.tiLeRa, 0.304, 0.0005);
  check('câu hỏi chiếm 1,9% đầu vào', k.tiLeHoi, 0.019, 0.0005);
  check('bộ đệm prompt cho chỉ dẫn hệ thống giảm 11,7%', k.tietKiem, 0.117, 0.0005);
  check('Ví dụ 2.1: chi phí tháng ≈ 3 043', k.thang, 3042.6, 0.05);
  const kc = U.chiPhi({ heThong: 600, soDoan: 5, doan: 350, lichSu: 800, cauHoi: 60, ra: 350,
    giaVao: 1, giaRa: 4, cache: true, giaCache: 0.1, soYeuCau: 1 });
  check('bật bộ đệm thì chi phí giảm đúng 11,7%', 1 - kc.moiYC / k.moiYC, k.tietKiem, 1e-12);
}

group('Ứng dụng LLM Mục 2.2 — KV cache của Llama 3 8B');
{
  const moi = U.kvMoiToken(32, 8, 128, 2);
  const GiB = Math.pow(2, 30);
  check('mỗi token 131 072 byte = 128 KiB', moi, 131072);
  check('2 048 token: 0,25 GiB', moi * 2048 / GiB, 0.25, 1e-12);
  check('8 192 token: 1 GiB', moi * 8192 / GiB, 1, 1e-12);
  check('32 768 token: 4 GiB', moi * 32768 / GiB, 4, 1e-12);
  check('131 072 token: 16 GiB', moi * 131072 / GiB, 16, 1e-12);
  check('không dùng GQA (32 đầu KV) thì lớn gấp 4', U.kvMoiToken(32, 32, 128, 2) / moi, 4);
}

group('Ứng dụng LLM Mục 9.7 — độ tin cậy của agent nhiều bước');
{
  const bang = [
    [1, [0.990, 0.950, 0.900]], [5, [0.951, 0.774, 0.590]], [10, [0.904, 0.599, 0.349]],
    [20, [0.818, 0.358, 0.122]], [50, [0.605, 0.077, 0.005]],
  ];
  for (const [n, vs] of bang) {
    [0.99, 0.95, 0.90].forEach((p, i) => {
      check(`p = ${p}, ${n} bước: ${vs[i]}`, U.agent(p, n, 0, 0).khong, vs[i], 0.0005);
    });
  }
  const kiemTra = [
    [0.0, 0, 0.358, 1.000], [0.5, 1, 0.587, 1.025], [0.8, 1, 0.785, 1.040],
    [0.8, 3, 0.811, 1.042], [0.95, 3, 0.949, 1.050],
  ];
  for (const [c, r, pc, goi] of kiemTra) {
    const k = U.agent(0.95, 20, c, r);
    check(`c = ${c}, r = ${r}: hoàn thành ${pc}`, k.co, pc, 0.0005);
    check(`c = ${c}, r = ${r}: ${goi} lời gọi mỗi bước`, k.goiMoiBuoc, goi, 0.0005);
  }
}

group('Ứng dụng LLM Mục 12.4 — cỡ bộ đánh giá');
{
  const ktc = [[50, 11.1], [100, 7.8], [200, 5.5], [500, 3.5], [1000, 2.5], [2000, 1.8]];
  for (const [n, v] of ktc) check(`n = ${n}: ± ${v} điểm`, U.ktc(0.8, n) * 100, v, 0.05);
  // Bảng Mục 12.4 là mô phỏng 2 000 lần mỗi n, nên chỉ so trong sai số mô phỏng.
  // A đúng 80%; B giữ 97% số câu A đúng và sửa 27% số câu A sai.
  const p01 = 0.2 * 0.27, p10 = 0.8 * 0.03;
  const luc = [[100, 0.086, 0.076], [300, 0.392, 0.154], [1000, 0.919, 0.410], [3000, 1.000, 0.857]];
  for (const [n, cap, rieng] of luc) {
    check(`n = ${n}: ghép cặp ≈ ${cap} (McNemar chính xác)`, U.lucGhepCap(n, p01, p10, 0.05), cap, 0.03);
    check(`n = ${n}: hai bộ độc lập ≈ ${rieng}`, U.lucDocLap(n, 0.8, 0.83, 0.05), rieng, 0.02);
  }
  check('ghép cặp cần ít hơn một phần ba số câu so với hai bộ độc lập',
    U.nGhepCap(p01, p10, 0.05, 0.8) / U.nDocLap(0.8, 0.83, 0.05, 0.8) < 1 / 3, true);
  check('normCdf(1,96) ≈ 0,975', U.normCdf(1.959964), 0.975, 1e-6);
}


console.log('\n' + (fail === 0 ? 'Tất cả ' + pass + ' phép kiểm tra đều đạt.' : pass + ' đạt, ' + fail + ' HỎNG.'));
process.exit(fail === 0 ? 0 : 1);
