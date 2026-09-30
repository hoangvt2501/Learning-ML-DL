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

const L = sandbox.window.QZ_LAB;
const M = sandbox.window.QZ_MLOPS;

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

console.log('\n' + (fail === 0 ? 'Tất cả ' + pass + ' phép kiểm tra đều đạt.' : pass + ' đạt, ' + fail + ' HỎNG.'));
process.exit(fail === 0 ? 0 : 1);
