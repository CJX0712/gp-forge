// 探针：独立重算一遍再比对，确认「全绿 ≠ 正确」
const fs = require('fs'), vm = require('vm'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx = { console }; ctx.globalThis = ctx; vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const G = ctx.GP;
const o = []; const say = s => o.push(s);

const X = [0.5, 1.4, 2.6, 3.9, 5.2, 6.8, 8.1, 9.3];
const y = [0.62, 1.05, 0.31, -0.72, -1.18, -0.35, 0.54, 1.02];
const p = { ls: 1.3, sf: 1.1, sn: 0.18 };

say('=== 数据 ===');
say('X = ' + X.map(v => v.toFixed(2)).join(' '));
say('y = ' + y.map(v => v.toFixed(2)).join(' '));
say('θ: ℓ = ' + p.ls + ', σf = ' + p.sf + ', σn = ' + p.sn);

// 核矩阵
const K = G.kernelMatrix(X, X, p);
say('');
say('=== 核矩阵 K（RBF）===');
K.forEach((r, i) => say('  ' + r.map(v => v.toFixed(4).padStart(8)).join(' ')));

// Cholesky 与 L Lᵀ 复原
const Ky = G.addNoise(K, p.sn), L = G.cholesky(Ky);
say('');
say('=== Cholesky 因子 L（下三角）===');
L.forEach(r => say('  ' + r.map(v => v.toFixed(5).padStart(10)).join(' ')));
let recErr = 0;
for (let i = 0; i < L.length; i++) for (let j = 0; j < L.length; j++) {
  let s = 0; for (let k = 0; k <= Math.min(i, j); k++) s += L[i][k] * L[j][k];
  recErr = Math.max(recErr, Math.abs(s - Ky[i][j]));
}
say('  ‖L·Lᵀ − (K+σₙ²I)‖∞ = ' + recErr.toExponential(3));
say('  log|K+σₙ²I| = ' + G.cholLogDet(L).toFixed(6));

// 独立实现：直接用求逆（高斯消元）算后验，与 Cholesky 路径对拍
function inverse(A) {
  const n = A.length, Inv = [];
  for (let j = 0; j < n; j++) {
    const e = new Array(n).fill(0); e[j] = 1;
    const col = G.gaussSolve(A, e);
    Inv.push(col);
  }
  // Inv 是行向量数组，转成标准矩阵
  const M = []; for (let i = 0; i < n; i++) { M.push(Inv.map(row => row[i])); }
  return M;
}
function predictViaInverse(Xtr, ytr, Xte, p) {
  const n = Xtr.length, Ky = G.addNoise(G.kernelMatrix(Xtr, Xtr, p), p.sn), Kinv = inverse(Ky);
  const alpha = G.gaussSolve(Ky, ytr);
  const mu = [], va = [];
  for (const xt of Xte) {
    const ks = Xtr.map(x => p.sf * p.sf * Math.exp(-0.5 * (x - xt) ** 2 / (p.ls * p.ls)));
    let m = 0; for (let i = 0; i < n; i++) m += ks[i] * alpha[i];
    mu.push(m);
    let v = p.sf * p.sf;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v -= ks[i] * Kinv[i][j] * ks[j];
    va.push(v);
  }
  return { mu, va, Kinv, alpha };
}
const Xte = [1.0, 2.0, 3.0, 4.5, 6.0, 7.5, 9.0];
const pr = G.gpPredict(X, y, Xte, p);
const ref = predictViaInverse(X, y, Xte, p);
say('');
say('=== 后验：Cholesky 路径 vs 独立求逆路径 ===');
say('  x        μ(Cholesky)   μ(求逆)       σ²(Cholesky)  σ²(求逆)');
let maxMu = 0, maxVa = 0;
for (let i = 0; i < Xte.length; i++) {
  maxMu = Math.max(maxMu, Math.abs(pr.mu[i] - ref.mu[i]));
  maxVa = Math.max(maxVa, Math.abs(pr.var[i] - ref.va[i]));
  say('  ' + Xte[i].toFixed(2).padStart(5) + '   ' + pr.mu[i].toFixed(8).padStart(12) + '   ' +
      ref.mu[i].toFixed(8).padStart(12) + '   ' + pr.var[i].toFixed(8).padStart(12) + '   ' + ref.va[i].toFixed(8).padStart(12));
}
say('  max|Δμ| = ' + maxMu.toExponential(3) + '   max|Δσ²| = ' + maxVa.toExponential(3));

// 训练点后验方差解析恒等式
const prTr = G.gpPredict(X, y, X, p), Kinv = G.inverseFromChol(prTr.fit.L);
say('');
say('=== 训练点：后验 μ 与方差解析恒等式 σₙ²(1−σₙ²[(K+σₙ²I)⁻¹]ᵢᵢ) ===');
say('  i    y         μ           σ²          σₙ²(1−σₙ²·Kinv_ii)   相对误差');
let idMax = 0;
for (let i = 0; i < X.length; i++) {
  const e = p.sn * p.sn * (1 - p.sn * p.sn * Kinv[i][i]);
  const rel = Math.abs(prTr.var[i] - e) / e;
  idMax = Math.max(idMax, rel);
  say('  ' + i + '   ' + y[i].toFixed(3).padStart(6) + '   ' + prTr.mu[i].toFixed(6).padStart(9) + '   ' +
      prTr.var[i].toFixed(8).padStart(10) + '   ' + e.toFixed(8).padStart(18) + '   ' + rel.toExponential(2));
}
say('  恒等式最大相对误差 = ' + idMax.toExponential(3));

// 信息单调性
say('');
say('=== 信息单调性：观测从 4 个增至 8 个，测试点后验方差应不增 ===');
const pa = G.gpPredict(X.slice(0, 4), y.slice(0, 4), Xte, p);
say('  x        σ²(n=4)      σ²(n=8)      增量');
let inc = -Infinity;
for (let i = 0; i < Xte.length; i++) {
  inc = Math.max(inc, pr.var[i] - pa.var[i]);
  say('  ' + Xte[i].toFixed(2).padStart(5) + '   ' + pa.var[i].toFixed(6).padStart(10) + '   ' +
      pr.var[i].toFixed(6).padStart(10) + '   ' + (pr.var[i] - pa.var[i]).toExponential(2).padStart(11));
}
say('  最大增量 = ' + inc.toExponential(3) + '（应 ≤ 0）');

// 边际似然梯度
say('');
say('=== 边际似然梯度：解析 vs 中心差分 ===');
const th = G.thetaFromParams(p), gA = G.logML(X, y, p, true).grad, names = ['log ℓ', 'log σf', 'log σn'];
say('  log p(y|X,θ) = ' + G.logML(X, y, p, false).val.toFixed(6));
for (let k = 0; k < 3; k++) {
  const eps = 1e-5, tp = th.slice(), tm = th.slice(); tp[k] += eps; tm[k] -= eps;
  const gN = (G.logML(X, y, G.paramsFromTheta(tp), false).val - G.logML(X, y, G.paramsFromTheta(tm), false).val) / (2 * eps);
  say('  ∂/∂' + names[k].padEnd(7) + ' 解析 = ' + gA[k].toFixed(8).padStart(13) +
      '  数值 = ' + gN.toFixed(8).padStart(13) +
      '  相对误差 = ' + (Math.abs(gA[k] - gN) / (Math.abs(gN) + 1e-12)).toExponential(2));
}

// 后验采样蒙特卡洛
say('');
say('=== 后验采样 20000 次 vs 解析 μ/σ ===');
const X3 = [1.0, 4.5, 9.0], pr3 = G.gpPredict(X, y, X3, p), C3 = G.gpPosteriorCov(X3, p, pr3);
say('  后验协方差 C =');
C3.forEach(r => say('    ' + r.map(v => v.toFixed(6).padStart(11)).join(' ')));
const NS = 20000, S = G.mvnSample(pr3.mu, C3, NS, G.mulberry32(2026));
const em = [0, 0, 0];
for (let t = 0; t < NS; t++) for (let i = 0; i < 3; i++) em[i] += S[t][i] / NS;
const ev = [0, 0, 0];
for (let t = 0; t < NS; t++) for (let i = 0; i < 3; i++) ev[i] += ((S[t][i] - em[i]) ** 2) / (NS - 1);
for (let i = 0; i < 3; i++) {
  say('  x=' + X3[i].toFixed(1) + '  μ解析 = ' + pr3.mu[i].toFixed(5) + '  经验 = ' + em[i].toFixed(5) +
      '  |  σ解析 = ' + Math.sqrt(pr3.var[i]).toFixed(5) + '  经验 = ' + Math.sqrt(ev[i]).toFixed(5) +
      '  相对误差 = ' + (Math.abs(Math.sqrt(ev[i]) - Math.sqrt(pr3.var[i])) / Math.sqrt(pr3.var[i]) * 100).toFixed(2) + '%');
}

// 超参还原
say('');
say('=== 超参还原：真值 ℓ=1.0 σf=1.0 σn=0.10，n=50，3 个起点取最优 ===');
const r0 = G.mulberry32(777);
const Xt = []; for (let i = 0; i < 50; i++) Xt.push(i * 0.2);
const zero = Xt.map(() => 0), truth = { ls: 1.0, sf: 1.0, sn: 0.10 };
const Ct = G.kernelMatrix(Xt, Xt, truth);   // 先验协方差：从 GP 先验采一条真实函数
const smp = G.mvnSample(zero, Ct, 1, r0)[0];
const yt = smp.map((v, i) => v + truth.sn * G.rnorm(r0));
const opt = G.optimizeAuto(Xt, yt, 120);
let minD = Infinity;
for (let i = 1; i < opt.hist.length; i++) minD = Math.min(minD, opt.hist[i] - opt.hist[i - 1]);
const pick = opt.hist.filter((_, i) => i % 20 === 0 || i === opt.hist.length - 1).map(v => v.toFixed(2));
say('  最优起点的上升曲线: ' + pick.join(' → '));
say('  最小 Δ = ' + minD.toExponential(2) + '（应 ≥ 0）');
say('  ℓ̂  = ' + opt.p.ls.toFixed(4) + '  (真值 1.0, |log 差| = ' + Math.abs(Math.log(opt.p.ls)).toFixed(3) + ')');
say('  σ̂f = ' + opt.p.sf.toFixed(4) + '  (真值 1.0)');
say('  σ̂n = ' + opt.p.sn.toFixed(4) + '  (真值 0.10)');
say('  logML(学习) = ' + opt.hist[opt.hist.length - 1].toFixed(3) +
    '   logML(真值) = ' + G.logML(Xt, yt, truth, false).val.toFixed(3));

fs.writeFileSync(path.join(__dirname, '_probe.txt'), o.join('\n') + '\n');
console.log(o.join('\n'));
