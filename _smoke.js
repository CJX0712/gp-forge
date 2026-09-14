// 无头验证：从 index.html 抽出 <script id="engine">，在 Node vm 里跑断言
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = __dirname;
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('engine script not found'); process.exit(1); }
const ctx = { console };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'engine.js' });
const G = ctx.GP;

let pass = 0, fail = 0; const fails = [];
function ok(cond, name, detail) { if (cond) pass++; else { fail++; fails.push(name + ' | ' + detail); } }
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ---------- 1) 内置 8 条不变量 ----------
G.selfTest().forEach((r, i) => ok(r.pass, 'selfTest#' + (i + 1) + ' ' + r.name, r.detail));

// ---------- 2) 随机压力测试 ----------
const rng = G.mulberry32(24680);
let nPred = 0;
for (let trial = 0; trial < 120; trial++) {
  const n = 1 + Math.floor(rng() * 12);
  const X = [], y = [];
  for (let i = 0; i < n; i++) { X.push(rng() * 10); y.push(2 * rng() - 1 + 0.2 * G.rnorm(rng)); }
  const p = { ls: 0.2 + rng() * 3, sf: 0.3 + rng() * 1.5, sn: 0.02 + rng() * 0.5 };
  const Xte = [0.5, 2.5, 5, 7.5, 9.5];

  // 核矩阵对称 + Cholesky 成功（含噪声后应严格正定）
  const K = G.kernelMatrix(X, X, p), Ky = G.addNoise(K, p.sn);
  let sym = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) sym = Math.max(sym, Math.abs(K[i][j] - K[j][i]));
  ok(sym === 0, `trial${trial} 核矩阵对称`, `sym=${sym}`);
  const L = G.cholesky(Ky);
  ok(L !== null, `trial${trial} K+σₙ²I 严格正定（Cholesky 零 jitter）`, `n=${n} ls=${p.ls.toFixed(2)}`);

  // Cholesky 解 == 高斯消元解
  if (L) {
    const aC = G.cholSolve(L, y), aG = G.gaussSolve(Ky, y);
    let rel = 0;
    for (let i = 0; i < n; i++) rel = Math.max(rel, Math.abs(aC[i] - aG[i]) / (Math.abs(aG[i]) + 1e-12));
    ok(rel < 1e-8, `trial${trial} Cholesky==高斯消元`, `rel=${rel}`);
    // L Lᵀ == Ky
    let err = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0; for (let k = 0; k <= Math.min(i, j); k++) s += L[i][k] * L[j][k];
      err = Math.max(err, Math.abs(s - Ky[i][j]));
    }
    ok(err < 1e-10, `trial${trial} L·Lᵀ == K+σₙ²I`, `err=${err}`);
  }

  // 后验：方差非负、均值有限
  const pr = G.gpPredict(X, y, Xte, p);
  ok(pr !== null, `trial${trial} 后验可计算`, '');
  if (pr) {
    nPred++;
    ok(pr.mu.every(Number.isFinite) && pr.var.every(v => Number.isFinite(v) && v >= 0),
       `trial${trial} μ/σ² 有限且方差非负`, JSON.stringify(pr.var.map(v => +v.toFixed(4))));
    ok(pr.var.every(v => v <= p.sf * p.sf + 1e-9), `trial${trial} 后验方差 ≤ 先验方差 σf²`, '');
  }

  // 解析梯度 == 中心差分
  const th = G.thetaFromParams(p), eps = 1e-5;
  const gA = G.logML(X, y, p, true).grad;
  let maxRel = 0;
  for (let k = 0; k < 3; k++) {
    const tp = th.slice(), tm = th.slice(); tp[k] += eps; tm[k] -= eps;
    const gN = (G.logML(X, y, G.paramsFromTheta(tp), false).val - G.logML(X, y, G.paramsFromTheta(tm), false).val) / (2 * eps);
    maxRel = Math.max(maxRel, Math.abs(gA[k] - gN) / (Math.abs(gN) + 1e-6));
  }
  ok(maxRel < 1e-4, `trial${trial} 边际似然梯度检验`, `maxRel=${maxRel}`);
}
ok(nPred >= 115, '后验覆盖数', 'nPred=' + nPred);

// ---------- 3) 信息单调性（随机 30 组） ----------
for (let t = 0; t < 30; t++) {
  const n1 = 2 + Math.floor(rng() * 4), n2 = n1 + 2 + Math.floor(rng() * 4);
  const X = [], y = [];
  for (let i = 0; i < n2; i++) { X.push(rng() * 10); y.push(2 * rng() - 1); }
  const p = { ls: 0.5 + rng() * 2, sf: 1, sn: 0.1 + rng() * 0.3 };
  const Xte = [1, 3, 5, 7, 9];
  const pa = G.gpPredict(X.slice(0, n1), y.slice(0, n1), Xte, p);
  const pb = G.gpPredict(X.slice(0, n2), y.slice(0, n2), Xte, p);
  let worst = -Infinity;
  for (let i = 0; i < Xte.length; i++) worst = Math.max(worst, pb.var[i] - pa.var[i]);
  ok(worst <= 1e-9, `mono#${t} 观测增多 → 后验方差不增`, `worstInc=${worst}`);
}

// ---------- 4) 边界条件 ----------
{
  // n = 0：应退化为先验 μ=0, σ²=σf²
  const p = { ls: 1, sf: 1.3, sn: 0.2 };
  const pr = G.gpPredict([], [], [1, 2, 3], p);
  ok(pr !== null && pr.mu.every(v => v === 0), 'n=0 后验均值 == 先验 0', JSON.stringify(pr && pr.mu));
  ok(pr && pr.var.every(v => close(v, p.sf * p.sf, 1e-12)), 'n=0 后验方差 == 先验 σf²', JSON.stringify(pr && pr.var));

  // n = 1：解析解 μ(x) = k(x,x1)/(k11+σn²)·y1
  const X1a = [2.0], y1 = [1.5], xt = 3.0;
  const pr1 = G.gpPredict(X1a, y1, [xt], p);
  const kk = p.sf * p.sf * Math.exp(-0.5 * 1 / (p.ls * p.ls));
  const muExp = kk / (p.sf * p.sf + p.sn * p.sn) * 1.5;
  ok(close(pr1.mu[0], muExp, 1e-12), 'n=1 后验均值 == 解析解', `${pr1.mu[0]} vs ${muExp}`);
  const varExp = p.sf * p.sf - kk * kk / (p.sf * p.sf + p.sn * p.sn);
  ok(close(pr1.var[0], varExp, 1e-12), 'n=1 后验方差 == 解析解', `${pr1.var[0]} vs ${varExp}`);

  // 重复 x（K 奇异）→ jitter 兜底，不得 NaN
  const Xd = [1, 1, 1, 4], yd = [1, 1.2, 0.8, -1];
  const prd = G.gpPredict(Xd, yd, [0, 2, 5], { ls: 1, sf: 1, sn: 1e-8 });
  ok(prd && prd.mu.every(Number.isFinite) && prd.var.every(Number.isFinite), '重复 x（奇异核）不产生 NaN',
     JSON.stringify(prd && prd.mu.map(v => +v.toFixed(4))));

  // 极端超参
  [[1e-3, 1, 0.1], [1e3, 1, 0.1], [1, 1e-3, 1e-3], [1, 3, 1.5]].forEach(([ls, sf, sn], i) => {
    const pr2 = G.gpPredict([1, 2, 3, 4], [1, -1, 0.5, 0], [0.5, 5], { ls, sf, sn });
    ok(pr2 && pr2.mu.every(Number.isFinite) && pr2.var.every(v => Number.isFinite(v) && v >= 0),
       `极端超参#${i} 数值有限`, `ls=${ls} sf=${sf} sn=${sn}`);
  });

  // 无噪插值
  const Xn = [0, 1, 2, 3, 4, 5], yn = [0.2, 1.1, -0.4, 0.7, -1.2, 0.3];
  const prn = G.gpPredict(Xn, yn, Xn, { ls: 1.2, sf: 1.5, sn: 1e-7 });
  let me = 0, mv = 0;
  for (let i = 0; i < Xn.length; i++) { me = Math.max(me, Math.abs(prn.mu[i] - yn[i])); mv = Math.max(mv, prn.var[i]); }
  ok(me < 1e-4 && mv < 1e-6, '无噪插值：μ==y 且 σ²==0', `maxErr=${me} maxVar=${mv}`);
}

// ---------- 5) 确定性 ----------
{
  const mk = () => { const r = G.mulberry32(7); const X = [], y = []; for (let i = 0; i < 8; i++) { X.push(i * 0.9); y.push(Math.sin(i) + 0.1 * G.rnorm(r)); } return { X, y }; };
  const a = mk(), b = mk();
  ok(a.y.every((v, i) => v === b.y[i]), '同 seed 数据确定性', '');
  const p = { ls: 1, sf: 1, sn: 0.1 };
  const s1 = G.mvnSample([0, 0], [[1, 0.3], [0.3, 1]], 200, G.mulberry32(11));
  const s2 = G.mvnSample([0, 0], [[1, 0.3], [0.3, 1]], 200, G.mulberry32(11));
  ok(s1.every((v, i) => v[0] === s2[i][0] && v[1] === s2[i][1]), '同 seed 采样确定性', '');
  const r1 = G.selfTest().map(r => r.detail).join('|'), r2 = G.selfTest().map(r => r.detail).join('|');
  ok(r1 === r2, 'selfTest 可复现', '');
}

// ---------- 6) 超参优化：单调 + 还原 ----------
for (let c = 0; c < 4; c++) {
  const r0 = G.mulberry32(500 + c);
  const truth = { ls: 0.5 + c * 0.6, sf: 1.0, sn: 0.08 };
  const X = [], y0 = [];
  for (let i = 0; i < 40; i++) X.push(i * 0.25);
  const zero = X.map(() => 0);
  const C = G.kernelMatrix(X, X, truth);   // 先验协方差：从 GP 先验采一条真实函数
  const s = G.mvnSample(zero, C, 1, r0)[0];
  for (let i = 0; i < X.length; i++) y0.push(s[i] + truth.sn * G.rnorm(r0));
  const opt = G.optimizeAuto(X, y0, 120);
  let minD = Infinity;
  for (let i = 1; i < opt.hist.length; i++) minD = Math.min(minD, opt.hist[i] - opt.hist[i - 1]);
  ok(minD > -1e-12, `opt#${c} 边际似然单调非减`, `minD=${minD}`);
  const mlTruth = G.logML(X, y0, truth, false).val, mlEnd = opt.hist[opt.hist.length - 1];
  ok(mlEnd >= mlTruth - 1e-6, `opt#${c} 优化后 logML ≥ 真值 logML`, `${mlEnd} vs ${mlTruth}`);
  ok(opt.p.ls > 1e-3 && opt.p.ls < 1e3 && opt.p.sf > 1e-3 && opt.p.sn > 1e-6, `opt#${c} 超参未跑飞`,
     `ls=${opt.p.ls.toFixed(3)} sf=${opt.p.sf.toFixed(3)} sn=${opt.p.sn.toFixed(4)}`);
}

fs.writeFileSync(path.join(dir, '_smoke.log'),
  `PASS ${pass} / ${pass + fail}\n` + (fail ? 'FAIL:\n' + fails.join('\n') : 'ALL GREEN') + '\n');
console.log(`PASS ${pass} / ${pass + fail}`);
if (fail) { console.log('FAIL:\n' + fails.join('\n')); process.exit(1); }
console.log('ALL GREEN');
