// 대시보드 시연 영상 녹화 (Playwright + 설치된 Chrome). 로딩·계산 대기 구간은 잘라내고 장면만 이어 붙여 mp4로.
// 사용: (대시보드 실행 중, localhost:8501)  node docs/발표/scripts/demo_record.js  → docs/발표/대시보드_시연_초안.mp4
// 장면: ① 구매 의사결정 → ② 머신러닝 예측(④) → ③ Portfolio Customization(⑤) → ④ 레짐 · 칼만 필터(②)
const { chromium } = require("playwright");
const path = require("path"), fs = require("fs"), { execFileSync } = require("child_process");
const URL = process.env.DASH_URL || "http://localhost:8501";
const OUTDIR = path.join(__dirname, "..");
const TMP = path.join(OUTDIR, "_demo_tmp");
const FFMPEG = process.env.FFMPEG || path.join(__dirname, "..", "..", "..", ".venv", "Lib", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-win-x86_64-v7.1.exe");
const VW = 1600, VH = 900;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 화면 위 자막 · 가짜 마우스 커서(헤드리스 녹화에는 커서가 안 찍힘)
const OVERLAY = () => {
  if (document.getElementById("demo-cap")) return;
  const cap = document.createElement("div"); cap.id = "demo-cap";
  cap.style.cssText = "position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:999999;background:rgba(30,58,47,.92);color:#fff;font:600 22px 'Malgun Gothic',sans-serif;padding:10px 26px;border-radius:12px;border-left:6px solid #C8A24A;box-shadow:0 6px 24px rgba(0,0,0,.35);opacity:0;transition:opacity .4s;pointer-events:none;white-space:nowrap";
  document.body.appendChild(cap);
  const cur = document.createElement("div"); cur.id = "demo-cur";
  cur.style.cssText = "position:fixed;left:-50px;top:-50px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:rgba(200,162,74,.55);border:2px solid #fff;z-index:999999;pointer-events:none;transition:transform .12s";
  document.body.appendChild(cur);
  addEventListener("mousemove", (e) => { cur.style.left = e.clientX + "px"; cur.style.top = e.clientY + "px"; }, true);
  addEventListener("mousedown", () => { cur.style.transform = "scale(.6)"; }, true);
  addEventListener("mouseup", () => { cur.style.transform = "scale(1)"; }, true);
};
const caption = (p, t) => p.evaluate((t) => { const c = document.getElementById("demo-cap"); if (!c) return; c.textContent = t; c.style.opacity = t ? 1 : 0; }, t);

async function moveTo(p, loc, steps = 25) {                      // 커서를 요소 가운데로 부드럽게 이동
  const b = await loc.boundingBox(); if (!b) return null;
  const x = b.x + b.width / 2, y = b.y + b.height / 2; await p.mouse.move(x, y, { steps }); return { x, y };
}
async function clickSmooth(p, loc) { await loc.scrollIntoViewIfNeeded(); const c = await moveTo(p, loc); if (c) { await sleep(250); await p.mouse.down(); await sleep(80); await p.mouse.up(); } }
async function wheel(p, dy, n = 12, gap = 60) { for (let i = 0; i < n; i++) { await p.mouse.wheel(0, dy / n); await sleep(gap); } }
async function scrollTo(p, loc) { await loc.evaluate((e) => e.scrollIntoView({ behavior: "smooth", block: "start" })); await sleep(1300); }
const visible = (p, sel, opts) => p.locator(sel, opts).filter({ visible: true }).first();
async function waitIdle(p, max = 120000) {                       // Streamlit 계산 끝(실행 중 표시 사라짐)까지
  const t0 = Date.now(), st = p.locator('[data-testid="stStatusWidget"]').filter({ visible: true });
  while (Date.now() - t0 < 3000 && !(await st.count())) await sleep(200);   // 실행 표시가 뜰 때까지 잠깐 기다린 뒤
  while (Date.now() - t0 < max) { if (!(await p.locator('[data-testid="stStatusWidget"]').filter({ visible: true }).count())) return; await sleep(500); }
}

async function scenes(p, mark) {
  const tab = (name) => p.getByRole("tab", { name, exact: true });
  // ① 구매 의사결정 (첫 탭)
  await p.mouse.move(VW / 2, VH / 2);
  mark("start"); await caption(p, "① 구매 의사결정 — 품목별 '지금 살까, 미룰까'를 매주 판단"); await sleep(1200);
  const cards = visible(p, "text=이번 주 판단"); await moveTo(p, cards); await sleep(1200);
  await wheel(p, 520); await sleep(2000); await wheel(p, 520); await sleep(1800); await wheel(p, -1040, 8); await sleep(400);
  // ② 머신러닝 예측: ④ 탭 → 모델 실행 (계산 대기는 잘라냄)
  await caption(p, "② 머신러닝 예측 — OLS · Ridge · Lasso · 나무 모델 등을 같은 조건으로 비교");
  await clickSmooth(p, tab("④ 수익률 예측")); await sleep(1800);
  const run = visible(p, "button", { hasText: "모델 실행" }); await clickSmooth(p, run);
  mark("cut"); await waitIdle(p); await p.getByText("성과 지표", { exact: true }).filter({ visible: true }).first().waitFor({ timeout: 120000 }); await waitIdle(p); await sleep(600);
  await scrollTo(p, p.getByText("성과 지표", { exact: true }).filter({ visible: true }).first()); mark("resume");
  await sleep(2200); await wheel(p, 600); await sleep(2200);
  // ③ Portfolio Customization: ⑤ 탭(A가 기본) → 효율적 투자선 3D 돌려 보기
  await caption(p, "③ Portfolio Customization — 전략을 골라 섞고, 효율적 투자선에서 내 위치 확인");
  await p.evaluate(() => document.querySelector('[data-testid="stMain"], section.main')?.scrollTo({ top: 0, behavior: "smooth" })); await sleep(700);
  await clickSmooth(p, tab("⑤ 포트폴리오")); await sleep(2200);
  await wheel(p, 500); await sleep(1500);
  const ef = visible(p, "text=효율적 투자선 — 고른 전략들로"); await scrollTo(p, ef); await sleep(600);
  const plot3 = visible(p, ".js-plotly-plot", {}).locator("xpath=.").last();
  const fr = p.locator(".js-plotly-plot").filter({ visible: true });
  let target = null;
  for (let i = 0; i < await fr.count(); i++) { const b = await fr.nth(i).boundingBox(); if (b && b.y > 150 && b.y < 600 && b.height > 300) { target = b; break; } }
  if (target) {
    const cx = target.x + target.width * 0.45, cy = target.y + target.height * 0.5;
    await p.mouse.move(cx, cy, { steps: 20 }); await p.mouse.down();
    await p.mouse.move(cx + 260, cy + 30, { steps: 60 }); await p.mouse.move(cx - 120, cy - 20, { steps: 60 }); await p.mouse.up(); await sleep(1500);
  }
  void plot3;
  // ④ 레짐 · 칼만 필터: ② 탭 → 모델 = Kalman TVP
  await caption(p, "④ 레짐 · 칼만 필터 — 위기 때 커지는 민감도(β)를 시간에 따라 추적");
  await p.evaluate(() => document.querySelector('[data-testid="stMain"], section.main')?.scrollTo({ top: 0, behavior: "smooth" })); await sleep(700);
  await clickSmooth(p, tab("② 레짐")); await sleep(1800);
  const sel = p.getByRole("combobox", { name: "모델", exact: true }).filter({ visible: true }).first(); await clickSmooth(p, sel); await sleep(700);
  const opt = p.getByRole("option", { name: "Kalman TVP" }); await clickSmooth(p, opt);
  mark("cut"); await waitIdle(p); await sleep(400); mark("resume");
  const runK = visible(p, "button", { hasText: "Kalman TVP 실행" }); await clickSmooth(p, runK);
  mark("cut");
  const beta = p.locator(".gtitle", { hasText: "Kalman 시변 계수" }).first(); await beta.waitFor({ timeout: 120000 }); await waitIdle(p); await sleep(600);
  await scrollTo(p, beta); await wheel(p, -120, 4); mark("resume");
  await sleep(3000);
  const k3 = p.getByText(/3D 보기 — 시변 계수/).filter({ visible: true }).first();
  if (await k3.count()) {                                      // 칼만 β 지형(3D)까지 내려가 한 번 돌려 봄
    await scrollTo(p, k3); await sleep(1200);
    const fr = p.locator(".js-plotly-plot").filter({ visible: true });
    for (let i = 0; i < await fr.count(); i++) {
      const b = await fr.nth(i).boundingBox();
      if (b && b.y > 60 && b.y < 400 && b.height > 300) {
        const cx = b.x + b.width * 0.5, cy = b.y + b.height * 0.5;
        await p.mouse.move(cx, cy, { steps: 20 }); await p.mouse.down(); await p.mouse.move(cx + 240, cy + 20, { steps: 60 }); await p.mouse.up(); break;
      }
    }
    await sleep(2500);
  } else { await wheel(p, 700); await sleep(3200); }
  await caption(p, ""); await sleep(600); mark("end");
}

(async () => {
  fs.rmSync(TMP, { recursive: true, force: true }); fs.mkdirSync(TMP, { recursive: true });
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  // 0) 예열: 같은 동작을 한 번 해 캐시를 채움 (녹화 안 함) → 녹화 때 계산 대기가 짧아짐
  if (!process.env.NO_WARM) {
    console.log("예열 중…");
    const w = await browser.newPage({ viewport: { width: VW, height: VH } });
    await w.goto(URL, { timeout: 180000 }); await w.getByRole("tab", { name: "★ 구매 의사결정" }).waitFor({ timeout: 180000 }); await waitIdle(w, 300000);
    await w.evaluate(OVERLAY); await scenes(w, () => {}); await w.close();
  }
  // 1) 녹화
  console.log("녹화 중…");
  const ctx = await browser.newContext({ viewport: { width: VW, height: VH }, recordVideo: { dir: TMP, size: { width: VW, height: VH } } });
  const p = await ctx.newPage(); const t0 = Date.now(); const marks = [];
  const mark = (k) => marks.push([k, (Date.now() - t0) / 1000]);
  await p.goto(URL, { timeout: 180000 }); await p.getByRole("tab", { name: "★ 구매 의사결정" }).waitFor({ timeout: 180000 }); await waitIdle(p, 300000); await sleep(1500);
  await p.evaluate(OVERLAY);
  await scenes(p, mark);
  const vpath = await p.video().path(); await ctx.close(); await browser.close();
  // 2) 잘라 붙이기: start~cut, resume~cut, …, resume~end
  const seg = []; let a = null;
  for (const [k, t] of marks) { if (k === "start" || k === "resume") a = t; else if (k === "cut" || k === "end") { seg.push([a, t]); a = null; } }
  const expr = seg.map(([s, e]) => `between(t,${s.toFixed(2)},${e.toFixed(2)})`).join("+");
  const out = path.join(OUTDIR, "대시보드_시연_초안.mp4");
  execFileSync(FFMPEG, ["-y", "-i", vpath, "-vf", `select='${expr}',setpts=N/FRAME_RATE/TB,fps=30`, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out], { stdio: "ignore" });
  const dur = seg.reduce((s, [x, y]) => s + (y - x), 0);
  console.log(`완료: ${out}  (약 ${dur.toFixed(0)}초, 구간 ${seg.map(([x, y]) => `${x.toFixed(1)}–${y.toFixed(1)}`).join(", ")})`);
  fs.rmSync(TMP, { recursive: true, force: true });
})().catch((e) => { console.error(e); process.exit(1); });
