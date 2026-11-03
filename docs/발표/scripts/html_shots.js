// HTML 발표(reveal.js)를 장마다 PNG로 캡처 → html_pdf.py가 PDF로 묶음 (HTML의 Plotly 3D 화면 그대로)
// 사용: node html_shots.js <발표.html> <저장 폴더>
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");
const [html, out] = [path.resolve(process.argv[2]), path.resolve(process.argv[3])];
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ channel: "msedge" });
  const p = await b.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1.5 });
  await p.goto(pathToFileURL(html).href);
  await p.waitForFunction(() => window.Reveal && Reveal.isReady && Reveal.isReady());
  await p.addStyleTag({ content: ".reveal .controls,.reveal .progress,.arch-nav,.spin-tip{display:none!important}" });   // 넘김 단추·진행 막대·회전 안내는 PDF에서 숨김
  await p.evaluate(() => Reveal.configure({ transition: "none", backgroundTransition: "none" }));
  const n = await p.evaluate(() => Reveal.getTotalSlides());
  for (let i = 0; i < n; i++) {
    await p.evaluate((k) => Reveal.slide(k), i);
    await p.waitForTimeout(2500);                                                                   // Plotly 그리기·구조도 맞춤 대기
    await p.screenshot({ path: path.join(out, `s${String(i + 1).padStart(2, "0")}.png`) });
    process.stdout.write(`${i + 1} `);
  }
  console.log(`\n${n}장 → ${out}`);
  await b.close();
})();
