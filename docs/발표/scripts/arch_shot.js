// 별첨 6(시스템 구조도) PPT용 그림: docs/architecture.html의 구조도(svg#arch)를 PNG로 캡처 → img/arch_map.png
// 실행: node arch_shot.js   (구조도를 고친 뒤 다시 실행. HTML 발표는 그림 대신 구조도 페이지를 그대로 넣음)
const path = require("path");
const { chromium } = require("playwright");
(async () => {
  const b = await chromium.launch({ channel: "msedge" });
  const p = await b.newPage({ viewport: { width: 1720, height: 2600 }, deviceScaleFactor: 1.5 });
  await p.goto(require("url").pathToFileURL(path.resolve(__dirname, "../../architecture.html")).href);
  await p.addStyleTag({ content: ".svgbox{max-height:none!important;overflow:visible!important}" });
  await p.waitForTimeout(1500);
  await p.locator("svg#arch").screenshot({ path: path.join(__dirname, "img", "arch_map.png") });
  console.log("img/arch_map.png");
  await b.close();
})();
