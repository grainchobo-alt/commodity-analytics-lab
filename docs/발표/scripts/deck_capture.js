// deck_v{N}.js를 그대로 실행하되 pptxgenjs 대신 '기록기'를 넣어 장마다 요소(글상자·도형·그림·표·차트·노트)를 JSON으로 저장.
// 사용: node deck_capture.js 4 out.json   → html_v4.py가 이 JSON으로 HTML 장을 같은 배치로 다시 그림
const Module = require("module");
const fs = require("fs");
const path = require("path");
const [ver, out] = [process.argv[2] || "4", process.argv[3] || path.join(__dirname, "capture.json")];

class Recorder {
  constructor() {
    this.slides = [];
    this.shapes = { RECTANGLE: "rect", ROUNDED_RECTANGLE: "roundRect", OVAL: "ellipse", LINE: "line" };
    this.charts = { BAR: "bar", LINE: "line" };
  }
  addSlide() {
    const s = { background: null, notes: "", items: [] }, push = (o) => s.items.push(o);
    this.slides.push(s);
    return {
      set background(v) { s.background = v; }, get background() { return s.background; },
      addText: (text, opts) => push({ kind: "text", text, opts }),
      addShape: (type, opts) => push({ kind: "shape", type, opts }),
      addImage: (opts) => push({ kind: "image", opts }),
      addTable: (rows, opts) => push({ kind: "table", rows, opts }),
      addChart: (type, data, opts) => push({ kind: "chart", type, data, opts }),
      addMedia: (opts) => push({ kind: "media", opts }),
      addNotes: (t) => { s.notes = t; },
    };
  }
  writeFile() { fs.writeFileSync(out, JSON.stringify({ slides: this.slides }), "utf8"); return Promise.resolve(out); }
}
const load = Module._load;
Module._load = function (req, ...rest) { return req === "pptxgenjs" ? Recorder : load.call(this, req, ...rest); };
process.argv[2] = out;                                                  // deck이 OUT으로 쓰는 인자
require(path.join(__dirname, `deck_v${ver}.js`));
