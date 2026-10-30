// 유리판 PPT: deck_v4.js를 그대로 실행하되 pptxgenjs를 감싸 배경·카드·제목만 유리 + 에디토리얼(세리프 제목)로 바꿈.
// 내용·배치는 기본판과 같음 → deck_v4.js를 고치면 두 판 모두 다시 만들 것. 배경 그림(img/glass_bg*.png)은 build.py --glass가 생성.
// 실행: node deck_v4_glass.js 출력.pptx   (보통은 build.py v4 --glass)
const path = require("path");
const Real = require("pptxgenjs");

const BG = path.join(__dirname, "img", "glass_bg.png"), BG_DARK = path.join(__dirname, "img", "glass_bg_dark.png");
const SERIF = "Batang";
const GLASS_FILL = { FFFFFF: 38, F1F3EF: 45, FBF4E1: 30, FBEDE7: 35 };     // 반투명으로 바꿀 상자 색 → 투명도(%)
const SHADOW = () => ({ type: "outer", color: "1E3A2F", opacity: 0.14, blur: 10, offset: 3, angle: 90 });            // pptxgenjs가 넘긴 객체를 제자리에서 단위 변환하므로 매번 새로 만듦

function glassSlide(s) {
  const bgDesc = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(s), "background");
  let dark = false;
  Object.defineProperty(s, "background", {
    get: () => bgDesc.get.call(s),
    set: (v) => { dark = !!v && v.color && v.color !== "FFFFFF"; bgDesc.set.call(s, { path: dark ? BG_DARK : BG }); },
  });
  const addShape = s.addShape.bind(s), addText = s.addText.bind(s), addImage = s.addImage.bind(s), addTable = s.addTable.bind(s);
  s.addShape = (shape, o = {}) => {
    const c = o.fill && o.fill.color;
    if (!dark && (shape === "rect" || shape === "roundRect") && c in GLASS_FILL && o.w > 0.6 && o.h > 0.35) {
      o = Object.assign({}, o, {
        fill: { color: c === "F1F3EF" ? "FFFFFF" : c, transparency: GLASS_FILL[c] },
        line: { color: c === "FBF4E1" ? "C8A24A" : "FFFFFF", transparency: c === "FBF4E1" ? 40 : 15, width: 1 },
        shadow: SHADOW(), rectRadius: o.rectRadius || 0.12,
      });
      shape = "roundRect";
    }
    return addShape(shape, o);
  };
  s.addText = (t, o = {}) => {
    if ((o.fontSize >= 24 && o.y < 1.1) || o.fontSize >= 36) o = Object.assign({}, o, { fontFace: SERIF });   // 제목만 세리프
    return addText(t, o);
  };
  s.addImage = (o) => {                                                       // 그림 뒤에 유리 액자
    if (!dark && o.w > 1.2 && o.h > 0.9)
      addShape("roundRect", { x: o.x - 0.05, y: o.y - 0.05, w: o.w + 0.1, h: o.h + 0.05, rectRadius: 0.1,   // 아래로는 넘치지 않게(바로 밑 설명 글 가림 방지)
        fill: { color: "FFFFFF", transparency: 25 }, line: { color: "FFFFFF", transparency: 10, width: 1 }, shadow: SHADOW() });
    return addImage(o);
  };
  s.addTable = (rows, o) => addTable(rows.map((r) => r.map((cell) => {         // 표: 흰·연회색 칸을 반투명으로
    const f = cell.options && cell.options.fill && cell.options.fill.color;
    if (f === "FFFFFF" || f === "F1F3EF")
      return Object.assign({}, cell, { options: Object.assign({}, cell.options, { fill: { color: "FFFFFF", transparency: f === "FFFFFF" ? 45 : 20 } }) });
    return cell;
  })), o);
  return s;
}

class GlassPres extends Real {
  addSlide(o) { return glassSlide(super.addSlide(o)); }
}
require.cache[require.resolve("pptxgenjs")].exports = GlassPres;
require(path.join(__dirname, "deck_v4.js"));
