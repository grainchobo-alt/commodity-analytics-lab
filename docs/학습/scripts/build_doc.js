// content.json + img/*.png → docs/학습/모델도감_학습노트.docx  (docx 패키지는 docs/발표/scripts/node_modules 재사용)
const fs = require("fs"), path = require("path");
const D = require(require.resolve("docx", { paths: [path.join(__dirname, "..", "..", "발표", "scripts")] }));
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, AlignmentType,
  HeadingLevel, LevelFormat, PageBreak, TableOfContents, Footer, PageNumber } = D;

const HERE = __dirname, IMG = path.join(HERE, "img");
const OUT = process.argv[2] || path.join(HERE, "..", "모델도감_학습노트.docx");
const blocks = JSON.parse(fs.readFileSync(path.join(HERE, "content.json"), "utf8"));
const F = "Malgun Gothic", INK = "222222", MUTED = "6B6B66", G = "1E3A2F";
const PAGE_W = 11906, MARGIN = 1134, CW = PAGE_W - 2 * MARGIN;          // A4, 좌우 2cm → 본문 폭 9638 DXA (6.69인치)
const BOX = { key: ["E8F0EC", G], ex: ["EAF2F7", "2F6E8F"], warn: ["FBEFE6", "D9582B"], fix: ["FFF6DA", "B8901C"] };

function runs(text, base = {}) {                                         // **굵게** 표기 → TextRun
  return String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true, ...base }) : new TextRun({ text: s, ...base }));
}
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function image(file, wIn, align = AlignmentType.CENTER, after = 60) {
  const [pw, ph] = pngSize(file), w = Math.min(wIn, 6.6) * 96;
  return new Paragraph({ alignment: align, spacing: { before: 60, after }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(w), height: Math.round(w * ph / pw) } })] });
}
const border = (c = "BFBFB8") => ({ style: BorderStyle.SINGLE, size: 4, color: c });
function table(rows, widths, small, hdr = true) {
  const n = rows[0].length, ws = widths || Array(n).fill(1), tot = ws.reduce((a, b) => a + b, 0);
  const cols = ws.map((w) => Math.floor(CW * w / tot)); cols[n - 1] += CW - cols.reduce((a, b) => a + b, 0);
  const fs_ = small ? 17 : 19;
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: cols,
    rows: rows.map((r, k) => { const i = hdr ? k : k + 1; return new TableRow({ tableHeader: i === 0, cantSplit: true, children: r.map((c, j) => new TableCell({
      width: { size: cols[j], type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 90, right: 90 },
      borders: { top: border(), bottom: border(), left: border(), right: border() },
      shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: G } : (j === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "F3F3EF" } : undefined),
      children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: runs(c, { size: fs_, font: F, color: i === 0 ? "FFFFFF" : INK, bold: i === 0 || (j === 0 && !hdr) || undefined }) })] })) }); }) });
}
function box(kind, title, items) {
  const [fill, col] = BOX[kind] || BOX.key;
  const kids = [new Paragraph({ spacing: { before: 0, after: 60 }, children: [new TextRun({ text: title, bold: true, color: col, size: 20, font: F })] })]
    .concat(items.map((t) => new Paragraph({ spacing: { before: 0, after: 50 }, numbering: items.length > 1 ? { reference: "dash", level: 0 } : undefined, children: runs(t, { size: 19, font: F }) })));
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: true, children: [new TableCell({
    width: { size: CW, type: WidthType.DXA }, margins: { top: 100, bottom: 100, left: 180, right: 160 },
    shading: { type: ShadingType.CLEAR, color: "auto", fill }, borders: { top: none, bottom: none, right: none, left: { style: BorderStyle.SINGLE, size: 24, color: col } }, children: kids })] })] });
}
const gap = () => new Paragraph({ spacing: { before: 0, after: 80 }, children: [] });

const body = [], numRefs = [];
for (const b of blocks) {
  switch (b.t) {
    case "title":
      body.push(new Paragraph({ spacing: { before: 2800, after: 300 }, children: [new TextRun({ text: b.text, bold: true, size: 60, color: G, font: F })] }),
        new Paragraph({ spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "C8A24A", space: 8 } }, children: [new TextRun({ text: b.sub, size: 26, color: INK, font: F })] }),
        image(path.join(IMG, "gnn_pipeline.png"), 6.0), new Paragraph({ spacing: { before: 400 }, children: [new TextRun({ text: b.meta, size: 20, color: MUTED, font: F })] }),
        new Paragraph({ children: [new PageBreak()] }));
      break;
    case "toc":
      body.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "목차", bold: true, size: 32, color: G, font: F })] }),
        new TableOfContents("목차", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({ children: [new TextRun({ text: "※ 목차가 비어 보이면 Word에서 목차를 클릭 → '필드 업데이트'를 누르세요.", size: 16, color: MUTED, font: F })] }));
      break;
    case "h1": body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b.text)] })); break;
    case "h2": body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b.text)] })); break;
    case "h3": body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(b.text)] })); break;
    case "p": body.push(new Paragraph({ spacing: { after: 100 }, indent: b.muted ? { left: 360 } : undefined,
      children: runs(b.text, b.muted ? { color: MUTED, size: 19, italics: true } : {}) })); break;
    case "ul": case "ol": {
      const ref = b.t === "ul" ? "bullet" : `num${body.length}`;
      if (b.t === "ol") numRefs.push(ref);
      b.items.forEach((t) => body.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: runs(t) })));
      break;
    }
    case "eq": body.push(image(path.join(IMG, "eq", b.name + ".png"), b.w, AlignmentType.CENTER, 100)); break;
    case "img":
      body.push(image(path.join(IMG, b.name + ".png"), b.w),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: b.cap, size: 17, color: MUTED, font: F })] }));
      break;
    case "table": body.push(table(b.rows, b.widths, b.small, b.hdr !== false), gap()); break;
    case "box": body.push(box(b.kind, b.title, b.items), gap()); break;
    case "pb": body.push(new Paragraph({ children: [new PageBreak()] })); break;
  }
}

const num = (ref) => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] });
const doc = new Document({
  creator: "Commodity Analytics Lab", title: "모델 도감 학습 노트",
  styles: {
    default: { document: { run: { font: F, size: 20, color: INK }, paragraph: { spacing: { line: 300 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 34, bold: true, color: G, font: F },
        paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 0, border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: "C8A24A", space: 6 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 27, bold: true, color: G, font: F },
        paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, color: "2F6E8F", font: F },
        paragraph: { spacing: { before: 220, after: 90 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullet", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 280 } } } }] },
    { reference: "dash", levels: [{ level: 0, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 300, hanging: 220 } } } }] },
    ...numRefs.map(num)] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1247, bottom: 1247, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "모델 도감 학습 노트 · ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("saved", OUT, body.length, "elements"); });
