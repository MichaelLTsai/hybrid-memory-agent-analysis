#!/usr/bin/env python
"""Table 1 with the stage breakdown: P1, P4, P5 and QA per functional group.

    ./venv_memos/bin/python make_stage_breakdown_table.py

Writes stage_breakdown_table.pptx: one slide holding a real PowerPoint table
(add_table, not text boxes) so it can be pasted into another deck and edited
there.

Two blocks of four columns. Reading a block left to right gives the
decomposition and then the outcome it explains: P1 + P4 + P5 + QA sums to 1.000
on every row, because the short-circuit attribution assigns each question to
exactly one stage and the three failure rates are the complement of accuracy.

Subsets are pooled by question count, which is the only pooling with a clean
reading: the cell is the rate over the union of that group's questions.

MemFail long_hop is left out of Multi-hop composition here. It reports the
benchmark's own summary_error, storage_error, retr_error and reason_error
rather than this study's probe stages, so it can contribute an accuracy but no
stage rates. Including it in QA alone would put QA over 62 questions while the
stages ran over 57, and the row would stop summing to one. The cost is that
this table's Multi-hop QA differs from the accuracy-only table, which does
include long_hop: Letta 0.421 here against 0.452 there. Temporal reasoning has
no MemFail subset, so that block matches the accuracy-only table exactly.

Numbers: memory_failure_matrix.xlsx, sheet "By Group", batch (6) "31B compare",
the batch where all five backends share gemma-4-31B-it.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

INK = RGBColor(0x14, 0x1E, 0x27)
INK2 = RGBColor(0x4C, 0x5C, 0x6B)
INK3 = RGBColor(0x7E, 0x8D, 0x9B)
ACCENT = RGBColor(0x16, 0x68, 0x5A)
WARN = RGBColor(0xAC, 0x47, 0x12)
BAND = RGBColor(0xF3, 0xF6, 0xFA)
WARNBAND = RGBColor(0xFA, 0xEE, 0xE6)
GOODBAND = RGBColor(0xE6, 0xF2, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT, MONO = "Helvetica Neue", "Menlo"

BACKENDS = ["Mem0 v1", "Mem0 v2", "StructMem", "A-MEM", "Letta"]
METRICS = [("P1 ↓", True), ("P4 ↓", True), ("P5 ↓", True), ("QA ↑", False)]
GROUPS = [
    ("Multi-hop composition", "LoCoMo multi-hop + HaluMem multi-hop", 57),
    ("Temporal reasoning",    "LongMemEval + LoCoMo temporal",        40),
]
# Question-count weighted over each group's subsets. Each row sums to 1.000.
DATA = {
    "Mem0 v1":   [0.211, 0.158, 0.351, 0.281,   0.925, 0.050, 0.025, 0.000],
    "Mem0 v2":   [0.088, 0.140, 0.439, 0.333,   0.900, 0.025, 0.000, 0.075],
    "StructMem": [0.055, 0.091, 0.527, 0.327,   0.075, 0.000, 0.100, 0.825],
    "A-MEM":     [0.088, 0.263, 0.439, 0.211,   0.850, 0.025, 0.100, 0.025],
    "Letta":     [0.105, 0.105, 0.368, 0.421,   0.775, 0.000, 0.200, 0.025],
}
WARN_CELLS = {("Letta", 4), ("Letta", 7)}        # Letta's temporal write failure
GOOD_CELLS = {("StructMem", 4), ("StructMem", 7)}  # the architecture that writes time
ROW_TINT = "Letta"


def edge(cell, name, width_pt, hexcolor):
    """Set one border on a cell; python-pptx exposes no API for this."""
    tcPr = cell._tc.get_or_add_tcPr()
    tag = qn(f"a:{name}")
    for old in tcPr.findall(tag):
        tcPr.remove(old)
    ln = tcPr.makeelement(tag, {"w": str(int(Pt(width_pt).emu)),
                                "cap": "flat", "cmpd": "sng", "algn": "ctr"})
    fill = ln.makeelement(qn("a:solidFill"), {})
    fill.append(ln.makeelement(qn("a:srgbClr"), {"val": hexcolor}))
    ln.append(fill)
    order = ["lnL", "lnR", "lnT", "lnB", "lnTlToBr", "lnBlToTr"]
    after = None
    for later in order[order.index(name) + 1:]:
        found = tcPr.find(qn(f"a:{later}"))
        if found is not None:
            after = found
            break
    tcPr.insert(list(tcPr).index(after) if after is not None else 0, ln)


def strip_edges(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for name in ("lnL", "lnR", "lnT", "lnB"):
        for old in tcPr.findall(qn(f"a:{name}")):
            tcPr.remove(old)


def write(cell, lines, size=9, bold=False, color=INK, font=FONT,
          align=PP_ALIGN.CENTER, fill=WHITE):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = cell.margin_right = Inches(0.02)
    cell.margin_top = cell.margin_bottom = Inches(0.015)
    tf = cell.text_frame
    tf.word_wrap = True
    if isinstance(lines, str):
        lines = [(lines, size, bold, color, font)]
    for i, spec in enumerate(lines):
        text, sz, bd, cl, fn = spec
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.name = fn
        r.font.size = Pt(sz)
        r.font.bold = bd
        r.font.color.rgb = cl


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    ncol = 1 + 4 * len(GROUPS)
    nrow = 2 + len(BACKENDS)
    gf = sl.shapes.add_table(nrow, ncol, Inches(1.20), Inches(1.86),
                             Inches(10.90), Inches(2.9))
    tbl = gf.table
    tbl.first_row = tbl.first_col = False
    tbl.horz_banding = tbl.vert_banding = False

    tbl.columns[0].width = Inches(1.70)
    for c in range(1, ncol):
        tbl.columns[c].width = Inches(1.15)
    for r, h in enumerate([0.46, 0.30] + [0.36] * len(BACKENDS)):
        tbl.rows[r].height = Inches(h)
    for r in range(nrow):
        for c in range(ncol):
            strip_edges(tbl.cell(r, c))

    # Row 0: the functional group, merged over its four measures, with the
    # subsets it pools and the question count it pools them over.
    write(tbl.cell(0, 0), [("", 9, False, INK, FONT)])
    for i, (name, subs, n) in enumerate(GROUPS):
        cell = tbl.cell(0, 1 + 4 * i)
        cell.merge(tbl.cell(0, 4 + 4 * i))
        write(cell, [(name, 12.5, True, ACCENT, FONT),
                     (f"{subs}   ·   micro-average, n = {n}",
                      8.5, False, INK3, FONT)])
        edge(cell, "lnB", 1.0, "16685A")

    # Row 1: the four measures. The stages are set softer than the outcome they
    # decompose, so a reader's eye lands on QA first and walks back.
    write(tbl.cell(1, 0), [("Backend", 10.5, True, INK2, FONT)],
          align=PP_ALIGN.LEFT)
    for i in range(len(GROUPS)):
        for j, (label, _lo) in enumerate(METRICS):
            write(tbl.cell(1, 1 + 4 * i + j),
                  [(label, 10.5, True, INK if j == 3 else INK2, MONO)])
    for c in range(ncol):
        edge(tbl.cell(1, c), "lnB", 1.5, "141E27")

    best = []
    for j, (_l, lower) in enumerate(METRICS * len(GROUPS)):
        vals = [DATA[b][j] for b in BACKENDS]
        best.append(min(vals) if lower else max(vals))

    for i, b in enumerate(BACKENDS):
        r = 2 + i
        tint = BAND if b == ROW_TINT else WHITE
        write(tbl.cell(r, 0), [(b, 11, True, INK, FONT)],
              align=PP_ALIGN.LEFT, fill=tint)
        for j, v in enumerate(DATA[b]):
            warn = (b, j) in WARN_CELLS
            good = (b, j) in GOOD_CELLS
            is_qa = j % 4 == 3
            colour = WARN if warn else (ACCENT if good else
                                        (INK if is_qa else INK2))
            write(tbl.cell(r, j + 1), [(
                f"{v:.3f}", 11,
                warn or good or abs(v - best[j]) < 1e-9, colour, MONO)],
                fill=WARNBAND if warn else (GOODBAND if good else tint))
        if i == len(BACKENDS) - 1:
            for c in range(ncol):
                edge(tbl.cell(r, c), "lnB", 1.5, "141E27")

    # One hairline between the two blocks
    for r in range(1, nrow):
        edge(tbl.cell(r, 4), "lnR", 0.75, "D3DCD8")

    for y, text, size, bold, color, italic in [
        (1.32, "Table 1.  Stage breakdown by functional group  "
               "(batch 6, all backends on gemma-4-31B-it)", 14, True, INK, False),
        (5.28, "P1 + P4 + P5 + QA sums to 1.000 on every row: each question is "
               "assigned to exactly one stage, so the three failure rates are the "
               "complement of accuracy. Bold is the best value in a column. Every "
               "cell is a micro-average: correct answers and questions are summed "
               "across the subsets before dividing, so each subset weighs exactly "
               "as much as the questions it holds. MemFail long_hop is excluded from "
               "multi-hop because it reports the benchmark's own error categories "
               "rather than probe stages, so it can contribute accuracy but no "
               "stages; temporal has no MemFail subset and is unaffected.",
         9.5, False, INK3, True),
    ]:
        tb = sl.shapes.add_textbox(Inches(1.20), Inches(y), Inches(10.90), Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = 0
        p = tf.paragraphs[0]
        p.line_spacing = 1.25
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color

    out = "stage_breakdown_table.pptx"
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    build()
