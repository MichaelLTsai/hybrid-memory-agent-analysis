#!/usr/bin/env python
"""The StructMem synthesis page for the results deck.

    ./venv_memos/bin/python make_structmem_synthesis_slide.py

Writes structmem_synthesis.pptx. Styled to sit with the five existing result
slides (End-to-End / Summary / Storage / Retrieval / Reasoning): same nav strip
with its progress fill, same title and italic subtitle, same booktabs table
with a spanning group header, same Note convention (bold best, underline
second best).

Where the numbers come from
---------------------------
Every value is a cell from those five slides, cross-checked against
memory_failure_matrix.xlsx sheet "Failure Matrix", batch 2 (08-19):

    row  9  Mem0 v1 · 0819
    row 10  Mem0 v2 · 0819
    row 14  StructMem E0 baseline   <- replaces row 11, whose run had an
                                       ingestion fault
    row 12  A-MEM · 0819
    row 13  Letta · 0819

Rank is StructMem's position among those five on that metric, recomputed here
from the full column rather than copied, so it cannot drift from the values.

One correction to the source deck
---------------------------------
The Storage slide prints StructMem's LongMemEval KU Accuracy as 0.4000. The
workbook holds 0.6000 in both places it appears (col 16 "LongMemEval KU acc"
and col 96 "[4] knowledge-update"), and the End-to-End slide prints
LongMemEval QA 0.6000 for StructMem. Since StructMem's LongMemEval run covers
only the 5 knowledge-update questions, QA and KU accuracy are by construction
the same number, so 0.4000 cannot be right. This slide uses 0.6000 and says so
in the Note. It changes StructMem's rank on that metric from 5th to 2nd, so it
is not cosmetic: if the deck is right and the workbook is wrong, flip
LME_KU_VALUE below and the ranks recompute.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

C = {
    "ink":   RGBColor(0x00, 0x00, 0x00),
    "ink2":  RGBColor(0x3A, 0x3A, 0x3A),
    "ink3":  RGBColor(0x8A, 0x8A, 0x8A),
    "rule":  RGBColor(0x00, 0x00, 0x00),
    "navon": RGBColor(0x8A, 0xA9, 0xF7),
    "navoff": RGBColor(0xF0, 0xF1, 0xF3),
    "navline": RGBColor(0x00, 0x00, 0x00),
}
FONT = "Helvetica Neue"
MONO = "Menlo"

W, H = 13.333, 7.5

#: See the module docstring. 0.6000 = workbook; 0.4000 = the Storage slide.
LME_KU_VALUE = 0.6000


def tbox(sl, x, y, w, h, text, size=11, bold=False, color=None, align=PP_ALIGN.LEFT,
         font=FONT, italic=False, underline=False):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.underline = underline
    r.font.color.rgb = color or C["ink"]
    return tb


def runs_box(sl, x, y, w, h, runs, align=PP_ALIGN.LEFT, line_spacing=1.0):
    """A text box built from (text, size, bold, italic, underline, colour, font)."""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    for text, size, bold, italic, underline, colour, font in runs:
        r = p.add_run()
        r.text = text
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.underline = underline
        r.font.color.rgb = colour
    return tb


def rule(sl, x1, y, x2, width=1.0, color=None):
    cx = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                 Inches(x1), Inches(y), Inches(x2), Inches(y))
    cx.line.color.rgb = color or C["rule"]
    cx.line.width = Pt(width)
    return cx


def nav(sl, progress=0.85):
    """Section strip. Sections before the active one are filled; the active one
    is filled to `progress` of its width, matching the source deck."""
    items = [("Problem & Motivation", 1.89), ("RQ", 1.04), ("Methodology", 2.79),
             ("Experimental Results & Analysis", 2.31), ("Proposed Improvements", 3.11),
             ("Validation & Conclusions", 2.19)]
    active = 3
    x = 0.0
    for i, (text, w) in enumerate(items):
        s = sl.shapes.add_shape(1, Inches(x), Inches(0.0), Inches(w), Inches(0.30))
        s.fill.solid()
        s.fill.fore_color.rgb = C["navon"] if i < active else C["navoff"]
        s.line.color.rgb = C["navline"]
        s.line.width = Pt(0.75)
        s.shadow.inherit = False
        tf = s.text_frame
        tf.margin_left = tf.margin_right = Pt(2)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(10)
        r.font.color.rgb = C["ink"]

        if i == active and progress > 0:
            f = sl.shapes.add_shape(1, Inches(x), Inches(0.0),
                                    Inches(w * progress), Inches(0.30))
            f.fill.solid()
            f.fill.fore_color.rgb = C["navon"]
            f.line.fill.background()
            f.shadow.inherit = False
            # Behind the label, in front of the cell.
            sl.shapes._spTree.remove(f._element)
            sl.shapes._spTree.insert(2, f._element)
        x += w


# ── The five backends, in the deck's row order ──────────────────────────────
BACKENDS = ["Mem0 v1", "Mem0 v2", "StructMem", "A-MEM", "Letta"]
SM = 2                                            # StructMem's index

#: (stage, metric label, lower_is_better, [LongMemEval, LoCoMo, HaluMem, MemFail])
#: Each inner list is the full column across the five backends, or None where
#: the benchmark defines no such metric. MemFail's official Summary / Retrieval
#: / Reasoning errors sit on the P1 / P4 / P5 rows, exactly as the source deck
#: pairs them.
TABLE = [
    ("End-to-end", "QA accuracy / correct ↑", False, [
        [0.4545, 0.4091, 0.6000, 0.5000, 0.4091],
        [0.4422, 0.6181, 0.6432, 0.4925, 0.6080],
        [0.2972, 0.3528, 0.5305, 0.4889, 0.5056],
        [0.7429, 0.7714, 0.8571, 0.7714, 0.5143]]),
    ("Summary", "P1 / summary error ↓", True, [
        [0.0909, 0.0909, 0.0000, 0.2273, 0.5455],
        [0.2640, 0.2525, 0.0573, 0.2256, 0.2374],
        [0.2483, 0.1655, 0.0480, 0.0414, 0.1483],
        [0.1143, 0.0571, 0.0000, 0.0000, 0.0000]]),
    ("Summary", "Extraction F1 ↑", False, [
        None,
        [0.6481, 0.7796, 0.8845, 0.7608, 0.7855],
        [0.6777, 0.8830, 0.9339, 0.9155, 0.8194],
        None]),
    ("Storage", "Update accuracy ↑", False, [
        [0.4286, 0.4286, LME_KU_VALUE, 0.5714, 0.7143],
        None,
        [0.5251, 0.7659, 0.7183, 0.8060, 0.5853],
        None]),
    ("Storage", "Storage error ↓", True, [
        None, None, None,
        [0.0000, 0.0286, 0.0000, 0.0000, 0.3429]]),
    ("Retrieval", "P4 / retrieval error ↓", True, [
        [0.0909, 0.1364, 0.0000, 0.0000, 0.0000],
        [0.1066, 0.0455, 0.0521, 0.1282, 0.0000],
        [0.2207, 0.2310, 0.2000, 0.1103, 0.0862],
        [0.0571, 0.0571, 0.0286, 0.0857, 0.0000]]),
    ("Reasoning", "P5 / reasoning error ↓", True, [
        [0.3636, 0.3636, 0.4000, 0.2727, 0.0455],
        [0.1827, 0.0808, 0.2240, 0.1436, 0.1515],
        [0.3897, 0.3897, 0.3440, 0.4586, 0.3621],
        [0.0857, 0.0857, 0.1143, 0.1429, 0.1429]]),
]

BENCH = ["LongMemEval", "LoCoMo", "HaluMem", "MemFail"]
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def rank_of(column, i, lower_is_better):
    """Competition rank of entry i, ties sharing the better position."""
    v = column[i]
    better = sum(1 for x in column
                 if (x < v if lower_is_better else x > v))
    return better + 1


FINDINGS = [
    ("End to end, StructMem wins every benchmark.",
     "LongMemEval 0.6000, LoCoMo 0.6432, HaluMem 0.5305, MemFail 0.8571: first "
     "place on all four. The summary stage explains most of it, with first place "
     "on five of its six metrics and a LoCoMo P1 failure of 0.0573 against 0.2256 "
     "to 0.2640 for everyone else. There is nothing left to win on the write path."),
    ("Reasoning is the only stage where it finishes last, and it finishes last twice.",
     "LongMemEval P5 0.4000 and LoCoMo P5 0.2240, the worst of the five on both, "
     "the latter nearly three times Mem0 v2's 0.0808. Storage is the other soft "
     "spot: it leads neither update metric, sitting 2nd on LongMemEval KU and 3rd "
     "on HaluMem update (0.7183 against A-MEM's 0.8060)."),
    ("Retrieval is not the problem, so the stale value is reaching the answer.",
     "LongMemEval P4 is 0.0000 and LoCoMo P4 is 0.0521: the evidence arrives. What "
     "arrives with it is the issue. StructMem writes 1.85 entries per LoCoMo turn "
     "against 0.06 to 1.00 for the others and retires none of them, so the old "
     "value and the new one reach the answering model side by side, unlabelled."),
    ("Therefore.",
     "The timestamp that orders those two entries is already on every StructMem "
     "payload and is never consulted at update time. M1 marks the old entry "
     "superseded instead of deleting it, so the extraction lead above is not "
     "traded away; M3 keeps the summaries in step; M4 uses the labels at read "
     "time, which is where P5 breaks."),
]


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    nav(sl)

    tbox(sl, 0.55, 0.72, 12.2, 0.50,
         "Experimental Results:  StructMem, Stage by Stage", size=26, bold=True)
    tbox(sl, 0.55, 1.26, 12.2, 0.26,
         "First on every end-to-end metric, and last on the stage that has to know "
         "which value is current", size=11.5, color=C["ink3"], italic=True)

    # ── Table ───────────────────────────────────────────────────────────────
    x0 = 0.55
    cw = [1.35, 2.55, 2.05, 2.05, 2.05, 2.18]
    xs, xx = [], x0
    for w in cw:
        xs.append(xx); xx += w
    right = xx
    val_left = xs[2]

    y = 1.88
    rule(sl, x0, y, right, 2.0)

    # Spanning group header over the value columns only, with its own rule.
    tbox(sl, val_left, y + 0.05, right - val_left, 0.22,
         "StructMem, and its rank among the five backends",
         size=11, bold=True, align=PP_ALIGN.CENTER)
    rule(sl, val_left, y + 0.30, right, 0.9)

    tbox(sl, xs[0], y + 0.16, cw[0], 0.22, "Stage", size=10.5, bold=True)
    tbox(sl, xs[1], y + 0.16, cw[1], 0.22, "Metric", size=10.5, bold=True)
    for b, x, w in zip(BENCH, xs[2:], cw[2:]):
        tbox(sl, x, y + 0.36, w, 0.20, b, size=10, bold=True, align=PP_ALIGN.CENTER)
    y += 0.62
    rule(sl, x0, y, right, 2.0)

    prev_stage = None
    for stage, metric, low, columns in TABLE:
        y += 0.10
        tbox(sl, xs[0], y, cw[0], 0.24,
             stage if stage != prev_stage else "", size=10.5, bold=True)
        prev_stage = stage
        tbox(sl, xs[1], y, cw[1], 0.24, metric, size=10, color=C["ink2"])

        for k, col in enumerate(columns):
            if col is None:
                tbox(sl, xs[2 + k], y, cw[2 + k], 0.24, "—", size=11,
                     font=MONO, align=PP_ALIGN.CENTER, color=C["ink3"])
                continue
            v = col[SM]
            r = rank_of(col, SM, low)
            runs_box(sl, xs[2 + k], y, cw[2 + k], 0.24, [
                (f"{v:.4f}", 11, r == 1, False, r == 2, C["ink"], MONO),
                (f"   {ORDINAL[r]}", 9, False, False, False, C["ink3"], FONT),
            ], align=PP_ALIGN.CENTER)
        y += 0.24
    y += 0.10
    rule(sl, x0, y, right, 2.0)

    # ── Findings ────────────────────────────────────────────────────────────
    fy = y + 0.22
    for head, body in FINDINGS:
        runs_box(sl, x0, fy, 12.2, 0.34, [
            (head + "  ", 9.5, True, False, False, C["ink"], FONT),
            (body, 9.5, False, False, False, C["ink2"], FONT),
        ], line_spacing=1.12)
        fy += 0.36

    # ── Note ────────────────────────────────────────────────────────────────
    tbox(sl, x0, 7.06, 12.2, 0.34,
         "Note. ↑ higher is better; ↓ lower is better. Best results are bold; "
         "second-best results are underlined; ties share the better rank. Dashes "
         "mark metrics the benchmark does not define. StructMem's LongMemEval run "
         "covers only the 5-question knowledge-update subset, so its LongMemEval QA "
         "and KU accuracy are the same 0.6000; the Storage slide's 0.4000 is a "
         "transcription error and is corrected here.",
         size=8.5, color=C["ink3"], italic=True)

    out = "structmem_synthesis.pptx"
    prs.save(out)
    print(f"Saved {out}")

    # Ranks are recomputed, so print them for checking against the deck.
    for stage, metric, low, columns in TABLE:
        cells = []
        for b, col in zip(BENCH, columns):
            if col is None:
                cells.append(f"{b}=—")
            else:
                cells.append(f"{b}={col[SM]:.4f}({ORDINAL[rank_of(col, SM, low)]})")
        print(f"  {stage:11s} {metric:26s} " + "  ".join(cells))


if __name__ == "__main__":
    build()
