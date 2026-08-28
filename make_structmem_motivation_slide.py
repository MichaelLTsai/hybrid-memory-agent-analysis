#!/usr/bin/env python
"""One slide: what the corrected StructMem baseline says, and why it motivates
the state-bank work.

    ./venv_memos/bin/python make_structmem_motivation_slide.py

Writes structmem_motivation.pptx. Booktabs styling, rules drawn as connectors,
everything native text boxes, so it stays editable in PowerPoint or Slides.
Ranking follows the paper convention: bold is the best value in a column,
underline the second best. No colour coding.

Which rows this slide uses
--------------------------
memory_failure_matrix.xlsx, sheet "Failure Matrix", batch 2 (08-19):

    row  9   Mem0 v1 · 0819          gemma-4-31B-it
    row 10   Mem0 v2 · 0819          gemma-4-31B-it
    row 14   StructMem E0 baseline   gemma-4-E4B-it   <- replaces row 11
    row 12   A-MEM · 0819            gemma-4-31B-it
    row 13   Letta · 0819            openai-proxy/gemma-4-31B-it

Row 11 ("StructMem · 0819") had an ingestion fault and per the user's decision
is replaced by the ablation control E0. build_matrix_excel.py already points
both rows at structmem-ablate_e0_baseline, so rows 11 and 14 now hold the same
values; row 14 is the one named here.

Why HaluMem and LongMemEval are not on this slide
-------------------------------------------------
They are not a like-for-like slice, so ranking five systems on them would be
misleading:

    LongMemEval   StructMem 5 questions, everyone else 22. The knowledge-update
                  subset is 5 questions against 7, so even that is not a shared
                  denominator.
    HaluMem       StructMem ran user #1 (77 sessions, 3,242 turns, 164
                  questions, 142 update points); the other four ran users #3
                  and #4 (6,170 turns, 360 questions, 299 update points). User
                  #1 carries 4 Dynamic Update questions against users #3+#4's
                  37, so StructMem's 0.0000 on that question type is 0 of 4 and
                  carries no weight at all.

    LoCoMo        1 conv (conv-26), 19 sessions, 419 turns, 199 questions.
    MemFail       35 questions across five subsets.
                  Both identical for all five systems. Those are the two the
                  slide ranks on. The extraction LLM still differs (E4B for
                  StructMem, 31B for the rest), which the scope note states.

Column provenance in the workbook
---------------------------------
    LoCoMo   F1 col 8, P1 col 9, P4 col 22, P5 col 28, error col 33,
             QA col 38, temporal col 100, entries/turn col 48
    MemFail  summary col 15, retrieval col 26, reasoning col 30,
             correct col 41, coexisting_facts col 112
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

C = {
    "ink":   RGBColor(0x14, 0x1E, 0x27),
    "ink2":  RGBColor(0x4C, 0x5C, 0x6B),
    "ink3":  RGBColor(0x7E, 0x8D, 0x9B),
    "rule":  RGBColor(0xDB, 0xE2, 0xE9),
    "rule2": RGBColor(0x14, 0x1E, 0x27),
    "navon": RGBColor(0x8A, 0xA9, 0xF7),
    "navoff": RGBColor(0xF0, 0xF1, 0xF3),
    "accent": RGBColor(0x16, 0x68, 0x5A),
    "band":  RGBColor(0xF3, 0xF6, 0xFA),
}
FONT = "Helvetica Neue"
MONO = "Menlo"

W, H = 13.333, 7.5


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


def rule(sl, x1, y, x2, width=1.0, color=None):
    cx = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                 Inches(x1), Inches(y), Inches(x2), Inches(y))
    cx.line.color.rgb = color or C["rule2"]
    cx.line.width = Pt(width)
    return cx


def band(sl, x, y, w, h):
    """Flat tint marking the StructMem row; sent behind the text."""
    s = sl.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = C["band"]
    s.line.fill.background()
    s.shadow.inherit = False
    sl.shapes._spTree.remove(s._element)
    sl.shapes._spTree.insert(2, s._element)
    return s


def nav(sl):
    items = [("Problem & Motivation", 2.05, False), ("RQ", 1.15, False),
             ("Methodology", 2.80, False), ("Experimental Results & Analysis", 2.30, True),
             ("Proposed Improvements", 2.65, False), ("Validation & Conclusions", 2.38, False)]
    x = 0.0
    for text, w, active in items:
        s = sl.shapes.add_shape(1, Inches(x), Inches(0.0), Inches(w), Inches(0.30))
        s.fill.solid()
        s.fill.fore_color.rgb = C["navon"] if active else C["navoff"]
        s.line.color.rgb = C["ink3"]
        s.line.width = Pt(0.5)
        s.shadow.inherit = False
        tf = s.text_frame
        tf.margin_left = tf.margin_right = Pt(2)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(9.5)
        r.font.color.rgb = C["ink"]
        x += w


def rank(values, lower_is_better):
    """The best and the second-best distinct value in a column."""
    ordered = sorted(set(values), reverse=not lower_is_better)
    return ordered[0], (ordered[1] if len(ordered) > 1 else None)


def draw_table(sl, x0, y, cols, rows, highlight, row_pitch=0.30, hdr=0.46,
               name_size=10.5, val_size=9.5):
    """Booktabs table. cols = (hdr1, hdr2, width, lower_is_better|None, fmt)."""
    xs, xx = [], x0
    for c in cols:
        xs.append(xx); xx += c[2]
    right = xx

    ranks = [rank([r[1][j] for r in rows], cols[j + 1][3]) if cols[j + 1][3] is not None
             else (None, None) for j in range(len(cols) - 1)]

    rule(sl, x0, y, right, 1.75)
    for (h1, h2, w, _, _), x in zip(cols, xs):
        al = PP_ALIGN.LEFT if x == x0 else PP_ALIGN.CENTER
        tbox(sl, x, y + 0.05, w, 0.18, h1, size=8.5, bold=True, align=al, color=C["ink3"])
        tbox(sl, x, y + 0.23, w, 0.18, h2, size=9.5, bold=True, align=al, color=C["ink2"])
    y += hdr
    rule(sl, x0, y, right, 0.9)

    for name, vals in rows:
        y += row_pitch - 0.26
        if name == highlight:
            band(sl, x0 - 0.06, y - 0.03, right - x0 + 0.12, 0.28)
        tbox(sl, xs[0], y, cols[0][2], 0.22, name, size=name_size, bold=True)
        for j, v in enumerate(vals):
            best, second = ranks[j]
            tbox(sl, xs[j + 1], y, cols[j + 1][2], 0.22, cols[j + 1][4].format(v),
                 size=val_size, font=MONO, align=PP_ALIGN.CENTER,
                 bold=(best is not None and v == best),
                 underline=(second is not None and v == second))
        y += 0.26
    rule(sl, x0, y, right, 1.75)
    return y, right


# ── Table 1 · LoCoMo, the slice all five systems share ──────────────────────
# 1 conv (conv-26), 19 sessions, 419 turns, 199 questions, identical for every
# system. Short-circuit attribution puts each error in exactly one stage, so
# P1 + P4 + P5 is the error column.
# Entries per turn carries no direction: denser is not better or worse on its
# own, it is context for why P5 behaves the way it does, so it is left unranked.
T1_COLS = [
    ("",           "",           1.50, None,  None),
    ("Extraction", "F1 ↑",       1.22, False, "{:.4f}"),
    ("Failure",    "P1 ↓",       1.22, True,  "{:.4f}"),
    ("Failure",    "P4 ↓",       1.22, True,  "{:.4f}"),
    ("Failure",    "P5 ↓",       1.22, True,  "{:.4f}"),
    ("Total",      "error ↓",    1.22, True,  "{:.4f}"),
    ("Answer",     "QA ↑",       1.22, False, "{:.4f}"),
    ("Temporal",   "acc ↑",      1.32, False, "{:.4f}"),
    ("Store",      "ent./turn",  1.38, None,  "{:.2f}"),
]
T1_ROWS = [
    ("Mem0 v1",   [0.6481, 0.2640, 0.1066, 0.1827, 0.5533, 0.4422, 0.0000, 0.68]),
    ("Mem0 v2",   [0.7796, 0.2525, 0.0455, 0.0808, 0.3788, 0.6181, 0.0541, 0.66]),
    ("StructMem", [0.8845, 0.0573, 0.0521, 0.2240, 0.3333, 0.6432, 0.3514, 1.85]),
    ("A-MEM",     [0.7608, 0.2256, 0.1282, 0.1436, 0.4974, 0.4925, 0.0270, 1.00]),
    ("Letta",     [0.7855, 0.2374, 0.0000, 0.1515, 0.3889, 0.6080, 0.0000, 0.06]),
]

# ── Table 2 · MemFail, the other shared slice ───────────────────────────────
# 35 questions across five subsets, identical for every system. It reproduces
# the LoCoMo pattern independently: StructMem best on the write and read
# stages, mid-pack on reasoning.
T2_COLS = [
    ("",        "",             1.05, None,  None),
    ("Summary", "error ↓",      1.00, True,  "{:.3f}"),
    ("Retrieval", "error ↓",    1.00, True,  "{:.3f}"),
    ("Reasoning", "error ↓",    1.00, True,  "{:.3f}"),
    ("Overall", "correct ↑",    1.00, False, "{:.3f}"),
    ("Coexist", "facts ↑",      1.05, False, "{:.3f}"),
]
T2_ROWS = [
    ("Mem0 v1",   [0.1143, 0.0571, 0.0857, 0.7429, 0.4000]),
    ("Mem0 v2",   [0.0571, 0.0571, 0.0857, 0.7714, 0.4000]),
    ("StructMem", [0.0000, 0.0286, 0.1143, 0.8571, 0.6000]),
    ("A-MEM",     [0.0000, 0.0857, 0.1429, 0.7714, 0.6000]),
    ("Letta",     [0.0000, 0.0000, 0.1429, 0.5143, 0.2000]),
]

FINDINGS = [
    ("StructMem already leads end to end. The ablation has to defend that.",
     "LoCoMo QA 0.6432 and MemFail 0.8571 are both the best in the field, on top "
     "of the best LoCoMo extraction: F1 0.8845 and P1 failure 0.0573 against "
     "0.2256 to 0.2640 for everyone else. There is nothing left to win on the "
     "write path. The question is what remains."),
    ("What remains is the answer stage, and only for StructMem.",
     "It is the one system whose largest error stage is P5: 0.2240 on LoCoMo, "
     "the worst of the five and nearly three times Mem0 v2's 0.0808, while its "
     "P4 failure is only 0.0521. The evidence is retrieved and the answer is "
     "still wrong. MemFail repeats it: summary error 0.0000 (tied best) and "
     "retrieval error 0.0286 (second), then reasoning error 0.1143, the weakest "
     "of its three stages."),
    ("The store is the densest in the study and labels nothing.",
     "StructMem writes 1.85 entries per LoCoMo turn against 0.06 to 1.00 for the "
     "others, and removes nothing, so the context reaching the answering model "
     "holds the old value and the new one side by side. Temporal accuracy 0.3514 "
     "is 6.5x the runner-up and still two questions in three wrong: the timestamp "
     "anchor works, and nothing acts on it."),
]


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    nav(sl)

    tbox(sl, 0.55, 0.50, 12.2, 0.46,
         "StructMem: the Errors That Are Left Sit at the Answer",
         size=23, bold=True)
    tbox(sl, 0.55, 0.98, 12.2, 0.26,
         "On the two benchmarks every system runs on an identical slice, StructMem "
         "leads end to end; it is also the only one whose error mass has moved "
         "past retrieval",
         size=11.5, color=C["ink3"], italic=True)

    # ── Table 1 ─────────────────────────────────────────────────────────────
    x0 = 0.55
    tbox(sl, x0, 1.42, 9.5, 0.22,
         "Table 1.  LoCoMo  (conv-26, 19 sessions, 419 turns, 199 questions; "
         "identical for all five systems)",
         size=10.5, bold=True, color=C["ink2"])
    yb, right1 = draw_table(sl, x0, 1.68, T1_COLS, T1_ROWS, "StructMem")
    tbox(sl, x0, yb + 0.07, 11.9, 0.20,
         "Bold is the best value in a column, underline the second best. Each error "
         "is attributed to exactly one stage, so P1 + P4 + P5 is the error column. "
         "Entries per turn is context, not a score, and is left unranked.",
         size=8.5, color=C["ink3"], italic=True)

    # ── Table 2 ─────────────────────────────────────────────────────────────
    tbox(sl, x0, 3.94, 6.5, 0.22,
         "Table 2.  MemFail  (35 questions, five subsets; identical for all five)",
         size=10.5, bold=True, color=C["ink2"])
    yb2, right2 = draw_table(sl, x0, 4.20, T2_COLS, T2_ROWS, "StructMem",
                             row_pitch=0.29, hdr=0.44)
    tbox(sl, x0, yb2 + 0.30, 6.5, 0.62,
         "Scope.  HaluMem and LongMemEval are deliberately absent: StructMem ran "
         "HaluMem user #1 (164 questions, 4 Dynamic Update) against the others' "
         "users #3 and #4 (360 questions, 37), and 5 LongMemEval questions against "
         "22, so neither supports a five-way ranking. StructMem also extracts with "
         "gemma-4-E4B-it, the rest with gemma-4-31B-it.",
         size=8.5, color=C["ink3"], italic=True)

    # ── Findings ────────────────────────────────────────────────────────────
    fx = 7.30
    tbox(sl, fx, 3.94, 5.5, 0.22, "Key findings", size=10.5, bold=True, color=C["ink2"])
    fy = 4.20
    for i, (head, body) in enumerate(FINDINGS, 1):
        tbox(sl, fx, fy, 0.28, 0.20, f"{i}", size=11, bold=True, color=C["accent"])
        tbox(sl, fx + 0.30, fy, 5.25, 0.20, head, size=10.2, bold=True)
        tb = sl.shapes.add_textbox(Inches(fx + 0.30), Inches(fy + 0.21),
                                   Inches(5.25), Inches(0.72))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = body
        r.font.name = FONT
        r.font.size = Pt(9.0)
        r.font.color.rgb = C["ink2"]
        p.line_spacing = 1.10
        fy += 0.88

    # ── Motivation band ─────────────────────────────────────────────────────
    rule(sl, 0.55, 6.94, 12.78, 0.75, C["rule"])
    tb = sl.shapes.add_textbox(Inches(0.55), Inches(7.00), Inches(12.2), Inches(0.46))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for lead, body, col in [
        ("Design implication.  ",
         "The timestamp that orders two competing entries is already on every "
         "StructMem payload and is never consulted at update time. The ablation "
         "adds a non-destructive state bank (M1) that marks the old entry "
         "superseded rather than deleting it, so the extraction lead in Table 1 is "
         "not traded away; summary synchronisation (M3); and state-aware retrieval "
         "(M4), which is where the 0.2240 P5 failure lives.", C["ink2"]),
        ("Guardrail.  ",
         "MemFail coexisting_facts, where StructMem is already at 0.6000, holds "
         "preferences that may legitimately hold at once. Any supersession rule "
         "has to lift temporal accuracy without turning those into false conflicts.",
         C["ink3"]),
    ]:
        p = tf.paragraphs[0] if not tf.paragraphs[0].runs else tf.add_paragraph()
        p.line_spacing = 1.12
        r = p.add_run(); r.text = lead
        r.font.name = FONT; r.font.size = Pt(8.5); r.font.bold = True; r.font.color.rgb = C["ink"]
        r = p.add_run(); r.text = body
        r.font.name = FONT; r.font.size = Pt(8.5); r.font.italic = True; r.font.color.rgb = col

    out = "structmem_motivation.pptx"
    prs.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    build()
