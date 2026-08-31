"""
Generates the proposal presentation deck (.pptx) for the NLP-with-LLMs course project.
Edit AUTHORS / INSTITUTE below, then re-run:  python build_slides.py
"""

import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------- edit these
TITLE = "TrustRAG"
SUBTITLE = "Cheap Claim-Level Hallucination Detection for Retrieval-Augmented Generation"
TAGLINE = "Can a 0.4B encoder catch what a 70B judge catches, at a fraction of the cost?"
COURSE = "NLP with LLMs  |  Course Project Proposal"
AUTHORS = "Adhithyan V (230121003)  \u00b7  Aditya Raj (230121004)"
PROGRAM = "B.Tech, Group of 2"
DATE = "27 August 2026"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "docs", "proposal", "TrustRAG_Proposal_Slides.pptx")
# ---------------------------------------------------------------------------

NAVY = RGBColor(0x12, 0x3A, 0x5F)
DEEP = RGBColor(0x0B, 0x22, 0x3A)
ACCENT = RGBColor(0x0F, 0x6F, 0xC5)
GREEN = RGBColor(0x1E, 0x84, 0x49)
AMBER = RGBColor(0xD6, 0x89, 0x10)
RED = RGBColor(0xC0, 0x39, 0x2B)
INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x6B, 0x7A, 0x8C)
PANEL = RGBColor(0xF1, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

SW = 13.333
_n = [1]


def tf_set(tf, runs, align=PP_ALIGN.LEFT, space_before=0, line_spacing=None):
    tf.word_wrap = True
    first = True
    for text, size, bold, color in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_before = Pt(space_before)
        if line_spacing:
            p.line_spacing = line_spacing
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = "Calibri"


def textbox(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            space_before=0, line_spacing=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.text_frame.vertical_anchor = anchor
    tf_set(tb.text_frame, runs, align, space_before, line_spacing)
    return tb


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def slide_frame(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    _n[0] += 1
    rect(s, 0, 0, 0.16, 7.5, NAVY)
    y = 0.42
    if kicker:
        textbox(s, 0.62, y, 11.5, 0.3, [(kicker.upper(), 12, True, ACCENT)])
        y += 0.36
    textbox(s, 0.6, y, 12.2, 0.75, [(title, 30, True, NAVY)])
    rect(s, 0.62, y + 0.82, 1.5, 0.045, ACCENT)
    textbox(s, 0.62, 6.98, 6.0, 0.3, [("TrustRAG  |  NLP with LLMs", 10, False, MUTED)])
    textbox(s, 11.6, 6.98, 1.2, 0.3, [(str(_n[0]), 10, False, MUTED)], align=PP_ALIGN.RIGHT)
    return s


def card(slide, x, y, w, h, head, body, accent=ACCENT, head_size=15, body_size=13):
    rect(slide, x, y, w, h, PANEL)
    rect(slide, x, y, 0.055, h, accent)
    tb = slide.shapes.add_textbox(Inches(x + 0.26), Inches(y + 0.16),
                                  Inches(w - 0.5), Inches(h - 0.3))
    runs = [(head, head_size, True, NAVY)]
    if body:
        runs.append((body, body_size, False, INK))
    tf_set(tb.text_frame, runs, space_before=5, line_spacing=1.05)
    return tb


def bullets(slide, x, y, w, h, items, size=15, gap=13, marker="\u25aa  "):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    first = True
    for lead, rest in items:
        p = tb.text_frame.paragraphs[0] if first else tb.text_frame.add_paragraph()
        p.space_before = Pt(0 if first else gap)
        first = False
        p.line_spacing = 1.08
        r = p.add_run()
        r.text = marker + lead
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = NAVY
        r.font.name = "Calibri"
        if rest:
            r2 = p.add_run()
            r2.text = rest
            r2.font.size = Pt(size)
            r2.font.color.rgb = INK
            r2.font.name = "Calibri"
    return tb


def steps(slide, x, y, w, h, items, size=14.5, gap=11):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    for i, (lead, rest) in enumerate(items):
        p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
        p.space_before = Pt(0 if i == 0 else gap)
        p.line_spacing = 1.07
        rn = p.add_run()
        rn.text = "%d.  " % (i + 1)
        rn.font.size = Pt(size)
        rn.font.bold = True
        rn.font.color.rgb = ACCENT
        rn.font.name = "Calibri"
        r = p.add_run()
        r.text = lead
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = NAVY
        r.font.name = "Calibri"
        r2 = p.add_run()
        r2.text = rest
        r2.font.size = Pt(size)
        r2.font.color.rgb = INK
        r2.font.name = "Calibri"
    return tb


def add_table(slide, x, y, w, h, headers, rows, col_w=None, fs=11.5, hfs=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(cw)))
    for i, htxt in enumerate(headers):
        c = tbl.cell(0, i)
        c.text = ""
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.margin_left = Inches(0.1)
        c.margin_right = Inches(0.08)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        r = c.text_frame.paragraphs[0].add_run()
        r.text = htxt
        r.font.size = Pt(hfs)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = "Calibri"
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            c = tbl.cell(ri, ci)
            c.text = ""
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if ri % 2 else PANEL
            c.margin_left = Inches(0.1)
            c.margin_right = Inches(0.08)
            c.margin_top = Inches(0.05)
            c.margin_bottom = Inches(0.05)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.line_spacing = 1.0
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(fs)
            r.font.color.rgb = INK
            r.font.name = "Calibri"
            if ci == 0:
                r.font.bold = True
                r.font.color.rgb = NAVY
    return tbl


# ============================================================ 1  TITLE
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, 7.5, DEEP)
rect(s, 0, 0, SW, 0.14, ACCENT)
textbox(s, 1.1, 1.55, 11, 0.4, [(COURSE.upper(), 13, True, RGBColor(0x7F, 0xB3, 0xE0))])
textbox(s, 1.05, 2.05, 11.2, 1.2, [(TITLE, 62, True, WHITE)])
textbox(s, 1.1, 3.25, 10.8, 0.8, [(SUBTITLE, 21, False, RGBColor(0xC9, 0xD6, 0xE2))])
rect(s, 1.12, 4.28, 1.8, 0.05, ACCENT)
textbox(s, 1.1, 4.55, 10.8, 0.5,
        [("\u201c" + TAGLINE + "\u201d", 17, False, RGBColor(0x9F, 0xC4, 0xE4))])
textbox(s, 1.1, 5.75, 11, 1.0,
        [(AUTHORS, 16, True, WHITE),
         (PROGRAM, 12.5, False, RGBColor(0x8E, 0xA3, 0xB8)),
         (DATE, 12.5, False, RGBColor(0x8E, 0xA3, 0xB8))], space_before=4)

# ============================================================ 2  MOTIVATION
s = slide_frame("RAG was meant to stop hallucination. It did not.", "Motivation")

textbox(s, 0.62, 1.78, 12.1, 0.4,
        [("The idea was simple. Give the model the source documents. Ask it to answer from them. It "
          "should then stop inventing facts.", 17, False, INK)])

card(s, 0.62, 2.5, 3.85, 2.2, "What the data shows",
     "RAGTruth marked every unsupported word in about 18,000 RAG answers.\n\n"
     "Unsupported text turned up in a large share of them, even when the right evidence was "
     "given.", RED, body_size=13)
card(s, 4.75, 2.5, 3.85, 2.2, "Two kinds of error",
     "Sometimes the model contradicts the evidence.\n\n"
     "More often it adds something extra: a date, a name, a reason. It sounds right. It fits the "
     "topic. It is not in the source.", AMBER, body_size=13)
card(s, 8.88, 2.5, 3.85, 2.2, "Where this gets used",
     "Hospitals. Law firms. Banks. Government helplines.\n\n"
     "Exactly the places where a made-up sentence costs the most.", ACCENT, body_size=13)

rect(s, 0.62, 4.98, 12.11, 1.35, PANEL)
rect(s, 0.62, 4.98, 0.055, 1.35, NAVY)
textbox(s, 0.98, 5.15, 11.5, 1.1,
        [("Why fluent answers are the dangerous ones", 16, True, NAVY),
         ("Told \u201cI could not find this in the documents\u201d, a user goes and checks. Handed a "
          "confident paragraph, they do not. The polish that makes these systems useful is the same "
          "thing that hides their mistakes.", 14.5, False, INK)], space_before=6, line_spacing=1.06)

# ============================================================ 3  THREE PROBLEMS
s = slide_frame("Three problems with today's tools", "Motivation")

card(s, 0.62, 1.85, 3.85, 2.6, "1.  Too coarse",
     "Most systems give one score for the whole answer.\n\n"
     "If sentence two is wrong, you still get one number.\n\n"
     "You cannot tell which part to distrust.", RED, body_size=14)
card(s, 4.75, 1.85, 3.85, 2.6, "2.  Too expensive",
     "LLM-as-judge calls a large model for every answer.\n\n"
     "SelfCheckGPT needs several samples. FActScore runs one query per fact.\n\n"
     "All of it costs seconds and money.", AMBER, body_size=14)
card(s, 8.88, 1.85, 3.85, 2.6, "3.  They do not act",
     "The score goes to a dashboard.\n\n"
     "A human reads it days later.\n\n"
     "The bad answer was still shown to the user.", ACCENT, body_size=14)

rect(s, 0.62, 4.8, 12.11, 1.5, NAVY)
textbox(s, 0.98, 5.0, 11.5, 1.25,
        [("We want to fix all three at once", 19, True, WHITE),
         ("Score each claim separately. Use a small model instead of a big one. Then use the score "
          "to change what the system says.", 15, False, RGBColor(0xC9, 0xD6, 0xE2)),
         ("Built on known results: NLI models are cheap and strong at checking facts (SummaC, "
          "AlignScore), and selective prediction turns a score into a rule for staying silent.",
          13, False, RGBColor(0x9F, 0xB6, 0xCB))], space_before=6, line_spacing=1.05)

# ============================================================ 4  PROBLEM
s = slide_frame("Problem Description", "What we are solving")

rect(s, 0.62, 1.72, 5.75, 2.5, PANEL)
rect(s, 0.62, 1.72, 0.055, 2.5, NAVY)
textbox(s, 0.98, 1.88, 5.25, 2.25,
        [("The setup", 16, True, NAVY),
         ("A user asks a question q.", 14, False, INK),
         ("The system retrieves evidence E.", 14, False, INK),
         ("A model writes an answer a.", 14, False, INK),
         ("We split a into claims c\u2081 \u2026 c\u2098.", 14, False, INK),
         ("A claim is unsupported if E does not back it. Learn f(c, E) that predicts this.",
          14, True, ACCENT)], space_before=6, line_spacing=1.05)

rect(s, 6.6, 1.72, 6.13, 2.5, PANEL)
rect(s, 6.6, 1.72, 0.055, 2.5, ACCENT)
textbox(s, 6.96, 1.88, 5.6, 2.25,
        [("Our idea", 16, True, NAVY),
         ("NLI models are trained on one question: does sentence B follow from sentence A? That is "
          "exactly our question.", 14, False, INK),
         ("So a small NLI model may work about as well as a huge one. It does not need to reason or "
          "write. It only compares two short sentences.", 14, False, INK),
         ("If we are wrong, we learn which errors need a big model. Either way we have a result.",
          13.5, True, NAVY)], space_before=6, line_spacing=1.05)

textbox(s, 0.62, 4.38, 12.1, 0.3, [("FOUR PARTS", 12.5, True, ACCENT)])
parts = [("Detection", "AUROC, AUPRC, F1"),
         ("Calibration", "make the score a\nreal probability"),
         ("Selective prediction", "show, flag, or\nhold back"),
         ("Cost", "ms per answer,\nGPU memory")]
for i, (h, b) in enumerate(parts):
    card(s, 0.62 + i * 3.09, 4.75, 2.86, 1.25, h, b, ACCENT, head_size=14.5, body_size=13)

textbox(s, 0.62, 6.2, 12.1, 0.4,
        [("Everything runs on one free Colab T4 with 16 GB. Every part is open-weight. The 70B model "
          "is only a comparison, never a component.", 14, True, MUTED)])

# ============================================================ 5  DATASET
s = slide_frame("Dataset Description", "What we train and test on")
textbox(s, 0.62, 1.7, 12.1, 0.4,
        [("The key point: both main datasets already contain LLM answers that humans have labelled, "
          "with the exact evidence the model saw. We do not have to run a generator to get our main "
          "result.", 15, True, NAVY)])

add_table(s, 0.62, 2.3, 12.11, 3.1,
          ["Dataset", "Size and access", "Labels", "Why we use it"],
          [["RAGTruth", "15,090 train / 2,700 test\nHF: wandb/RAGTruth-processed",
            "Character-level spans marking unsupported text, plus 4 type labels. Answers from GPT-4, "
            "GPT-3.5, Llama-2, Mistral",
            "Main dataset. Trains and tests the detector, and tells us which error types we miss"],
           ["HaluEval", "10,000 QA rows\nHF: pminervini/HaluEval, config 'qa'",
            "Each question has one correct and one hallucinated answer, plus the supporting text. "
            "Balanced classes",
            "Transfer test. We never train on it. Shows the method works on a different generator"],
           ["HotpotQA", "113k questions; 2 correct and 8 distractor paragraphs each",
            "Answer spans and sentence-level supporting facts",
            "Live demo. We index 500 documents so the demo checks fresh text, not stored answers"],
           ["FEVER", "185k claims over Wikipedia",
            "SUPPORTED / REFUTED / NOT ENOUGH INFO with evidence",
            "Used indirectly. Our NLI model was already trained on it, so it works with no extra "
            "training"]],
          col_w=[1.3, 2.55, 3.9, 4.36], fs=11, hfs=12)

textbox(s, 0.62, 5.62, 12.1, 0.8,
        [("One detail that matters: we split train and validation by source document, not by "
          "response. Splitting by response puts the same evidence passage on both sides. That leaks "
          "information and would make our numbers look better than they are.", 14, False, INK)],
        line_spacing=1.06)

# ============================================================ 6  METHOD
s = slide_frame("Methodology", "How the system works")

textbox(s, 0.62, 1.72, 6.4, 0.3, [("FIVE STEPS", 12.5, True, ACCENT)])
steps(s, 0.62, 2.08, 6.4, 3.9, [
    ("Split. ", "Cut the answer into sentence-level claims. Cut the evidence into sentences too."),
    ("Check. ", "Run an NLI model on every claim and evidence sentence pair. We use "
                "DeBERTa-v3-large (0.4B). It returns entailment, neutral and contradiction."),
    ("Score. ", "For each claim take the best evidence sentence. Support = highest P(entail) minus "
                "P(contradict)."),
    ("Combine. ", "Feed six features into a logistic regression. Calibrate with isotonic regression."),
    ("Decide. ", "Two thresholds turn that probability into an action."),
], size=14.5, gap=12)

textbox(s, 7.3, 1.72, 5.43, 0.3, [("THE SIX FEATURES", 12.5, True, ACCENT)])
feats = [
    ("Support score", "best P(entail) \u2212 P(contradict)"),
    ("Peak contradiction", "conflict vs. omission"),
    ("Mean top-3 entailment", "one sentence, or the passage?"),
    ("Embedding similarity", "bge-small cosine"),
    ("Claim specificity", "names and numbers are faked more"),
    ("Claim length", "keeps the rest fair"),
]
tb = s.shapes.add_textbox(Inches(7.3), Inches(2.08), Inches(5.43), Inches(2.4))
tb.text_frame.word_wrap = True
for i, (h, b) in enumerate(feats):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.space_before = Pt(0 if i == 0 else 9)
    p.line_spacing = 1.05
    r = p.add_run()
    r.text = "%d.  %s:  " % (i + 1, h)
    r.font.size = Pt(13.5)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r.font.name = "Calibri"
    r2 = p.add_run()
    r2.text = b
    r2.font.size = Pt(13.5)
    r2.font.color.rgb = INK
    r2.font.name = "Calibri"

textbox(s, 7.3, 4.62, 5.43, 0.3, [("WHAT THE THRESHOLDS DO", 12.5, True, ACCENT)])
acts = [("Low score \u2192 ", "show the claim", GREEN),
        ("Middle \u2192 ", "show it, flag it, attach the evidence", AMBER),
        ("High score \u2192 ", "hold it back, mark answer incomplete", RED)]
tb = s.shapes.add_textbox(Inches(7.3), Inches(4.98), Inches(5.43), Inches(1.2))
tb.text_frame.word_wrap = True
for i, (h, b, col) in enumerate(acts):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.space_before = Pt(0 if i == 0 else 8)
    p.line_spacing = 1.05
    r = p.add_run()
    r.text = h
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = col
    r.font.name = "Calibri"
    r2 = p.add_run()
    r2.text = b
    r2.font.size = Pt(14)
    r2.font.color.rgb = INK
    r2.font.name = "Calibri"

textbox(s, 0.62, 6.12, 6.4, 0.7,
        [("We compare claims against single sentences, not whole passages. NLI gets much weaker on "
          "long text.", 13, True, MUTED)], line_spacing=1.05)

# ============================================================ 7  EVALUATION
s = slide_frame("How we will measure it", "Evaluation")

textbox(s, 0.62, 1.7, 6.0, 0.3, [("SIX BASELINES", 12.5, True, ACCENT)])
add_table(s, 0.62, 2.06, 6.05, 2.9,
          ["Baseline", "What it tells us"],
          [["Random guess", "The floor. HaluEval is balanced, so 0.50"],
           ["Word overlap", "How much is solved by matching words"],
           ["Embedding similarity", "Is similarity enough without entailment?"],
           ["NLI, whole passage", "How much sentence-level scoring helps"],
           ["Llama-3.2-3B judge", "Does a small LLM beat a small encoder?"],
           ["Llama-3.3-70B judge", "The ceiling, and the cost we avoid"]],
          col_w=[2.15, 3.9], fs=11.5, hfs=12)

textbox(s, 6.95, 1.7, 5.8, 0.3, [("WHAT WE REPORT", 12.5, True, ACCENT)])
mets = [
    ("Detection: ", "AUROC, AUPRC, F1, split by error type"),
    ("Calibration: ", "ECE with 15 bins, Brier score"),
    ("Selective prediction: ", "risk-coverage curve, error left at 90% coverage"),
    ("Transfer: ", "train on RAGTruth, test on HaluEval"),
    ("Cost: ", "ms per answer, GPU memory, cost per 1,000 queries"),
    ("Human check: ", "both of us label 100 claims, report Cohen's \u03ba"),
]
bullets(s, 6.95, 2.06, 5.8, 2.9, mets, size=13.5, gap=11)

rect(s, 0.62, 5.2, 12.11, 1.4, PANEL)
rect(s, 0.62, 5.2, 0.055, 1.4, AMBER)
textbox(s, 0.98, 5.36, 11.5, 1.15,
        [("Ablations: each answers a question we expect to be asked", 15, True, NAVY),
         ("Verifier size, 0.4B against 0.18B  \u00b7  sentence pooling against whole-passage  \u00b7  "
          "drop each feature in turn  \u00b7  calibration method  \u00b7  results split by which model "
          "wrote the answer.", 14, False, INK)], space_before=6, line_spacing=1.05)

# ============================================================ 8  RISKS
s = slide_frame("What could go wrong", "Risks")

risks = [
    ("Bad sentence splitting", AMBER,
     "If the splitter cuts claims badly, our detector looks worse than it really is.\n\n"
     "We check 100 splits by hand and measure an upper bound using correct splits, so the two "
     "errors stay separate."),
    ("Colab timeouts", AMBER,
     "Long runs can die partway through.\n\n"
     "Every scoring script saves progress to Drive every 200 examples and picks up from the last "
     "checkpoint."),
    ("Class imbalance", AMBER,
     "Unsupported claims are the minority in RAGTruth, so accuracy alone would look flattering.\n\n"
     "We track AUPRC next to AUROC, use class weights, and keep balanced HaluEval as a sanity "
     "check."),
    ("Our idea may be wrong", GREEN,
     "If NLI loses badly to the 70B judge, we say so.\n\n"
     "The per-type results still show which errors need a large model, which is a useful finding "
     "on its own."),
]
for i, (head, col, body) in enumerate(risks):
    x = 0.62 + (i % 2) * 6.28
    y = 1.82 + (i // 2) * 2.22
    card(s, x, y, 5.85, 2.05, head, body, col, head_size=16, body_size=12.5)

rect(s, 0.62, 6.3, 12.11, 0.52, NAVY)
textbox(s, 0.98, 6.38, 11.5, 0.45,
        [("The idea is testable either way, so no outcome leaves us with nothing to say.",
          14.5, True, WHITE)])

# ============================================================ 9  WHY
s = slide_frame("Why we want to do this", "Now, and after the course")

textbox(s, 0.62, 1.78, 5.85, 0.3, [("NOW", 13, True, ACCENT)])
now = [
    ("Puts the whole stack in one system. ", "Retrieval, transformer encoders, quantisation, calibration and abstention stop being separate topics. They become one thing we had to debug."),
    ("Teaches evaluation, not just models. ", "The hard part is not getting a model to run. It is splits that do not leak, baselines that are hard to beat, and reporting the number that could prove us wrong."),
    ("Gives us something concrete to show. ", "A benchmark table and a working demo beat a notebook with one accuracy figure."),
    ("It cannot quietly fail. ", "The idea is testable either way, so there is no version of this where we end up with nothing to say."),
]
bullets(s, 0.62, 2.15, 5.85, 4.0, now, size=13.5, gap=12)

textbox(s, 6.9, 1.78, 5.85, 0.3, [("AFTER", 13, True, ACCENT)])
later = [
    ("It matches real jobs. ", "LLM evaluation, trust and safety, and AI reliability are named roles now. Checking RAG answers is what they do."),
    ("It applies to real systems. ", "Almost any company shipping a RAG assistant ends up needing this check. Most build one poorly."),
    ("It can grow into research. ", "Extends into other languages, multi-hop claims, or learned claim splitting. Enough for a workshop paper or a thesis."),
    ("It does not go out of date. ", "The checker wraps any generator. Work that fine-tunes one model does not survive being replaced."),
]
bullets(s, 6.9, 2.15, 5.85, 4.0, later, size=13.5, gap=12)

rect(s, 0.62, 6.24, 12.11, 0.55, NAVY)
textbox(s, 0.98, 6.32, 11.5, 0.4,
        [("If a 0.4B encoder does most of what a 70B judge does, checking answers becomes cheap "
          "enough to run on every single response.", 14.5, True, WHITE)])

# ============================================================ 10  REFERENCES
s = slide_frame("Key references", "Full list of 16 in the written proposal")
refs = [
    ("Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 33.",
     "https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html"),
    ("Niu et al. (2024). RAGTruth: A Hallucination Corpus for Trustworthy Retrieval-Augmented Language Models. ACL 2024.",
     "https://aclanthology.org/2024.acl-long.585/"),
    ("Li et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs. EMNLP 2023.",
     "https://aclanthology.org/2023.emnlp-main.397/"),
    ("Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection. EMNLP 2023.",
     "https://aclanthology.org/2023.emnlp-main.557/"),
    ("Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision. EMNLP 2023.",
     "https://aclanthology.org/2023.emnlp-main.741/"),
    ("Laban et al. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection. TACL 10.",
     "https://aclanthology.org/2022.tacl-1.10/"),
    ("Zha et al. (2023). AlignScore: Evaluating Factual Consistency with a Unified Alignment Function. ACL 2023.",
     "https://aclanthology.org/2023.acl-long.634/"),
    ("Geifman & El-Yaniv (2017). Selective Classification for Deep Neural Networks. NeurIPS 30.",
     "https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html"),
    ("He et al. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training. ICLR 2023.",
     "https://arxiv.org/abs/2111.09543"),
    ("Thorne et al. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT 2018.",
     "https://aclanthology.org/N18-1074/"),
]
tb = s.shapes.add_textbox(Inches(0.62), Inches(1.85), Inches(12.1), Inches(4.8))
tb.text_frame.word_wrap = True
for i, (txt, url) in enumerate(refs):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.space_before = Pt(0 if i == 0 else 11)
    p.line_spacing = 1.0
    num = p.add_run()
    num.text = "[%d]  " % (i + 1)
    num.font.size = Pt(12.5)
    num.font.bold = True
    num.font.color.rgb = ACCENT
    num.font.name = "Calibri"
    body = p.add_run()
    body.text = txt + "  "
    body.font.size = Pt(12.5)
    body.font.color.rgb = INK
    body.font.name = "Calibri"
    link = p.add_run()
    link.text = url
    link.font.size = Pt(11)
    link.font.name = "Calibri"
    link.hyperlink.address = url

prs.save(OUT)
print("Saved:", OUT)
