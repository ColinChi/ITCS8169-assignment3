"""Build the two-page report as PDF (report/assignment3_report.pdf) and Word (report/assignment3_report.docx).

The layout is that of the Assignment 2 report: Letter paper, 1-inch margins, 12 pt serif text, sans-serif
headings, one table with its caption above. Both renderers use the same text and the same Table 1 data.
All numbers are read from results/ and from the final checkpoint (run run_methods.py, eval_lora.py,
run_ablations.py, scripts/summarize.py, scripts/oracle_probe.py and evaluate.py first). The repository URL comes
from report/report_metadata.json.
Text uses a small inline markup: <b>, <i>, <sub>, <super>, <link href=...>.

Usage: python scripts/build_report.py [--pdf] [--docx]     (default: both; the Word file needs python-docx)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from html.parser import HTMLParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "report" / "assignment3_report.pdf"
OUT_DOCX = ROOT / "report" / "assignment3_report.docx"
TITLE = "One-Shot Learning with Foundation Models"
MAIN = "siglip2_so400m"
COL_WIDTHS_IN = [3.1, 0.7, 1.0, 0.7, 1.0]
H2_COLOR = "#333333"
NB = "\u00a0"  # non-breaking space: keeps "91.0 ± 1.0%" on one line

# Liberation Serif/Sans are the fonts of the Assignment 2 report. They are embedded when the .ttf files are
# found; otherwise the PDF uses the metric-compatible built-in Times/Helvetica, which gives the same layout.
FONT_DIRS = [os.environ.get("REPORT_FONT_DIR", ""), ROOT / "report" / "fonts", ROOT / "cache" / "fonts",
             "/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2",
             "/usr/share/fonts/liberation", "/usr/share/fonts/liberation-serif", "/usr/share/fonts/liberation-sans",
             Path.home() / ".local/share/fonts", Path.home() / ".fonts"]
FONT_FILES = {"serif": "LiberationSerif-Regular.ttf", "serif_b": "LiberationSerif-Bold.ttf",
              "serif_i": "LiberationSerif-Italic.ttf", "serif_bi": "LiberationSerif-BoldItalic.ttf",
              "sans_b": "LiberationSans-Bold.ttf", "sans_bi": "LiberationSans-BoldItalic.ttf"}
BUILTIN_FONTS = {"serif": "Times-Roman", "serif_b": "Times-Bold", "serif_i": "Times-Italic",
                 "serif_bi": "Times-BoldItalic", "sans_b": "Helvetica-Bold", "sans_bi": "Helvetica-BoldOblique"}


def f2(x) -> str:
    return f"{float(x):.2f}"


def pm(r) -> str:
    return f"{r['mean']:.1f}{NB}±{NB}{r['std']:.1f}"


def load_acc() -> dict:
    """Per-method test accuracies of the main backbone: seed 0, the ten draws and their statistics.
    Keys: methods of run_methods.py and eval_lora.py; "<prompts>:<method>" for run_ablations.py."""
    acc = {}
    for name in (MAIN, f"{MAIN}_lora", f"{MAIN}_ablation"):
        for r in csv.DictReader(open(ROOT / "results" / f"acc_{name}.csv")):
            d = torch.tensor([float(v) for k, v in r.items() if k[0] == "s" and k[1:].isdigit()])
            acc[r["method"]] = {"seed0": d[0].item(), "mean": d.mean().item(), "std": d.std().item(),
                                "min": d.min().item(), "max": d.max().item(), "draws": d}
    return acc


def shot_lambdas() -> dict:
    """CLAP's lambda_c of each class (zero-shot probability of the true class of its training image), computed
    from the text classifier and the shot features stored in the final checkpoint."""
    ck = torch.load(ROOT / "checkpoints" / f"{MAIN}_seed0.pt")
    t, s = ck["text_classifier"].float(), ck["shot_feats"].float()
    p = (ck["logit_scale"] * s @ t.T).softmax(-1)
    return {c: p[i, :, i].mean().item() for i, c in enumerate(ck["classes"])}


def content() -> dict:
    """Everything that appears in the report, independent of the output format."""
    meta = json.loads((ROOT / "report" / "report_metadata.json").read_text())
    A = load_acc()
    bb = {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / "summary_backbones.csv"))}
    fit = list(csv.DictReader(open(ROOT / "results" / f"prompt_fit_{MAIN}.csv")))
    pred = list(csv.DictReader(open(ROOT / "results" / f"predictions_{MAIN}_seed0.csv")))
    final = {k: sum(r[k] == r["label"] for r in pred) for k in ("zero-shot", "inductive", "transductive")}
    acc = {k: 100 * v / len(pred) for k, v in final.items()}
    ok = {k: {c: sum(r[k] == c for r in pred if r["label"] == c) for c in {r["label"] for r in pred}} for k in final}
    zs_ok, ind_ok = ok["zero-shot"], ok["inductive"]
    lam = shot_lambdas()
    url = meta["github_url"]

    # prompts with synonyms (fixed in advance) / class-name prompt "a photo of a {}." (run_ablations.py: "single")
    zs, zs_bal, clap, bal, st, trans = (A[k] for k in ("zs", "zs+avg+bal", "clap", "clap+avg+bal", "clap+avg+st",
                                                       "clap+avg+trans"))
    lora, lora_bal, const, free = A["lora_clap"], A["lora_clap+avg+bal"], A["syn:clap_const"], A["syn:clap_free"]
    n_zs_bal, n_bal, n_lora_bal, n_st = (A[f"single:{k}"] for k in ("zs+avg+bal", "clap+avg+bal",
                                                                     "lora_clap+avg+bal", "clap+avg+st"))
    count = lambda a, b: int((a["draws"] > b["draws"]).sum())  # noqa: E731  draws in which a is better than b
    of10 = lambda n: "all ten draws" if n == 10 else f"{n} of 10 draws"  # noqa: E731
    syn_fits = sum(min(("syn", "raw", "single"), key=lambda k: float(r[k])) == "syn" for r in fit)
    # the qualitative statements of the analysis, checked against the numbers they summarise
    assert A["proto"]["mean"] < zs["seed0"] - 5 and clap["seed0"] == clap["max"]
    assert lam["InsideCity"] == min(lam.values()) and zs_ok["Street"] > ind_ok["Street"]
    assert st["mean"] == max(r["mean"] for r in (bal, st, trans)) and st["min"] < clap["min"] < bal["min"]
    assert abs(const["mean"] - clap["mean"]) < 0.5 < clap["mean"] - free["mean"]
    assert count(clap, lora) >= 7 and count(lora_bal, bal) >= 7 and lora_bal["seed0"] < bal["seed0"]
    assert abs(A["clap_var"]["mean"] - clap["mean"]) < 0.5 and A["clap+avg"]["mean"] < clap["mean"]
    assert max(float(bb["siglip2_g"]["clap_mean"]), float(bb["pe_g14"]["clap_mean"])) < clap["mean"]
    assert n_zs_bal["seed0"] == acc["transductive"] and n_bal["mean"] > bal["mean"] and n_bal["seed0"] > bal["seed0"]
    assert n_lora_bal["min"] == max(r["min"] for r in (n_bal, n_lora_bal, n_st)) and n_st["min"] < n_bal["min"]
    assert n_st["seed0"] == max(r["seed0"] for r in (n_bal, n_lora_bal, n_st)) and syn_fits > 5

    setup = (
        "The data consist of 2,400 training images (150 per class, 16 classes) and 400 test images. Only one "
        "training image per class is used: it is drawn at random with a fixed seed, and the other 2,384 training "
        "images are never read, with or without labels. The submitted model uses seed 0; seeds 1–9 measure how much "
        "the result depends on the drawn images, and results are given as mean ± standard deviation over the ten "
        "draws. Because a validation set would need a second labeled image per class, every method uses its "
        "published defaults, and all reported numbers are test accuracies. The encoder, the prompts, the classifier "
        "head and seed 0 were fixed in advance. Test images are classified independently (inductive) or jointly as "
        f'an unlabeled set (transductive). Code and trained model: <link href="{url}" color="#1155cc">{url}</link>.')

    baseline = (
        "The foundation model is SigLIP 2 (SO400M/16, 384 px), a contrastive vision-language model whose image "
        "encoder (428M parameters) and text encoder map images and text to a shared 1,152-d space. Its weights stay "
        "frozen. A text classifier is built from the class names: each name and a few synonyms are inserted into 10 "
        "templates such as “a black and white photo of a {}”, and their normalized embeddings are averaged into one "
        "class vector. A test image is assigned to the class with the most similar vector. This zero-shot "
        f"classifier reaches {f2(zs['seed0'])}% test accuracy. A control uses only the prompt “a photo of a {{}}.” "
        "with the class name.")

    sub = lambda s: f'<sub rise="1.5" size="7">{s}</sub>'  # noqa: E731
    lam_c = f"<i>λ</i>{sub('c')}"
    strategy = [
        "Sixteen images are too few to train a network or, without validation data, to fine-tune one safely. The "
        "encoder is therefore kept frozen, the text classifier serves as a prior, and the single image corrects it "
        "only where it is wrong; only a linear head of 16 × 1,152 weights is trained.",

        "<b>CLAP (class-adaptive linear probe).</b> The head <i>W</i> is a linear classifier on the image "
        "embeddings, initialized with the text classifier <i>t</i>. It is trained on 66 views of each training "
        "image (the original, its mirror image and 64 random crops of 50–100% of the area) by minimizing the "
        f"cross-entropy plus the penalty Σ{sub('c')}{NB}{lam_c}{NB}‖<i>W</i>{sub('c')} − <i>t</i>{sub('c')}‖"
        "<super rise=\"4.5\" size=\"7\">2</super>, which pulls each class vector back to its text embedding. The "
        f"weight {lam_c} is the probability that the text classifier assigns to the correct class of the training "
        f"image of class <i>c</i>. If the text recognizes this image, {lam_c} is close to 1 and the class "
        "vector stays anchored to the text; if not, the name describes the class poorly and the image may move the "
        "vector. The objective is convex and is solved to convergence with L-BFGS in about a second, so no learning "
        "rate or number of epochs has to be chosen without validation data.",

        "<b>Balanced assignment (test-time adaptation).</b> In the transductive setting, each test embedding is "
        "averaged over 34 views, and the class probabilities of the 400 test images are rescaled jointly by Sinkhorn "
        "iterations, as in PT-MAP, until every class receives the same total probability. This uniform class prior "
        "assumes a balanced test set; it counteracts the typical one-shot failure, in which an atypical training "
        "image pulls a whole group of test images into its class.",

        "<b>Alternatives.</b> Evaluated with their published settings: a nearest-prototype classifier on the image "
        "alone; Tip-Adapter; CLIP-LoRA, which fine-tunes the image encoder (rank-2 LoRA, 373k parameters) before "
        "the head is fitted; 16 Stable Diffusion XL variations of each training image as extra views; and the "
        "transductive methods TransCLIP and self-training on confident predictions.",
    ]

    def row(label, key, bold=()):
        a, b = A[key], A[f"single:{key}"]
        cells = [label]
        for r in (a, b):
            cells += [f2(r["seed0"]), pm(r) if r["std"] > 0 else "–"]  # zero-shot rows do not depend on the draw
        return {"kind": "data", "cells": cells, "bold": set(bold)}

    sec = lambda t: {"kind": "section", "cells": [t]}  # noqa: E731
    table = [
        {"kind": "group", "cells": ["", "Prompts with synonyms", "", "Class-name prompt", ""]},
        {"kind": "header", "cells": ["Method", "Seed 0", "Mean ± std", "Seed 0", "Mean ± std"]},
        sec("Zero-shot (no training image)"),
        row("Text classifier", "zs"),
        row("Text classifier + balanced assignment", "zs+avg+bal"),
        sec("One-shot (one training image per class)"),
        row("CLAP head", "clap", bold=[1]),
        row("CLAP head + balanced assignment", "clap+avg+bal", bold=[1]),
        row("CLAP head + self-training", "clap+avg+st"),
        row("LoRA encoder + CLAP head + balanced assignment", "lora_clap+avg+bal"),
    ]
    caption = ("<b>Table 1.</b> Test accuracy (%). Seed 0: submitted one-shot set; mean ± std: ten random sets. "
               "Bold: submitted model.")

    analysis = [
        "<b>Text prior and class balance do most of the work; one image refines them.</b> The balanced assignment "
        f"alone raises zero-shot accuracy from {f2(zs['seed0'])}% to {f2(zs_bal['seed0'])}%. An image prototype "
        f"alone reaches {pm(A['proto'])}% and Tip-Adapter {pm(A['tip'])}%; the CLAP head exceeds zero-shot by "
        f"{clap['mean'] - zs['seed0']:.1f} points on average, in {of10(count(clap, zs))}, mainly in Forest "
        f"({zs_ok['Forest']}{NB}→{NB}{ind_ok['Forest']} of 25 correct for seed 0) and InsideCity "
        f"({zs_ok['InsideCity']}{NB}→{NB}{ind_ok['InsideCity']}). From the raw images, the submitted model reaches "
        f"{acc['inductive']:.2f}% ({final['inductive']}/400) and, with the balanced assignment, "
        f"{acc['transductive']:.2f}% ({final['transductive']}/400); Table 1 (cached features) differs by one image.",

        "<b>Which image is drawn matters, and transductive methods amplify it.</b> The CLAP head ranges from "
        f"{f2(clap['min'])}% to {f2(clap['max'])}% over ten draws, and seed 0 is among the best. Even there, the "
        f"InsideCity image resembles a street, and Street falls from {zs_ok['Street']} to {ind_ok['Street']} "
        f"correct. Self-training and TransCLIP ({pm(trans)}%) reinforce such errors: self-training has the highest "
        f"mean but falls to {f2(st['min'])}% in the worst draw. The balanced assignment never falls below "
        f"{f2(bal['min'])}% and was therefore chosen.",

        "<b>The anchor matters; its weighting and the encoder size do not.</b> Without the anchor the head loses "
        f"{clap['mean'] - free['mean']:.1f} points ({pm(free)}%), but a constant {lam_c} = 1 equals the adaptive "
        f"weight ({pm(const)}%). LoRA lowers inductive accuracy in {of10(count(clap, lora))}; with the balanced "
        f"assignment it is better in {of10(count(lora_bal, bal))}, but not for seed 0. Generated variations "
        f"({pm(A['clap_var'])}%) and averaged test views ({pm(A['clap+avg'])}%) do not help, and encoders 2.7 and "
        f"4.4 times larger reach {float(bb['siglip2_g']['clap_mean']):.1f}% and "
        f"{float(bb['pe_g14']['clap_mean']):.1f}%.",

        "<b>Improvements found on the test set.</b> With the class-name prompt, balanced zero-shot accuracy "
        f"({f2(n_zs_bal['seed0'])}%) equals that of the submitted model, and the balanced CLAP head reaches "
        f"{f2(n_bal['seed0'])}% (mean {pm(n_bal)}%). LoRA makes it more reliable ({pm(n_lora_bal)}%, at worst "
        f"{f2(n_lora_bal['min'])}%); self-training is highest but unstable ({f2(n_st['seed0'])}% for seed 0, "
        f"{f2(n_st['min'])}% at worst). On the 16 training images, the prompts with synonyms fit better in "
        f"{of10(syn_fits)}. These variants, like the balanced assignment, were selected on test results and are "
        "upper estimates; the submitted model is unchanged.",
    ]

    conclusion = (
        f"One labeled image per class raises the accuracy of a frozen SigLIP 2 from {f2(zs['seed0'])}% to "
        f"{acc['inductive']:.2f}%, and from {f2(zs_bal['seed0'])}% to {acc['transductive']:.2f}% with the balanced "
        f"assignment; test-informed variants reach {n_bal['seed0']:.0f}–{n_lora_bal['mean']:.0f}%. <i>Model "
        "complexity:</i> the trainable part should be as small as the data; an 18k-parameter convex head is "
        "more reliable than 373k LoRA parameters or a larger encoder. <i>Data:</i> one image and the wording of a "
        "prompt each change accuracy by about a point, so results should be averaged over draws. "
        "<i>Computation:</i> one strong frozen encoder suffices; fine-tuning and image generation cost far more for "
        "no reliable gain.")

    blocks = [("h1", "1. Experiment Setup and Evaluation"), ("p", setup),
              ("h1", "2. Implementation Details"), ("h2", "2.1 Zero-shot baseline"), ("p", baseline),
              ("h2", "2.2 One-Shot Learning Strategy"), *[("p", t) for t in strategy],
              ("h1", "3. Results and Analysis"), ("table", None), *[("p", t) for t in analysis],
              ("h1", "4. Conclusion"), ("p", conclusion)]
    return {"meta": meta, "blocks": blocks, "table": table, "caption": caption}


# ----------------------------------------------------------------------------------------------- PDF

def pdf_fonts() -> dict:
    """Register Liberation Serif/Sans if all six files are found, else fall back to the built-in fonts."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    found = {k: next((Path(d) / f for d in FONT_DIRS if d and (Path(d) / f).is_file()), None)
             for k, f in FONT_FILES.items()}
    if not all(found.values()):
        print("Liberation fonts not found; using the built-in Times/Helvetica (same metrics)")
        return BUILTIN_FONTS
    names = {k: Path(f).stem for k, f in FONT_FILES.items()}
    for k, path in found.items():
        pdfmetrics.registerFont(TTFont(names[k], str(path)))
    pdfmetrics.registerFontFamily(names["serif"], normal=names["serif"], bold=names["serif_b"],
                                  italic=names["serif_i"], boldItalic=names["serif_bi"])
    return names


def render_pdf(c: dict) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Table, TableStyle

    F = pdf_fonts()
    body = ParagraphStyle("body", fontName=F["serif"], fontSize=12, leading=13.8, alignment=TA_JUSTIFY,
                          spaceAfter=6)
    title = ParagraphStyle("title", fontName=F["serif_b"], fontSize=17.5, leading=20, alignment=TA_CENTER,
                           spaceAfter=8.5)
    h1 = ParagraphStyle("h1", fontName=F["sans_b"], fontSize=13, leading=15, spaceBefore=8.2, spaceAfter=1.7)
    h2 = ParagraphStyle("h2", fontName=F["sans_bi"], fontSize=12, leading=13.8, spaceBefore=6.2, spaceAfter=0.8,
                        textColor=colors.HexColor(H2_COLOR))
    h2_first = ParagraphStyle("h2_first", parent=h2, spaceBefore=3)  # directly below a section heading
    cap = ParagraphStyle("cap", fontName=F["serif"], fontSize=10, leading=12, spaceBefore=4.4, spaceAfter=4)

    def P(text, style):  # the built-in fonts have no double bar
        return Paragraph(text.replace("‖", "||") if F is BUILTIN_FONTS else text, style)

    gray, last = colors.Color(0.6, 0.6, 0.6), len(c["table"]) - 1
    rows, heights, first_section = [], [], True
    style = [("FONT", (0, 0), (-1, -1), F["serif"], 10, 10), ("FONT", (0, 0), (-1, 1), F["serif_b"], 10, 10),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3.6), ("RIGHTPADDING", (0, 0), (-1, -1), 3.6),
             ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.9),
             ("BOTTOMPADDING", (0, 1), (-1, 1), 3.2), ("BOTTOMPADDING", (0, last), (-1, last), 3.5),
             ("SPAN", (1, 0), (2, 0)), ("SPAN", (3, 0), (4, 0)), ("LINEABOVE", (0, 0), (-1, 0), 1.25, colors.black),
             ("LINEBELOW", (1, 0), (2, 0), 0.25, gray), ("LINEBELOW", (3, 0), (4, 0), 0.25, gray),
             ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black),
             ("LINEBELOW", (0, last), (-1, last), 1.25, colors.black)]
    for i, r in enumerate(c["table"]):
        rows.append(r["cells"] + [""] * (5 - len(r["cells"])))
        heights.append(13.9 if r["kind"] == "header" else 13.6 if i == last else 13.0)
        style += [("FONT", (j, i), (j, i), F["serif_b"], 10, 10) for j in r.get("bold", ())]
        if r["kind"] == "section":
            style += [("SPAN", (0, i), (-1, i)), ("FONT", (0, i), (-1, i), F["serif_bi"], 10, 10),
                      ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f2f2"))]
            if not first_section:
                style.append(("LINEABOVE", (0, i), (-1, i), 0.25, gray))
            first_section = False
    table = Table(rows, colWidths=[w * inch for w in COL_WIDTHS_IN], rowHeights=heights, spaceAfter=5.5)
    table.setStyle(TableStyle(style))

    story, prev = [P(TITLE, title)], None
    for kind, text in c["blocks"]:
        if kind == "table":
            story[-1] = KeepTogether([story[-1], P(c["caption"], cap), table])
        elif kind == "h1":
            story.append(P(text, h1))
        elif kind == "h2":
            story.append(P(text, h2_first if prev == "h1" else h2))
        else:
            story.append(P(text, body))
        prev = kind

    author = c["meta"]["student_name"]
    doc = BaseDocTemplate(str(OUT_PDF), pagesize=letter, title=TITLE,
                          author="" if author == "STUDENT NAME" else author)
    frame = Frame(inch, inch, 6.5 * inch, 9 * inch + 1, leftPadding=0, rightPadding=0, topPadding=0,
                  bottomPadding=0)
    doc.addPageTemplates(PageTemplate(frames=[frame]))
    doc.build(story)
    print(f"wrote {OUT_PDF} ({doc.page} pages)")


# ---------------------------------------------------------------------------------------------- DOCX

class _Runs(HTMLParser):
    """Split the inline markup into (text, bold, italic, subscript, superscript, link) runs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.runs, self.state, self.link = [], {"b": 0, "i": 0, "sub": 0, "super": 0}, None

    def handle_starttag(self, tag, attrs):
        if tag in self.state:
            self.state[tag] += 1
        if tag == "link":
            self.link = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag in self.state:
            self.state[tag] -= 1
        if tag == "link":
            self.link = None

    def handle_data(self, data):
        s = self.state
        self.runs.append((data, s["b"] > 0, s["i"] > 0, s["sub"] > 0, s["super"] > 0, self.link))


def render_docx(c: dict) -> None:
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    # Times New Roman / Arial are shown as the metric-compatible Liberation Serif / Sans where they are missing
    serif, sans = "Times New Roman", "Arial"
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
    doc.core_properties.title = TITLE
    author = c["meta"]["student_name"]
    doc.core_properties.author = "" if author == "STUDENT NAME" else author

    def plain_fonts(rfonts, name):  # theme fonts would take precedence over the named font
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(k), name)
        for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            rfonts.attrib.pop(qn(k), None)

    def set_style(style, name, size, before, after, bold=None, italic=None, color=None):
        f, pf = style.font, style.paragraph_format
        f.name, f.size, f.bold, f.italic = name, Pt(size), bold, italic
        plain_fonts(style.element.rPr.find(qn("w:rFonts")), name)
        if color:
            f.color.rgb = RGBColor.from_string(color[1:].upper())
        pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0

    # Word and Writer add the space after a paragraph to the space before the next one
    for rfonts in doc.styles.element.find(qn("w:docDefaults")).iter(qn("w:rFonts")):
        plain_fonts(rfonts, serif)
    set_style(doc.styles["Normal"], serif, 12, 0, 6)
    set_style(doc.styles["Heading 1"], sans, 13, 2.2, 1.7, bold=True, italic=False, color="#000000")
    set_style(doc.styles["Heading 2"], sans, 12, 0.2, 0.8, bold=True, italic=True, color=H2_COLOR)

    def hyperlink(par, url, text):
        r_id = par.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        link = OxmlElement("w:hyperlink")
        link.set(qn("r:id"), r_id)
        run = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        for tag, val in (("w:color", "1155CC"), ("w:u", "single")):
            e = OxmlElement(tag)
            e.set(qn("w:val"), val)
            rpr.append(e)
        run.append(rpr)
        t = OxmlElement("w:t")
        t.text = text
        t.set(qn("xml:space"), "preserve")
        run.append(t)
        link.append(run)
        par._p.append(link)

    def add_markup(par, text, size=None, bold_all=False, italic_all=False):
        parser = _Runs()
        parser.feed(text)
        for txt, b, i, sub, sup, link in parser.runs:
            if link:
                hyperlink(par, link, txt)
                continue
            f = par.add_run(txt).font
            if size:
                f.size = Pt(size)
            if b or bold_all:
                f.bold = True
            if i or italic_all:
                f.italic = True
            if sub:
                f.subscript = True
            if sup:
                f.superscript = True
        return par

    def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, before=None, after=None):
        p = doc.add_paragraph()
        p.alignment = align
        if before is not None:
            p.paragraph_format.space_before = Pt(before)
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        return add_markup(p, text, size)

    def format_cell(cell, top=None, bottom=None, fill=None):
        """Horizontal rules (size in 1/8 pt, hex color) and shading; appended in schema order."""
        tcpr = cell._tc.get_or_add_tcPr()
        tb = OxmlElement("w:tcBorders")
        for edge, spec in (("top", top), ("left", None), ("bottom", bottom), ("right", None)):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single" if spec else "nil")
            if spec:
                e.set(qn("w:sz"), str(spec[0]))
                e.set(qn("w:color"), spec[1])
            tb.append(e)
        tcpr.append(tb)
        if fill:
            shd = OxmlElement("w:shd")
            for k, val in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", fill)):
                shd.set(qn(k), val)
            tcpr.append(shd)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    def add_table():
        cap = para(c["caption"], WD_ALIGN_PARAGRAPH.LEFT, 10, 2.7, 4)
        cap.paragraph_format.keep_with_next = True
        rows = c["table"]
        t = doc.add_table(rows=len(rows), cols=5)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False  # writes <w:tblLayout w:type="fixed"/>
        tblpr = t._tbl.tblPr
        tblpr.find(qn("w:tblW")).set(qn("w:type"), "dxa")
        tblpr.find(qn("w:tblW")).set(qn("w:w"), str(round(sum(COL_WIDTHS_IN) * 1440)))
        mar = OxmlElement("w:tblCellMar")
        for k, w in (("top", 0), ("left", 72), ("bottom", 0), ("right", 72)):
            e = OxmlElement(f"w:{k}")
            e.set(qn("w:w"), str(w))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tblpr.find(qn("w:tblLook")).addprevious(mar)  # schema order: tblCellMar precedes tblLook
        for col, w in zip(t.columns, COL_WIDTHS_IN):
            col.width = Inches(w)
        last, first_section = len(rows) - 1, True
        for i, r in enumerate(rows):
            t.rows[i].height, t.rows[i].height_rule = Pt(13), WD_ROW_HEIGHT_RULE.AT_LEAST
            cells = t.rows[i].cells
            for cell, w in zip(cells, COL_WIDTHS_IN):
                cell.width = Inches(w)
            kind, texts = r["kind"], r["cells"]
            top = (10, "000000") if i == 0 else ((2, "999999") if kind == "section" and not first_section else None)
            bottom = (4, "000000") if kind == "header" else ((10, "000000") if i == last else None)
            if kind == "section":
                cells, first_section = [cells[0].merge(cells[4])], False
            elif kind == "group":  # one label above each pair of columns
                cells = [cells[0], cells[1].merge(cells[2]), cells[3].merge(cells[4])]
                texts = [texts[0], texts[1], texts[3]]
            for j, (cell, text) in enumerate(zip(cells, texts)):
                format_cell(cell, top, (2, "999999") if kind == "group" and j else bottom,
                            "F2F2F2" if kind == "section" else None)
                cp = cell.paragraphs[0]
                cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                cp.paragraph_format.space_after = Pt(0)
                cp.paragraph_format.keep_with_next = i < last
                add_markup(cp, text, 10, bold_all=kind != "data" or j in r["bold"], italic_all=kind == "section")

    para(f"<b>{TITLE}</b>", WD_ALIGN_PARAGRAPH.CENTER, 17.5)
    prev = None
    for kind, text in c["blocks"]:
        if kind == "table":
            add_table()
        elif kind in ("h1", "h2"):
            h = doc.add_heading(text, level=int(kind[1]))
            if (prev, kind) == ("h1", "h2"):  # directly below a section heading
                h.paragraph_format.space_before = Pt(1.3)
        else:
            para(text, before=5.5 if prev == "table" else None)
        prev = kind

    zoom = doc.settings.element.find(qn("w:zoom"))  # default template omits the required percent
    if zoom is not None:
        zoom.set(qn("w:percent"), "100")
    doc.save(OUT_DOCX)
    print(f"wrote {OUT_DOCX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--docx", action="store_true")
    a = ap.parse_args()
    c = content()
    if a.pdf or not a.docx:
        render_pdf(c)
    if a.docx or not a.pdf:
        try:
            import docx  # noqa: F401
        except ImportError:
            if a.docx:
                raise
            print(f"python-docx is not installed: {OUT_DOCX.name} was NOT rebuilt"
                  + (" and may no longer match the PDF" if OUT_DOCX.exists() else ""))
        else:
            render_docx(c)


if __name__ == "__main__":
    main()
