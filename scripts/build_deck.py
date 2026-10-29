"""Build the team deck from the course template.

Usage (from final-team/):
    .venv/bin/python scripts/build_deck.py [--team N] [--member1 "Фамилия Имя"] [--member2 "Фамилия Имя"]
                                           [--github URL] [--space URL] [--out FILE] [--check] [--dump FILE]

Every metric on the slides is read from reports/*.json (and a few tables from reports/*.md) at
build time, so the script can simply be re-run when new results land. Missing optional reports
(xlmr_results.json, final_choice.json, related_work.md) are rendered as "в работе" placeholders.
The template file is only read, never written.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import re
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMG = REPORTS / "img"
TEMPLATE = ROOT.parent / "tasks" / "Итоговое задание (Презентация-описание работы).pptx"

FONT = "Arial"           # the template uses Proxima Nova, which only Google Slides has
SLIDE_W, SLIDE_H = 9144000, 5143500
LEFT = 357000
RIGHT = 8790000
DECOR_X = 6400000          # body text must stay left of this on slides with right-side decor
TABLE_STYLE = "{99CE2E37-16BC-4408-9B30-6D05F93488BB}"   # the template's own "Table_0"
HEADER_FILL = RGBColor(0xEE, 0xEF, 0xF2)
CHOSEN_FILL = RGBColor(0xE3, 0xEC, 0xFA)
GREY = RGBColor(0x60, 0x60, 0x60)

PREC_MIN, FPR_MAX = 0.99, 0.005   # threshold-selection constraints used in train_classic.py
PENDING = "в работе"
MIN_PT = 12.0
BODY_PT = 14.0           # body text on content slides
TABLE_PT = 12.0
TOP = 760000             # first content line under a slide title
BOTTOM = SLIDE_H - 150000
GAP = 150000             # vertical space between blocks
LINK_COLOR = RGBColor(0x1A, 0x56, 0xDB)
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)|(https?://[^\s<>»)]*[^\s<>»).,;:])")
TITLE_SRC: list = []     # template title shape cloned onto added slides
TOUCHED: list = []       # txBody elements written by this script (only these are overflow-checked)
BROKEN_WORDS: list = []  # table words wider than their column: the renderer splits them mid-word

MODEL_NAMES = {
    "logreg": "TF-IDF + LogReg",
    "linsvc": "TF-IDF + LinearSVC",
    "cnb": "TF-IDF + ComplementNB",
    "e5": "e5-small + LogReg",
    "xlmr": "XLM-R (дообучение)",
}
MODEL_ORDER = ["logreg", "linsvc", "cnb", "e5", "xlmr"]

CLASS_RU_FALLBACK = {
    "ads": "реклама", "job": "предложение работы", "dating": "знакомства",
    "finance": "финансы/мошенничество", "phishing": "фишинг",
    "newsletter": "рассылки/уведомления", "other": "другое",
}
STUDENT_NAMES = {"tfidf_logreg": "TF-IDF + LogReg", "e5_logreg": "e5-small + LogReg",
                 "ensemble": "среднее двух учеников"}
STUDENT_DESC = {"tfidf_logreg": "TF-IDF слов и символов + LogReg",
                "e5_logreg": "эмбеддинги e5-small + LogReg",
                "ensemble": "среднее вероятностей TF-IDF + LogReg и e5-small + LogReg"}
NLI_NAMES = {"zs": "mDeBERTa-v3-base-xnli", "mdeberta": "mDeBERTa-v3-base-xnli",
             "large": "DeBERTa-v3-large-zeroshot", "avg": "среднее mDeBERTa и DeBERTa-large"}


# ----------------------------------------------------------------------------- data loading

def load_json(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[warn] cannot read {p}: {e}", file=sys.stderr)
        return None


def read_text(name: str) -> str:
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def md_table_after(md: str, marker: str) -> list[list[str]]:
    """Rows (header first, separator dropped) of the first markdown table after a marker line."""
    lines = md.splitlines()
    start = next((i for i, l in enumerate(lines) if marker in l), None)
    if start is None:
        return []
    rows: list[list[str]] = []
    for l in lines[start + 1:]:
        s = l.strip()
        if not s.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip().replace("**", "") for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def regex(md: str, pattern: str, default=None):
    m = re.search(pattern, md, flags=re.S)
    return m.groups() if m else default


class Data:
    def __init__(self) -> None:
        self.classic = load_json("classic_results.json") or {}
        self.e5 = load_json("e5small_results.json")
        self.xlmr = load_json("xlmr_results.json")
        self.cat = load_json("category_results.json")
        self.spambase = load_json("spambase_results.json")
        self.final = load_json("final_choice.json")
        self.data_md = read_text("data.md")
        self.exp_md = read_text("experiments.md")
        self.smoke_md = read_text("service_smoke.md")
        self.related_md = read_text("related_work.md")
        self.latency = load_json("latency.json") or {}
        mp = ROOT / "models" / "final_classic.joblib"
        self.model_mb = f"{mp.stat().st_size / 2**20:.1f}".replace(".", ",") if mp.exists() else None

    def results(self, key: str):
        """{'cv': {'default', 'tuned'}, 'test': {...}} for a model key, or None if not available."""
        if key in ("logreg", "linsvc", "cnb"):
            try:
                return {"cv": self.classic["cv"][key], "test": self.classic["test"][key]}
            except KeyError:
                return None
        src = {"e5": self.e5, "xlmr": self.xlmr}.get(key)
        if src and "cv" in src and "test" in src:
            return {"cv": src["cv"], "test": src["test"]}
        return None

    @staticmethod
    def norm_key(name: str) -> str:
        n = name.lower()
        for pat, key in (("xlm", "xlmr"), ("e5", "e5"), ("svc", "linsvc"), ("svm", "linsvc"),
                         ("cnb", "cnb"), ("complement", "cnb"), ("nb", "cnb"), ("logreg", "logreg"),
                         ("logistic", "logreg")):
            if pat in n:
                return key
        return n

    def chosen(self) -> tuple[str, str | None]:
        if self.final and self.final.get("model"):
            key = self.norm_key(str(self.final["model"]))
            if self.results(key):
                return key, self.final.get("reason_ru")
        return self.classic.get("chosen", {}).get("model", "linsvc"), None


def class_ru() -> dict:
    try:
        sys.path.insert(0, str(ROOT))
        from src.category.common import CLASS_RU  # noqa: E402
        return dict(CLASS_RU)
    except Exception:
        return CLASS_RU_FALLBACK
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)


# ----------------------------------------------------------------------------- formatting

def num(x, nd: int = 4) -> str:
    return PENDING if x is None else f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd: int = 2) -> str:
    return PENDING if x is None else f"{x * 100:.{nd}f}%".replace(".", ",")


def intru(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ----------------------------------------------------------------------------- pptx helpers

def set_run_font(run, size: float, bold: bool = False, color: RGBColor | None = None) -> None:
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = False
    if color is not None:
        f.color.rgb = color
    else:
        f.color.theme_color = MSO_THEME_COLOR.DARK_1
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)


def plain(text: str) -> str:
    """Text as rendered: link markup reduced to its label, bold markers dropped."""
    return LINK_RE.sub(lambda m: m.group(1) or m.group(3), text).replace("**", "")


def add_runs(p, text: str, size: float, bold: bool = False, color=None) -> None:
    """Text with **bold** segments as runs; [label](url) and bare http(s) URLs become hyperlinks."""
    def run(t: str, b: bool, url: str | None = None) -> None:
        if not t:
            return
        r = p.add_run()
        r.text = t
        set_run_font(r, size, bold=b, color=LINK_COLOR if url else color)
        if url:
            r.hyperlink.address = url
            r.font.underline = True

    for i, part in enumerate(text.split("**")):
        b, pos = bold or i % 2 == 1, 0
        for m in LINK_RE.finditer(part):
            run(part[pos:m.start()], b)
            run(m.group(1) or m.group(3), b, m.group(2) or m.group(3))
            pos = m.end()
        run(part[pos:], b)


def set_bullet(p, char: str | None, marl: int = 171450) -> None:
    pPr = p._p.get_or_add_pPr()
    for tag in ("a:buFont", "a:buChar", "a:buNone", "a:buAutoNum"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    if char is None:
        pPr.set("marL", "0")
        pPr.set("indent", "0")
        etree.SubElement(pPr, qn("a:buNone"))
    else:
        pPr.set("marL", str(marl))
        pPr.set("indent", str(-marl))
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def clear_tf(tf) -> None:
    txBody = tf._txBody
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)


def write_tf(tf, items: list[str], size: float = 12, color=None, align=None) -> None:
    """Fill a text frame. Item markup: '## heading', '- bullet', '' small gap, '~ caption', else plain."""
    clear_tf(tf)
    tf.word_wrap = True
    TOUCHED.append(tf._txBody)
    first = True
    for item in items:
        p = tf._txBody.add_p()
        from pptx.text.text import _Paragraph
        para = _Paragraph(p, tf)
        para.line_spacing = 1.0
        if item == "":
            para.space_before = Pt(0)
            set_bullet(para, None)
            r = para.add_run()
            r.text = ""
            set_run_font(r, size * 0.5)
            first = False
            continue
        if item.startswith("## "):
            para.space_before = Pt(0 if first else 5)
            set_bullet(para, None)
            add_runs(para, item[3:], size, bold=True, color=color)
        elif item.startswith("- "):
            para.space_before = Pt(2)
            set_bullet(para, "•")
            add_runs(para, item[2:], size, color=color)
        elif item.startswith("~- "):
            para.space_before = Pt(1)
            set_bullet(para, "•")
            add_runs(para, item[3:], max(MIN_PT, size - 1), color=GREY)
        elif item.startswith("~ "):
            para.space_before = Pt(0)
            set_bullet(para, None)
            add_runs(para, item[2:], max(MIN_PT, size - 1), color=GREY)
        else:
            para.space_before = Pt(0 if first else 3)
            set_bullet(para, None)
            add_runs(para, item, size, color=color)
        if align is not None:
            para.alignment = align
        first = False


def next_shape_id(slide) -> int:
    ids = [int(el.get("id")) for el in slide.shapes._spTree.iter(qn("p:cNvPr")) if el.get("id", "").isdigit()]
    return max(ids + [1]) + 1


def add_textbox(slide, left, top, width, height, items, size=12, name=None, color=None, align=None):
    tb = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = tb.text_frame
    bp = tf._txBody.bodyPr
    for k in ("lIns", "tIns", "rIns", "bIns"):
        bp.set(k, "0")
    tf.vertical_anchor = MSO_ANCHOR.TOP
    write_tf(tf, items, size, color=color, align=align)
    if name:
        tb.name = name
    return tb


def add_table(slide, rows: list[list[str]], left, top, col_w: list[int], size=10.5,
              header=True, bold_rows=(), fill_rows=(), align_num=True, name=None):
    n_r, n_c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_r, n_c, Emu(left), Emu(top), Emu(sum(col_w)), Emu(200000 * n_r))
    if name:
        gf.name = name
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for k in ("firstRow", "bandRow", "firstCol", "lastRow", "lastCol", "bandCol"):
        tblPr.set(k, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = TABLE_STYLE
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Emu(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Emu(54000)
            cell.margin_top = cell.margin_bottom = Emu(22000)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_head = header and i == 0
            al = PP_ALIGN.LEFT if (j == 0 or not align_num) else PP_ALIGN.CENTER
            # An empty cell gets a space: LibreOffice sizes an empty paragraph at 18pt and the row grows.
            write_tf(cell.text_frame, [str(val) or " "], size, align=al)
            for word in str(val).split():
                if word_w(word, size, is_head or i in bold_rows) > (col_w[j] - 108000) / 12700:
                    BROKEN_WORDS.append(f"table '{name}' col {j}: '{word}' does not fit")
            if is_head or i in bold_rows:
                for r in cell.text_frame.paragraphs[0].runs:
                    r.font.bold = True
            if is_head:
                cell.fill.solid()
                cell.fill.fore_color.rgb = HEADER_FILL
            elif i in fill_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = CHOSEN_FILL
            else:
                cell.fill.background()
    # Row heights from the text estimate, so the declared frame matches what will be rendered.
    total = 0
    for i, row in enumerate(rows):
        bold = (header and i == 0) or i in bold_rows
        h = max(est_cell_height(str(v), size, col_w[j] - 108000, bold) for j, v in enumerate(row)) + 44000
        tbl.rows[i].height = Emu(int(h))
        total += int(h)
    gf.height = Emu(total)
    return gf


def auto_col_widths(rows: list[list[str]], size: float, avail: int, header: bool = True,
                    bold_rows=()) -> list[int]:
    """Column widths that keep body cells on one line and never break a word, filling `avail`."""
    pad = 108000 + 30000
    nat, minw = [], []
    for j in range(len(rows[0])):
        col = [str(r[j]) for r in rows]
        bold = [(header and i == 0) or i in bold_rows for i in range(len(col))]
        body = [word_w(v, size, bold[i]) for i, v in enumerate(col) if not (header and i == 0)]
        words = [word_w(w, size, bold[i]) for i, v in enumerate(col) for w in v.split()]
        nat.append(int(max(body + words + [0]) * 12700) + pad)
        minw.append(int(max(words + [0]) * 12700) + pad)
    total = sum(nat)
    if total <= avail:
        return [w + (avail - total) * w // total for w in nat]
    widths, over = nat[:], total - avail
    for j in sorted(range(len(widths)), key=lambda k: widths[k] - minw[k], reverse=True):
        cut = min(over, widths[j] - minw[j])
        widths[j] -= cut
        over -= cut
        if over <= 0:
            break
    return widths


def table_height(rows: list[list[str]], col_w: list[int], size: float, header: bool = True,
                 bold_rows=()) -> int:
    return sum(max(est_cell_height(str(v), size, col_w[j] - 108000, (header and i == 0) or i in bold_rows)
                   for j, v in enumerate(row)) + 44000 for i, row in enumerate(rows))


def crop_panel(img: Image.Image, pad: int = 16) -> Image.Image:
    """Crop a page screenshot to its white card (the app itself) plus a small margin."""
    box = img.convert("L").point(lambda v: 255 if v >= 254 else 0).getbbox()
    if not box:
        return img
    l, t, r, b = box
    return img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))


def trim_bg(img: Image.Image, pad: int = 16, tol: int = 8) -> Image.Image:
    """Cut the uniform background (the colour of the top-left pixel) around a screenshot."""
    from PIL import ImageChops
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    box = ImageChops.difference(img, bg).convert("L").point(lambda v: 255 if v > tol else 0).getbbox()
    if not box:
        return img
    l, t, r, b = box
    return img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))


def add_picture_fit(slide, img: Image.Image, left, top, max_w, max_h, border=True):
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    pw, ph = int(w * scale), int(h * scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    pic = slide.shapes.add_picture(buf, Emu(left), Emu(top), Emu(pw), Emu(ph))
    if border:
        pic.line.color.rgb = RGBColor(0xC4, 0xC4, 0xC4)
        pic.line.width = Pt(0.75)
    return pic


def open_img(name: str, crop=None) -> Image.Image | None:
    p = IMG / name
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    if crop:
        l, t, r, b = crop
        im = im.crop((l, t, min(r, im.width), min(b, im.height)))
    return im


def shape_by_name(slide, name: str):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    raise KeyError(name)


def remove_shape(shape) -> None:
    el = shape._element
    el.getparent().remove(el)


def delete_slide(prs, slide) -> None:
    lst = prs.slides._sldIdLst
    for sldId in list(lst):
        if prs.part.related_part(sldId.rId) is slide.part:
            prs.part.drop_rel(sldId.rId)
            lst.remove(sldId)
            return


def new_slide_after(prs, after, title: str, title_src):
    """Content slide on layout '1.Фон' with a title box cloned from a template title, placed after `after`."""
    layout = next(l for l in prs.slide_layouts if l.name == "1.Фон")
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        remove_shape(ph)
    el = copy.deepcopy(title_src._element)
    s.shapes._spTree.append(el)
    tshape = s.shapes[-1]
    el.find(".//" + qn("p:cNvPr")).set("id", str(next_shape_id(s)))
    tshape.name = "Title"
    runs = tshape.text_frame.paragraphs[0].runs
    runs[0].text = title
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    # A long title would wrap onto the content below: shrink it to one line.
    sz = runs[0].font.size.pt if runs[0].font.size else 28.0
    avail = (int(tshape.width) - 2 * 91440) / 12700
    need = word_w(title, sz, True)
    if need > avail:
        runs[0].font.size = Pt(round(sz * avail / need * 0.97, 1))
    lst = prs.slides._sldIdLst
    new_el = lst[-1]
    lst.remove(new_el)
    ids = list(lst)
    pos = next(i for i, sid in enumerate(ids) if prs.part.related_part(sid.rId) is after.part)
    lst.insert(pos + 1, new_el)
    return s


def clear_notes(prs) -> None:
    for s in prs.slides:
        if s.has_notes_slide:
            tf = s.notes_slide.notes_text_frame
            if tf is not None:
                tf.text = ""


# ----------------------------------------------------------------------------- overflow estimate

def char_w(ch: str, size: float) -> float:
    if ch == " " or ch == " ":
        return 0.27 * size
    if ch in ".,:;!|'il1()[]":
        return 0.3 * size
    if ch.isupper() or ch.isdigit() or ch in "%@&MWmwЖШЩЮжшщю":
        return 0.64 * size
    return 0.54 * size


_FONTS: dict = {}


def word_w(word: str, size: float, bold: bool) -> float:
    """Width in pt measured with Liberation Sans (same metrics as Arial); char_w estimate without it."""
    from PIL import ImageFont
    if bold not in _FONTS:
        name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
        found = [d / name for d in (Path.home() / ".local/share/fonts", Path("/usr/share/fonts/truetype/liberation"))
                 if (d / name).exists()]
        _FONTS[bold] = ImageFont.truetype(str(found[0]), 100) if found else None
    f = _FONTS[bold]
    if f is None:
        return sum(char_w(c, size) for c in word) * (1.1 if bold else 1.0)
    return f.getlength(word) * size / 100


def wrap_lines(text: str, size: float, width_pt: float, bold: bool = False) -> int:
    total = 0
    sp = word_w(" ", size, bold)
    for raw in re.split(r"[\n\x0b]", text):
        n, cur = 1, 0.0
        for word in raw.split(" "):
            w = word_w(word, size, bold)
            if cur == 0:
                cur = w
            elif cur + sp + w <= width_pt:
                cur += sp + w
            else:
                n += 1
                cur = w
            while cur > width_pt and width_pt > 0:
                n += 1
                cur -= width_pt
        total += n
    return total


def est_cell_height(text: str, size: float, width_emu: int, bold: bool = False) -> int:
    lines = wrap_lines(text, size, width_emu / 12700, bold)
    return int(lines * size * 1.2 * 12700)


def para_size(p, default=18.0) -> float:
    for r in p.runs:
        if r.font.size:
            return r.font.size.pt
    return default


def est_tf_height(tf, width_emu: int) -> int:
    h = 0.0
    for p in tf.paragraphs:
        size = para_size(p)
        pPr = p._p.pPr
        marl = int(pPr.get("marL", "0")) if pPr is not None else 0
        sb = p.space_before.pt if p.space_before is not None else 0
        ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
        lines = wrap_lines(p.text, size, (width_emu - marl) / 12700)
        h += sb + lines * size * 1.2 * ls
    return int(h * 12700)


def est_items_height(items: list[str], size: float, width_emu: int) -> int:
    """Height write_tf() will need for these items, in EMU (same markup and spacing rules)."""
    h, first = 0.0, True
    for item in items:
        if item == "":
            h += size * 0.5 * 1.2
            first = False
            continue
        s, sb, marl, bold, text = size, (0 if first else 3), 0, False, item
        if item.startswith("## "):
            sb, bold, text = (0 if first else 5), True, item[3:]
        elif item.startswith("- "):
            sb, marl, text = 2, 171450, item[2:]
        elif item.startswith("~- "):
            s, sb, marl, text = max(MIN_PT, size - 1), 1, 171450, item[3:]
        elif item.startswith("~ "):
            s, sb, text = max(MIN_PT, size - 1), 0, item[2:]
        h += sb + wrap_lines(plain(text), s, (width_emu - marl) / 12700, bold) * s * 1.2
        first = False
    return int(h * 12700 * 1.04)


def insets(tf) -> tuple[int, int, int, int]:
    bp = tf._txBody.bodyPr
    return (int(bp.get("lIns", 91440)), int(bp.get("tIns", 45720)),
            int(bp.get("rIns", 91440)), int(bp.get("bIns", 45720)))


def check_deck(prs) -> list[str]:
    warns: list[str] = list(BROKEN_WORDS)
    for si, s in enumerate(prs.slides, 1):
        rects = []
        for sh in s.shapes:
            l, t, w, h = int(sh.left or 0), int(sh.top or 0), int(sh.width or 0), int(sh.height or 0)
            kind = "decor"
            bottom = t + h
            if sh.has_text_frame and sh.text_frame.text.strip() and not any(
                    x is sh.text_frame._txBody for x in TOUCHED):
                kind = "template"
            elif sh.has_text_frame and sh.text_frame.text.strip():
                li, ti, ri, bi = insets(sh.text_frame)
                need = est_tf_height(sh.text_frame, w - li - ri) + ti + bi
                if need > h:
                    warns.append(f"slide {si}: '{sh.name}' text needs ~{need / 914400 * 2.54:.2f} cm, "
                                 f"box is {h / 914400 * 2.54:.2f} cm")
                for p in sh.text_frame.paragraphs:
                    sz = para_size(p, 0)
                    if p.text.strip() and sz and sz < MIN_PT:
                        warns.append(f"slide {si}: '{sh.name}' font {sz}pt < 10pt")
                kind = "text"
                bottom = t + need
            elif sh.has_table:
                kind = "table"
                for ri_, row in enumerate(sh.table.rows):
                    for ci, cell in enumerate(row.cells):
                        cw = sh.table.columns[ci].width - cell.margin_left - cell.margin_right
                        need = est_tf_height(cell.text_frame, cw) + cell.margin_top + cell.margin_bottom
                        if need > row.height + 10000:
                            warns.append(f"slide {si}: table '{sh.name}' row {ri_} col {ci} needs more height")
                bottom = t + sum(r.height for r in sh.table.rows)
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                kind = "picture"
            elif sh.has_text_frame:
                kind = "decor"   # empty ellipse etc.
            if bottom > SLIDE_H - 60000 and kind not in ("decor", "template"):
                warns.append(f"slide {si}: '{sh.name}' bottom {bottom / 914400 * 2.54:.2f} cm "
                             f"is past the slide edge {SLIDE_H / 914400 * 2.54:.2f} cm")
            if l + w > SLIDE_W:
                warns.append(f"slide {si}: '{sh.name}' sticks out on the right")
            rects.append((sh.name, kind, l, t, l + w, bottom))
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if {a[1], b[1]} <= {"decor", "template"}:
                    continue
                if a[2] < b[4] and b[2] < a[4] and a[3] < b[5] and b[3] < a[5]:
                    warns.append(f"slide {si}: '{a[0]}' ({a[1]}) overlaps '{b[0]}' ({b[1]})")
    return warns


def dump_text(prs) -> str:
    out = []
    for si, s in enumerate(prs.slides, 1):
        out.append(f"=== Слайд {si}")
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                out.append(sh.text_frame.text.replace("\x0b", "\n"))
            elif sh.has_table:
                for row in sh.table.rows:
                    out.append(" | ".join(c.text_frame.text for c in row.cells))
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                out.append("[картинка]")
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- slide content

def fill_title(T, args) -> None:
    box = shape_by_name(T[0], "Google Shape;120;p26")
    paras = box.text_frame.paragraphs
    for p, text in ((paras[1], f"Команда {args.team}"), (paras[2], f"{args.member1}, {args.member2}")):
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    for p in (paras[0], paras[3]):
        p._p.getparent().remove(p._p)
    TOUCHED.append(box.text_frame._txBody)


def fill_roles(T, args) -> None:
    s = T[4]
    remove_shape(shape_by_name(s, "Google Shape;151;p30"))
    gf = shape_by_name(s, "Google Shape;152;p30")
    tbl = gf.table
    trs = tbl._tbl.tr_lst
    for tr in trs[3:]:
        tbl._tbl.remove(tr)
    col_w = [2300000, 6130000]
    tbl._tbl.tblGrid.findall(qn("a:gridCol"))[0].set("w", str(col_w[0]))
    tbl._tbl.tblGrid.findall(qn("a:gridCol"))[1].set("w", str(col_w[1]))
    rows = [
        ("Фамилия Имя", "Описание работы"),
        (args.member1, "ML-инженер, Data Engineer, Data Scientist. Сбор и очистка 4 корпусов, "
                       "эксперименты, сервис на FastAPI в Docker, публикация на Hugging Face"),
        (args.member2, "Аналитик и менеджер. Постановка задачи, обзор решений, метрики и порог, "
                       "проверка разметки категорий, защита"),
    ]
    for i, (a, b) in enumerate(rows):
        for j, val in enumerate((a, b)):
            cell = tbl.cell(i, j)
            tf = cell.text_frame
            clear_tf(tf)
            p = tf._txBody.add_p()
            from pptx.text.text import _Paragraph
            para = _Paragraph(p, tf)
            para.alignment = PP_ALIGN.CENTER if i == 0 else PP_ALIGN.LEFT
            r = para.add_run()           # one plain run per cell: easy to edit by hand
            r.text = val
            set_run_font(r, 16, bold=(i == 0))
    total = 0
    for i, row in enumerate(rows):
        h = max(est_cell_height(v, 16, col_w[j] - 182850) for j, v in enumerate(row)) + 182850
        tbl.rows[i].height = Emu(h)
        total += h
    gf.left, gf.top, gf.width, gf.height = Emu(LEFT), Emu(950000), Emu(sum(col_w)), Emu(total)


class Flow:
    """Places text, tables and pictures top to bottom on a slide and opens a continuation slide
    (same title plus "(продолжение)") when the next block does not fit."""

    def __init__(self, prs, slide, cont_title: str, top: int = TOP, x: int = LEFT, first_width: int | None = None):
        self.prs, self.slide, self.cont_title = prs, slide, cont_title
        self.y, self.x = max(top, title_bottom(slide) + 60000), x
        self.width = first_width or (RIGHT - x)
        self.used = False

    def room(self) -> int:
        return BOTTOM - self.y

    def new_slide(self) -> None:
        self.slide = new_slide_after(self.prs, self.slide, self.cont_title, TITLE_SRC[0])
        self.y, self.x, self.width, self.used = TOP, LEFT, RIGHT - LEFT, False

    def text(self, items: list[str], size: float = BODY_PT, name: str | None = None) -> None:
        chunk: list[str] = []

        def flush() -> None:
            if chunk:
                h = est_items_height(chunk, size, self.width)
                add_textbox(self.slide, self.x, self.y, self.width, h, list(chunk), size=size, name=name)
                self.y += h + GAP
                self.used = True
                chunk.clear()

        i = 0
        while i < len(items):
            # a heading moves together with the line after it
            group = items[i:i + 2] if items[i].startswith("## ") and i + 1 < len(items) else [items[i]]
            if est_items_height(chunk + group, size, self.width) > self.room() and (chunk or self.used):
                # break at the start of the current section rather than leave its tail alone on the next slide
                k = max((j for j, it in enumerate(chunk) if it.startswith("## ")), default=0)
                if k > 0 and not items[i].startswith("## "):
                    i -= len(chunk) - k
                    del chunk[k:]
                flush()
                self.new_slide()
                if items[i] == "":
                    i += 1
                continue
            chunk.extend(group)
            i += len(group)
        flush()

    def fit_table(self, rows, room: int, bold_rows=(), max_size: float = BODY_PT) -> tuple[float, list[int], int] | None:
        """Largest size from max_size down to TABLE_PT that fits `room`: first with every body cell on
        one line, then allowing cells to wrap as long as no word is broken."""
        for one_line in (True, False):
            size = max_size
            while size >= TABLE_PT:
                col_w = auto_col_widths(rows, size, self.width, bold_rows=bold_rows)
                need = 0
                for j in range(len(rows[0])):
                    cells = [(str(r[j]), i == 0 or i in bold_rows) for i, r in enumerate(rows)]
                    words = [word_w(w, size, b) for v, b in cells for w in v.split()]
                    body = [word_w(v, size, b) for i, (v, b) in enumerate(cells) if i] if one_line else []
                    need += int(max(words + body + [0]) * 12700) + 138000
                h = table_height(rows, col_w, size, bold_rows=bold_rows)
                if need <= self.width and h <= room:
                    return size, col_w, h
                size -= 1
        return None

    def table(self, rows: list[list[str]], name: str | None = None, reserve: int = 0,
              max_size: float = BODY_PT, caption: str | None = None, **kw) -> None:
        """`reserve`: height kept free below the table for what follows it on the same slide.
        `caption` goes right under the table and moves with it."""
        bold_rows = kw.get("bold_rows", ())
        cap = [f"~ {caption}"] if caption else []
        cap_h = est_items_height(cap, BODY_PT, self.width) + 40000 if cap else 0
        fit = (self.fit_table(rows, self.room() - reserve - cap_h, bold_rows, max_size)
               or self.fit_table(rows, self.room() - cap_h, bold_rows, max_size))
        if fit is None and self.used:
            self.new_slide()
            fit = self.fit_table(rows, self.room() - cap_h, bold_rows, max_size)
        if fit is None:
            col_w = auto_col_widths(rows, TABLE_PT, self.width, bold_rows=bold_rows)
            fit = TABLE_PT, col_w, table_height(rows, col_w, TABLE_PT, bold_rows=bold_rows)
        size, col_w, h = fit
        add_table(self.slide, rows, self.x, self.y, col_w, size=size, name=name, **kw)
        self.y += h
        if cap:
            self.y += 40000
            add_textbox(self.slide, self.x, self.y, self.width, cap_h - 40000, cap, size=BODY_PT,
                        name=f"{name}_caption" if name else None)
            self.y += cap_h - 40000
        self.y += GAP
        self.used = True

    def picture(self, img: Image.Image, caption: str | None = None, name: str | None = None) -> None:
        cap_h = est_items_height([f"~ {caption}"], BODY_PT, self.width) + 40000 if caption else 0
        scale = min(self.width / img.width, (self.room() - cap_h) / img.height)
        if self.used and scale * img.height < 0.6 * (BOTTOM - TOP - cap_h):
            self.new_slide()
        pic = add_picture_fit(self.slide, img, self.x, self.y, self.width, self.room() - cap_h)
        pic.left = Emu(self.x + (self.width - pic.width) // 2)
        if name:
            pic.name = name
        self.y += pic.height + 40000
        if caption:
            add_textbox(self.slide, self.x, self.y, self.width, cap_h, [f"~ {caption}"], size=BODY_PT,
                        align=PP_ALIGN.CENTER)
            self.y += cap_h
        self.y += GAP
        self.used = True


def title_bottom(slide) -> int:
    """Lowest edge of the title text boxes (those starting in the top strip of the slide)."""
    return max([int(sh.top) + int(sh.height) for sh in slide.shapes
                if sh.has_text_frame and sh.text_frame.text.strip() and int(sh.top or 0) < 600000] + [0])


def decor_width(x: int = LEFT) -> int:
    """Text width that stays left of the template's illustration in the lower right corner."""
    return DECOR_X - x - 150000


def fill_task(prs, T, D: Data):
    lat = D.latency
    items = [
        "## Задача на языке ML",
        "- Пользователь не должен тратить время на спам и попадаться на фишинг, но и терять нужные "
        "письма нельзя. Цена ошибок асимметрична.",
        "- Бинарная классификация текста письма, спам (1) или нет (0). На выходе вероятность и решение "
        "по порогу.",
        "- Вторая модель относит письмо к одной из 7 категорий.",
        "## Вход модели",
        "- Тема и тело после разбора MIME (HTML в текст, base64, quoted-printable).",
        "## Требуемая скорость",
        f"- Десятки мс на письмо на CPU. TF-IDF + LinearSVC тратит {num(lat.get('linsvc_batch1_ms'), 1)} мс "
        f"на письмо по одному и {num(lat.get('linsvc_batch1600_ms_per_email'), 1)} мс в пакете из 1600.",
        "## Метрики и почему",
        "- Precision и FPR важнее всего, потерять нужное письмо дороже, чем пропустить спам.",
        "- Также Recall, F1, ROC-AUC и PR-AUC.",
        f"- Порог берем по OOF 5-fold CV, максимум Recall при Precision ≥ {num(PREC_MIN, 2)} "
        f"и FPR ≤ {pct(FPR_MAX, 1)}.",
    ]
    remove_shape(shape_by_name(T[7], "Google Shape;176;p33"))
    f = Flow(prs, T[7], "Описание задачи (продолжение)", first_width=decor_width())
    f.text(items, name="task_text")
    return f.slide


def parse_related(md: str) -> tuple[str, list[str]] | None:
    if not md.strip():
        return None
    paras, sources, buf = [], [], []
    for line in md.splitlines():
        s = line.strip()
        if re.match(r"^([-*]|\d+[.)])\s+", s):
            sources.append(re.sub(r"^([-*]|\d+[.)])\s+", "", s))
        elif s.startswith("#"):
            continue
        elif s:
            buf.append(s)
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    if not paras and not sources:
        return None
    return (paras[0] if paras else ""), sources[:5]


def fill_related(prs, T, D: Data):
    remove_shape(shape_by_name(T[8], "Google Shape;225;p34"))
    rel = parse_related(D.related_md)
    if rel is None:
        items = [f"Анализ существующих решений {PENDING}.",
                 "~ Блок заполняется из reports/related_work.md при пересборке."]
    else:
        para, sources = rel
        items = [para] if para else []
        if sources:
            items += ["", "## Источники"] + [f"~- {s}" for s in sources]
    f = Flow(prs, T[8], "Результаты анализа существующих решений (продолжение)", first_width=decor_width())
    f.text(items, name="related_work_block")
    return f.slide


def fill_dataset(prs, T, D: Data):
    s = T[9]
    remove_shape(shape_by_name(s, "Google Shape;231;p35"))
    rows_md = md_table_after(D.data_md, "После очистки и дедупликации")
    rows = [["Источник", "Писем после очистки", "Спам", "Ham", "Доля спама"]]
    total_idx = None
    if rows_md:
        for r in rows_md[1:]:
            rows.append(r[:5])
            if r[0].lower().startswith("итого"):
                total_idx = len(rows) - 1
                rows[-1][0] = "Итого"
    else:  # final numbers from reports/data.md
        rows += [["Enron-Spam", "28472", "13271", "15201", "46,6%"],
                 ["SpamAssassin", "4431", "925", "3506", "20,9%"],
                 ["Ling-Spam", "2768", "432", "2336", "15,6%"],
                 ["SMS Spam Collection", "5098", "606", "4492", "11,9%"],
                 ["Итого", "40769", "15234", "25535", "37,4%"]]
        total_idx = len(rows) - 1
    for r in rows[1:]:
        for j in (1, 2, 3):
            if r[j].isdigit():
                r[j] = intru(r[j])

    cm = g(D.classic, "test", "linsvc", "default", "cm")
    t_ham, t_spam = (sum(cm[0]), sum(cm[1])) if cm else (None, None)
    sb = g(D.spambase, "logreg", "default", "cm")
    sb_n = sum(map(sum, sb)) if sb else None
    items = [
        "## Тест курса",
        f"- {intru(t_ham + t_spam) if cm else PENDING} писем .eml, ham {intru(t_ham) if cm else ''}, "
        f"спам {intru(t_spam) if cm else ''}.",
        "## SpamBase",
        f"- {intru(sb_n) if sb else PENDING} строк, 57 признаков, отдельный бейзлайн. "
        f"CV F1 у LogReg {num(g(D.spambase, 'logreg', 'default', 'f1'), 3)}, "
        f"у бустинга {num(g(D.spambase, 'histgb', 'default', 'f1'), 3)}.",
        "## Характер данных",
        "- Все на английском. SpamAssassin в виде сырых MIME, Enron-Spam и Ling-Spam уже токенизированы, "
        "SMS короткие строки.",
        "## Преобразования",
        "- Разбор MIME, декодирование, HTML в текст (BeautifulSoup), тема по RFC 2047.",
        "- Из заголовков оставлен только Subject, метки [SPAM] и X-Spam убраны, чтобы метка не утекла.",
        "- Удалены пустые письма, 4385 точных дубликатов (md5 токенов) и 1518 почти-дубликатов "
        "(MinHash LSH, Жаккар 0,85).",
        "- Удалены 1499 обучающих писем, совпадающих с тестом курса.",
        "- Нижний регистр, URL и e-mail заменены токенами, цифры на 0, обрезка до 6000 символов.",
    ]
    f = Flow(prs, s, "Описание используемого набора данных (продолжение)")
    f.table(rows, bold_rows=(total_idx,) if total_idx else (), name="sources_table",
            reserve=est_items_height(items[:4], BODY_PT, f.width) + GAP)
    f.text(items, name="dataset_text")
    return f.slide


def model_rows_cv(D: Data, chosen: str) -> tuple[list[list[str]], int | None]:
    rows = [["Модель (OOF, подобранный порог)", "Precision", "Recall", "F1", "FPR", "ROC-AUC"]]
    chosen_row = None
    for key in MODEL_ORDER:
        r = D.results(key)
        if r is None:
            if key == "xlmr":
                rows.append([MODEL_NAMES[key]] + [PENDING] * 5)
            continue
        m = r["cv"]["tuned"]
        rows.append([MODEL_NAMES[key], num(m["precision"]), num(m["recall"]), num(m["f1"]),
                     pct(m["fpr"]), num(m["roc_auc"])])
        if key == chosen:
            chosen_row = len(rows) - 1
    return rows, chosen_row


def arch_text(key: str, D: Data) -> list[str]:
    nf = D.classic.get("n_features")
    thr_p = g(D.classic, "chosen", "threshold_proba")
    tfidf = (f"- TF-IDF слов (1-2-граммы, до 200 тыс.) и символьных n-грамм 3-5 (до 300 тыс.), "
             f"всего {intru(nf) if nf else PENDING} признаков, sublinear tf, L2.")
    if key in ("linsvc", "logreg", "cnb"):
        clf = {"linsvc": "LinearSVC, C = 0,5", "logreg": "LogisticRegression, C = 10",
               "cnb": "ComplementNB, alpha = 0,1"}[key]
        out = ["## Архитектура", tfidf, f"- {clf}."]
        if key == "linsvc":
            out.append(f"- Калибровка Платта на OOF дает вероятность, порог {num(thr_p, 3)}.")
        return out
    if key == "e5":
        return ["## Архитектура", "- Замороженный энкодер multilingual-e5-small (384-мерный вектор, "
                "256 токенов), поверх LogisticRegression, C = 10."]
    ep, ml = g(D.xlmr, "epochs"), g(D.xlmr, "max_len")
    return ["## Архитектура", f"- {g(D.xlmr, 'model', default='xlm-roberta-base')}, дообучение "
            f"всей сети, {ep} эпохи, до {ml} токенов, AdamW, lr 3e-5."]


def fill_model(prs, T, D: Data):
    s = T[10]
    remove_shape(shape_by_name(s, "Google Shape;238;p36"))
    chosen, reason = D.chosen()
    rows, crow = model_rows_cv(D, chosen)
    m = g(D.results(chosen) or {}, "cv", "tuned", default={})
    if reason is None:
        e5r = g(D.results("e5") or {}, "cv", "tuned", "recall")
        reason = (f"Наибольший Recall {num(m.get('recall'))} при ограничении на FP на OOF, ROC-AUC "
                  f"{num(m.get('roc_auc'))}. Около {D.ms or PENDING} мс на письмо на CPU, артефакт "
                  f"{D.model_mb or PENDING} МБ, легко упаковать в сервис.")
        if e5r is not None and chosen in ("linsvc", "logreg"):
            reason += f" Замороженные эмбеддинги e5 при том же FPR дают Recall {num(e5r)}."
    n_train = g(D.classic, "cv", "linsvc", "default", "cm")
    n_train = sum(map(sum, n_train)) if n_train else None
    items = [f"## Выбрана {MODEL_NAMES.get(chosen, chosen)}", reason]
    if D.xlmr is None:
        items.append(f"~ XLM-R: обучение {PENDING}, строка таблицы обновится при пересборке.")
    items += arch_text(chosen, D) + [
        "## Процесс обучения",
        "- 5-fold CV со стратификацией по источнику и метке, "
        + ("векторизатор обучается внутри фолда." if chosen in ("linsvc", "logreg", "cnb")
           else "на каждом фолде модель обучается заново."),
        f"- Финальная модель обучена на всех {intru(n_train) if n_train else PENDING} письмах.",
    ]
    f = Flow(prs, s, "Описание выбранной модели (продолжение)")
    f.table(rows, fill_rows=(crow,) if crow else (), bold_rows=(crow,) if crow else (), name="cv_table",
            reserve=est_items_height(items[:2], BODY_PT, f.width) + GAP,
            caption=f"5-fold CV, OOF-предсказания, порог под Precision ≥ {num(PREC_MIN, 2)} "
                    f"и FPR ≤ {pct(FPR_MAX, 1)}")
    f.text(items, name="model_text")
    return f.slide


def fill_test(prs, T, D: Data):
    s = T[11]
    remove_shape(shape_by_name(s, "Google Shape;245;p37"))
    chosen, _ = D.chosen()
    cm = g(D.classic, "test", "linsvc", "default", "cm")
    n_ham, n_spam = (sum(cm[0]), sum(cm[1])) if cm else ("?", "?")
    rows = [["Модель", "Порог", "Accuracy", "Precision", "Recall", "F1", "FPR"]]
    fills = []
    for key in MODEL_ORDER:
        r = D.results(key)
        for thr, label in (("default", "по умолч."), ("tuned", "с OOF")):
            if r is None:
                if key == "xlmr":
                    rows.append([MODEL_NAMES[key] if thr == "default" else "", label] + [PENDING] * 5)
                continue
            mm = r["test"][thr]
            rows.append([MODEL_NAMES[key] if thr == "default" else "", label, num(mm["accuracy"]),
                         num(mm["precision"]), num(mm["recall"]), num(mm["f1"]), pct(mm["fpr"])])
            if key == chosen:
                fills.append(len(rows) - 1)
    r = D.results(chosen) or {}
    mt = g(r, "test", "tuned", default={})
    md = g(r, "test", "default", default={})
    notes = [
        f"- Курс требует точность выше 86%, у {MODEL_NAMES.get(chosen, chosen).split(' + ')[-1]} "
        f"{pct(md.get('accuracy'), 1)} и {pct(mt.get('accuracy'), 1)}.",
        f"- ROC-AUC {num(mt.get('roc_auc'))}, PR-AUC {num(mt.get('pr_auc'))}.",
        f"- FPR на тесте {pct(mt.get('fpr'))} против {pct(g(r, 'cv', 'tuned', 'fpr'))} на CV, "
        "тест из другого распределения.",
    ]
    f = Flow(prs, s, "Результаты эксперимента (продолжение)")
    f.table(rows, fill_rows=fills, name="test_table", reserve=est_items_height(notes, BODY_PT, f.width) + GAP,
            caption=f"Тест курса, {intru(n_ham + n_spam) if cm else PENDING} писем (ham {intru(n_ham) if cm else ''}, "
                    f"спам {intru(n_spam) if cm else ''}), в обучении и выборе порога не участвовал")
    f.text(notes, name="test_notes")
    return f.slide


def fill_errors(prs, s, D: Data):
    lin = g(D.classic, "test", "linsvc", "tuned", default={})
    cm = lin.get("cm") or [[0, 0], [0, 0]]
    rows_md = md_table_after(D.exp_md, "Доля верных ответов по происхождению")
    rows = [["Класс", "Происхождение письма", "Писем", "Верно"]]
    rows += [r[:4] for r in rows_md[1:]] if rows_md else [["", PENDING, "", ""]]
    cm_rows = [["факт \\ прогноз", "ham", "спам"],
               ["ham", str(cm[0][0]), str(cm[0][1])],
               ["спам", str(cm[1][0]), str(cm[1][1])]]
    col = 4700000
    left = Flow(prs, s, "Результаты эксперимента, разбор ошибок (продолжение)", first_width=col)
    left.table(cm_rows, name="cm_table", max_size=13)
    left.table(rows, name="origin_table", max_size=13, caption="Тест курса, TF-IDF + LinearSVC, порог с OOF")
    items = [
        f"## Ложных срабатываний {cm[0][1]}",
        "- Свежие транзакционные письма (Apple ID, JetBrains, Kaggle, SoloLearn), два русских "
        "уведомления, три поста рассылок 2002 года.",
        "- Корпуса 1999-2005 годов, свежий ham модель не видела.",
        f"## Пропущено спама {cm[1][0]}",
        "- Фишинг под PayPal, eBay и банки. Вежливого служебного текста в обучении почти нет.",
        "- Русские, японские, китайские письма, а в обучении только английский.",
        "- Короткие письма почти без текста.",
    ]
    x = LEFT + col + 250000
    right = Flow(prs, s, "Результаты эксперимента, разбор ошибок (продолжение)", x=x)
    right.text(items, name="err_notes")
    order = [sl.slide_id for sl in prs.slides]
    return max((left.slide, right.slide), key=lambda sl: order.index(sl.slide_id))


def fill_category(prs, s, D: Data):
    c = D.cat
    if not c:
        add_textbox(s, LEFT, TOP, RIGHT - LEFT, 600000, [f"Модель категорий {PENDING}"], size=BODY_PT)
        return s
    names = class_ru()
    weak = c.get("weak", {})
    wt = c.get("weak_train", {})
    chosen_st = c.get("chosen") or next(iter(c.get("student", {})), None)
    tags = []
    for v in weak:
        t = v.split("_")[0]
        if t not in tags:
            tags.append(t)
    nli = ", ".join(dict.fromkeys(NLI_NAMES.get(t, t) for t in tags if t != "avg")) or PENDING
    n_gold = c.get("n_gold", {})
    gd, gt = n_gold.get("dev"), n_gold.get("test")
    classes = ", ".join(names.values())
    counts = wt.get("class_counts", {})
    st_test = g(c, "student", chosen_st, "test", default={})
    top = [
        f"Под семь классов ({classes}) готовой разметки нет.",
        f"- Слабые метки дают zero-shot NLI ({nli}) и правила по ключевым словам. Метку получили "
        f"{intru(wt['n']) if 'n' in wt else PENDING} писем, "
        f"{intru(wt['n_dropped_low_conf']) if 'n_dropped_low_conf' in wt else PENDING} отброшены как неуверенные.",
        f"- Ученик на слабых метках, {STUDENT_DESC.get(chosen_st, chosen_st)}.",
        f"- Ручной gold из {(gd or 0) + (gt or 0)} писем, {gd} для выбора варианта и {gt} только для отчета. "
        f"На gold test ученик дает macro-F1 {num(st_test.get('macro_f1'), 3)} и accuracy "
        f"{num(st_test.get('accuracy'), 3)}.",
    ]
    by_lang = st_test.get("accuracy_by_lang")
    if isinstance(by_lang, dict) and "ru" in by_lang:
        top[-1] += f" На русских письмах accuracy {num(by_lang['ru'], 3)}."

    def weak_name(v: str) -> str:
        parts = v.split("_")
        base = "zero-shot " + {"zs": "mDeBERTa", "mdeberta": "mDeBERTa", "large": "DeBERTa-large",
                               "avg": "среднее"}.get(parts[0], parts[0])
        if "cal" in parts:
            base += ", калибр."
        if "rules" in parts:
            base += " + правила"
        return base

    shown = [v for v in weak if "_" not in v]
    if wt.get("variant") in weak and wt["variant"] not in shown:
        shown.append(wt["variant"])
    rows = [["Метод (gold)", "macro-F1 dev", "macro-F1 test", "Accuracy test"]]
    fills = []
    for v in shown[:5]:
        rows.append([weak_name(v), num(g(weak, v, "dev", "macro_f1"), 3), num(g(weak, v, "test", "macro_f1"), 3),
                     num(g(weak, v, "test", "accuracy"), 3)])
    for st, res in c.get("student", {}).items():
        if "+" in st and st != chosen_st:  # extra ablation variants stay in the report only
            continue
        rows.append([f"ученик {STUDENT_NAMES.get(st, st)}", num(g(res, "dev", "macro_f1"), 3),
                     num(g(res, "test", "macro_f1"), 3), num(g(res, "test", "accuracy"), 3)])
        if st == chosen_st:
            fills.append(len(rows) - 1)
    pc = st_test.get("per_class", {})
    rows2 = [["Класс (gold test)", "Precision", "Recall", "F1", "Писем", "Слабых меток"]]
    for k, v in pc.items():
        rows2.append([names.get(k, k), num(v.get("precision"), 2), num(v.get("recall"), 2),
                      num(v.get("f1"), 2), str(v.get("support", "")), intru(counts[k]) if k in counts else ""])
    f = Flow(prs, s, "Дополнительная модель категорий (продолжение)")
    f.text(top, name="cat_text")
    f.table(rows, fill_rows=fills, name="cat_methods")
    f.table(rows2, name="cat_per_class")
    return f.slide


def fill_code(prs, T, D: Data, args):
    remove_shape(shape_by_name(T[12], "Google Shape;252;p38"))
    items = [
        f"## Репозиторий: {args.github}",
        f"## Демо (Hugging Face Space): {args.space}",
        "## Структура",
        "- src/data (скачивание корпусов, разбор MIME, очистка, дедупликация)",
        "- src/models (TF-IDF, классические модели, e5, XLM-R, бейзлайн SpamBase)",
        "- src/category (zero-shot разметка, правила, модель категорий)",
        "- service (FastAPI, Dockerfile, тесты, сборка Space)",
        "- scripts/run_all.sh (весь пайплайн, готовые шаги пропускаются)",
        "- reports (отчеты по этапам и JSON со всеми метриками)",
        "## Воспроизведение",
        "- bash scripts/run_all.sh",
        "- docker build -f service/Dockerfile -t antispam . && docker run -p 7860:7860 antispam",
    ]
    f = Flow(prs, T[12], "Код экспериментов (продолжение)")
    f.text(items, name="code_text")
    return f.slide


def fill_packaging(prs, T, D: Data, args):
    s = T[13]
    remove_shape(shape_by_name(s, "Google Shape;259;p39"))
    thr = g(D.classic, "chosen", "threshold_proba")
    cat_on = '"category_loaded":true' in D.smoke_md.replace(" ", "")
    items = [
        "## Логика работы",
        "- На вход JSON с текстом письма или пакетом писем.",
        f"- Нормализация как при обучении, TF-IDF слов и символов, LinearSVC, калибровка Платта, "
        f"сравнение с порогом {num(thr, 3)}.",
        "- В ответе is_spam, proba и threshold. Пустой или слишком длинный текст дает 422.",
        "## Эндпоинты FastAPI",
        "- GET / (демо-страница)",
        "- GET /health (статус, модель, порог)",
        "- POST /predict (одно письмо)",
        "- POST /predict_batch (пакет писем)",
        "- POST /category (категория письма)" + ("" if cat_on else f", модель категорий {PENDING}"),
    ]
    f = Flow(prs, s, "Упаковка модели (продолжение)")
    f.text(items, name="packaging_text")
    last = f.slide

    demo = open_img("service_demo_spam.png")
    if demo:
        sd = new_slide_after(prs, last, "Демо-страница сервиса", TITLE_SRC[0])
        cap = f"Демо онлайн {args.space}" if args.space.startswith("http") else None
        Flow(prs, sd, "Демо-страница сервиса").picture(crop_panel(demo), cap, name="demo_picture")
        last = sd
    pred = open_img("service_docs_predict.png", crop=(0, 1440, 2088, 2250))
    if pred:
        sa = new_slide_after(prs, last, "Запрос к API в Swagger UI", TITLE_SRC[0])
        Flow(prs, sa, "Запрос к API в Swagger UI").picture(trim_bg(pred), name="api_picture")
        last = sa

    size_mb = regex(D.smoke_md, r"Размер образа:\s*(\d+)\s*MB", (None,))[0]
    items2 = [
        "## Docker",
        "- python:3.12-slim, пользователь без root (uid 1000), порт 7860.",
        "- В образ попадают только код и модели (белый список в Dockerfile.dockerignore).",
        f"- Образ {size_mb or PENDING} МБ, из них модель спама {D.model_mb or PENDING} МБ и модель категорий "
        f"{num(D.latency.get('category_artifacts_mb'), 0)} МБ (e5 в int8 ONNX).",
        "- docker build -f service/Dockerfile -t antispam .",
        "- docker run -p 7860:7860 antispam",
        "## Публикация",
        "- Hugging Face Space (sdk docker), тот же Dockerfile.",
        f"- {args.space}",
        "## Проверка",
        "- pytest для API. Smoke-тест в контейнере верно отвечает на 5 писем из 5, на пустой текст 422.",
    ]
    sk = new_slide_after(prs, last, "Docker и публикация", TITLE_SRC[0])
    f2 = Flow(prs, sk, "Docker и публикация (продолжение)")
    f2.text(items2, name="docker_text")
    return f2.slide


def fill_conclusions(prs, T, D: Data):
    remove_shape(shape_by_name(T[14], "Google Shape;318;p40"))
    lin = D.results("linsvc") or {}
    e5 = D.results("e5") or {}
    loso = g(D.classic, "loso", default={})
    auc = {src: g(loso, src, "linsvc", "roc_auc") for src in ("enron", "sms", "spamassassin")}
    items = [
        f"- Нужны несколько корпусов. В LOSO на отложенном Enron ROC-AUC {num(auc['enron'], 2)}, "
        f"на SMS {num(auc['sms'], 2)}, на SpamAssassin FPR {pct(g(loso, 'spamassassin', 'linsvc', 'fpr'), 0)}.",
        f"- Порог с CV не выдержал сдвига. FPR вырос с {pct(g(lin, 'cv', 'tuned', 'fpr'))} на OOF до "
        f"{pct(g(lin, 'test', 'tuned', 'fpr'))} на тесте, Recall упал с "
        f"{num(g(lin, 'cv', 'tuned', 'recall'), 3)} до {num(g(lin, 'test', 'tuned', 'recall'), 3)}.",
        f"- TF-IDF сильнее замороженных эмбеддингов. При FPR {pct(FPR_MAX, 1)} на OOF Recall "
        f"{num(g(lin, 'cv', 'tuned', 'recall'), 3)} против {num(g(e5, 'cv', 'tuned', 'recall'), 3)} у e5.",
    ]
    xl = D.results("xlmr")
    if xl:
        xru = regex(D.exp_md, r"на русских письмах: (\d+,\d)% верных ответов против (\d+,\d)%", (None, None))
        items.append(f"- XLM-R почти равен LinearSVC на CV, но на тесте сильнее, accuracy "
                     f"{num(g(xl, 'test', 'tuned', 'accuracy'), 3)} против {num(g(lin, 'test', 'tuned', 'accuracy'), 3)}, "
                     f"на русских письмах {xru[0] or PENDING}% против {xru[1] or PENDING}%.")
    if D.cat:
        st = g(D.cat, "student", D.cat.get("chosen"), "test", "macro_f1")
        wv = g(D.cat, "weak_train", "variant")
        wf = g(D.cat, "weak", wv, "test", "macro_f1")
        items.append(f"- Ученик на слабых метках дает macro-F1 {num(st, 2)} на gold против {num(wf, 2)} "
                     "у лучшей слабой разметки.")
    items += [
        "## Что сделали бы дальше",
        "- Добавить свежий ham, фишинг и русские письма, перепроверить порог на целевом потоке.",
        "- Мультиязычный энкодер в ансамбле с TF-IDF, отметки «спам» и «не спам» от пользователей.",
    ]
    f = Flow(prs, T[14], "Выводы (продолжение)", first_width=decor_width())
    f.text(items, name="conclusions_text")
    return f.slide


def fill_questions(prs, T, D: Data):
    remove_shape(shape_by_name(T[15], "Google Shape;326;p41"))
    ru = regex(D.exp_md, r"на (\d+) русских письмах теста.*?(\d+,\d)% верных ответов.*?против (\d+,\d)% у LinearSVC",
               (None, None, None))
    en = regex(D.exp_md, r"английской части картина обратная, (\d+,\d)% против (\d+,\d)%", (None, None))
    xru = regex(D.exp_md, r"на русских письмах: (\d+,\d)% верных ответов против (\d+,\d)%", (None, None))
    lin = D.results("linsvc") or {}
    wt = g(D.cat, "weak_train", default={}) or {}
    st = g(D.cat, "student", g(D.cat, "chosen"), "test", "macro_f1")
    wf = g(D.cat, "weak", wt.get("variant"), "test", "macro_f1")
    items = [
        "## Как сделать систему мультиязычной?",
        f"- Обучение пока только на английском. На {ru[0] or PENDING} русских письмах теста LinearSVC "
        f"прав в {ru[2] or PENDING}% случаев, мультиязычный e5 в {ru[1] or PENDING}%, на английских "
        f"наоборот ({en[1] or PENDING}% против {en[0] or PENDING}%).",
        "- XLM-R, дообученный только на английском спаме, верно отвечает на "
        f"{xru[0] or PENDING}% русских писем теста.",
        "- Нужны русские корпуса или машинный перевод английских писем.",
        "- Детектор языка отправляет английские письма в TF-IDF, остальные в энкодер.",
        "## Как адаптировать систему под concept drift?",
        f"- Сдвиг уже виден, FPR {pct(g(lin, 'cv', 'tuned', 'fpr'))} на CV и "
        f"{pct(g(lin, 'test', 'tuned', 'fpr'))} на тесте.",
        "- Следить за распределением proba, долей спама по дням и жалобами пользователей.",
        "- Переобучать на скользящем окне и пересчитывать порог на свежих данных.",
        "- Валидировать по времени (обучение на прошлом, проверка на будущем), а не случайными фолдами.",
        "## Как реализовать дообучение или полуконтролируемое обучение?",
        "- Кнопки «спам» и «не спам» дают новые метки. От отравления защищают лимит меток на пользователя "
        "и учет только отметок, согласованных несколькими людьми.",
        "- Self-training. Модель размечает поток, в обучение идут только уверенные примеры "
        "(proba > 0,99 или < 0,01), качество проверяем на отложенной ручной выборке.",
        f"- Так обучена наша модель категорий, macro-F1 на gold вырос с {num(wf, 2)} до {num(st, 2)}.",
        "- SGDClassifier.partial_fit для инкрементального обучения, энкодер дообучать по расписанию.",
        "## Какую бизнес-проблему решаем и как это задача ML?",
        "- Пользователь тратит время на спам и рискует попасть на фишинг. Потерять счет или приглашение "
        "на собеседование хуже, чем пропустить рекламу.",
        "- В терминах ML это бинарная классификация с асимметричной ценой ошибок.",
        "- Спам уходит в папку «Спам», а не удаляется.",
        "- Категории (7 классов) помогают сортировать почту и считать долю фишинга.",
    ]
    f = Flow(prs, T[15], "Дополнительные вопросы (продолжение)", first_width=decor_width())
    f.text(items, name="questions_text")
    return f.slide


# ----------------------------------------------------------------------------- main

def fix_app_xml(path: Path, n_slides: int) -> None:
    """The template's docProps/app.xml still names its editor and slide counts; replace it.
    Template text in Proxima Nova is switched to FONT: without that font viewers fall back to
    something wider and the 80% line spacing of the section titles makes the lines overlap."""
    import zipfile

    app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
           'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
           f'<Slides>{n_slides}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides>'
           '<PresentationFormat>On-screen Show (16:9)</PresentationFormat></Properties>')
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = app.encode() if item.filename == "docProps/app.xml" else src.read(item.filename)
            if item.filename.startswith("ppt/") and item.filename.endswith(".xml"):
                data = data.replace(b'typeface="Proxima Nova"', f'typeface="{FONT}"'.encode())
                data = data.replace(b'<a:spcPct val="80000"/>', b'<a:spcPct val="100000"/>')
            dst.writestr(item, data)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="{{КОМАНДА}}")
    ap.add_argument("--member1", default="{{ФАМИЛИЯ_ИМЯ_1}}")
    ap.add_argument("--member2", default="{{ФАМИЛИЯ_ИМЯ_2}}")
    ap.add_argument("--github", default="https://github.com/hexzedels/antispam")
    ap.add_argument("--space", default="https://huggingface.co/spaces/hexzedels/antispam")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="estimate overflow and print the text of every slide")
    ap.add_argument("--dump", type=Path, default=None, help="write all slide text to this .txt file")
    args = ap.parse_args()

    D = Data()
    prs = Presentation(str(TEMPLATE))
    T = list(prs.slides)                       # template slides by original position (0-based)
    TITLE_SRC.append(shape_by_name(T[7], "Google Shape;193;p33"))

    # Each fill_* returns the last slide it used: added and continuation slides go right after it.
    fill_title(T, args)
    fill_roles(T, args)
    fill_task(prs, T, D)
    fill_related(prs, T, D)
    fill_dataset(prs, T, D)
    fill_model(prs, T, D)
    last = fill_test(prs, T, D)
    last = fill_errors(prs, new_slide_after(prs, last, "Результаты эксперимента, разбор ошибок", TITLE_SRC[0]), D)
    fill_category(prs, new_slide_after(prs, last, "Дополнительная модель категорий", TITLE_SRC[0]), D)
    fill_code(prs, T, D, args)
    fill_packaging(prs, T, D, args)
    fill_conclusions(prs, T, D)
    fill_questions(prs, T, D)
    # "Регламент выполнения задания" holds instructions to students. Deleted after the new slides
    # are added, otherwise add_slide reuses an existing part name (slideN.xml).
    delete_slide(prs, T[5])
    clear_notes(prs)

    cp = prs.core_properties
    cp.author = "МКБ251"
    cp.last_modified_by = "МКБ251"
    cp.comments = ""
    cp.title = "Итоговое задание: спам-фильтр и категории писем"
    cp.subject = "Применение ML в кибербезопасности"
    cp.keywords = ""

    out = args.out or ROOT / f"{args.member1.strip().replace(' ', '_')}_МКБ251_Итоговое задание.pptx"
    prs.save(str(out))
    fix_app_xml(out, len(prs.slides))
    print(f"saved {out} ({len(prs.slides)} slides)")
    if D.xlmr is None:
        print("[info] reports/xlmr_results.json missing: XLM-R rows rendered as placeholders")
    if not D.related_md:
        print("[info] reports/related_work.md missing: slide 'existing solutions' is a placeholder")

    text = dump_text(prs)
    if args.dump:
        args.dump.write_text(text, encoding="utf-8")
    if args.check:
        warns = check_deck(prs)
        print(f"\n--- overflow check: {len(warns)} warning(s)")
        for w in warns:
            print("[warn]", w)
        print("\n--- slide text\n" + text)


if __name__ == "__main__":
    main()
