"""Generate docs/MANUAL.md (the Markdown version of the manual) from the typeset manual.

The LaTeX manual is the single source: this script turns uccap_manual.tex into GitHub
Markdown with the same chapters, sections, tables, figures and numbering, so the two
versions cannot drift apart. The TikZ diagrams are exported as PNG files to
docs/figures/manual/.

Run after building the PDF (xelatex twice, which writes uccap_manual.aux):

    cd docs/manual_tex && xelatex uccap_manual.tex && xelatex uccap_manual.tex
    python to_markdown.py

Needs xelatex, pdftoppm (poppler) and pandoc.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent
TEX = HERE / "uccap_manual.tex"
AUX = HERE / "uccap_manual.aux"
FIG_DIR = DOCS / "figures" / "manual"
OUT = DOCS / "MANUAL.md"


# ---------------------------------------------------------------- helpers

def braced(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group starting at s[i] == '{', and the index after it."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def replace_command(s: str, name: str, fn, nargs: int = 1) -> str:
    """Replace \\name{a}{b}... by fn(a, b, ...)."""
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    out, pos = [], 0
    while True:
        m = pat.search(s, pos)
        if not m:
            break
        j = m.end()
        if j >= len(s) or s[j] != "{":
            out.append(s[pos:j])
            pos = j
            continue
        args = []
        for _ in range(nargs):
            while s[j] in " \n":
                j += 1
            a, j = braced(s, j)
            args.append(a)
        out.append(s[pos:m.start()])
        out.append(fn(*args))
        pos = j
    out.append(s[pos:])
    return "".join(out)


def environments(s: str, name: str):
    """(start, end, inner) of each \\begin{name}...\\end{name}, outermost first."""
    begin, end = "\\begin{" + name + "}", "\\end{" + name + "}"
    pos = 0
    while True:
        a = s.find(begin, pos)
        if a < 0:
            return
        b = s.find(end, a)
        yield a, b + len(end), s[a + len(begin):b]
        pos = b + len(end)


def replace_env(s: str, name: str, fn) -> str:
    out, pos = [], 0
    for a, b, inner in list(environments(s, name)):
        out += [s[pos:a], fn(inner)]
        pos = b
    out.append(s[pos:])
    return "".join(out)


def labels() -> dict[str, str]:
    text = AUX.read_text(encoding="utf-8")
    return {k: v for k, v in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", text)}


# ---------------------------------------------------------------- TikZ diagrams

def export_diagrams(preamble: str, pictures: list[str]) -> list[str]:
    """Render each tikzpicture to docs/figures/manual/diagram-NN.png; returns the file names."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    head = preamble.replace("\\begin{document}", "")
    head += ("\n\\usepackage[active,tightpage]{preview}\n\\PreviewEnvironment{tikzpicture}\n"
             "\\setlength\\PreviewBorder{6pt}\n\\begin{document}\n")
    body = "\n\n".join("\\begin{tikzpicture}" + p + "\\end{tikzpicture}" for p in pictures)
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        (t / "d.tex").write_text(head + body + "\n\\end{document}\n", encoding="utf-8")
        for _ in range(2):
            subprocess.run(["xelatex", "-interaction=nonstopmode", "d.tex"], cwd=t, check=False,
                           stdout=subprocess.DEVNULL)
        subprocess.run(["pdftoppm", "-r", "180", "-png", "d.pdf", "p"], cwd=t, check=True)
        pages = sorted(t.glob("p-*.png"))
        if len(pages) != len(pictures):
            raise RuntimeError(f"expected {len(pictures)} diagrams, got {len(pages)}")
        names = []
        for i, page in enumerate(pages, 1):
            name = f"diagram-{i:02d}.png"
            shutil.copy(page, FIG_DIR / name)
            names.append(name)
    return names


# ---------------------------------------------------------------- conversion

def convert() -> None:
    tex = TEX.read_text(encoding="utf-8")
    preamble, body = tex.split("\\begin{document}", 1)
    preamble += "\\begin{document}"
    body = body.split("\\end{document}")[0]
    lab = labels()

    # front page and table of contents are replaced by the Markdown header
    body = re.sub(r"\\begin\{titlepage\}.*?\\end\{titlepage\}", "", body, flags=re.S)
    for cmd in ("tableofcontents", "frontmatter", "mainmatter", "backmatter", "clearpage", "cleardoublepage",
                "newpage", "medskip", "smallskip", "bigskip", "centering", "noindent"):
        body = re.sub(r"\\" + cmd + r"(?![A-Za-z])", "", body)
    body = re.sub(r"\\newcommand\{\\faq\}\[1\]\{.*?\}\n", "", body)

    # code blocks become placeholders so that pandoc leaves them alone
    blocks: list[str] = []

    def block(lang):
        def fn(inner):
            inner = re.sub(r"^\[\{[^\n]*?\}\]|^\[[^\]\n]*\]", "", inner)  # outbox title, optionally braced
            blocks.append(f"```{lang}\n{inner.strip(chr(10))}\n```")
            return f"\n\nUCCAPBLOCK{len(blocks) - 1}END\n\n"
        return fn
    body = replace_env(body, "pycode", block("python"))
    body = replace_env(body, "shcode", block("sh"))
    body = replace_env(body, "outboxkeep", block("text"))
    body = replace_env(body, "outbox", block("text"))

    def api(sig):
        sig = sig.replace("\\_", "_").replace("$\\rightarrow$", "→").replace("\\{", "{").replace("\\}", "}")
        sig = re.sub(r"\\textbf\{([^}]*)\}", r"\1", sig)
        sig = re.sub(r"\\\\\s*\\hspace\*?\{[^}]*\}", "\n    ", sig).replace("\\\\", "\n")
        blocks.append(f"```python\n{sig.strip()}\n```")
        return f"\n\nUCCAPBLOCK{len(blocks) - 1}END\n\n"
    body = replace_command(body, "apihead", api)

    # cross references, with the numbers of the PDF
    body = replace_command(body, "eqref", lambda k: f"({lab.get(k, '?')})")
    body = replace_command(body, "figref", lambda k: f"Figure {lab.get(k, '?')}")
    body = replace_command(body, "tabref", lambda k: f"Table {lab.get(k, '?')}")
    body = replace_command(body, "chapref", lambda k: f"Chapter {lab.get(k, '?')}")
    body = replace_command(body, "secref", lambda k: f"Section {lab.get(k, '?')}")
    body = re.sub(r"(~?)\\ref\{([^}]*)\}", lambda m: (" " if m.group(1) else "") + lab.get(m.group(2), "?"), body)

    # TikZ diagrams -> PNG
    pictures = [inner for _, _, inner in environments(body, "tikzpicture")]
    names = export_diagrams(preamble, pictures)
    it = iter(names)
    body = replace_env(body, "tikzpicture", lambda inner: f"\\includegraphics{{figures/manual/{next(it)}}}")

    # figures and tables: caption as a numbered paragraph, no floats
    def figure(inner):
        cap = ""
        m = re.search(r"\\caption\{", inner)
        if m:
            cap, j = braced(inner, m.end() - 1)
            inner = inner[:m.start()] + inner[j:]
        key = re.search(r"\\label\{([^}]*)\}", cap + inner)
        num = lab.get(key.group(1), "") if key else ""
        cap = re.sub(r"\\label\{[^}]*\}", "", cap)
        inner = re.sub(r"\\label\{[^}]*\}", "", inner)
        return inner, (f"\\textbf{{{{Figure}} {num}.}} {cap}" if num else cap)

    def fig_env(inner):
        content, cap = figure(re.sub(r"^\[[^\]]*\]", "", inner))
        content = re.sub(r"\\includegraphics(\[[^\]]*\])?\{\.\./figures/", r"\\includegraphics{figures/", content)
        return f"\n\n{content}\n\n{cap}\n\n"

    def tab_env(inner):
        content, cap = figure(re.sub(r"^\[[^\]]*\]", "", inner))
        cap = cap.replace("{Figure}", "{Table}")
        return f"\n\n{cap}\n\n{content}\n\n"
    body = replace_env(body, "figure", fig_env)
    body = replace_env(body, "table", tab_env)

    # tables: tabularx / custom environments -> plain tabular
    def columns(spec):
        spec = re.sub(r"\*\{(\d+)\}\{([^}]*)\}", lambda m: m.group(2) * int(m.group(1)), spec)
        spec = re.sub(r"[@<>]\{[^}]*\}", "", spec)  # >{\raggedright...} is not a column
        spec = re.sub(r"[PpmbP]\{[^}]*\}", "l", spec)
        return "l" * len(re.findall(r"[lcrLRXl]", spec))

    def tabularx(s):
        out, pos = [], 0
        for a, b, inner in list(environments(s, "tabularx")):
            _, j = braced(inner, inner.index("{"))
            spec, k = braced(inner, inner.index("{", j))
            out += [s[pos:a], "\\begin{tabular}{" + columns(spec) + "}" + inner[k:] + "\\end{tabular}"]
            pos = b
        out.append(s[pos:])
        return "".join(out)
    body = tabularx(body)
    body = replace_env(body, "params", lambda inner: "\\begin{tabular}{lll}\\toprule Parameter & Default & "
                       "Description\\\\\\midrule" + inner + "\\bottomrule\\end{tabular}")
    body = replace_env(body, "attrs", lambda inner: "\\begin{tabular}{ll}\\toprule Attribute or method & "
                       "Description\\\\\\midrule" + re.sub(r"^\[[^\]]*\]", "", inner) +
                       "\\bottomrule\\end{tabular}")
    body = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", body)
    body = re.sub(r"\\begin\{(enumerate|itemize)\}\[[^\]]*\]", r"\\begin{\1}", body)
    body = replace_command(body, "paragraph", lambda x: f"\\textbf{{{x}}}")
    body = replace_command(body, "multicolumn", lambda n, spec, x: x + " &" * (int(n) - 1), nargs=3)

    # callouts and small macros
    def callout(default):
        def fn(inner):
            m = re.match(r"^\[([^\]]*)\]", inner)
            title = m.group(1) if m else default
            inner = inner[m.end():] if m else inner
            return f"\\begin{{quote}}\\textbf{{{title}.}} {inner.strip()}\\end{{quote}}"
        return fn
    body = replace_env(body, "keypoint", callout("Key point"))
    body = replace_env(body, "caution", callout("Caution"))
    body = replace_env(body, "chapterintro", lambda inner: f"\\emph{{{inner.strip()}}}")
    body = replace_command(body, "faq", lambda q: f"\\subsection*{{Q. {q}}}")
    body = replace_command(body, "code", lambda x: "\\texttt{" + x.replace("--", "UCCAPDASHDASH") + "}")
    body = replace_command(body, "thead", lambda x: x)
    body = replace_command(body, "msg", lambda x: "\\emph{" + x + "}")
    body = re.sub(r"\\pkg\{\}", "uccap", body)
    body = re.sub(r"\\pkg(?![A-Za-z])", "uccap", body)
    body = body.replace("\\accepttag", "\\textbf{ACCEPT}").replace("\\rejecttag", "\\textbf{REJECT}")
    body = body.replace("\\Ppkhat", "\\widehat{P}_{pk}").replace("\\Cpkhat", "\\widehat{C}_{pk}")
    body = replace_command(body, "doi", lambda d: f"\\href{{https://doi.org/{d}}}{{doi:{d}}}")
    body = replace_command(body, "arxiv", lambda d: f"\\href{{https://arxiv.org/abs/{d}}}{{arXiv:{d}}}")
    body = replace_command(body, "colorbox", lambda c, x: x, nargs=2)
    body = re.sub(r"\\(small|footnotesize|scriptsize|sffamily|bfseries)(?![A-Za-z])", "", body)
    body = re.sub(r"\\vspace\*?\{[^}]*\}", "", body)

    # citations as [n] with the numbers of the PDF, and the bibliography as a numbered list
    bib = dict(re.findall(r"\\bibcite\{([^}]*)\}\{([^}]*)\}", AUX.read_text(encoding="utf-8")))

    def cite(m):
        nums = ", ".join(bib.get(k.strip(), "?") for k in m.group(2).split(","))
        return f"[{nums}{', ' + m.group(1) if m.group(1) else ''}]"

    body = re.sub(r"\\cite(?:\[([^\]]*)\])?\{([^}]*)\}", cite, body)

    def bibliography(m):
        items = re.split(r"\\bibitem\{([^}]*)\}", m.group(1))[1:]
        return "\n\n".join(f"[{bib.get(key, '?')}] {text.strip()}"
                            for key, text in zip(items[::2], items[1::2]))

    body = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", bibliography, body,
                  flags=re.S)

    # numbered headings, as in the PDF
    chap, sec, sub, appendix = 0, 0, 0, False
    out, pos = [], 0
    for m in re.finditer(r"\\(addchap|chapter|section|subsection|appendix)(\*?)(?![A-Za-z])", body):
        out.append(body[pos:m.start()])
        kind, star = m.group(1), m.group(2)
        pos = m.end()
        if kind == "appendix":
            appendix, chap = True, 0
            continue
        title, pos = braced(body, pos)
        if kind == "addchap" or star:
            level = {"addchap": "chapter", "chapter": "chapter", "section": "section", "subsection": "subsection"}[kind]
            out.append(f"\\{level}{{{title}}}")
            continue
        if kind == "chapter":
            chap, sec, sub = chap + 1, 0, 0
            number = chr(64 + chap) if appendix else str(chap)
            out.append(f"\\chapter{{{'Appendix ' if appendix else ''}{number}{'. ' if appendix else ' '}{title}}}")
        elif kind == "section":
            sec, sub = sec + 1, 0
            number = f"{chr(64 + chap) if appendix else chap}.{sec}"
            out.append(f"\\section{{{number} {title}}}")
        else:
            sub += 1
            number = f"{chr(64 + chap) if appendix else chap}.{sec}.{sub}"
            out.append(f"\\subsection{{{number} {title}}}")
    out.append(body[pos:])
    body = "".join(out)
    body = re.sub(r"\\label\{[^}]*\}", "", body)

    md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=none", "--shift-heading-level-by=1"],
                        input=body, capture_output=True, text=True, check=True).stdout
    # display math as ```math blocks, which GitHub renders reliably
    md = re.sub(r"[ \t]*\$\$(.+?)\$\$[ \t]*", lambda m: "\n\n```math\n" + m.group(1).strip() + "\n```\n\n", md,
                flags=re.S)
    md = md.replace("UCCAPDASHDASH", "--")
    for i, b in enumerate(blocks):
        md = md.replace(f"UCCAPBLOCK{i}END", b)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"!\[image\]\(([^)]*)\)\n\n\*\*Figure ([\d.]+)\.\*\*",
                r"![Figure \2](\1)\n\n**Figure \2.**", md)

    toc = [f"- [{h}](#{anchor(h)})" for h in re.findall(r"^## (.+)$", md, flags=re.M)]
    header = ("# uccap User Manual\n\n"
              "*Process Capability Analysis, Risk-Calibrated Approval and Hybrid Statistical Learning Decision Support with Finite Samples*\n\n"
              "> This is the Markdown version of the typeset manual "
              "[`docs/manual_tex/uccap_manual.pdf`](manual_tex/uccap_manual.pdf): same chapters, tables, "
              "figures and numbering. It is generated from the LaTeX source by "
              "`docs/manual_tex/to_markdown.py`; edit the LaTeX source, not this file.\n\n"
              "## Contents\n\n" + "\n".join(toc) + "\n\n")
    OUT.write_text(header + md.strip() + "\n", encoding="utf-8")
    print(f"{OUT} written ({len(md.split())} words, {len(names)} diagrams)")


def anchor(heading: str) -> str:
    """GitHub's heading anchor."""
    h = heading.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


if __name__ == "__main__":
    convert()
