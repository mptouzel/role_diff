# Role differentiation as ignition of a collective information engine

Manuscript package (REVTeX 4.2).

## Files

- `main.tex`        — the article. Sections I–VI plus Appendices A–E.
- `supplement.tex`  — Supplemental Material (Notation plus Secs. S1–S8),
                      deposited separately.
- `references.bib`  — shared bibliography. Contains the `SM` entry that the
                      article cites for the Supplemental Material.
- `popular_summary.txt` — non-technical summary, 147 words.
- `figures/`, `code/` — figures and the scripts that generate them.
- `data/`           — `fig1_data.npz`, written by `code/sim_fig1.py` and read
                      by `code/make_fig1.py`.
- `code/scratch/`   — working notes and exploratory scripts. Not part of the
                      submission and not used to build any figure.
- `requirements.txt`, `Makefile` — pinned figure environment and build targets.

## Compiling

The two documents are independent: each compiles on its own with the usual
`pdflatex → bibtex → pdflatex → pdflatex`. There is no `xr` cross-document
dependency, so on Overleaf you can set either file as the main document.

### Compiling locally

On Ubuntu/Debian (including WSL), install TeX Live plus the REVTeX class:

```
sudo apt-get update
sudo apt-get install -y \
  texlive-latex-base \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-publishers \
  texlive-bibtex-extra \
  latexmk
```

`texlive-publishers` provides the `revtex4-2` document class; `texlive-bibtex-extra`
provides the APS bibliography styles. Then build either document with:

```
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

`latexmk` runs the `pdflatex → bibtex → pdflatex → pdflatex` sequence
automatically and only reruns the steps that are out of date.

For editing, VS Code with the
[LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
extension picks up `latexmk` automatically and gives a live PDF preview
(`Ctrl+Alt+V`) with SyncTeX jump-to-source.

The article refers to the Supplemental Material by citation, e.g.
`\smref{S2}` renders as "[12, Sec. S2]" and `\smeq{S2}` as "[12, Eq. (S2)]".
The Supplemental Material refers back by explicit section number ("Sec. V A of
the main text"); if the article's section numbering changes, update those by
hand — they are plain text, not references.

## Structure

Article: I Introduction · II Model · III Ignition (III A gain matrix and
critical couplings) · IV Cascade (IV A counting function, IV B five classes) ·
V Schema-resource selection of the cascade class (V A fixed point, V B closed
pool, V C regime map) ·
VI Monitoring (VI A consequences for monitoring and design) · VII Discussion.
Appendices: A microfoundations · B simulation details · C platform control
surface · D schema directions and the vector form.

Supplemental Material: S1 model ladder, coordination payoff, correlated
equilibrium · S2 pairing statistics and the Λ factorization · S3
associative-memory mapping and role capacity · S4 status ledger · S5
monotonicity of the pinning fixed point · S6 validity of the linearization.

## Figure sources

| Figure | Script |
|---|---|
| Fig. 1 (loop and ignition) | `code/sim_fig1.py`, `code/make_fig1.py` |
| Fig. 2 (cascade classes)   | `code/make_cascade_figures.py` |
| Fig. 3 (pinning saddle)    | `code/make_fig_pinning_saddle.py` |
| Fig. 4 (characterization)  | `code/make_fig_characterization.py` |
| Fig. 5 (monitoring)        | `code/make_fig_monitoring.py` |
| Pairing-steering numbers (Appendix C) | `code/pairing_steering.py` |

Note: `figures/fig4_monitoring.pdf` is now Figure 5 (the saddle figure moved
into the body ahead of it). The filename is unchanged.

## Reproducing the figures

The committed figures were produced with **matplotlib 3.8.4** (recorded in the
`Creator` metadata of every file in `figures/`). matplotlib governs text and
glyph rendering, so a different version reproduces the same data with every
label subtly restyled. Each figure script prints a warning when the installed
version differs.

```bash
# Use an interpreter that ships ensurepip. On Debian/Ubuntu the default
# python3 may not: check with  python3 -c "import ensurepip"  and fall
# back to a version that does, e.g. /usr/bin/python3.10.
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
make figures PY=.venv/bin/python     # make pdf builds both documents
```

Two failure modes are worth naming, both seen on WSL:

- **No `pip` inside the venv.** If `.venv/bin/pip` is absent, the interpreter
  lacked `ensurepip`. Delete `.venv` and rebuild with one that has it. Do not
  fall back to a bare `pip`, which resolves elsewhere on `PATH` (a conda
  install, say) and puts the packages outside the venv. Always use
  `.venv/bin/python -m pip`.
- **Rebuilding over an existing `.venv`.** Remove it first. Layering a second
  interpreter on top leaves two `lib/pythonX.Y` trees, and NumPy's bundled
  OpenBLAS can end up in the tree its package is not in, giving
  `libopenblas...so: cannot open shared object file` on import.

