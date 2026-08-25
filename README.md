# Role differentiation as ignition of a collective information engine

Manuscript package (REVTeX 4.2).

## Files

- `main.tex`        — the article. Sections I–VII plus Appendices A–D.
- `supplement.tex`  — Supplemental Material (Secs. S1–S4), deposited separately.
- `references.bib`  — shared bibliography. Contains the `SM` entry that the
                      article cites for the Supplemental Material.
- `popular_summary.txt` — non-technical summary, 147 words.
- `figures/`, `code/` — figures and the scripts that generate them.

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
| Fig. 3 (characterization)  | `code/make_fig_characterization.py` |
| Fig. 4 (pinning saddle)    | `code/make_fig_pinning_saddle.py` |
| Fig. 5 (monitoring)        | `code/make_fig_monitoring.py` |
| Pairing-steering numbers (Appendix C) | `code/pairing_steering.py` |

Note: `figures/fig4_monitoring.pdf` is now Figure 5 (the saddle figure moved
into the body ahead of it). The filename is unchanged.
