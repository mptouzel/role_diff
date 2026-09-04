# Build targets for the manuscript and its figures.
# Figures are pinned to matplotlib 3.8.4 (see requirements.txt); a mismatch is
# reported by each script rather than silently restyling every label.

PY ?= python3

.PHONY: all figures pdf promo clean

all: figures pdf

## regenerate every figure in figures/ (sim_fig1 writes the data make_fig1 reads)
figures:
	$(PY) code/sim_fig1.py
	$(PY) code/make_fig1.py
	$(PY) code/make_cascade_figures.py
	$(PY) code/make_fig_pinning_saddle.py
	$(PY) code/make_fig_characterization.py
	$(PY) code/make_fig_monitoring.py

## promotional animation (not part of the submission)
promo:
	$(PY) promo/sim_ramp.py
	$(PY) promo/make_ramp_animation.py

## build both documents
pdf:
	latexmk -pdf -interaction=nonstopmode main.tex
	latexmk -pdf -interaction=nonstopmode supplement.tex

## remove build droppings, keeping main.pdf and supplement.pdf
clean:
	latexmk -c main.tex supplement.tex
	rm -f pdflatex*.fls *.synctex.gz.sum.synctex
	rm -rf code/__pycache__
