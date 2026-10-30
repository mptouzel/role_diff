# Build targets for the manuscript and its figures.
# Figures are pinned to matplotlib 3.8.4 (see requirements.txt); a mismatch is
# reported by each script rather than silently restyling every label.

PY ?= python3
# learn-data runs from code/learn (its scripts import core), so a relative PY is made absolute
LPY = $(if $(findstring /,$(PY)),$(abspath $(PY)),$(PY))

.PHONY: all figures learn-data pdf promo clean

all: figures pdf

## regenerate every figure in figures/ (sim_fig1 writes the data make_fig1 reads)
figures:
	$(PY) code/sim_fig1.py
	$(PY) code/make_fig1.py
	$(PY) code/make_cascade_figures.py
	$(PY) code/make_fig_pinning_saddle.py
	$(PY) code/make_fig_characterization.py
	$(PY) code/make_fig_monitoring.py
	$(PY) code/make_fig_learning.py

## rerun the learned-policy simulations that make_fig_learning.py reads (~1.5 h)
learn-data:
	cd code/learn && "$(LPY)" exp_fixed_k.py
	cd code/learn && "$(LPY)" exp_learn.py sweep
	cd code/learn && "$(LPY)" exp_learn.py sign
	cd code/learn && "$(LPY)" exp_learn.py es 0 20
	cd code/learn && "$(LPY)" exp_tilt.py basin
	cd code/learn && "$(LPY)" exp_tilt.py learn
	cd code/learn && "$(LPY)" exp_tilt.py fresh

## promotional animations (not part of the submission)
## cascade: the spectrum lifting off one mode at a time
## ring:    the degenerate pair, which does not give two role axes
## geometry: the 3D objects -- identity cloud, schema, the split into lobes
promo:
	$(PY) promo/sim_ramp.py
	$(PY) promo/make_ramp_animation.py
	$(PY) promo/sim_ring.py
	$(PY) promo/make_ring_animation.py
	$(PY) promo/sim_geometry.py
	$(PY) promo/make_geometry_animation.py

## build both documents
pdf:
	latexmk -pdf -interaction=nonstopmode main.tex
	latexmk -pdf -interaction=nonstopmode supplement.tex

## remove build droppings, keeping main.pdf and supplement.pdf
clean:
	latexmk -c main.tex supplement.tex
	rm -f pdflatex*.fls *.synctex.gz.sum.synctex
	rm -rf code/__pycache__
