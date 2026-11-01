# Needs a TeX distribution with latexmk, koma-script, natbib, chicago, cjk.
# Figures need a Python with seaborn.
PY ?= $(HOME)/miniconda3/bin/python3

thesis.pdf: thesis.tex references.bib $(wildcard chapters/*.tex appendix/*.tex backmatter/*.tex tables/*.tex)
	latexmk -pdf -interaction=nonstopmode thesis.tex

clean:
	latexmk -C thesis.tex

# Aggregates from the company export and notebooks into data/*.csv
data:
	$(PY) scripts/extract.py

# data/*.csv into numbers.tex, the macros the prose uses for every number
numbers:
	$(PY) scripts/macros.py

# data/*.csv and harness/out into figures/*.pdf (previews in figures/preview)
figures:
	$(PY) scripts/figures.py

# data/*.csv and harness/out into tables/*.tex, \input by the chapters
tables:
	$(PY) scripts/tables.py

.PHONY: clean data numbers figures tables
