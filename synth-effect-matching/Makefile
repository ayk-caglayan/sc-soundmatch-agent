NAME = ReferencesClaudeCode
#NAME = ReferencesCodex

BIB = References

RM = rm

# TeX Live 2020 pdflatex works, but TeX Live 2020 pdflatex fails ("Undefined control sequence ... \current@color")
#PDFLATEX = pdflatex -interaction=nonstopmode
PDFLATEX = pdflatex

all p : $(NAME).pdf

q : tclean p

$(NAME).pdf : $(NAME).tex *.tex $(BIB).bib
	$(PDFLATEX) $(NAME)
	bibtex $(NAME)
	$(PDFLATEX) $(NAME)
	$(PDFLATEX) $(NAME)

$(NAME).aux: $(NAME).tex
	$(PDFLATEX) $(NAME)

$(NAME).bbl: $(NAME).aux
	bibtex $(NAME)

pnb: $(NAME).bbl
	$(PDFLATEX) $(NAME)
	$(PDFLATEX) $(NAME)

test:
	faust2plot -double $(NAME).dsp
	$(NAME)

pdf p: $(NAME).pdf

pdfo po: p
	open $(NAME).pdf

DSP = tilt

clean:
	-$(RM) $(NAME) $(NAME).dvi $(NAME).aux *~ $(NAME).log $(NAME).out $(NAME).bbl $(NAME).d $(NAME).bak $(NAME).blg
	-$(RM) *~ $(DSP) $(DSP).m $(DSP).dsp.cpp
	-$(RM) -rf $(DSP)-svg/ $(ARXIV_DIR)/

tc tclean: clean
	-$(RM) $(NAME).pdf
	-$(RM) $(ARXIV_DIR).tar.gz

svgFigure fig/svgFigure.pdf: figsvg/svgFigure.dsp
	./myf2ff figsvg/svgFigure.dsp
	rsvg-convert --format=pdf figsvg/svgFigure-svg/process.svg > figsvg/svgFigure.pdf

install:
	/bin/mv bode-one-pole.tikz bode-one-pole.tikz-0
	/bin/sed bode-one-pole.tikz-0 bode-one-pole.tikz

f:
	faust $(DSP).dsp

f2o:
	faust2octave $(DSP).dsp

c:
	faust2plot $(DSP).dsp
	$(DSP)

d:
	faust2firefox $(DSP).dsp

# PDF to text conversion
PDFS = $(wildcard *.pdf)
TXTS = $(patsubst %.pdf,txt/%.txt,$(PDFS))

txt: $(TXTS)

txt/%.txt: %.pdf | txt/
	pdf2txt.py "$<" > "$@"

txt/:
	mkdir -p txt

.PHONY: txt

u update:
	/bin/cp -p tilt.dsp tilt.dsp.txt

uc updateAndCommit: update
	git commit -am 'update'
	git push

ARXIV_FILES = IEEEtran.cls abbreviations-verbose.tex bode_pink.m gnuplot-lua-tikz-common.tex gnuplot-lua-tikz.sty $(BIB).bib $(NAME).tex tikz tikzscale.sty $(NAME).bbl

ARXIV_DIR = ToArXivFromJulius

# ArXiv will not run bibtex for you!:

a arxiv: $(NAME).bbl
	-/bin/rm -rf $(ARXIV_DIR)/
	mkdir $(ARXIV_DIR)
	/bin/cp -rp $(ARXIV_FILES) $(ARXIV_DIR)/
	/usr/bin/tar -czvf $(ARXIV_DIR).tar.gz $(ARXIV_DIR)
