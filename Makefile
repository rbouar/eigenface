.PHONY: report


TOPDF=pdflatex
OPT=-output-directory
TARGET=target

REPORTTEXT=src/report/eigenfaces-text.tex
REPORTPDF=target/eigenfaces-text.pdf

PRESENTATIONSRC=src/beamer/presentation.tex
PRESENTATIONPDF=target/presentation.pdf

all: report presentation

report:
	mkdir -p $(TARGET)
	$(TOPDF) $(OPT) $(TARGET) $(REPORTTEXT)
	cp $(REPORTPDF) .

presentation:
	mkdir -p $(TARGET)
	$(TOPDF) $(OPT) $(TARGET) $(PRESENTATIONSRC)
	cp $(PRESENTATIONPDF) .
