.PHONY: report


TOPDF=pdflatex
OPT=-output-directory
TARGET=target
REPORTTEXT=src/report/eigenfaces-text.tex
REPORTPDF=target/eigenfaces-text.pdf



report:
	$(TOPDF) $(OPT) $(TARGET) $(REPORTTEXT)
	cp $(REPORTPDF) .
