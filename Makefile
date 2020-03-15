.PHONY: report


TOPDF=pdflatex
OPT=-output-directory
TARGET=target
REPORTTEXT=src/report/eigenfaces-text.tex
REPORTPDF=target/eigenfaces-text.pdf



report:
	mkdir -p $(TARGET)
	$(TOPDF) $(OPT) $(TARGET) $(REPORTTEXT)
	cp $(REPORTPDF) .
