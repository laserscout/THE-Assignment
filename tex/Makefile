.PHONY: main.pdf all clean

all: main.pdf

FLAGS = -pdf -pdflatex="xelatex" -use-make \
	-quiet -synctex=1

main.pdf: main.tex
	latexmk $(FLAGS) main.tex #; skimview $@

clean:
	latexmk -CA
