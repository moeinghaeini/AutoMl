help:
	@echo "Makefile help: (Tested on Linux)"
	@echo "* install	to install the requirements into your current virtaul env"
	@echo "* test		to run the tests"
	@echo "* check 	to run the code style checker"

install: | package-install download

package-install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests

check:
	pycodestyle --max-line-length=120 src
	@echo "All good!"

clean-tex:
	rm *.log *.aux *.out

tex:
	pdflatex *.tex
	make clean-tex

download:
	wget --no-clobber --directory-prefix "data" "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
	tar -xf "data/fcnet_tabular_benchmarks.tar.gz" --skip-old-files --directory "data" --strip-components=1


.PHONY: install test check all clean-tex tex help
.DEFAULT_GOAL := help
