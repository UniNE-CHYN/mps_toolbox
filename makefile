install:
	pip3 uninstall -y mpstool
	pip3 install .

pep8:
	autopep8 --in-place mpstool/*.py
	autopep8 --in-place tests/*.py

test:
	pytest tests/*.py
	pycodestyle mpstool/*.py
	pycodestyle tests/*.py
