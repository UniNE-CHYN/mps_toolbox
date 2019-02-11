install:
	pip3 install -U .

uninstall:
	pip3 uninstall -y mpstool


pep8:
	autopep8 --in-place mpstool/*.py
	autopep8 --in-place tests/*.py

test:
	pytest tests/*.py
	pycodestyle mpstool/*.py
	pycodestyle tests/*.py
