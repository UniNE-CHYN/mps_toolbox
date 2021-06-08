install:
	pip3 install -U .

uninstall:
	pip3 uninstall -y mpstool

pep8:
	autopep8 --in-place mpstool/*.py
	autopep8 --in-place tests/*.py

test:
	flake8 mpstool tests --statistics
	PYTHONPATH=. pytest -v --cov mpstool

major:
	bumpversion major

minor:
	bumpversion minor

requirements:
	pip freeze > requirements.txt

