lint:
	PYTHONPATH=`pwd` python3 -m black src --check
	PYTHONPATH=`pwd` python3 -m ruff check src
                                                                                                                         
format:
	PYTHONPATH=`pwd` python3 -m black src
	PYTHONPATH=`pwd` python3 -m ruff check src --fix

version:
	python -m setuptools_scm
