# Tests other than tox
pytest
echo "flake8:"
flake8 src tests
echo "mypy:"
mypy src tests