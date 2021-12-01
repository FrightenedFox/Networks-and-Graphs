import pytest
import pandas as pd
from optinet.create_graph import CitiesNodes

df = pd.read_csv("tests/test_data/Cities.csv", sep=";", decimal=",")


@pytest.fixture()
def graph():
    return CitiesNodes(df)
