import os
import tempfile
import numpy as np

import pytest
from app import create_app


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # create a temporary file to isolate the database for each test
    db_fd, db_path = tempfile.mkstemp()

    # create the app with common test config
    app = create_app({
        'TESTING': True,
        'DATABASE': db_path,
    })

    yield app

    # close and remove the temporary database
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_empty_db_contents(client):
    with client as c:
        rv = c.get('/list-db-contents')
        print(rv.response)
        assert rv.get_json() == []


def test_new_observation(client):
    with client as c:
        rv = c.post('/predict', json={"id": 0, "observation": {"m_or_f": "m", "person_attributes": "driving", "seat": "front_left",
                    "other_person_location": "N/A", "other_factor_1": "N/A", "other_factor_2": "N/A", "other_factor_3": "N/A", "age_in_years": "50"
                                                               }})
        resp = rv.get_json()
        # the return json only includes the probability
        assert list(resp.keys()) == ['proba']
        # probability is a float between 0 and 1
        assert resp['proba'] >= 0 and resp['proba'] <= 1 and type(resp['proba']) == float


def test_duplicate_observation(client):
    with client as c:
        rv = c.post('/predict', json={"id": 0, "observation": {"m_or_f": "m", "person_attributes": "driving", "seat": "front_left",
                    "other_person_location": "N/A", "other_factor_1": "N/A", "other_factor_2": "N/A", "other_factor_3": "N/A", "age_in_years": "50"
                                                               }})
        resp = rv.get_json()
        # the return json only includes the probability
        assert list(resp.keys()) == ['proba']
        # probability is a float between 0 and 1
        assert resp['proba'] >= 0 and resp['proba'] <= 1 and type(resp['proba']) == float


def test_na_observation(client):
    with client as c:
        rv = c.post('/predict', json={"id": 12, "observation": {"m_or_f": "m", "person_attributes": np.nan, "seat": "front_left",
                    "other_person_location": "N/A", "other_factor_1": "N/A", "other_factor_2": "N/A", "other_factor_3": np.nan, "age_in_years": np.nan
                                                                }})
        resp = rv.get_json()
        # the return json only includes the probability
        assert list(resp.keys()) == ['proba']
        # probability is a float between 0 and 1
        assert resp['proba'] >= 0 and resp['proba'] <= 1 and type(resp['proba']) == float
