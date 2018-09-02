import os
import tempfile

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
        rv = c.post('/predict', json={"id": 0, "observation": {"Age": 22.0, "Cabin": 'null',
                                                               "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}})
        resp = rv.get_json()
        assert resp == {
            # 'error': 'Observation ID: "0" already exists', 'proba': 0.09500452074453283
            'proba': 0.09500452074453283
        }


def test_duplicate_observation(client):
    with client as c:
        rv = c.post('/predict', json={"id": 0, "observation": {"Age": 22.0, "Cabin": 'null',
                                                               "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}})
        resp = rv.get_json()
        assert resp == {
            # 'error': 'Observation ID: "0" already exists', 'proba': 0.09500452074453283
            'proba': 0.09500452074453283
        }
