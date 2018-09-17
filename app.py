import os
import sys
import logging
import json
import pickle
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


def create_app(test_config=None):
    app = Flask(__name__)
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)

    ########################################
    # Begin database stuff

    # when testing, use temporary database defined in test config
    if test_config is not None and test_config['TESTING']:
        DB = SqliteDatabase(test_config['DATABASE'])

    elif 'DATABASE_URL' in os.environ:
        db_url = os.environ['DATABASE_URL']
        dbname = db_url.split('@')[1].split('/')[1]
        user = db_url.split('@')[0].split(':')[1].lstrip('//')
        password = db_url.split('@')[0].split(':')[2]
        host = db_url.split('@')[1].split('/')[0].split(':')[0]
        port = db_url.split('@')[1].split('/')[0].split(':')[1]
        DB = PostgresqlDatabase(
            dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        )

    else:
        DB = SqliteDatabase('predictions.db')

    class Prediction(Model):
        observation_id = IntegerField(unique=True)
        observation = TextField()
        proba = FloatField()
        true_class = IntegerField(null=True)

        class Meta:
            database = DB

    DB.create_tables([Prediction], safe=True)

    # End database stuff
    ########################################

    ########################################
    # Unpickle the previously-trained model

    with open('pipeline/columns.json') as fh:
        columns = json.load(fh)

    pipeline = joblib.load('pipeline/pipeline.pickle')

    with open('pipeline/dtypes.pickle', 'rb') as fh:
        # convert ints to float to allow np
        dtypes = pickle.load(fh).apply(lambda x: float if x == int else x)

    # End model un-pickling
    ########################################

    ########################################
    # Begin webserver stuff

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            app.logger.info('prediction request: {}'.format(request.get_json()))
        except Exception:
            app.logger.error('error parsing prediction request!')
        # flask provides a deserialization convenience function called
        # get_json that will work if the mimetype is application/json
        obs_dict = request.get_json()
        _id = obs_dict['id']
        observation = obs_dict['observation']

        # for non-object columns, try to convert the values individually, and if it fails, use nan instead
        # this makes sure that an incoming string will be interpreted as NaNs in numerical columns
        for key, value in list(zip(dtypes.keys(), dtypes.values)):
            if (value != object):
                try:
                    observation[key] = value(observation[key])
                except ValueError:
                    observation[key] = np.nan
        # now do what we already learned in the notebooks about how to transform
        # a single observation into a dataframe that will work with a pipeline
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
        # now get ourselves an actual prediction of the positive class
        proba = float(pipeline.predict_proba(obs)[0, 1])
        response = {'proba': proba}
        p = Prediction(
            observation_id=_id,
            proba=proba,
            observation=request.data
        )
        try:
            p.save()
        except IntegrityError:
            error_msg = 'Observation ID: "{}" already exists'.format(_id)
            response['error'] = error_msg
            print(error_msg)
            DB.rollback()
        return jsonify(response)

    @app.route('/update', methods=['POST'])
    def update():
        try:
            app.logger.info('update request: {}'.format(request.get_json(silent=True)))
        except Exception:
            app.logger.error('error parsing update request!')

        obs = request.get_json()
        try:
            p = Prediction.get(Prediction.observation_id == obs['id'])
            p.true_class = obs['true_class']
            p.save()
            return jsonify(model_to_dict(p))
        except Prediction.DoesNotExist:
            error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
            return jsonify({'error': error_msg})

    @app.route('/list-db-contents')
    def list_db_contents():
        return jsonify([
            model_to_dict(obs) for obs in Prediction.select()
        ])

    # End webserver stuff
    ########################################

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
