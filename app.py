import os
import json
import joblib
import numpy as np
import pandas as pd
from threading import Thread
from dotenv import load_dotenv

from flask import Flask, jsonify, request
from utils.base_logger import logger
from train_model import LeadScore
from utils.data_utils import scale_request, get_bad_columns, process_df

LEAD_SCORE_OUTPUT_CSV = '/data/output.csv'
LEAD_SCORE_OUTPUT_MODEL = '/models/lead_score_model.sav'

lead_score = LeadScore()
load_dotenv()
app = Flask(__name__)


@app.route('/api/check-lead-score', methods=['POST'])
def get_score():
    if request.method == 'POST':

        model = joblib.load(LEAD_SCORE_OUTPUT_MODEL)

        if 'file' in request.files:
            file_csv = request.files['file']
            full_data = pd.read_csv(file_csv)
            file_to_send = full_data.copy()

            drop_columns = get_bad_columns()

            full_data.drop(labels=drop_columns, axis=1, inplace=True, errors='ignore')

            full_data = process_df(full_data)
            predictions = model.predict_proba(full_data)[:, 1]

            predict_flags = []
            for prediction in predictions:
                if prediction >= 0.5:
                    predict_flag = 1
                else:
                    predict_flag = 0
                predict_flags.append(predict_flag)

            file_to_send.insert(len(file_to_send.columns), 'predictions', predictions)
            file_to_send.insert(len(file_to_send.columns), 'predict_flags', predict_flags)

            parsed = json.loads(file_to_send.to_json(orient="records"))
            return json.dumps(parsed, indent=4)

        else:
            # get the un-normalized data from the post request
            data = request.get_json(force=True)

            request_json = list(data.values())

            last_row = scale_request(request_json)
            # load model assets

            # get probability prediction from the model
            prediction = model.predict_proba([last_row])[:, 1]

            # flag with custom classifier threshold
            predict_flag = np.where(
                prediction >= 0.5,
                1,
                0
            )

            # tag prediction with date
            predict_date = pd.to_datetime('now', utc=True)

            # prepare output to give back as json
            return_data = {
                'predict_prob': str(prediction),
                'predict_date': str(predict_date),
                'predict_flag': str(predict_flag[0])
            }

            return json.dumps(return_data)


@app.route('/api/train-lead-score', methods=['POST'])
def train_score():
    if request.method == 'POST':
        # Check if a file was passed in the POST request
        if 'file' not in request.files:
            return 'Error: No file part', 400

        file = request.files['file']

        # Check if the file has a CSV extension
        if not file.filename.endswith('.csv'):
            return 'Error: Invalid file type', 400

        if not os.path.exists('./data'):
            os.mkdir('./data')
        file.save(LEAD_SCORE_OUTPUT_CSV)

        try:
            Thread(target=lead_score.train_lead_score_model).start()

            response_text = f'Training of the model has successfully started!'
            logger.info(f'Response text: {response_text}')

        except Exception as error:
            response_text = f"Model training failed due to: {error}"
            logger.info(f'Response text: {response_text}')

        return jsonify(response_text)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=os.getenv("LEAD_SCORE_PORT"))
