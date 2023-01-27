import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_utils import (send_training_failed_request, send_training_finished_request)
from utils.base_logger import logger
from utils.data_utils import (dummy_wrapper, process_df, get_bad_columns, convert_to_binary)

LEAD_SCORE_OUTPUT_CSV = '/data/output.csv'
LEAD_SCORE_OUTPUT_MODEL = '/models/lead_score_model.sav'


class LeadScore:
    @staticmethod
    def train_lead_score_model():
        try:
            full_data = pd.read_csv(LEAD_SCORE_OUTPUT_CSV)

            last_column = full_data.columns[len(full_data.columns.tolist()) - 1]
            full_data = convert_to_binary(full_data, last_column)

            y_train = full_data[last_column]
            full_data.drop(labels=last_column, axis=1, inplace=True, errors='ignore')

            drop_columns = get_bad_columns()

            full_data.drop(labels=drop_columns, axis=1, inplace=True, errors='ignore')

            full_data = process_df(full_data)

            full_data = dummy_wrapper(full_data, full_data.columns.tolist())

            x_train = full_data.values

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                              test_size=0.3, random_state=101)
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)

            score = xgb_clf.score(x_val, y_val)
            predict = xgb_clf.predict(x_val)
            accuracy = accuracy_score(y_val, predict)

            print("Score: " + str(score))
            print("Accuracy: " + str((accuracy * 100.0)))

            if not os.path.exists('/models'):
                os.mkdir('/models')

            model_names = [
                LEAD_SCORE_OUTPUT_MODEL
            ]

            model_objects = [
                xgb_clf
            ]

            for i, v in enumerate(model_names):
                joblib.dump(model_objects[i], v)

            logger.info(f"Finished model training")
            send_training_finished_request()

        except Exception as error:
            send_training_failed_request(stage_failed="models training",
                                         error_message=error)
            logger.warning(f"Model training failed due to: {error}")

            response_text = f"Model training failed due to: {error}"
            logger.info(f'Response text: {response_text}')
