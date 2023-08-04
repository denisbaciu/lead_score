# Agnostic LEAD SCORE

<img width="812" alt="image" src="https://github.com/denisbaciu/lead_score/assets/98389187/eadfcbae-f6c5-4b4a-a506-1b3de7dd6f48">

## Unique Features
Our model distinguishes itself with its adaptability and versatility. It can ingest any CSV file, accommodating any number of features and any type of data, all without needing pre-setting or manual tuning. This unparalleled flexibility drastically reduces the need for intensive data cleaning and pre-processing, allowing you to concentrate more on deriving insights from the model's predictions.

## Pre-requisites
The model requires training data in the form of a CSV file. This data serves as the basis for learning and prediction. The model is designed to use the last column of the CSV file for predictions and training, simplifying data preparation for users.

## Pre-processing
The model automates the pre-processing stage to identify the most relevant features from the data. It performs tokenization, categorization, and normalization of data, and intelligently selects the top 10 most impactful features for model training. These steps ensure that the model comprehends and uses the most pertinent information for generating accurate predictions.

## Model Training and Saving
During the model training phase, the dataset is partitioned into a 70% training set and a 30% test set. The model leverages the powerful XGBoost (Extreme Gradient Boosting) algorithm for this process. After each training cycle, the model is saved on the server in the .sav format, allowing for easy retrieval and reuse. This feature ensures consistency in predictions and eliminates the need for retraining the model with the same data.

An endpoint is exposed which, when invoked, loads the saved model to make further predictions. This design provides a seamless workflow for recurrent predictions and operational efficiency.

## Data Validation and Prediction
The model validates the incoming data at the prediction stage, ensuring data integrity and avoiding potential errors that could disrupt the model's performance. This robust validation checks the columns, data types, and the overall integrity of the data, making sure it aligns with the model's expectations.

When new training data is provided, this process repeats, ensuring the model is always working with high-quality, reliable data.

## Integration with Databases
For larger and more complex datasets, the model can be wired to a database, bypassing the CSV file step. Clients can run insights on vast datasets by directly calling the API, which initiates model training and predictions. This feature provides a scalable solution for big data analytics and machine learning, catering to the evolving needs of data-driven organizations.

## Use Case Domains
With an average accuracy rate of 87%, our model can serve diverse domains, from predicting credit defaults to diagnosing heart failure, or any other use case your organization might have. Its ability to predict across various domains makes our lead scoring model an indispensable tool for any industry seeking data-driven insights and predictions.

http://127.0.0.1:5001/api/train-lead-score
<img width="866" alt="image" src="https://user-images.githubusercontent.com/98389187/219647288-cc6026c4-f01b-4287-8253-c1345da6af45.png">

http://127.0.0.1:5001/api/check-lead-score
<img width="813" alt="image" src="https://user-images.githubusercontent.com/98389187/219647354-33e53286-ddc5-4fcd-9822-506f28de73f2.png">
