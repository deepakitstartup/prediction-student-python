from flask import Flask, jsonify, request
from statsmodels.iolib.smpickle import load_pickle
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

app = Flask(__name__)
df = pd.read_csv("student-mat.csv", delimiter=';')

# cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob',
#                 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
#             'nursery', 'higher', 'internet', 'romantic']
#
# num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
#            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

cat_cols = []
num_cols = ['age', 'G2']
target = 'G3'


def train_model(d, numeric_var_cols, categorical_var_cols, target):
    """ Read data file as DataFrame """
    x = pd.get_dummies(d[np.append(categorical_var_cols, numeric_var_cols)], drop_first=True)
    y = d[target]
    print("y shape - ", y.shape)
    # scaler = pp.StandardScaler()
    # for i in numeric_var_cols:
    #    x[i] = scaler.fit_transform(x[i])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

    print(len(x_train), len(y_train))
    print(len(x_test), len(y_test))

    #reg_model = lm.LinearRegression(fit_intercept=True, normalize=False)
    reg_model = sm.OLS(y_train, x_train)
    res = reg_model.fit()
    res.save("trained_model.pickle")
    print(res.summary())
    s = "The Prediction model is trained at::" +str(datetime.now())+ ". Trained model score is " + str(round(res.rsquared*100, 2)) + " %"
    return s


@app.route('/predict_model', methods=['GET'])
def predict_model():
    if request.method == 'GET':
        loaded_res = load_pickle("trained_model.pickle")

        age = int(request.args.get("age"))
        g2_score = int(request.args.get("g2_score"))
        l = [age, g2_score]
        rec = np.array(l)
        prediction = np.round(loaded_res.predict(rec))
        return "Predicted Final Score of student is: " + str(prediction[0]) +". The driving parameters are Age, Maths & Science Score"


@app.route('/train_model', methods=['GET'])
def modelling():
    if(request.method == 'GET'):
        print("\nStudent Performance Prediction")
        s = train_model(df, num_cols, cat_cols, target)
        return s



# driver function
if __name__ == '__main__':
    # creating a Flask app
    app.run(debug=True)
