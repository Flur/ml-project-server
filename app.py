from flask import Flask
from flask import request
import lightgbm as lgb
import numpy as np

app = Flask(__name__)

bst = lgb.Booster(model_file='final_lgbm_model.txt')

@app.route('/getPredictions', methods=['POST'])
def get_predictions():
    np.set_printoptions(suppress=True)

    body = request.get_json()

    ypred = bst.predict(body.get('data'), num_iteration=bst.best_iteration)

    return {"prediction": str(ypred)}
