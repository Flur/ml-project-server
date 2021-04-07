from flask import Flask
from flask import request
import lightgbm as lgb

app = Flask(__name__)

bst = lgb.Booster(model_file='final_lgbm_model.txt')

@app.route('/getPredictions', methods=['POST'])
def get_predictions():
    body = request.get_json()

    ypred = bst.predict(body.get('data'), num_iteration=bst.best_iteration)

    return {"data":ypred.tolist()}
