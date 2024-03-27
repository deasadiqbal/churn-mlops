from flask import Flask, render_template, request, jsonify
import os
import joblib

from churn_mlops.utils.common import read_yaml
from churn_mlops.constants import CONFIG_FILE_PATH

root_dir = "webapp"

static_dir = os.path.join(root_dir, "static")
template_dir = os.path.join(root_dir, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

# Class to handle exceptions when the values entered are not numeric
class NotANumber(Exception):
    def __init__(self, message = " Values entered are not Numeric"):
        self.message = message
        super().__init__(self.message)

# Function to check if the values entered are numeric
def check_if_numeric(data):
    for _, val in data.items():
        try:
            val = float(val)
        except Exception as e:
            raise NotANumber
    return True

# Fucntion to get prediction from the model
def get_prediction(data):
    config = read_yaml(CONFIG_FILE_PATH)
    model_path = config.model_dir
    RF_mdoel = joblib.load(model_path)
    pred = RF_mdoel.predict(data).tolist()[0]
    return pred

# Function to get response
def get_response(data):
    try:
        if check_if_numeric(data):
            data = data.values()
            data = [list(map(float, data))]
            response = get_prediction(data)
            return response
    except NotANumber as e:
        response = {"error": e.message}
        return response
    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
       
       try:
           if request.form:
               data = dict(request.form)
               response = get_response(data)
               return render_template("index.html", response=response)
       except Exception as e:
           print(e)
           return render_template("404.html")
    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)