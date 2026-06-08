from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    prediction = model.predict(np.array(data))
    labels = ["setosa", "versicolor", "virginica"]
    result = [labels[i] for i in prediction]
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)