from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('california_house_price_prediction_model.pkl')

def predict():
    data = request.get_json(force=True)
    features = pd.DataFrame(data['features'], index=[0])
    prediction = model.predict(features)
    
    return jsonify({'prediction':prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
