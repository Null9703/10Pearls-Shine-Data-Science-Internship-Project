from flask import Flask, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open("best_logistic_regression_model.pkl", 'rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()