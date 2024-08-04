from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("best_logistic_regression_model.pkl", 'rb'))
poly = pickle.load(open('poly_features.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    health_status = {
        'status': 'healthy',
        'message': 'Application is running'
    }
    return jsonify(health_status)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    # print(type(form_data))
    # print(form_data)
    data = {
        'gender': [form_data['gender']],
        "SeniorCitizen": [int(form_data['SeniorCitizen'])],
        "Partner": [form_data['Partner']],
        'Dependents': [form_data['Dependents']],
        'tenure': [int(form_data['tenure'])],
        'PhoneService': [form_data['PhoneService']],
        'MultipleLines': [form_data['MultipleLines']],
        'InternetService': [form_data['InternetService']],
        'OnlineSecurity': [form_data['OnlineSecurity']],
        'OnlineBackup': [form_data['OnlineBackup']],
        'DeviceProtection': [form_data['DeviceProtection']],
        'TechSupport': [form_data['TechSupport']],
        'StreamingTV': [form_data['StreamingTV']],
        'StreamingMovies': [form_data['StreamingMovies']],
        'Contract': [form_data['Contract']],
        'PaperlessBilling': [form_data['PaperlessBilling']],
        'PaymentMethod': [form_data['PaymentMethod']],
        'MonthlyCharges': [float(form_data['MonthlyCharges'])],
        'TotalCharges': [float(form_data['TotalCharges'])]
    }
    df = pd.DataFrame(data)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('uint8')
    print("Input:")
    for i in df:
        print(i, ":", df[i].iloc[0])
    
    # one hot encoding
    new_data = {
        'gender_Male': [int(df.gender[0] == 'Male')],
        'Partner_Yes': [int(df.Partner[0] == 'Yes')],
        'Dependents_Yes': [int(df.Dependents[0] == 'Yes')],
        'PhoneService_Yes': [int(df.PhoneService[0] == 'Yes')],
        "MultipleLines_No phone service": [int(df.MultipleLines[0] == 'No phone service')],
        "MultipleLines_Yes": [int(df.MultipleLines[0] == 'Yes')],
        "InternetService_Fiber optic": [int(df.InternetService[0] == 'Fiber optic')],
        "InternetService_No": [int(df.InternetService[0] == 'No')],
        "OnlineSecurity_No internet service": [int(df.OnlineSecurity[0] == 'No internet service')],
        "OnlineSecurity_Yes": [int(df.OnlineSecurity[0] == 'Yes')],
        "OnlineBackup_No internet service": [int(df.OnlineBackup[0] == "No internet service")],
        "OnlineBackup_Yes": [int(df.OnlineBackup[0] == "Yes")],
        'DeviceProtection_No internet service': [int(df.DeviceProtection[0] == "No internet service")],
        'DeviceProtection_Yes': [int(df.DeviceProtection[0] == "Yes")],
        'TechSupport_No internet service': [int(df.TechSupport[0] == "No internet service")],
        'TechSupport_Yes': [int(df.TechSupport[0] == "Yes")],
        'StreamingTV_No internet service': [int(df.StreamingTV[0] == "No internet service")],
        'StreamingTV_Yes': [int(df.StreamingTV[0] == "Yes")],
        'StreamingMovies_No internet service': [int(df.StreamingMovies[0] == "No internet service")],
        'StreamingMovies_Yes': [int(df.StreamingMovies[0] == "Yes")],
        'Contract_One year': [int(df.Contract[0] == "One year")],
        'Contract_Two year': [int(df.Contract[0] == "Two year")],
        'PaperlessBilling_Yes': [int(df.PaperlessBilling[0] == "Yes")],
        'PaymentMethod_Credit card (automatic)': [int(df.PaymentMethod[0] == "Credit card (automatic)")],
        'PaymentMethod_Electronic check': [int(df.PaymentMethod[0] == "Electronic check")],
        'PaymentMethod_Mailed check': [int(df.PaymentMethod[0] == "Mailed check")]
    }
    temp_df = pd.DataFrame(new_data).astype('uint8')
    df2 = pd.concat([df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']], temp_df], axis = 1)
    df2.rename(columns = {'gender_Male': 'gender', 'Partner_Yes': 'Partner', 'Dependents_Yes': 'Dependents', 'PhoneService_Yes': 'PhoneService', 'PaperlessBilling_Yes': 'PaperlessBilling'}, inplace = True)
    
    # Polynomial Features
    temp = pd.DataFrame(poly.transform(df2[["tenure", "MonthlyCharges", "TotalCharges"]]), columns = poly.get_feature_names_out())
    df2.drop(["tenure", "MonthlyCharges", "TotalCharges"], axis = 1, inplace = True)
    df2 = pd.concat([df2, temp], axis = 1)
    
    # Feature Importance
    # drop_features = ['MultipleLines_No phone service', 'InternetService_No', 'OnlineBackup_No internet service', 'DeviceProtection_No internet service', 'TechSupport_No internet service', 'StreamingMovies_No internet service']
    # df2.drop(drop_features, axis = 1)
    # df2.drop(['tenure^2', 'MonthlyCharges^2', 'TotalCharges^2', 'MonthlyCharges TotalCharges', 'tenure TotalCharges', 'tenure MonthlyCharges', "Churn_Yes"], axis = 1)
    
    # Standard Scaler
    df2[poly.get_feature_names_out()] = scaler.transform(df2[poly.get_feature_names_out()])
    # print(df2.dtypes)
    
    # Model Prediction
    prediction = model.predict(df2)[0]
    prediction = "Customer will likely churn." if prediction == 1 else "Customer will likely be retained."
    
    print("Preprocessed Input:")
    for i in df2:
        print(i, ":", df2[i].iloc[0])
    
    print("Prediction:", prediction)
    # Check the 'Accept' header
    if request.headers.get('Accept') == 'application/json':
        return jsonify({"prediction": prediction})
    else:
        return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run()