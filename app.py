from flask import Flask,render_template,request
import pickle
import json
import numpy as np

with open("artifacts/columns_name.json","r") as json_file:
    col_name=json.load(json_file)

model=pickle.load(open("artifacts/model.pkl",'rb'))

col_name_list=col_name["col_name"]
print(col_name_list)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    data = request.form
    input_data=np.zeros(len(col_name_list))
    input_data[0]=data["Gender"]
    input_data[1]=data["Married"]
    input_data[2]=data["Dependents"]
    input_data[3]=data["Education"]
    input_data[4]=data["Self_Employed"]
    input_data[5]=data["Loan_Amount_Term"]
    input_data[6]=data["Credit_History"]
    input_data[7]=data["Property_Area"]
    #####log Transform for loan amount
    loan_amount=int(data["loan_amt"])
    loan_amount_log=np.log(loan_amount)
    input_data[8]=data["loan_amount_log"]

    total_income=int(data["applocant_income"])+int(data["co_applicant_income"])
    input_data[9]=data["total_income"]

    print(input_data)
    result = model.predict([input_data])

    if result[0]==0:
        print ("Loan rejected")
    else:
        print (f"loan approved")


    return "API SUCCESS"

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)