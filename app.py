from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/decision_tree.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        age = int(request.form["Age"])
        income = float(request.form["Income"])
        loan = float(request.form["LoanAmount"])
        credit = int(request.form["CreditScore"])

        data = pd.DataFrame([[age, income, loan, credit]],
                            columns=[
                                "Age",
                                "Income",
                                "LoanAmount",
                                "CreditScore"
                            ])

        result = model.predict(data)

        if result[0] == 1:
            prediction = "✅ Loan Approved"
        else:
            prediction = "❌ Loan Rejected"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)