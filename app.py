from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("gradient_boosting_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None

    if request.method == "POST":
        # Get user input from the HTML form
        company = request.form["Company"]
        type_name = request.form["TypeName"]
        ram = int(request.form["Ram"])
        weight = float(request.form["Weight"])
        touchscreen = int(request.form["Touchscreen"])
        ips = int(request.form["Ips"])
        ppi = float(request.form["ppi"])
        cpu_brand = request.form["CpuBrand"]
        gpu_brand = request.form["GpuBrand"]
        os = request.form["OS"]

        # Create a DataFrame with the user input
        new_data = pd.DataFrame({
            "Company": [company],
            "TypeName": [type_name],
            "Ram": [ram],
            "Weight": [weight],
            "Touchscreen": [touchscreen],
            "Ips": [ips],
            "ppi": [ppi],
            "Cpu brand": [cpu_brand],
            "Gpu Brand": [gpu_brand],
            "os": [os]
        })

        # Make a prediction using the loaded model
        predicted_price = model.predict(new_data)

    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
