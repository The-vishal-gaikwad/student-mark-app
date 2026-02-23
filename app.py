import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

html_template = """
<h2>Student Marks Predictor</h2>
<form method="POST">
    Study Hours: <input type="number" name="hours" step="0.1" required>
    <input type="submit" value="Predict">
</form>
{% if prediction %}
<h3>Predicted Marks: {{ prediction }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        result = model.predict(np.array([[hours]]))
        prediction = round(result[0], 2)

    return render_template_string(html_template, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)