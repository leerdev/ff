from flask import Flask, redirect, request, render_template
import pickle
import numpy as np

# instance of flask app
app = Flask(__name__)

# loading model
my_model = pickle.load(
    open('my_model.model', 'rb')
)


@app.route('/')
def hello_hse():
    return render_template("f_fire_tmpl.html")


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    form_values = [np.array([int(x) for x in request.form.values()])]
    p = round(my_model.predict_proba(form_values)[0][1], 2)
    message = "Probability of forest fire is HIGH: " + str(p) if p > 0.5 else "Probability of forest fire is LOW: " + str(p)
    return render_template('f_fire_tmpl.html', pred=message)


if __name__ == '__main__':
    app.run()
