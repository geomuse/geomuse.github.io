from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    username = "Alice"
    color = "blue"
    return render_template("index.html", username=username, color=color)

@app.route("/profile/<username>/<int:age>")
def profile(username,age):
    return render_template('form.html',username=username, age=age)

if __name__ == "__main__":

    app.run(debug=True)