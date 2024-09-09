from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', name="John")
# index.html 以 bash.html 为准

if __name__ == '__main__':
    app.run(debug=True)