from flask import Flask, render_template , abort

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("indexx.html")

# 自定义 404 错误页面
@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404

@app.route('/divide/<int:num1>/<int:num2>')
def divide(num1,num2):
    num1 , num2 = int(num1) , int(num2)
    if num2 == 0 :
        abort(403)
    return f'{num1/num2}'

@app.errorhandler(403)
def not_found_error(error):
    return render_template("404.html"), 403

if __name__ == "__main__":

    app.run(debug=True)