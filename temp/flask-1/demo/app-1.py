from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def form_example():
    if request.method == "POST":
        name = request.form.get("name")  # 获取表单输入
        number = request.form.get('number')
        return f"Hello, {name}! Your {int(number)**3}"
    return render_template("form-1.html")  # 显示 HTML 表单

if __name__ == "__main__":
    app.run(debug=True)