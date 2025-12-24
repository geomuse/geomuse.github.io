from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST':
        # 从表单中获取数据，使用 HTML 中的 'name' 属性
        name = request.form['username']
        # 您现在可以处理数据（例如，保存到数据库）
        return f'Hello, {name}! 您的表单已提交。'
    
    # 对于 GET 请求，渲染表单模板
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)