from flask import Flask, render_template, redirect, url_for
from forms import SimpleUserForm # 假设 forms.py 在同一目录下

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SimpleUserForm()

    if form.validate_on_submit():
        # 数据有效且已提交
        name = form.username.data
        
        # 在实际应用中，您会将此数据保存到数据库
        print(f"用户注册: {name}") 
        
        # 重定向用户以防止刷新时重复提交
        return redirect(url_for('success', user=name))
    
    # 渲染模板，用于 GET 请求或验证失败时
    return render_template('form_wtf.html', form=form)

@app.route('/success/<user>')
def success(user):
    return f'成功! 欢迎您, {user}!'

@app.route('/')
def home():
    return f'im home.'

if __name__ == '__main__':
    # 运行应用
    # debug=True 允许在代码修改后自动重载，仅用于开发环境
    app.run(debug=True)