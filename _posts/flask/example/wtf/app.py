from flask import Flask, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, Email, ValidationError
import re

# 1. 模拟数据存储 (代替数据库)
# 键: 用户名, 值: 密码
USERS_DB = {'admin': '123456', 'testuser': 'password'}

# --- 2. 表单类定义与自定义验证器 ---

# 📌 自定义验证器函数（确保用户名唯一性）
def validate_username_unique(form, field):
    if field.data in USERS_DB:
        # 如果用户名已存在于我们的“数据库”中，则抛出 ValidationError
        raise ValidationError('该用户名已被占用，请选择另一个。')

# 📌 自定义验证器方法（在LoginForm中检查用户名/密码是否匹配）
class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired('请输入用户名。')])
    password = PasswordField('密码', validators=[DataRequired('请输入密码。')])
    submit = SubmitField('登录')
    
    # 自动执行的自定义方法验证：validate_<字段名>
    def validate_username(self, field):
        # 验证用户名是否存在
        if field.data not in USERS_DB:
            raise ValidationError('用户名不存在或密码错误。')

    def validate_password(self, field):
        # 只有在用户名存在的情况下才检查密码（避免两次抛出错误）
        username = self.username.data
        if username in USERS_DB:
            # 检查密码是否匹配
            if field.data != USERS_DB.get(username):
                raise ValidationError('用户名不存在或密码错误。')

class RegisterForm(FlaskForm):
    username = StringField(
        '用户名', 
        validators=[
            DataRequired('请输入用户名。'), 
            Length(min=4, max=25, message='用户名长度必须在4到25个字符之间。'),
            validate_username_unique # 👈 应用自定义验证器函数
        ]
    )
    email = StringField(
        '邮箱', 
        validators=[
            DataRequired('请输入邮箱。'), 
            Email('邮箱格式不正确。')
        ]
    )
    password = PasswordField(
        '密码', 
        validators=[
            DataRequired('请输入密码。'), 
            Length(min=6, message='密码至少需要6个字符。'),
            EqualTo('confirm_password', message='两次输入的密码必须匹配。')
        ]
    )
    confirm_password = PasswordField('确认密码', validators=[DataRequired('请再次输入密码。')])
    submit = SubmitField('注册')


# --- 3. Flask 应用配置和视图函数 ---

app = Flask(__name__)
# ⚠️ 必须设置 SECRET_KEY，用于保护会话和 CSRF Token
app.config['SECRET_KEY'] = 'simple_secure_key_12345' 

# 注册路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    
    # form.validate_on_submit() 处理 POST 请求并运行所有验证器
    if form.validate_on_submit():
        # 验证通过，执行注册逻辑
        username = form.username.data
        password = form.password.data 
        
        # 模拟：将新用户添加到我们的“数据库”
        USERS_DB[username] = password
        
        print(f"用户 {username} 注册成功！当前的数据库: {USERS_DB}")
        
        flash(f'用户 {username} 注册成功！请登录。', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html', form=form)


# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    
    if form.validate_on_submit():
        # 登录表单的所有验证器（包括 validate_username 和 validate_password）都已通过
        flash('登录成功！欢迎回来。', 'success')
        # 实际应用中：在这里设置会话，标记用户已登录
        return redirect(url_for('index'))
            
    return render_template('login.html', form=form)

# 首页路由
@app.route('/')
def index():
    return "<h1>欢迎来到首页！</h1><p>请访问 <a href='/login'>登录</a> 或 <a href='/register'>注册</a></p>"

if __name__ == '__main__':
    app.run(debug=True)
