from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 为了安全性，请使用随机生成的密钥

# 模拟用户数据
users = {'geo': 'kali'}

@app.route('/')
def index():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 验证用户名和密码
        if username in users and users[username] == password:
            session['username'] = username  # 保存用户名到会话
            return redirect(url_for('index'))
        else:
            error = '无效的用户名或密码，请重试。'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)  # 清除会话中的用户名
    return redirect(url_for('login'))

if __name__ == '__main__':
    
    app.run(debug=True)