from flask import Flask , render_template
# from flask_sqlalchemy import SQLAlchemy

from datetime import datetime
app = Flask(__name__)

# 資料庫設定：使用 SQLite（可換成 PostgreSQL, MySQL）
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 稍後我們會導入 models
# from models import User

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/users')
def show_users():
    users = User.query.all()  # 查询全部用户
    return render_template('users.html', users=users)

if __name__ == '__main__':

    from models import db, User

    # 初始化資料庫
    db.init_app(app)

    with app.app_context():
        db.create_all()  # 建立所有模型對應的資料表 
        # if not User.query.filter_by(username="jack").first():
        #     user = User(
        #         username="jack",
        #         password="1234",
        #         email="jack111@example.com",
        #         created_at=datetime.utcnow()
        #     )
        #     db.session.add(user)
        #     db.session.commit()
        #     print("使用者 jack 已插入")
        # else:
        #     print("使用者 jack 已存在")

        # users = User.query.all()
        # print(users)

    app.run(debug=True)
