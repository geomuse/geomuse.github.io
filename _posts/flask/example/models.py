from flask_sqlalchemy import SQLAlchemy
from datetime import datetime  # 加入這行

db = SQLAlchemy()  # 不要綁 app

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)

    # ✅ 新增時間欄位：自動儲存建立時間
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username} | {self.email}>'
