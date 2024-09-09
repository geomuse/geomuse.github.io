from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'
# db = SQLAlchemy(app)

# 数据库模型
# class Todo(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     content = db.Column(db.String(200), nullable=False)
#     completed = db.Column(db.Boolean, default=False)

# 首页，展示待办事项
@app.route('/')
def index():
    # todos = Todo.query.all()
    return render_template('index.html')

# 添加新的待办事项
@app.route('/add', methods=['POST'])
def add():
    todo_content = request.form['content']
    # new_todo = Todo(content=todo_content)
    
    # try:
    #     db.session.add(new_todo)
    #     db.session.commit()
    #     return redirect('/')
    # except:
    #     return "There was an issue adding your task"

# 删除待办事项
# @app.route('/delete/<int:id>')
# def delete(id):
#     todo_to_delete = Todo.query.get_or_404(id)

#     try:
#         db.session.delete(todo_to_delete)
#         db.session.commit()
#         return redirect('/')
#     except:
#         return "There was an issue deleting the task"

# 标记为已完成
# @app.route('/complete/<int:id>')
# def complete(id):
#     todo = Todo.query.get_or_404(id)
#     todo.completed = not todo.completed
    
#     try:
#         db.session.commit()
#         return redirect('/')
#     except:
#         return "There was an issue marking the task as complete"

if __name__ == '__main__':
    
    app.run(debug=True)

