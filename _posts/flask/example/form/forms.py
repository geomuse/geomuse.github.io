from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class SimpleUserForm(FlaskForm):
    # 带有标签和验证器（例如，不能为空）的字段
    username = StringField('用户名', validators=[DataRequired(), Length(min=3, max=25)])
    
    # 提交按钮
    submit = SubmitField('注册')