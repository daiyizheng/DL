from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String,Integer,Column
from wtforms.validators import DataRequired,EqualTo,Length
from flask import Flask,render_template,url_for,redirect,request,jsonify,flash
from wtforms import StringField,SubmitField,TextAreaField,TelField,PasswordField,BooleanField
from flask_login import current_user,login_user,logout_user,login_required,LoginManager,UserMixin

app=Flask(__name__)
app.secret_key='view'

class Config:
    SQLALCHEMY_DATABASE_URI='mysql+pymysql://root:root@127.0.0.1:3306/main'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

app.config.from_object(Config)
mysql=SQLAlchemy(app)
login_manager=LoginManager(app)

class UserForm(FlaskForm):
    username=StringField(label='用户名: ',validators=[DataRequired()])
    password=PasswordField(label='密码: ',validators=[DataRequired()])
    remember=BooleanField(label='记住')
    submit=SubmitField(label='登录')

class User(mysql.Model,UserMixin):
    __tablename__='users'
    id=Column(Integer,primary_key=True)
    username=Column(String(128),unique=True)
    password=Column(String(128),unique=True)

    def __repr__(self):
        return '<User: %s:%s>'%(self.username,self.password)

@login_manager.user_loader
def load_user(user_id):
    user=User.query.get(int(user_id))
    return user

@app.route('/login',methods=['POST','GET'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('show_user'))
    form=UserForm()
    if request.method=='POST':
        if form.validate_on_submit():
            username=form.username.data
            password=form.password.data
            remember=form.remember.data
            print('username: {}'.format(username))
            print('password: {}'.format(password))
            print('remember: {}'.format(remember))
            #数据库查询
            user=User.query.filter_by(username=username).first()
            if user:
                if username==user.username and password==user.password:
                # if username==username:
                    #login_user表示让用户登录。保存到当前会话当中（session），这样才能加载和访问id
                    login_user(user,remember)
                    flash('登录成功')
                    return redirect(url_for('show_user'))
            else:
                flash('账户名或者密码错误')
                redirect(url_for('login'))
    return render_template('login.html',form=form)

login_manager.login_view='login'
#login-message：用户重定向到登录页面时闪出的消息
login_manager.login_message='Please restore login!'

@app.route('/showuser',methods=['POST','GET'])
@login_required
def show_user():
    #获得当前登录的用户
    username=current_user.username
    return render_template('template/flask_login.html',username=username)

@app.route('/logout',methods=['POST','GET'])
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    print('flask')
    new_user=User(username='tom',password='123')
    mysql.drop_all()
    mysql.create_all()
    mysql.session.add(new_user)
    mysql.session.commit()
    mysql.session.close()
    app.run(debug=True)
