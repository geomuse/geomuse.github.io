from flask import Flask , render_template

app = Flask(__name__)

name = 'geo'

movies = [
    {'title' : 'A' , 'year' : '2025' , 'star' : '5'} , 
    {'title' : 'B' , 'year' : '2025' , 'star' : '5'} ,
    {'title' : 'C' , 'year' : '2026' , 'star' : '5'} ,
]

@app.route('/')
def index():
    return render_template('index.html',name=name,movies=movies)

if __name__ == '__main__' :

    app.run(debug=True)