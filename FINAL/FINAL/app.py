from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
from back.chatbot import predict_class

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/response', methods=['POST'])
def getvalue():
    text = request.get_json().get("message")
    response = predict_class(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)