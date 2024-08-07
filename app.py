from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from chatbot import predict_class, get_response

app = Flask(__name__)
CORS(app)

# Define route to handle chatbot requests
@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    intents_list = predict_class(message)
    response = get_response(intents_list)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("No, we don't make 'yo mama' jokes here (maybe). Ask us anything!")
    app.run(debug=True)
