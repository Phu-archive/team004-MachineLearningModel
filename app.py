from flask import Flask, jsonify, request
from functionalities import getResponse, loadTextGetVector

app = Flask(__name__)

@app.route('/response', methods=["POST"])
def getBotResponse():
    data = request.get_json()
    response = {
        "status": "success",
        "message": "Invalid Payload"
    }

    if not data:
        return jsonify(response), 400

    text = data.get('text')
    temperature = data.get('temperature')

    if not text:
        response["message"] = "Text not Found"
        return jsonify(response), 400

    if not temperature:
        temperature = 0.5

    try:
        temperature = float(temperature)
    except ValueError:
        response["message"] = "Temperature isn't Float"
        return jsonify(response), 400


    outText = getResponse(text.lower(), temp=temperature)

    response["status"] = "success"
    response["message"] = "successfully added the response"
    response["response_text"] = outText
    return jsonify(response), 200


@app.route('/vector', methods=["POST"])
def getVector():
    data = request.get_json()
    response = {
        "status": "success",
        "message": "Invalid Payload"
    }

    if not data:
        return jsonify(response), 400

    text = data.get('text')

    if not text:
        response["message"] = "Text not Found"
        return jsonify(response), 400

    outVec = loadTextGetVector(text.lower())

    response["status"] = "success"
    response["message"] = "successfully get the vector of the value"
    response["response_vector"] = outVec
    return jsonify(response), 200

@app.route('/')
def main():
    response = {
        "status": "success",
        "message": "Pong"
    }
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
