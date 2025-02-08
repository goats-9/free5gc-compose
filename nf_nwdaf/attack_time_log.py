from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging to output to the console
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['POST'])
def receive_data():
    if request.is_json:
        data = request.get_json()
        logging.info("Received JSON data: %s", data)
        
        # TODO: Add your processing logic here
        # For example, store data in a database, trigger alerts, etc.

        return jsonify({"status": "success"}), 200
    else:
        logging.warning("Received non-JSON data")
        return jsonify({"error": "Invalid JSON"}), 400

if __name__ == '__main__':
    # Run the Flask app on all available interfaces on port 8080
    app.run(host='0.0.0.0', port=8080)
