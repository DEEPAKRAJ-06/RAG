import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Ensure worker is initialized
worker.init_llm()

# ✅ Use `/tmp/uploads/` instead of `./uploads` (since `/tmp/` is writable)
UPLOAD_FOLDER = "/tmp/uploads"

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"Uploads directory created or exists: {UPLOAD_FOLDER}")
except Exception as e:
    print(f"Error creating uploads directory: {str(e)}")

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json.get('userMessage', '')
    
    if not user_message:
        return jsonify({"botResponse": "Please enter a valid message."}), 400
    
    bot_response = worker.process_prompt(user_message)

    return jsonify({"botResponse": bot_response}), 200

# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly. Please try again."
        }), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        file.save(file_path)
        worker.process_document(file_path)

        return jsonify({
            "botResponse": "Thank you for providing your PDF document. I have analyzed it, so now you can ask me any "
                           "questions regarding it!"
        }), 200
    except Exception as e:
        return jsonify({"botResponse": f"Error saving the file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, port=8000, host='0.0.0.0')
