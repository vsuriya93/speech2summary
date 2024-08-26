from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from utils import speech2text, rawtext2summary
import json
import os


app = Flask(__name__)
app.secret_key = 'sairam'  # Necessary for flashing messages


# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    #todo: add more checks here + error handling and refactor into utils
    """
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash(f'File {filename} uploaded successfully!')
        response = app.response_class(
            # Indent by 4 spaces
            response=json.dumps(process(file_path), indent=4),
            mimetype='application/json'
        )
        return response
        # todo: design a landing page to render json response
        # return redirect(url_for('index'))


def process(file_path):
    print("processing file: " + file_path)

    # store information in dict
    process_metadata = {}
    process_metadata["file_path"] = file_path

    input_text_for_summarization = speech2text(file_path, process_metadata)
    summary = rawtext2summary(input_text_for_summarization, process_metadata)
    return process_metadata


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
