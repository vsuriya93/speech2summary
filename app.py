from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from faster_whisper import WhisperModel
import json
import os

app = Flask(__name__)
app.secret_key = 'sairam'  # Necessary for flashing messages

model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
summary_model = T5ForConditionalGeneration.from_pretrained(model_name)

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
        # return redirect(url_for('index'))


def process(file_path):
    print("processing file: " + file_path)

    # store information in dict
    process_metadata = {}
    process_metadata["file_path"] = file_path

    # todo: add more checks here
    segments, info = model.transcribe(file_path, beam_size=5)
    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))

    process_metadata["language"] = info.language
    process_metadata["language_probability"] = info.language_probability

    texts = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))
        texts.append(segment.text)

    input_text_for_summarization = ''.join(' '.join(texts)).strip()
    print("Original Text:")
    print(input_text_for_summarization)

    process_metadata["audio_file_summary"] = input_text_for_summarization

    # Tokenize and summarize the input text using T5
    inputs = tokenizer.encode("summarize: " + input_text_for_summarization,
                              return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=1000, min_length=10,
                                         length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("\nSummary: ", summary)
    process_metadata["t5_summary"] = summary

    return process_metadata
