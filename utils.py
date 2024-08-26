from transformers import T5Tokenizer, T5ForConditionalGeneration
from faster_whisper import WhisperModel

model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
summary_model = T5ForConditionalGeneration.from_pretrained(model_name)


def speech2text(file_path, process_metadata) -> str:
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
    return input_text_for_summarization


def rawtext2summary(input_text, process_metadata) -> str:
    # Tokenize and summarize the input text using T5
    inputs = tokenizer.encode("summarize: " + input_text,
                              return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=1000, min_length=10,
                                         length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("\nSummary: ", summary)
    process_metadata["t5_summary"] = summary
    return summary
