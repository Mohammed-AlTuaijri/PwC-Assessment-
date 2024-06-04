from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from scraper import scrape_website, visited
from vector_db import store_in_vector_db, query_vector_db
import openai
import os
import logging
import replicate
from openai import OpenAI
import tiktoken
import time

logging.basicConfig(level=logging.DEBUG)

openai.api_key = os.getenv('OPENAI_API_KEY')
replicate_api_key = os.getenv('REPLICATE_API_TOKEN')

if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if replicate_api_key is None:
    raise ValueError("REPLICATE_API_KEY environment variable not set.")

app = Flask(__name__)

# Route to serve the HTML file
@app.route('/')
def Assignment():
    return send_from_directory('.', 'Assignment.html')

# SSE endpoint for testing
@app.route('/stream')
def stream():
    """Simple stream endpoint for testing SSE."""
    def event_stream():
        while True:
            time.sleep(1)
            yield f'data: The current time is {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n'
    return Response(event_stream(), mimetype="text/event-stream")

# Endpoint to scrape the website
@app.route('/scrape', methods=['POST'])
def scrape():
    global visited
    visited.clear()
    logging.debug("Received request data: %s", request.data)
    try:
        base_url = request.json['url']
        path_prefix = '/en/information-and-services'
        text = scrape_website(base_url, base_url, path_prefix, depth=10)
        print("DATA SCRAPED")

        # Save the scraped text to a file
        file_path = 'scraped_text_dp10.txt'
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)

        return jsonify({"message": "Data scraped and saved to file"})
    except Exception as e:
        logging.error("Error processing request: %s", str(e))
        return jsonify({"error": str(e)}), 400

# Endpoint to embed text and store it in the vector database
@app.route('/embed', methods=['POST'])
def embed():
    try:
        file_path = 'scraped_text_dp10.txt'
        if not os.path.exists(file_path):
            return jsonify({"error": "Scraped text file does not exist. Please run the scrape endpoint first."}), 400
            
        with open(file_path, 'r', encoding='utf-8') as file:
            text_from_file = file.read()

        logging.debug(f"Text read from file for embedding: {text_from_file[:1000]}...")
        store_in_vector_db(text_from_file)
        return jsonify({"message": "Text from file embedded and stored in vector database"})
    
    except Exception as e:
        logging.error("Error processing request: %s", str(e))
        return jsonify({"error": str(e)}), 400

# Define the maximum context length for the models
MODEL_MAX_TOKENS = {
    "gpt-3.5-turbo": 16284,
    "gpt-4": 8092,
    "meta/llama-2-70b-chat": 2996
}
# Define maximum output tokens for the models
MAX_OUTPUT_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 1024,
    "meta/llama-2-70b-chat": 200
}
# Tokenizer setup using tiktoken for the models
TOKEN_ENCODINGS = {
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
    "meta/llama-2-70b-chat": tiktoken.encoding_for_model("gpt-3.5-turbo"),  # Replace with actual encoding if available
}

def combine_and_truncate_prompt(user_prompt, results, model_name):
    additional_context = " ".join(results)
    combined_prompt = f"""
    You are being provided with a user's query and relevant information. Please respond to the user's query based only on the information provided below. Do not use any external knowledge or your own knowledge.

    User's Query: {user_prompt}

    Information Provided:
    {additional_context}

    Response:
    """
    encoding = TOKEN_ENCODINGS[model_name]
    max_input_tokens = MODEL_MAX_TOKENS[model_name] - MAX_OUTPUT_TOKENS[model_name]  # Calculate max input tokens
    tokens = encoding.encode(combined_prompt)
    if len(tokens) > max_input_tokens:
        truncated_tokens = tokens[:max_input_tokens]  # Truncate to max input tokens
        combined_prompt = encoding.decode(truncated_tokens)
    return combined_prompt

# Endpoint to query GPT-3.5
@app.route('/query_gpt3', methods=['POST'])
def query_gpt3():
    user_prompt = request.json['prompt']
    results = query_vector_db(user_prompt)
    truncated_combined_prompt = combine_and_truncate_prompt(user_prompt, results, "gpt-3.5-turbo")
    
    # Log the combined prompts being sent to each model
    with open('gpt-3.5-turbo_prompts.txt', 'w', encoding='utf-8') as file:
        file.write(f"Prompt for {"gpt-3.5-turbo"}:\n\n{truncated_combined_prompt}\n\n")

    client = OpenAI()
    def generate():
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": truncated_combined_prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield f"{chunk.choices[0].delta.content} ".encode('utf-8') 
    return Response(generate(), content_type='text/event-stream')

# Endpoint to query GPT-4
@app.route('/query_gpt4', methods=['POST'])
def query_gpt4():
    user_prompt = request.json['prompt']
    results = query_vector_db(user_prompt)
    truncated_combined_prompt = combine_and_truncate_prompt(user_prompt, results, "gpt-4")
    
    # Log the combined prompts being sent to each model
    with open('gpt-4_prompt.txt', 'w', encoding='utf-8') as file:
        file.write(f"Prompt for {"gpt-4"}:\n\n{truncated_combined_prompt}\n\n")

    client = OpenAI()
    def generate():
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": truncated_combined_prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield f"{chunk.choices[0].delta.content} ".encode('utf-8')
    return Response(generate(), content_type='text/event-stream')

# Endpoint to query LLaMA-2
@app.route('/query_llama2', methods=['POST'])
def query_llama2():
    user_prompt = request.json['prompt']
    results = query_vector_db(user_prompt)
    truncated_combined_prompt = combine_and_truncate_prompt(user_prompt, results, "meta/llama-2-70b-chat")

    # Log the combined prompts being sent to each model
    with open('llama-2-70b-chat_prompt', 'w', encoding='utf-8') as file:
        file.write(f"Prompt for {"llama-2-70b-chat"}:\n\n{truncated_combined_prompt}\n\n")

    def generate():
        prediction = replicate.run(
            "meta/llama-2-70b-chat",
            input={"prompt": truncated_combined_prompt, "temperature": 1},
            stream=True,
        )
        for message in prediction:
            yield f"{message} "
    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug = True)