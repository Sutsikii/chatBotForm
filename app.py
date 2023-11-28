import json
from flask import Flask, render_template, request, jsonify
from llamaapi import LlamaAPI
import html

app = Flask(__name__)

llama = LlamaAPI("LL-KhzmmTe3QZLdq9V2Hx0AoFARaVgB5SKvz4fH4BvaoZaYjaSXaQWydVLGb9r3hgoG")

conversation_context = []

@app.route('/')
def home():
    with open('data.json', 'r', encoding='utf-8') as file:
        data_json_content = file.read()
    return render_template('index.php', data_json_content=data_json_content)

@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')

    llama2_response = call_llama2_api(user_text)
    bot_response = process_llama2_response(llama2_response)

    conversation_context.append({"user_input": user_text, "bot_response": bot_response})

    return jsonify({"bot_response": bot_response})

def call_llama2_api(user_input):
    api_request_json = {
        "messages": [
            {"role": "user", "content": user_input},
        ],
        "functions": [
        ],
        "stream": False,
    }

    response = llama.run(api_request_json)
    print(f"Llama2 Response: {response.text}")
    return response

def process_llama2_response(llama2_response):
    try:
        data = llama2_response.json()
        if 'choices' in data and data['choices']:
            assistant_message = data['choices'][0]['message']['content']
            assistant_message_escaped = html.escape(assistant_message)
            return assistant_message_escaped
        else:
            return "I cannot provide any information at this time. Please try again later."
    except Exception as e:
        print(f"Error processing Llama2 response: {e}")
        return "An error occurred while processing the response."
    
    
if __name__ == '__main__':
    app.run(debug=True)
