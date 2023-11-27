import json
from flask import Flask, render_template, request, jsonify
from llamaapi import LlamaAPI

app = Flask(__name__)

llama = LlamaAPI("LL-KhzmmTe3QZLdq9V2Hx0AoFARaVgB5SKvz4fH4BvaoZaYjaSXaQWydVLGb9r3hgoG")

# Utiliser une liste pour stocker le contexte de la conversation
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
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "days": {
                            "type": "number",
                            "description": "for how many days ahead you wants the forecast",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                },
                "required": ["location", "days"],
            }
        ],
        "stream": False,
        "function_call": "get_current_weather",
    }

    response = llama.run(api_request_json)
    return response

def process_llama2_response(llama2_response):
    data = llama2_response.json()
    if 'choices' in data and data['choices']:
        return data['choices'][0]['message']['content']
    else:
        return "Je ne peux pas fournir d'information pour le moment. Veuillez réessayer plus tard."

if __name__ == '__main__':
    app.run(debug=True)
