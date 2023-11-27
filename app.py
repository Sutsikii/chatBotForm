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
                            "description": "for how many days ahead you want the forecast",
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
    print(f"Llama2 Response: {response.text}")  # Ajout de ce print statement
    return response


def process_llama2_response(llama2_response):
    try:
        data = llama2_response.json()
        if 'choices' in data and data['choices']:
            # Check if the user asked a question related to PHP
            if any(phrase in data['choices'][0]['message']['content'].lower() for phrase in ['php', 'programming language', 'web development']):
                # If the user asked a question related to PHP, use the LlamaAPI to retrieve information from an external API
                api_request_json = {
                    "messages": [
                        {"role": "user", "content": "What is PHP?"},
                    ],
                    "functions": [
                        {
                            "name": "get_information",
                            "description": "Get information about a topic",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "The topic to get information about"
                                    },
                                    "source": {
                                        "type": "string",
                                        "description": "The source of the information"
                                    }
                                },
                                "required": ["topic", "source"]
                            }
                        }
                    ],
                    "stream": False,
                    "function_call": "get_information"
                }
                response = llama.run(api_request_json)
                if response.ok:
                    information = response.json()["data"]["results"][0]["information"]
                    return information
            else:
                return data['choices'][0]['message']['content']
        else:
            return "Je ne peux pas fournir d'information pour le moment. Veuillez réessayer plus tard."
    except Exception as e:
        print(f"Error processing Llama2 response: {e}")
        return "An error occurred while processing the response."

if __name__ == '__main__':
    app.run(debug=True)
