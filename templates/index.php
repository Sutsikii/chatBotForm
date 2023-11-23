<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <title>Chat Bot</title>
</head>

<body>
    <div id="container">
        <h1>Chat Bot Form</h1>
        <div id="chat-container">
            <div id="chat"></div>
            <div id="bar">
            <input type="text" class="bar" id="user-input" placeholder="Je sais faire un formulaire php..." onkeydown="handleKeyDown(event)">
            <button class="bar" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='script.js') }}"></script>

</body>
</html>