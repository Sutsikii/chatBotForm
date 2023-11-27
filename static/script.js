function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    document.getElementById("user-input").value = "";
    document.getElementById("chat").innerHTML += "<p>Vous : " + userInput + "</p>";

    var chatDiv = document.getElementById("chat");
    
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/get?msg=" + encodeURIComponent(userInput), true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var botResponse = xhr.responseText;
            var botResponseObj = JSON.parse(botResponse);
            document.getElementById("chat").innerHTML += "<p>Boty : " + botResponseObj.bot_response + "</p>";
            chatDiv.scrollTop = chatDiv.scrollHeight;
        }
    };
    xhr.send();
}

function handleKeyDown(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}
