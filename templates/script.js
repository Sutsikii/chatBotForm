function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    document.getElementById("user-input").value = "";
    document.getElementById("chat").innerHTML += "<p>User: " + userInput + "</p>";
    
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/get?msg=" + encodeURIComponent(userInput), true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var botResponse = xhr.responseText;
            document.getElementById("chat").innerHTML += "<p>Bot: " + botResponse + "</p>";
        }
    };
    xhr.send();
}