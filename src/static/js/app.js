// Debug logging to verify that app.js is loaded
console.log("app.js loaded");

document.addEventListener("DOMContentLoaded", function() {
    // Poll the gesture sentence endpoint every second
    setInterval(function() {
        fetch('/gesture_sentence')
            .then(response => response.json())
            .then(data => {
                document.getElementById('gesture_sentence').textContent = data.gesture_sentence;
            })
            .catch(error => console.error('Error fetching gesture sentence:', error));
    }, 1000);

    // Poll the state info endpoint every second
    setInterval(function() {
        fetch('/state_info')
            .then(response => response.json())
            .then(data => {
                document.getElementById('state_text').textContent = data.state_text;
                document.getElementById('remaining_time').textContent = `Remaining Time: ${data.remaining_time}s`;
                document.getElementById('word_count').textContent = data.word_count_text;
            })
            .catch(error => console.error('Error fetching state info:', error));
    }, 1000);

    document.getElementById("toggleButton").addEventListener("click", function () {
        fetch("/toggle_detection", { method: "POST" })
            .then(response => response.json())
            .then(data => console.log("Detection toggled:", data))
            .catch(error => console.error("Error:", error));
    });
    
    document.getElementById("resetButton").addEventListener("click", function () {
        fetch("/reset_gesture", { method: "POST" })
            .then(response => response.json())
            .then(data => console.log("Gesture reset:", data))
            .catch(error => console.error("Error:", error));
    });
    

    // Chatbot button event: use the current gesture sentence as input for the chatbot
    document.getElementById('chatButton').addEventListener('click', () => {
        console.log("Chat button clicked");
        fetch('/chat_from_gesture')
            .then(response => response.json())
            .then(data => {
                const output = "ASL Input: " + data.gesture_sentence + "\n" +
                               "English Interpretation: " + data.english_interpretation + "\n" +
                               "Chat Response: " + data.english_response + "\n" +
                               "ASL Response: " + data.asl_response;
                const chatOutput = document.getElementById('chatOutput');
                chatOutput.textContent = output;
                // Auto-scroll to the bottom if needed
                chatOutput.scrollTop = chatOutput.scrollHeight;
            })
            .catch(error => console.error('Error fetching chatbot response:', error));
    });
});
