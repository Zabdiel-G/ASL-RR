// Toggle detection state on button click
document.getElementById('toggleButton').addEventListener('click', () => {
  fetch('/toggle_detection', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      console.log("Detection toggled:", data);
      document.getElementById('stateText').innerText = data.is_detecting ? "Capturing" : "Idle";
    })
    .catch(error => console.error('Error toggling detection:', error));
});

// Reset gesture on button click
document.getElementById('resetButton').addEventListener('click', () => {
  fetch('/reset_gesture', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      console.log("Gesture reset:", data);
      updateGestureSentence(data.gesture_sentence);
    })
    .catch(error => console.error('Error resetting gesture:', error));
});

// Adjust the font size to make the text fit inside the container
function adjustFontSize() {
  const container = document.getElementById('gestureSentenceText');
  let fontSize = 24; // initial font size in pixels
  container.style.fontSize = fontSize + 'px';
  // Reduce the font size until the text fits inside the container or until a minimum font size is reached
  while (container.scrollWidth > container.clientWidth && fontSize > 12) {
    fontSize -= 1;
    container.style.fontSize = fontSize + 'px';
  }
}

// Update the gesture sentence and adjust the font size
function updateGestureSentence(text) {
  const container = document.getElementById('gestureSentenceText');
  container.innerText = text || "No gestures detected";
  adjustFontSize();
}

// Poll the gesture sentence endpoint every second to update the displayed sentence
setInterval(() => {
  fetch('/gesture_sentence')
    .then(response => response.json())
    .then(data => {
      updateGestureSentence(data.gesture_sentence);
    })
    .catch(error => console.error('Error fetching gesture sentence:', error));
}, 1000);

// Chatbot button event: use the current gesture sentence as input for the chatbot
document.getElementById('chatButton').addEventListener('click', () => {
  console.log("Chat button clicked");
  fetch('/chat_from_gesture')
    .then(response => response.json())
    .then(data => {
      const output = data.asl_response;
      const chatOutput = document.getElementById('chatOutput');
      chatOutput.textContent = output;
      // Auto-scroll to the bottom if needed
      chatOutput.scrollTop = chatOutput.scrollHeight;
    })
    .catch(error => console.error('Error fetching chatbot response:', error));
});