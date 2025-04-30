function appendUserMessage(message) {
  const chatOutput = document.getElementById('chatOutput');
  const div = document.createElement('div');
  div.className = 'chat-message user';
  div.innerHTML = `
    <span class="label">You:</span><br>
    &gt; ${message}`;
  chatOutput.appendChild(div);
  chatOutput.scrollTop = chatOutput.scrollHeight;
}

function appendBotMessage(aslGloss, englishText) {
  const chatOutput = document.getElementById('chatOutput');
  const div = document.createElement('div');
  div.className = 'chat-message bot';
  div.innerHTML = `
    <span class="label">ChatBot:</span><br>
    &gt; Gloss: ${aslGloss}<br>
    &gt; English: ${englishText}`;
  chatOutput.appendChild(div);
  chatOutput.scrollTop = chatOutput.scrollHeight;
}

document.getElementById('toggleButton').addEventListener('click', () => {
  const btn = document.getElementById('toggleButton');
  btn.disabled = true;
  btn.style.backgroundColor = '#ccc'; // processing state
  fetch('/toggle_detection', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      document.getElementById('stateText').innerText = data.is_detecting ? "Capturing" : "Idle";
      btn.style.backgroundColor = data.is_detecting ? '#4CAF50' : '#f44336';
    })
    .catch(error => {
      console.error('Error toggling detection:', error);
      btn.style.backgroundColor = '#f44336';
    })
    .finally(() => {
      // Re-enable button after a short delay
      setTimeout(() => btn.disabled = false, 1000);
    });
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

// For User Input Translation 
document.getElementById('translateButton').addEventListener('click', () => {
  const englishSentence = document.getElementById('englishInput').value;
  if (!englishSentence) {
    alert("Please type a sentence.");
    return;
  }
  const btn = document.getElementById('translateButton');
  btn.disabled = true;
  btn.style.backgroundColor = '#ccc'; // Show a processing state

  fetch('/translate_english', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ english_sentence: englishSentence })
  })
  .then(response => response.json())
  .then(data => {
    const userInput = englishSentence;
    const aslResponse = data.asl_response;
    const englishResponse = data.english_response;
  
    // Update individual panels
    // document.getElementById('aslGlossText').innerText = aslResponse;
    // document.getElementById('englishResponseText').innerText = englishResponse;
  
    // Update chat log
    appendUserMessage(userInput);
    appendBotMessage(aslResponse, englishResponse);
  })
  .catch(error => console.error("Error translating sentence:", error))
  .finally(() => {
    btn.disabled = false;
    btn.style.backgroundColor = '#4CAF50';
  });
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
  container.innerText = text;
  adjustFontSize();
}


setInterval(() => {
  fetch('/gesture_sentence')
    .then(response => response.json())
    .then(data => {
      updateGestureSentence(data.gesture_sentence);
    })
    .catch(error => console.error('Error fetching gesture sentence:', error));
}, 1000);


document.getElementById('chatButton').addEventListener('click', () => {
  console.log("Chat button clicked");
  fetch('/chat_from_gesture')
    .then(response => response.json())
    .then(data => {
      const aslGloss = data.asl_response;
      const english = data.english_response;
    
      // document.getElementById('aslGlossText').innerText = aslGloss;
      // document.getElementById('englishResponseText').innerText = english;
    
      appendUserMessage(data.gesture_sentence || "...");
      appendBotMessage(aslGloss, english);
    })
    .catch(error => console.error('Error fetching chatbot response:', error));
});


