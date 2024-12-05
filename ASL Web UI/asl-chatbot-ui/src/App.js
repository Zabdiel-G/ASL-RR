import './App.css';
import React, { useState } from 'react';
import VideoCapture from './VideoCapture';
import ChatResponse from './ChatResponse';

function App() {
  const [response, setResponse] = useState('');

  const handleASLResponse = (aslText) => {
    // Send the recognized ASL text to the backend and get the chatbot's response
    fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ aslText }),
    })
      .then((res) => res.json())
      .then((data) => setResponse(data.response))
      .catch((err) => console.error(err));
  };

  return (
    <div className="App">
      <h1>User ASL Input</h1>
      <VideoCapture onASLRecognized={handleASLResponse} />
      <ChatResponse response={response} />
    </div>
  );
}

export default App;
