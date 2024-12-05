import React, { useEffect, useState } from 'react';

const ChatResponse = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    // Fetching data from your backend API
    fetch('http://localhost:5000/api/asl')
      .then(response => response.json())
      .then(data => {
        console.log(data); // Log the data for debugging
        setData(data);     // Set the data to state
      })
      .catch(error => {
        console.error('Error fetching the API:', error);
      });
  }, []); // Empty dependency array to run only once on mount

  return (
    <div>
      <h1>ASL Chatbot Response</h1>
      {data ? <p>{data.message}</p> : <p></p>}
    </div>
  );
};

export default ChatResponse;

