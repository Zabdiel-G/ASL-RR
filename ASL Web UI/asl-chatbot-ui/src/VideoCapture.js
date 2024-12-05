import React, { useRef, useEffect } from 'react';
import './VideoCapture.css'

function VideoCapture({ onASLRecognized }) {
  const videoRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
      })
      .catch((err) => console.error('Error accessing webcam:', err));
  }, []);

  const handleASLRecognition = () => {
    // Placeholder function for ASL recognition
    const recognizedASL = 'Hello'; // Mocked ASL recognition result
    onASLRecognized(recognizedASL);
  };

  return (
    <div>
      <video ref={videoRef} autoPlay width="640" height="480"></video>
      <button class="button" onClick={handleASLRecognition}>Capture ASL</button>
    </div>
  );
}

export default VideoCapture;

