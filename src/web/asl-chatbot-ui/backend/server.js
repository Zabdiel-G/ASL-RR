const { spawn } = require('child_process');
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());

// Route to trigger ASL recognition
app.get('/api/asl', (req, res) => {
  const pythonProcess = spawn('python', ['./path/to/asl_recognition.py']);

  pythonProcess.stdout.on('data', (data) => {
    res.json({ message: data.toString() });
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Error: ${data}`);
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

