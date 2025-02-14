document.addEventListener("DOMContentLoaded", function() {
    // Update the gesture sentence every second
    setInterval(function() {
        fetch('/gesture_sentence')
            .then(response => response.json())
            .then(data => {
                document.getElementById('gesture_sentence').textContent = data.gesture_sentence;
            });
    }, 1000);  // Update every second

    // Update state, remaining time, and word count every second
    setInterval(function() {
        fetch('/state_info')
            .then(response => response.json())
            .then(data => {
                document.getElementById('state_text').textContent = data.state_text;
                document.getElementById('remaining_time').textContent = `Remaining Time: ${data.remaining_time}s`;
                document.getElementById('word_count').textContent = data.word_count_text;
            });
    }, 1000);  // Update every second
});
