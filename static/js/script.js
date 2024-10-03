const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const activateCameraBtn = document.getElementById('activate-camera-btn');
let stream = null;

// Activate camera on button click
activateCameraBtn.addEventListener('click', function() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(s) {
            stream = s; // Store the stream to stop later
            video.srcObject = stream;
            video.style.display = 'block';
            captureBtn.style.display = 'block';  // Show the capture button once camera is activated
        }).catch(function(error) {
            document.getElementById('error-message').textContent = 'Unable to access camera: ' + error.message;
        });
    } else {
        document.getElementById('error-message').textContent = 'Camera not supported by this browser.';
    }
});

// Capture image from camera
captureBtn.addEventListener('click', function() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Convert to Blob and send to backend
    canvas.toBlob(function(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'camera-image.jpg');

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('error-message').textContent = '';
            if (data.error) {
                document.getElementById('error-message').textContent = data.error;
            } else {
                document.getElementById('result-message').textContent = 'Predicted Diseases: ' + data.predicted_class;
            }

            // Deactivate the camera after capturing
            video.style.display = 'none';
            captureBtn.style.display = 'none'; // Hide capture button
            if (stream) {
                let tracks = stream.getTracks();
                tracks.forEach(track => track.stop()); // Stop the camera
            }
        })
        .catch(error => {
            document.getElementById('error-message').textContent = 'An error occurred: ' + error.message;
        });
    }, 'image/jpeg');
});

// File upload
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('error-message').textContent = '';
        if (data.error) {
            document.getElementById('error-message').textContent = data.error;
        } else {
            document.getElementById('result-message').textContent = 'Predicted Class: ' + data.predicted_class;
        }
    })
    .catch(error => {
        document.getElementById('error-message').textContent = 'An error occurred: ' + error.message;
    });
});
