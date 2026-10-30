# FSL Sign Lab — Moving Gesture Prototype

A browser-based educational prototype for collecting and classifying short moving hand gestures. It uses MediaPipe Hands to extract hand landmarks and TensorFlow.js to train a multi-class sequence classifier in the browser.

## What it does
- Uses a webcam to track up to two hands.
- Lets each user create custom word or phrase labels.
- Records each training example as 30 consecutive frames (126 landmark values per frame, with zero values when no hand is detected).
- Trains a neural network on the ordered sequence of frames rather than a single static pose.
- Predicts from a rolling 30-frame window while the camera is running.
- Stores labels and training sequences in that browser's local storage. No training data is uploaded by this app.

## Run locally
1. Extract the ZIP.
2. Open the folder in VS Code.
3. Use the Live Server extension to open `index.html` (or host it on HTTPS).
4. Allow camera access.
5. Add at least two labels.
6. For each label, click **Record full movement**, wait for the countdown, and perform the entire sign while keeping your hands visible.
7. Record multiple examples per label; 15–30 or more varied examples per label is a better starting point than one or two.
8. Click **Train model**, then perform the full signs in front of the camera.

## GitHub Pages
Upload `index.html`, `style.css`, `app.js`, and `README.md` to the root of a GitHub repository. Open **Settings → Pages**, choose deployment from the `main` branch and `/(root)`, then save. Wait for deployment and open the published HTTPS URL. Camera access generally requires HTTPS or localhost.

## Limitations
- This is a prototype, not a production-grade FSL translator. The sequence model is a simple dense classifier over 30 frames, not a purpose-built LSTM/GRU model. It can learn movement patterns in the captured window, but results depend heavily on the quality, consistency, and quantity of examples.
- Signs longer or shorter than the captured window may be poorly recognized. The recording currently uses a fixed 30-frame window.
- Landmarks are raw normalized coordinates; differences in camera placement, body position, hand visibility, and signer can affect performance.
- A high training score does not prove real-world accuracy. Evaluate with new recordings not used in training.
- Data is saved locally in each browser. Classmates must collect their own examples; data does not sync between devices.
- This is not a certified interpreter and should not be relied on for emergencies.