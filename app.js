const video = document.getElementById("input_video");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const cameraStatus = document.getElementById("cameraStatus");
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const labelForm = document.getElementById("labelForm");
const wordInput = document.getElementById("wordInput");
const classSelect = document.getElementById("classSelect");
const labelsEl = document.getElementById("labels");
const recordButton = document.getElementById("recordButton");
const deleteClassButton = document.getElementById("deleteClassButton");
const recordStatus = document.getElementById("recordStatus");
const sampleSummary = document.getElementById("sampleSummary");
const trainButton = document.getElementById("trainButton");
const resetButton = document.getElementById("resetButton");
const trainingStatus = document.getElementById("trainingStatus");

const FRAME_COUNT = 30;
const FRAME_SIZE = 126;
const SEQUENCE_SIZE = FRAME_COUNT * FRAME_SIZE;
const DATA_KEY = "fsl_sign_lab_sequences_v2";
const LABEL_KEY = "fsl_sign_lab_labels_v2";
let labels = JSON.parse(localStorage.getItem(LABEL_KEY) || "[]");
let samples = JSON.parse(localStorage.getItem(DATA_KEY) || "[]");
let latestHands = [];
let model = null;
let cameraReady = false;
let recording = false;
let capturing = false;
let predicting = false;
let recordingFrames = [];
let recordingDetectedFrames = 0;
let motionBuffer = [];
let lastPredictionAt = 0;
let lastWindowKey = "";

function saveLocalData() {
  localStorage.setItem(LABEL_KEY, JSON.stringify(labels));
  localStorage.setItem(DATA_KEY, JSON.stringify(samples));
}
function prettyLabel(label) { return label.replaceAll("_", " "); }
function zeroFrame() { return Array(FRAME_SIZE).fill(0); }
function updateUI() {
  labelsEl.innerHTML = "";
  if (!labels.length) labelsEl.innerHTML = '<span class="muted">No labels added yet.</span>';
  else labels.forEach(label => {
    const chip = document.createElement("span");
    chip.className = "label-chip";
    chip.textContent = prettyLabel(label);
    labelsEl.appendChild(chip);
  });
  const oldValue = classSelect.value;
  classSelect.innerHTML = "";
  labels.forEach(label => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = prettyLabel(label);
    classSelect.appendChild(option);
  });
  if (labels.includes(oldValue)) classSelect.value = oldValue;
  classSelect.disabled = !labels.length || !cameraReady || recording;
  recordButton.disabled = !labels.length || !cameraReady || recording;
  deleteClassButton.disabled = !labels.length || recording;
  const counts = labels.map(label => ({ label, count: samples.filter(sample => sample.label === label).length }));
  sampleSummary.innerHTML = "";
  counts.forEach(item => {
    const row = document.createElement("div");
    row.className = "summary-row";
    const name = document.createElement("span"); name.textContent = prettyLabel(item.label);
    const count = document.createElement("strong"); count.textContent = `${item.count} sequence${item.count === 1 ? "" : "s"}`;
    row.append(name, count); sampleSummary.appendChild(row);
  });
  const enough = labels.length >= 2 && labels.every(label => samples.some(sample => sample.label === label));
  trainButton.disabled = !enough || recording;
  if (!enough) trainingStatus.textContent = "Add at least two labels and record at least one complete moving sequence for each. Aim for 15–30 sequences per label for better results.";
}
function normalizeLabel(raw) { return raw.trim().toUpperCase().replace(/\s+/g, "_").replace(/[^A-Z0-9_'-]/g, ""); }
labelForm.addEventListener("submit", event => {
  event.preventDefault();
  const label = normalizeLabel(wordInput.value);
  if (!label) return;
  if (labels.includes(label)) {
    recordStatus.textContent = "That word already exists. Select it from the list.";
    classSelect.value = label; return;
  }
  labels.push(label); saveLocalData(); wordInput.value = "";
  model?.dispose(); model = null; motionBuffer = [];
  predictionEl.textContent = "Model needs retraining";
  confidenceEl.textContent = "You added a new label.";
  updateUI(); classSelect.value = label;
});
classSelect.addEventListener("change", updateUI);

function getHandData(handsDetected) {
  if (!handsDetected || handsDetected.length === 0) return null;
  const values = [];
  for (let i = 0; i < 2; i++) {
    if (handsDetected[i]) {
      for (const point of handsDetected[i]) values.push(point.x, point.y, point.z);
    } else {
      for (let j = 0; j < 63; j++) values.push(0);
    }
  }
  return values;
}
function drawResults(results) {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.save(); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  latestHands = results.multiHandLandmarks || [];
  for (const hand of latestHands) {
    drawConnectors(ctx, hand, HAND_CONNECTIONS, { color: "#8be0c1", lineWidth: 4 });
    drawLandmarks(ctx, hand, { color: "#ff6b6b", lineWidth: 2, radius: 3 });
  }
  ctx.restore();

  const frame = getHandData(latestHands);
  const currentFrame = frame || zeroFrame();
  if (recording && capturing) {
    recordingFrames.push(currentFrame);
    if (frame) recordingDetectedFrames++;
    recordStatus.textContent = `Recording movement… ${Math.min(recordingFrames.length, FRAME_COUNT)}/${FRAME_COUNT} frames`;
    if (recordingFrames.length >= FRAME_COUNT) finishRecording();
  }
  motionBuffer.push(currentFrame);
  if (motionBuffer.length > FRAME_COUNT) motionBuffer.shift();
  const now = Date.now();
  if (model && motionBuffer.length === FRAME_COUNT && !recording && !predicting && now - lastPredictionAt > 350) {
    lastPredictionAt = now;
    const key = motionBuffer[FRAME_COUNT - 1].slice(0, 12).join(",");
    if (key !== lastWindowKey) {
      lastWindowKey = key;
      runPrediction(motionBuffer.flat());
    }
  } else if (!latestHands.length && model && !recording && !predicting) {
    predictionEl.textContent = "Waiting for a moving sign…";
    confidenceEl.textContent = "Perform a complete gesture in the camera view.";
  }
}
async function runPrediction(sequenceData) {
  if (!model || !sequenceData || predicting) return;
  predicting = true;
  try {
    const result = tf.tidy(() => model.predict(tf.tensor2d([sequenceData], [1, SEQUENCE_SIZE])));
    const scores = await result.data(); result.dispose();
    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    predictionEl.textContent = prettyLabel(labels[best] || "UNKNOWN");
    confidenceEl.textContent = `Model score: ${(scores[best] * 100).toFixed(1)}% · Estimate only; this is not verified accuracy.`;
  } catch (error) {
    console.error(error); confidenceEl.textContent = "Prediction error. Please train the sequence model again.";
  } finally { predicting = false; }
}
function finishRecording() {
  if (!recording) return;
  recording = false; capturing = false;
  if (recordingDetectedFrames < 12) {
    recordStatus.textContent = "Too few frames contained a detected hand. Keep your hand visible and record the full movement again.";
  } else {
    const sequence = recordingFrames.slice(0, FRAME_COUNT).flat();
    samples.push({ label: classSelect.value, sequence });
    saveLocalData(); model?.dispose(); model = null; motionBuffer = [];
    predictionEl.textContent = "Model needs retraining";
    confidenceEl.textContent = "A full movement sequence was saved.";
    recordStatus.textContent = `Saved moving sequence for ${prettyLabel(classSelect.value)}.`;
  }
  recordingFrames = []; recordingDetectedFrames = 0;
  updateUI();
}
recordButton.addEventListener("click", async () => {
  if (recording || !cameraReady) return;
  recording = true; capturing = false; recordingFrames = []; recordingDetectedFrames = 0;
  updateUI();
  try {
    for (let count = 3; count > 0; count--) {
      recordStatus.textContent = `Get ready to perform the complete sign… ${count}`;
      await new Promise(resolve => setTimeout(resolve, 700));
    }
    capturing = true;
    recordStatus.textContent = `Perform the full sign now. Capturing ${FRAME_COUNT} frames…`;
  } catch (error) {
    recording = false; capturing = false; recordingFrames = []; updateUI();
  }
});
deleteClassButton.addEventListener("click", () => {
  const selected = classSelect.value; if (!selected) return;
  if (!confirm(`Delete ${prettyLabel(selected)} and all its recorded sequences from this device?`)) return;
  labels = labels.filter(label => label !== selected);
  samples = samples.filter(sample => sample.label !== selected);
  model?.dispose(); model = null; saveLocalData(); motionBuffer = [];
  predictionEl.textContent = "Model needs retraining"; confidenceEl.textContent = ""; updateUI();
});
trainButton.addEventListener("click", async () => {
  const validSamples = samples.filter(sample => labels.includes(sample.label) && sample.sequence?.length === SEQUENCE_SIZE);
  const counts = labels.map(label => validSamples.filter(sample => sample.label === label).length);
  if (labels.length < 2 || counts.some(count => count < 1)) {
    trainingStatus.textContent = "Record at least one complete movement sequence for every one of at least two labels."; return;
  }
  trainButton.disabled = true; recordButton.disabled = true; trainingStatus.textContent = "Preparing movement sequences…";
  let xs, ys;
  try {
    const xValues = validSamples.map(sample => sample.sequence);
    const yValues = validSamples.map(sample => labels.indexOf(sample.label));
    xs = tf.tensor2d(xValues, [xValues.length, SEQUENCE_SIZE]);
    const indices = tf.tensor1d(yValues, "int32"); ys = tf.oneHot(indices, labels.length); indices.dispose();
    model?.dispose(); model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [SEQUENCE_SIZE], units: 128, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: labels.length, activation: "softmax" }));
    model.compile({ optimizer: tf.train.adam(0.0005), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
    trainingStatus.textContent = "Training the sequence classifier…";
    await model.fit(xs, ys, {
      epochs: 70, shuffle: true,
      validationSplit: validSamples.length >= labels.length * 5 ? 0.2 : 0,
      callbacks: { onEpochEnd: (epoch, logs) => {
        const accuracy = logs.acc ?? logs.accuracy ?? 0;
        trainingStatus.textContent = `Epoch ${epoch + 1}/70 · training accuracy ${(accuracy * 100).toFixed(1)}%`;
      }}
    });
    motionBuffer = [];
    trainingStatus.textContent = "Sequence training finished on this device. Test by performing each full sign several times. Training score is not the same as test accuracy.";
    predictionEl.textContent = "Sequence model ready";
    confidenceEl.textContent = "Perform a complete moving gesture; the model evaluates a rolling window of frames.";
  } catch (error) {
    console.error(error); trainingStatus.textContent = `Training failed: ${error.message}`;
  } finally { xs?.dispose(); ys?.dispose(); updateUI(); }
});
resetButton.addEventListener("click", () => {
  if (!confirm("Clear all labels and movement sequences saved by this demo on this browser?")) return;
  labels = []; samples = []; model?.dispose(); model = null; motionBuffer = []; saveLocalData();
  predictionEl.textContent = "Not trained yet"; confidenceEl.textContent = "Train a model after recording examples.";
  recordStatus.textContent = "Local data cleared."; trainingStatus.textContent = "Add at least two labels and record complete moving sequences for each."; updateUI();
});
const hands = new Hands({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
hands.onResults(drawResults);
const camera = new Camera(video, { onFrame: async () => { await hands.send({ image: video }); }, width: 640, height: 480 });
camera.start().then(() => {
  cameraReady = true; cameraStatus.textContent = "Camera ready";
  statusEl.textContent = "Hand tracking is ready. Allow camera permission if prompted."; updateUI();
}).catch(error => {
  console.error(error); cameraStatus.textContent = "Camera unavailable";
  statusEl.textContent = "Camera could not start. Open this site over HTTPS, allow camera access, and close other apps using the camera.";
});
updateUI();