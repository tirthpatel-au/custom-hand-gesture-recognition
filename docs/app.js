import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm";

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_ROOT =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const TIP_IDS = {
  thumb: 4,
  index: 8,
  middle: 12,
  ring: 16,
  pinky: 20,
};

const PIP_IDS = {
  thumb: 3,
  index: 6,
  middle: 10,
  ring: 14,
  pinky: 18,
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const mirrorToggle = document.getElementById("mirrorToggle");
const statusText = document.getElementById("statusText");
const gestureName = document.getElementById("gestureName");
const gestureMeta = document.getElementById("gestureMeta");
const permissionOverlay = document.getElementById("permissionOverlay");

let handLandmarker = null;
let animationFrameId = null;
let webcamStream = null;
let running = false;
let lastTimestamp = -1;

function setStatus(message) {
  statusText.textContent = message;
}

function updateGesturePanel(name, meta) {
  gestureName.textContent = name;
  gestureMeta.textContent = meta;
}

async function createHandLandmarker() {
  setStatus("Loading MediaPipe vision runtime...");
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  setStatus("Loading hand landmark model...");
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_ASSET_PATH,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
  setStatus("Ready. Start the camera to test gestures.");
}

function isFingerUp(landmarks, finger, handedness, mirrored) {
  if (finger === "thumb") {
    let effective = handedness;
    if (mirrored) {
      effective = handedness === "Right" ? "Left" : "Right";
    }
    if (effective === "Right") {
      return landmarks[TIP_IDS.thumb].x < landmarks[PIP_IDS.thumb].x;
    }
    return landmarks[TIP_IDS.thumb].x > landmarks[PIP_IDS.thumb].x;
  }
  return landmarks[TIP_IDS[finger]].y < landmarks[PIP_IDS[finger]].y;
}

function detectFingersUp(landmarks, handedness, mirrored) {
  return {
    thumb: isFingerUp(landmarks, "thumb", handedness, mirrored),
    index: isFingerUp(landmarks, "index", handedness, mirrored),
    middle: isFingerUp(landmarks, "middle", handedness, mirrored),
    ring: isFingerUp(landmarks, "ring", handedness, mirrored),
    pinky: isFingerUp(landmarks, "pinky", handedness, mirrored),
  };
}

function classifyGesture(landmarks, handedness, mirrored) {
  const fingersUp = detectFingersUp(landmarks, handedness, mirrored);
  const patterns = [
    ["Fist", { thumb: false, index: false, middle: false, ring: false, pinky: false }, 0.98],
    ["Open Palm", { thumb: true, index: true, middle: true, ring: true, pinky: true }, 0.98],
    ["Thumbs Up", { thumb: true, index: false, middle: false, ring: false, pinky: false }, 0.95],
    ["Peace", { thumb: false, index: true, middle: true, ring: false, pinky: false }, 0.96],
    ["Rock", { thumb: false, index: true, middle: false, ring: false, pinky: true }, 0.92],
    ["Pointing Up", { thumb: false, index: true, middle: false, ring: false, pinky: false }, 0.94],
  ];

  const thumbTip = landmarks[TIP_IDS.thumb];
  const indexTip = landmarks[TIP_IDS.index];
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const pinchDistance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  const palmScale = Math.hypot(wrist.x - middleMcp.x, wrist.y - middleMcp.y);

  if (
    palmScale > 0 &&
    pinchDistance / palmScale < 0.35 &&
    fingersUp.middle &&
    fingersUp.ring &&
    fingersUp.pinky
  ) {
    return { name: "Okay", confidence: 0.93, fingersUp };
  }

  for (const [name, pattern, confidence] of patterns) {
    const isMatch = Object.keys(pattern).every((key) => fingersUp[key] === pattern[key]);
    if (isMatch) {
      return { name, confidence, fingersUp };
    }
  }

  const count = Object.values(fingersUp).filter(Boolean).length;
  return { name: `${count} finger(s)`, confidence: count ? 0.55 : 0.5, fingersUp };
}

function drawHand(landmarks, handedness, mirrored, gesture) {
  const width = canvas.width;
  const height = canvas.height;
  const xPoints = landmarks.map((point) => point.x * width);
  const yPoints = landmarks.map((point) => point.y * height);
  const padding = 20;
  const left = Math.max(0, Math.min(...xPoints) - padding);
  const top = Math.max(0, Math.min(...yPoints) - padding);
  const right = Math.min(width, Math.max(...xPoints) + padding);
  const bottom = Math.min(height, Math.max(...yPoints) + padding);
  const labelHeight = 68;
  const labelWidth = Math.max(250, right - left);
  const labelTop = Math.max(0, top - labelHeight);

  ctx.strokeStyle = "#50d046";
  ctx.lineWidth = 3;
  ctx.strokeRect(left, top, right - left, bottom - top);

  ctx.fillStyle = "#50bb20";
  ctx.fillRect(left, labelTop, labelWidth, Math.min(labelHeight, top || labelHeight));

  ctx.fillStyle = "#ffffff";
  ctx.font = "700 18px 'Space Grotesk', sans-serif";
  ctx.fillText(`${gesture.name} (${gesture.confidence.toFixed(2)})`, left + 12, labelTop + 28);
  ctx.font = "500 16px 'Space Grotesk', sans-serif";
  ctx.fillText(`Hand: ${handedness}${mirrored ? " (mirrored)" : ""}`, left + 12, labelTop + 52);

  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    ctx.strokeStyle = "rgba(244, 244, 244, 0.95)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  }

  landmarks.forEach((point, index) => {
    const isTip = Object.values(TIP_IDS).includes(index);
    ctx.fillStyle = isTip ? "#ffd44f" : "#ff5f5f";
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, isTip ? 7 : 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function renderLoop() {
  if (!running || !handLandmarker) {
    return;
  }

  if (video.readyState >= 2) {
    resizeCanvas();
    clearCanvas();

    const timestamp = performance.now();
    if (timestamp > lastTimestamp) {
      lastTimestamp = timestamp;
      const result = handLandmarker.detectForVideo(video, timestamp);

      if (result.handLandmarks.length > 0) {
        const firstGesture = classifyGesture(
          result.handLandmarks[0],
          result.handedness[0][0].categoryName,
          mirrorToggle.checked,
        );

        updateGesturePanel(
          firstGesture.name,
          `${result.handedness[0][0].categoryName} hand | confidence ${firstGesture.confidence.toFixed(2)}`,
        );

        result.handLandmarks.forEach((landmarks, index) => {
          const handedness = result.handedness[index][0].categoryName;
          const gesture = classifyGesture(landmarks, handedness, mirrorToggle.checked);
          drawHand(landmarks, handedness, mirrorToggle.checked, gesture);
        });
      } else {
        updateGesturePanel("Waiting...", "Show one hand clearly in front of the camera.");
      }
    }
  }

  animationFrameId = window.requestAnimationFrame(renderLoop);
}

async function startCamera() {
  if (!handLandmarker) {
    await createHandLandmarker();
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
  } catch (error) {
    setStatus("Camera access was denied or is unavailable.");
    permissionOverlay.classList.remove("hidden");
    updateGesturePanel("Blocked", "Allow camera access to use the live demo.");
    return;
  }

  video.srcObject = webcamStream;
  await video.play();
  const transform = mirrorToggle.checked ? "scaleX(-1)" : "scaleX(1)";
  video.style.transform = transform;
  canvas.style.transform = transform;
  permissionOverlay.classList.add("hidden");
  running = true;
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus("Live detection is running.");
  renderLoop();
}

function stopCamera() {
  running = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }
  video.srcObject = null;
  clearCanvas();
  startButton.disabled = false;
  stopButton.disabled = true;
  permissionOverlay.classList.remove("hidden");
  setStatus("Camera stopped.");
  updateGesturePanel("Waiting...", "Restart the camera to try the demo again.");
}

mirrorToggle.addEventListener("change", () => {
  const transform = mirrorToggle.checked ? "scaleX(-1)" : "scaleX(1)";
  video.style.transform = transform;
  canvas.style.transform = transform;
});

startButton.addEventListener("click", () => {
  startCamera().catch((error) => {
    console.error(error);
    setStatus("The browser demo failed to start.");
    updateGesturePanel("Error", "Refresh the page or check the browser console.");
  });
});

stopButton.addEventListener("click", stopCamera);
window.addEventListener("resize", resizeCanvas);

createHandLandmarker().catch((error) => {
  console.error(error);
  setStatus("Failed to load MediaPipe assets.");
  updateGesturePanel("Load error", "Refresh the page or check your network connection.");
});
