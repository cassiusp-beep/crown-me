// Crown Me: finds every face in a photo, checks which ones are Cassius with a
// Teachable Machine image model, and draws a crown on his head only.
// tmImage and tmPose are globals from the classic scripts in index.html.
import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

// ---------- Settings ----------
const IMAGE_MODEL_URL = "https://teachablemachine.withgoogle.com/models/q5b7GQQSp/";
const POSE_MODEL_URL = "https://teachablemachine.withgoogle.com/models/oMQMyNhPT/";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CROWN_THRESHOLD = 0.8; // P(Cassius) needed for a crown (raised from 0.6: see README "Crop fidelity")
const POSE_THRESHOLD = 0.6; // arms_up confidence needed for sparkles
const WRIST_MIN_SCORE = 0.3; // at least one wrist must be this visible before the pose is trusted
const CROP_PADDING = 0.4; // extra margin around the face crop (40% matched the training photos better than 15%)
const CROP_SIZE = 224; // Teachable Machine image models take 224x224 input
const MAX_EDGE = 1600; // big phone photos are scaled down to this long edge for speed

// Webcam timing: landmarks run every frame, but the classifiers are slower, so they run less often
const CLASSIFY_EVERY_MS = 400;
const POSE_EVERY_MS = 1000;
const SMOOTHING = 0.5; // how much each new webcam prediction counts vs. the running average
const CROWN_OFF_THRESHOLD = CROWN_THRESHOLD - 0.1; // a crown stays on until P(Cassius) drops below this

// Landmark indices on MediaPipe's 478-point face mesh
const LM = { forehead: 10, rightEye: 33, leftEye: 263, rightSide: 234, leftSide: 454 };

const CROWN_COLORS = {
  gold: { light: "#fff1a8", mid: "#f4c430", dark: "#b8860b", outline: "#6e5100" },
  green: { light: "#b6f5cc", mid: "#2ecc71", dark: "#1e8449", outline: "#0f4d2a" },
};

// "Under the hood" view: crop boxes, key landmarks, and the exact crops the model sees.
// Toggle it with the button, or start with it on by adding ?debug=1 to the URL.
let debug = new URLSearchParams(location.search).has("debug");

// ---------- Page elements ----------
const $ = (id) => document.getElementById(id);
const els = {
  loading: $("loading"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  status: $("status"),
  stage: $("stage"),
  canvas: $("result"),
  downloadBtn: $("download-btn"),
  debugBtn: $("debug-btn"),
  debugLegend: $("debug-legend"),
  results: $("results"),
  faceList: $("face-list"),
  poseResult: $("pose-result"),
  tabUpload: $("tab-upload"),
  tabWebcam: $("tab-webcam"),
  uploadPanel: $("upload-panel"),
  webcamPanel: $("webcam-panel"),
  cameraBtn: $("camera-btn"),
  captureBtn: $("capture-btn"),
  video: $("video"),
};
const ctx = els.canvas.getContext("2d");

let faceLandmarker, imageModel, poseModel;
let runId = 0; // lets a newer upload cancel an older one that's still running
let lastPhoto = null; // the latest photo result, so the "Under the hood" toggle can redraw it
let landmarkerMode = "IMAGE"; // FaceLandmarker needs "IMAGE" for photos and "VIDEO" for the webcam

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("is-error", isError);
}

// ---------- Step 0: Load all three models ----------
// They load in parallel. Each checklist item flips to ✓ as soon as its model is ready.
function track(name, promise) {
  const item = els.loading.querySelector(`[data-model="${name}"]`);
  return promise.then(
    (model) => {
      item.dataset.state = "ready";
      return model;
    },
    (err) => {
      item.dataset.state = "failed";
      throw err;
    }
  );
}

async function createFaceLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
    runningMode: "IMAGE",
    numFaces: 5,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    // Some phones and older laptops can't use the GPU path, so fall back to CPU
    console.warn("GPU face landmarker failed, using CPU", err);
    return await FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

async function loadModels() {
  try {
    [faceLandmarker, imageModel, poseModel] = await Promise.all([
      track("face", createFaceLandmarker()),
      track("image", tmImage.load(IMAGE_MODEL_URL + "model.json", IMAGE_MODEL_URL + "metadata.json")),
      track("pose", tmPose.load(POSE_MODEL_URL + "model.json", POSE_MODEL_URL + "metadata.json")),
    ]);
    els.loading.hidden = true;
    els.fileInput.disabled = false;
    els.dropzone.classList.remove("is-disabled");
    els.cameraBtn.disabled = false;
    setStatus("Ready. Choose a photo to get started.");
  } catch (err) {
    console.error(err);
    els.loading.querySelector(".loading-title").textContent = "Couldn't load the models.";
    setStatus("Check your internet connection and reload the page.", true);
  }
}

// ---------- Step 1: Upload the photo and draw it on the canvas ----------
function readImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("decode failed"));
    };
    img.src = url;
  });
}

// Keeps the photo's natural aspect ratio but caps the long edge at MAX_EDGE.
// CSS then shrinks the canvas to fit the page, while downloads keep full resolution.
function imageToCanvas(img) {
  const scale = Math.min(1, MAX_EDGE / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function handleFile(file) {
  if (!file) return;
  const myRun = ++runId;
  els.downloadBtn.disabled = true;

  let img;
  try {
    img = await readImage(file);
  } catch {
    setStatus("Couldn't read that file. Try a JPG or PNG (iPhone HEIC photos may need converting).", true);
    return;
  }

  // A clean copy of the photo. Crops always come from here, never from the canvas with crowns on it.
  const source = imageToCanvas(img);
  els.canvas.width = source.width;
  els.canvas.height = source.height;
  ctx.drawImage(source, 0, 0);
  els.stage.hidden = false;
  setStatus("Looking for faces…");

  // Let the browser paint the photo before the models run
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  if (myRun !== runId) return;

  try {
    await useLandmarkerMode("IMAGE");
    if (myRun !== runId) return;
    const faces = detectFacesTiled(source);
    const classified = [];
    for (const face of faces) classified.push(await classifyFace(source, face));
    const pose = await classifyPose(source);
    if (myRun !== runId) return;

    for (const face of classified) face.crown = crownColor(face.probs);
    renderScene(source, classified, pose);
    showResults(classified, pose);
    lastPhoto = { source, faces: classified, pose };
    els.downloadBtn.disabled = false;

    if (faces.length === 0) {
      setStatus("No faces found. Try a clearer, front-facing photo.");
    } else {
      const crowned = classified.filter((f) => f.crown).length;
      setStatus(crowned ? "Long live the king. 👑" : "No Cassius in this one, so no crown.");
    }
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong while analyzing that photo. Try another one.", true);
  }
}

// One FaceLandmarker serves both modes; switch its running mode only when it changes
async function useLandmarkerMode(mode) {
  if (landmarkerMode === mode) return;
  await faceLandmarker.setOptions({ runningMode: mode });
  landmarkerMode = mode;
}

// ---------- Step 2: Find every face with MediaPipe FaceLandmarker ----------
// Returns each face's 478 landmarks converted from 0–1 coordinates to pixels.
function detectFaces(source, timestamp) {
  const result =
    timestamp === undefined ? faceLandmarker.detect(source) : faceLandmarker.detectForVideo(source, timestamp);
  const w = source.width || source.videoWidth;
  const h = source.height || source.videoHeight;
  return result.faceLandmarks.map((landmarks) => {
    const points = landmarks.map((p) => ({ x: p.x * w, y: p.y * h }));
    return { points, box: cropBox(points) };
  });
}

// The face detector shrinks the whole photo to a small square, so in wide group shots some faces
// end up too small to find. For photos we also scan four overlapping sections (each 60% of the
// width and height, so every face appears larger) and add any face the full-photo pass missed.
// Sections can produce false faces (in testing, a hand with a ring), so each new face must be
// found again when we zoom in on it before it counts.
function confirmFace(source, box) {
  const side = box.side * 2;
  const zoom = document.createElement("canvas");
  zoom.width = zoom.height = 256;
  const c = zoom.getContext("2d");
  c.fillStyle = "#000";
  c.fillRect(0, 0, 256, 256);
  c.drawImage(source, box.cx - side / 2, box.cy - side / 2, side, side, 0, 0, 256, 256);
  return faceLandmarker.detect(zoom).faceLandmarks.length > 0;
}

function detectFacesTiled(source) {
  const found = detectFaces(source);
  const tw = Math.round(source.width * 0.6);
  const th = Math.round(source.height * 0.6);
  for (const [fx, fy] of [[0, 0], [0.4, 0], [0, 0.4], [0.4, 0.4]]) {
    const x = Math.round(source.width * fx);
    const y = Math.round(source.height * fy);
    const tile = document.createElement("canvas");
    tile.width = tw;
    tile.height = th;
    tile.getContext("2d").drawImage(source, x, y, tw, th, 0, 0, tw, th);
    for (const face of detectFaces(tile)) {
      const points = face.points.map((p) => ({ x: p.x + x, y: p.y + y }));
      const box = cropBox(points);
      // Skip faces we already have (same face seen in the full photo or another section)
      const duplicate = found.some(
        (f) => Math.hypot(f.box.cx - box.cx, f.box.cy - box.cy) < Math.max(f.box.side, box.side) * 0.5
      );
      if (!duplicate && confirmFace(source, box)) found.push({ points, box });
    }
  }
  return found;
}

// ---------- Step 3: Crop a square around the face (forehead to chin, ear to ear) ----------
// The bounding box of all landmarks spans forehead to chin and ear to ear. We make it square
// and add CROP_PADDING (40%) around it. The model was trained on wider screenshots, and in testing
// 40% padding misread fewer strangers than a tight 15% crop.
function cropBox(points) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const side = Math.max(maxX - minX, maxY - minY) * (1 + CROP_PADDING);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return { x: cx - side / 2, y: cy - side / 2, side, cx, cy };
}

// Resizes the crop to 224x224 on an offscreen canvas. Any part of the square that
// falls outside the photo stays black.
function cropFace(source, box) {
  const crop = document.createElement("canvas");
  crop.width = crop.height = CROP_SIZE;
  const c = crop.getContext("2d");
  c.fillStyle = "#000";
  c.fillRect(0, 0, CROP_SIZE, CROP_SIZE);
  c.drawImage(source, box.x, box.y, box.side, box.side, 0, 0, CROP_SIZE, CROP_SIZE);
  return crop;
}

// ---------- Step 4: Ask the image model who this is ----------
// The model's saved labels are cut off ("cassius_happ..."), so we map them back to full names.
function canonicalLabel(raw) {
  if (raw.startsWith("cassius_happ")) return "cassius_happy";
  if (raw.startsWith("cassius_neut")) return "cassius_neutral";
  return raw.replace(/\.\.\.$/, "");
}

async function predictCrop(crop) {
  const predictions = await imageModel.predict(crop);
  const probs = { cassius_happy: 0, cassius_neutral: 0, not_cassius: 0 };
  for (const p of predictions) probs[canonicalLabel(p.className)] = p.probability;
  return probs;
}

async function classifyFace(source, face) {
  const crop = cropFace(source, face.box);
  return { ...face, crop, probs: await predictCrop(crop) };
}

function topClass(probs) {
  return Object.entries(probs).reduce((best, cur) => (cur[1] > best[1] ? cur : best));
}

// Crown rule: add the two Cassius classes together to decide whether it's Cassius at all,
// then the bigger of the two picks the color. (A face scored 45% happy / 45% neutral is
// clearly Cassius even though neither class alone would pass the threshold.)
function crownColor(probs, threshold = CROWN_THRESHOLD) {
  const pCassius = probs.cassius_happy + probs.cassius_neutral;
  if (pCassius < threshold) return null;
  return probs.cassius_happy > probs.cassius_neutral ? "gold" : "green";
}

// ---------- Step 7: Run the pose model on the image ----------
// Teachable Machine trained the pose model on the center square of each photo, and the model
// reads PoseNet's raw heatmap grid, so position in the square matters. We crop the same way.
// (Tested on the 133 pose training photos: whole photos got 25/66 arms_down right, the center square 66/66.)
function centerSquare(source) {
  const side = Math.min(source.width, source.height);
  const square = document.createElement("canvas");
  square.width = square.height = side;
  square.getContext("2d").drawImage(
    source, (source.width - side) / 2, (source.height - side) / 2, side, side, 0, 0, side, side
  );
  return square;
}

// PoseNet always returns *some* pose. If the arms are out of frame it guesses where they are,
// and those guesses can look like "arms up". So we only trust the pose when a wrist is visible.
async function classifyPose(source) {
  const { pose, posenetOutput } = await poseModel.estimatePose(centerSquare(source));
  const wrists = pose
    ? pose.keypoints.filter((k) => k.part === "leftWrist" || k.part === "rightWrist")
    : [];
  if (!wrists.some((k) => k.score >= WRIST_MIN_SCORE)) {
    return null;
  }
  const predictions = await poseModel.predict(posenetOutput);
  const [label, probability] = topClass(
    Object.fromEntries(predictions.map((p) => [p.className, p.probability]))
  );
  return { label, probability, sparkles: label === "arms_up" && probability >= POSE_THRESHOLD };
}

// ---------- Drawing ----------
function renderScene(source, faces, pose, time = 0) {
  ctx.drawImage(source, 0, 0, els.canvas.width, els.canvas.height);
  for (const face of faces) {
    if (!face.crown) continue;
    const frame = crownFrame(face.points);
    ctx.save();
    ctx.translate(frame.x, frame.y);
    ctx.rotate(frame.angle);
    drawCrown(ctx, frame.width, CROWN_COLORS[face.crown]);
    if (pose && pose.sparkles) drawSparkles(ctx, frame.width, time);
    ctx.restore();
  }
  if (debug) drawDebug(faces);
}

// ---------- Step 5: Work out where the crown goes ----------
// Landmark 10 (top of forehead) is the anchor, the distance from 234 to 454 is the head width,
// and the line between the outer eye corners (33 → 263) gives the head's tilt.
function crownFrame(points) {
  const top = points[LM.forehead];
  const rightEye = points[LM.rightEye];
  const leftEye = points[LM.leftEye];
  const headWidth = Math.hypot(
    points[LM.leftSide].x - points[LM.rightSide].x,
    points[LM.leftSide].y - points[LM.rightSide].y
  );
  return {
    x: top.x,
    y: top.y,
    angle: Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x),
    width: headWidth * 1.2,
  };
}

// ---------- Step 6: Draw the crown procedurally ----------
// Drawn in the crown's own rotated coordinates: (0, 0) is the forehead anchor and
// negative y points up out of the head. Five points, a band, and small jewels.
function drawCrown(c, width, colors) {
  const w = width;
  const h = w * 0.55; // tallest point
  const base = -w * 0.06; // lift the band slightly above the forehead so it sits on the hair
  const band = h * 0.28;
  const peakX = [-0.5, -0.25, 0, 0.25, 0.5].map((f) => f * w);
  const peakH = [0.82, 0.7, 1, 0.7, 0.82].map((f) => base - f * h);
  const valleyH = base - band - h * 0.12;

  c.lineJoin = "round";
  c.lineWidth = Math.max(1.5, w * 0.018);
  c.strokeStyle = colors.outline;

  // Crown body: a zigzag of 5 points above the band
  const body = new Path2D();
  body.moveTo(-w / 2, base);
  body.lineTo(peakX[0], peakH[0]);
  for (let i = 1; i < 5; i++) {
    body.lineTo((peakX[i - 1] + peakX[i]) / 2, valleyH);
    body.lineTo(peakX[i], peakH[i]);
  }
  body.lineTo(w / 2, base);
  body.closePath();

  const gradient = c.createLinearGradient(0, base - h, 0, base);
  gradient.addColorStop(0, colors.light);
  gradient.addColorStop(0.55, colors.mid);
  gradient.addColorStop(1, colors.dark);
  c.fillStyle = gradient;
  c.fill(body);
  c.stroke(body);

  // Band along the bottom
  const bandPath = new Path2D();
  bandPath.rect(-w / 2, base - band, w, band);
  c.fillStyle = colors.dark;
  c.fill(bandPath);
  c.stroke(bandPath);

  // Jewels: three on the band, one pearl on each point
  const jewelR = w * 0.045;
  [
    [-w * 0.28, "#d62839"],
    [0, "#2a6fdb"],
    [w * 0.28, "#d62839"],
  ].forEach(([x, color]) => jewel(c, x, base - band / 2, jewelR, color, colors.outline));
  peakX.forEach((x, i) => jewel(c, x, peakH[i], jewelR * 0.75, "#fdfdf7", colors.outline));
}

function jewel(c, x, y, r, color, outline) {
  c.beginPath();
  c.arc(x, y, r, 0, Math.PI * 2);
  c.fillStyle = color;
  c.fill();
  c.lineWidth = Math.max(1, r * 0.3);
  c.strokeStyle = outline;
  c.stroke();
  // Small highlight so it reads as a gem
  c.beginPath();
  c.arc(x - r * 0.3, y - r * 0.3, r * 0.3, 0, Math.PI * 2);
  c.fillStyle = "rgba(255,255,255,0.8)";
  c.fill();
}

// Bonus from the pose model: sparkles around the crown when Cassius has his arms up.
// `time` makes them twinkle in webcam mode; photos use time = 0.
const SPARKLE_SPOTS = [
  [-0.85, -0.45, 0.13],
  [0.85, -0.6, 0.11],
  [-0.55, -1.0, 0.09],
  [0.6, -1.05, 0.12],
  [0.05, -1.3, 0.1],
  [-1.0, 0.05, 0.08],
  [1.0, 0.0, 0.09],
];

function drawSparkles(c, width, time) {
  SPARKLE_SPOTS.forEach(([fx, fy, fr], i) => {
    const twinkle = time ? 0.65 + 0.35 * Math.sin(time / 180 + i * 1.7) : 1;
    sparkle(c, fx * width, fy * width, fr * width * twinkle);
  });
}

function sparkle(c, x, y, r) {
  c.save();
  c.translate(x, y);
  c.beginPath();
  c.moveTo(0, -r);
  c.quadraticCurveTo(0, 0, r, 0);
  c.quadraticCurveTo(0, 0, 0, r);
  c.quadraticCurveTo(0, 0, -r, 0);
  c.quadraticCurveTo(0, 0, 0, -r);
  c.shadowColor = "rgba(255, 215, 80, 0.9)";
  c.shadowBlur = r * 0.8;
  c.fillStyle = "#fffbe6";
  c.fill();
  c.restore();
}

// Debug overlay: the crop square the model sees, plus the five landmarks used for placement
function drawDebug(faces) {
  const lw = Math.max(2, els.canvas.width / 400);
  faces.forEach((face, i) => {
    const b = face.box;
    ctx.lineWidth = lw;
    ctx.strokeStyle = face.crown ? "#00e5ff" : "#ff4081";
    ctx.strokeRect(b.x, b.y, b.side, b.side);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = `${Math.round(lw * 8)}px system-ui, sans-serif`;
    ctx.fillText(`Face ${i + 1}`, b.x + lw * 2, b.y + b.side - lw * 3);
    Object.values(LM).forEach((idx) => {
      const p = face.points[idx];
      ctx.beginPath();
      ctx.arc(p.x, p.y, lw * 2, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

// ---------- Results panel ----------
const pct = (p) => `${Math.round(p * 100)}%`;

function showResults(faces, pose) {
  els.faceList.replaceChildren(
    ...faces.map((face, i) => {
      const li = document.createElement("li");
      if (!face.probs) {
        li.textContent = `Face ${i + 1}: checking…`;
        return li;
      }
      const [label, probability] = topClass(face.probs);
      if (debug && face.crop) {
        face.crop.className = "face-thumb";
        li.append(face.crop);
      }
      const name = document.createElement("span");
      name.className = "face-name";
      name.textContent = `Face ${i + 1}`;
      const badge = document.createElement("span");
      badge.className = "badge" + (face.crown ? ` badge-${face.crown}` : "");
      badge.textContent = `${label} (${pct(probability)})${face.crown ? " 👑" : ""}`;
      const detail = document.createElement("span");
      detail.className = "face-probs";
      detail.textContent =
        `happy ${pct(face.probs.cassius_happy)} · neutral ${pct(face.probs.cassius_neutral)}` +
        ` · not Cassius ${pct(face.probs.not_cassius)}`;
      li.append(name, badge, detail);
      return li;
    })
  );
  if (faces.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No faces found.";
    els.faceList.append(li);
  }
  els.poseResult.textContent = pose
    ? `Pose: ${pose.label} (${pct(pose.probability)})${pose.sparkles ? " ✨" : ""}`
    : "Pose: arms not in view";
  els.results.hidden = false;
}

// ---------- Live webcam mode ----------
// Same pipeline as photos, run on video frames. Face landmarks run every frame so crowns follow
// your head smoothly; the face and pose classifiers run a few times a second, and each face's
// predictions are averaged over time so the crown doesn't flicker on and off.
const cam = {
  stream: null,
  running: false,
  frozen: false,
  raf: 0,
  tracks: [], // one entry per face being followed: position, averaged probs, crown color
  pose: null,
  lastClassify: 0,
  lastPose: 0,
  classifyBusy: false,
  poseBusy: false,
};

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("The webcam needs a secure page: open this over https:// or on localhost.", true);
    return;
  }
  els.cameraBtn.disabled = true;
  setStatus("Asking for camera permission…");
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    els.cameraBtn.disabled = false;
    const messages = {
      NotAllowedError: "Camera permission was blocked. Allow it in your browser's site settings and try again.",
      NotFoundError: "No camera found on this device.",
      NotReadableError: "The camera is busy in another app. Close it and try again.",
    };
    setStatus(messages[err.name] || "Couldn't start the camera.", true);
    return;
  }
  // The user may have switched back to Upload while the permission prompt was open
  if (els.webcamPanel.hidden) {
    cam.stream.getTracks().forEach((t) => t.stop());
    cam.stream = null;
    els.cameraBtn.disabled = false;
    return;
  }

  els.video.srcObject = cam.stream;
  await els.video.play();
  await useLandmarkerMode("VIDEO");
  runId++; // cancel any photo still being analyzed
  lastPhoto = null;

  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;
  els.canvas.classList.add("is-mirrored"); // selfie view on screen only
  els.stage.hidden = false;
  els.results.hidden = false;
  els.faceList.replaceChildren();
  els.poseResult.textContent = "";
  els.downloadBtn.disabled = true;
  Object.assign(cam, { running: true, frozen: false, tracks: [], pose: null, lastClassify: 0, lastPose: 0 });

  els.cameraBtn.textContent = "Stop camera";
  els.cameraBtn.disabled = false;
  els.captureBtn.textContent = "Capture";
  els.captureBtn.disabled = false;
  setStatus("Camera on. Crowns update live.");
  cam.raf = requestAnimationFrame(webcamFrame);
}

function stopCamera(message = "Camera off.") {
  if (!cam.stream) return;
  cancelAnimationFrame(cam.raf);
  cam.stream.getTracks().forEach((t) => t.stop());
  cam.stream = null;
  cam.running = false;
  els.video.srcObject = null;
  els.cameraBtn.textContent = "Start camera";
  els.captureBtn.textContent = "Capture";
  els.captureBtn.disabled = true;
  // Keep the last frame on screen so a captured photo can still be downloaded
  setStatus(message);
}

function webcamFrame(now) {
  if (!cam.running) return;
  cam.raf = requestAnimationFrame(webcamFrame);
  if (cam.frozen || els.video.readyState < 2) return;

  // Step 2 on this frame: find faces, then match each one to the face it was last frame
  const faces = detectFaces(els.video, now);
  followFaces(faces, now);

  // Steps 3–4 and 7, throttled. They run in the background while frames keep drawing.
  if (!cam.classifyBusy && now - cam.lastClassify >= CLASSIFY_EVERY_MS) classifyWebcamFaces(faces);
  if (!cam.poseBusy && now - cam.lastPose >= POSE_EVERY_MS) classifyWebcamPose();

  // Steps 5–6: draw the frame with crowns from the latest (smoothed) predictions
  renderScene(els.video, faces, cam.pose, now);
}

// Faces don't come back in a fixed order, so each face is paired with the nearest face
// from the previous frame. That lets its averaged prediction follow it around.
function followFaces(faces, now) {
  const unclaimed = new Set(cam.tracks);
  for (const face of faces) {
    let best = null;
    let bestDist = face.box.side * 0.6; // farther than this counts as a new face
    for (const t of unclaimed) {
      const d = Math.hypot(t.cx - face.box.cx, t.cy - face.box.cy);
      if (d < bestDist) {
        best = t;
        bestDist = d;
      }
    }
    if (best) unclaimed.delete(best);
    else cam.tracks.push((best = { probs: null, crown: null, crop: null }));
    Object.assign(best, { cx: face.box.cx, cy: face.box.cy, lastSeen: now });
    face.track = best;
    face.probs = best.probs;
    face.crown = best.crown;
    face.crop = best.crop;
  }
  // Forget faces that left the frame more than a second ago
  cam.tracks = cam.tracks.filter((t) => now - t.lastSeen < 1000);
}

async function classifyWebcamFaces(faces) {
  cam.classifyBusy = true;
  cam.lastClassify = performance.now();
  try {
    // Crop every face from the same frame first, then classify them one by one
    const crops = faces.map((face) => cropFace(els.video, face.box));
    for (let i = 0; i < faces.length; i++) {
      const probs = await predictCrop(crops[i]);
      const t = faces[i].track;
      // Running average, then hysteresis: turning a crown on needs CROWN_THRESHOLD,
      // but it only turns off below CROWN_OFF_THRESHOLD
      t.probs = t.probs
        ? Object.fromEntries(Object.keys(probs).map((k) => [k, t.probs[k] * (1 - SMOOTHING) + probs[k] * SMOOTHING]))
        : probs;
      t.crown = crownColor(t.probs, t.crown ? CROWN_OFF_THRESHOLD : CROWN_THRESHOLD);
      t.crop = crops[i];
      Object.assign(faces[i], { probs: t.probs, crown: t.crown, crop: t.crop });
    }
    if (cam.running && !cam.frozen) showResults(faces, cam.pose);
  } catch (err) {
    console.error(err);
  } finally {
    cam.classifyBusy = false;
  }
}

async function classifyWebcamPose() {
  cam.poseBusy = true;
  cam.lastPose = performance.now();
  try {
    // PoseNet reads a canvas more reliably than a <video>, so snapshot the frame first
    const frame = document.createElement("canvas");
    frame.width = els.video.videoWidth;
    frame.height = els.video.videoHeight;
    frame.getContext("2d").drawImage(els.video, 0, 0);
    cam.pose = await classifyPose(frame);
  } catch (err) {
    console.error(err);
  } finally {
    cam.poseBusy = false;
  }
}

// Capture freezes the current crowned frame so it can be downloaded; Resume goes live again
function toggleCapture() {
  cam.frozen = !cam.frozen;
  els.captureBtn.textContent = cam.frozen ? "Resume" : "Capture";
  els.downloadBtn.disabled = !cam.frozen;
  setStatus(cam.frozen ? "Captured. Download it, or resume the live view." : "Camera on. Crowns update live.");
}

function switchMode(mode) {
  lastPhoto = null;
  const webcam = mode === "webcam";
  els.tabUpload.setAttribute("aria-selected", String(!webcam));
  els.tabWebcam.setAttribute("aria-selected", String(webcam));
  els.uploadPanel.hidden = webcam;
  els.webcamPanel.hidden = !webcam;
  if (!webcam) stopCamera("");
  runId++; // cancel any photo still being analyzed
  els.canvas.classList.remove("is-mirrored");
  els.stage.hidden = true;
  els.results.hidden = true;
  els.downloadBtn.disabled = true;
  if (els.loading.hidden) {
    setStatus(webcam ? "Start the camera to go live." : "Ready. Choose a photo to get started.");
  }
}

els.tabUpload.addEventListener("click", () => switchMode("upload"));
els.tabWebcam.addEventListener("click", () => switchMode("webcam"));
els.cameraBtn.addEventListener("click", () => (cam.running ? stopCamera() : startCamera()));
els.captureBtn.addEventListener("click", toggleCapture);
// Don't keep the camera running in a background tab
document.addEventListener("visibilitychange", () => {
  if (document.hidden && cam.running) stopCamera("Camera turned off while the tab was hidden.");
});

// ---------- Under the hood toggle ----------
function setDebug(on) {
  debug = on;
  els.debugBtn.setAttribute("aria-pressed", String(on));
  els.debugBtn.textContent = on ? "Hide under the hood" : "Under the hood";
  els.debugLegend.hidden = !on;
  // Photos are redrawn right away; the webcam picks it up on its next frame
  if (lastPhoto && !cam.running) {
    renderScene(lastPhoto.source, lastPhoto.faces, lastPhoto.pose);
    showResults(lastPhoto.faces, lastPhoto.pose);
  }
}
els.debugBtn.addEventListener("click", () => setDebug(!debug));
setDebug(debug);

// ---------- Download ----------
els.downloadBtn.addEventListener("click", () => {
  els.canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "crowned.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
});

// ---------- Upload + drag and drop wiring ----------
els.fileInput.addEventListener("change", () => {
  handleFile(els.fileInput.files[0]);
  els.fileInput.value = ""; // allow picking the same file again
});

["dragenter", "dragover"].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((type) =>
  els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("is-dragover"))
);
els.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (els.fileInput.disabled) return;
  handleFile(e.dataTransfer.files[0]);
});
// A photo dropped outside the drop zone shouldn't make the browser navigate away
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

loadModels();
