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
const SHOULDER_MIN_SCORE = 0.3; // both shoulders must be this visible to count as "a body"
const CROP_PADDING = 0.4; // extra margin around the face crop (40% matched the training photos better than 15%)
const CROP_SIZE = 224; // Teachable Machine image models take 224x224 input
const MAX_EDGE = 1600; // big phone photos are scaled down to this long edge for speed

// Landmark indices on MediaPipe's 478-point face mesh
const LM = { forehead: 10, rightEye: 33, leftEye: 263, rightSide: 234, leftSide: 454 };

const CROWN_COLORS = {
  gold: { light: "#fff1a8", mid: "#f4c430", dark: "#b8860b", outline: "#6e5100" },
  green: { light: "#b6f5cc", mid: "#2ecc71", dark: "#1e8449", outline: "#0f4d2a" },
};

// Add ?debug=1 to the URL to see crop boxes, key landmarks, and the exact crops the model sees
const DEBUG = new URLSearchParams(location.search).has("debug");

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
  results: $("results"),
  faceList: $("face-list"),
  poseResult: $("pose-result"),
};
const ctx = els.canvas.getContext("2d");

let faceLandmarker, imageModel, poseModel;
let runId = 0; // lets a newer upload cancel an older one that's still running

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
    const faces = detectFaces(source);
    const classified = [];
    for (const face of faces) classified.push(await classifyFace(source, face));
    const pose = await classifyPose(source);
    if (myRun !== runId) return;

    for (const face of classified) face.crown = crownColor(face.probs);
    renderScene(source, classified, pose);
    showResults(classified, pose);
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

async function classifyFace(source, face) {
  const crop = cropFace(source, face.box);
  const predictions = await imageModel.predict(crop);
  const probs = { cassius_happy: 0, cassius_neutral: 0, not_cassius: 0 };
  for (const p of predictions) probs[canonicalLabel(p.className)] = p.probability;
  return { ...face, crop, probs };
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

// ---------- Step 7: Run the pose model on the whole image ----------
// PoseNet always returns *some* pose, even for a face-only close-up, so we only trust it
// when both shoulders are clearly visible. Otherwise we skip quietly.
async function classifyPose(source) {
  const { pose, posenetOutput } = await poseModel.estimatePose(source);
  const shoulders = pose
    ? pose.keypoints.filter((k) => k.part === "leftShoulder" || k.part === "rightShoulder")
    : [];
  if (shoulders.length < 2 || shoulders.some((k) => k.score < SHOULDER_MIN_SCORE)) {
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
  if (DEBUG) drawDebug(faces);
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
      const [label, probability] = topClass(face.probs);
      const li = document.createElement("li");
      if (DEBUG && face.crop) {
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
    : "Pose: no body detected";
  els.results.hidden = false;
}

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
