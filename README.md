# Crown Me 👑

Crown Me is a small static web app. Upload a photo or turn on your webcam, and it finds every face, checks which faces are Cassius with a Teachable Machine image model, and draws a crown on his head only. The crown's color shows his expression:

| Prediction | Crown |
|---|---|
| `cassius_happy` | gold |
| `cassius_neutral` | green |
| `not_cassius` | none |

A face gets a crown only when the model is at least 80% sure it's Cassius (happy + neutral combined). A second Teachable Machine model (pose) looks at the whole photo. If it sees `arms_up` with at least 60% confidence, sparkles appear around the crown.

Everything runs in the browser. Photos and camera frames are never uploaded anywhere.

## Two modes

- **Upload photo**: choose a photo or drag one onto the page. You get the crowned image and a results panel listing every face with its prediction.
- **Live webcam**: click **Start camera** and allow access. Crowns follow your head live. Click **Capture** to freeze a frame, then **Download image** to save it. **Resume** goes live again.
  - The camera needs a secure page. `localhost`/`127.0.0.1` (Live Server) and Vercel's `https://` both qualify. Opening the page from another device via your laptop's IP address (`http://192.168…`) won't allow the camera.
  - The live view is mirrored like a selfie camera. Downloads are saved unmirrored, the way the scene actually looks.

## Files

| File | What it does |
|---|---|
| `index.html` | Page layout; loads TensorFlow.js and the Teachable Machine libraries |
| `app.js` | The full pipeline (load models → find faces → crop → classify → draw), commented step by step, plus the webcam loop |
| `style.css` | Styling, including the mobile layout |

No frameworks, no npm, no build step.

## Run it locally (VS Code + Live Server)

1. Install the **Live Server** extension in VS Code (by Ritwick Dey).
2. Open the `crown_app` folder in VS Code.
3. Right-click `index.html` → **Open with Live Server**, or click **Go Live** in the status bar.
4. The page opens at `http://127.0.0.1:5500`. Wait for all three models to show ✓, then choose a photo.

Open it through Live Server, not by double-clicking `index.html`. Browsers block the model downloads on `file://` pages.

Without VS Code: run `python3 -m http.server` in this folder and open `http://localhost:8000`.

## Deploy to Vercel

Option A, from GitHub:
1. Go to [vercel.com/new](https://vercel.com/new) and import this repository.
2. Set Framework Preset to **Other**. Leave Build Command empty and Output Directory as the root (`./`).
3. Click **Deploy**.

Option B, from the command line:
```sh
npm i -g vercel   # once
vercel            # from inside this folder; answers: no build command, current directory
vercel --prod
```

## Debug mode

Add `?debug=1` to the URL, e.g. `http://127.0.0.1:5500/?debug=1`. You'll see:
- the square crop around each face (cyan = crowned, pink = not crowned),
- the five landmarks used to place the crown (10, 33, 263, 234, 454),
- a thumbnail of the exact 224×224 image the model classified, in the results panel.

Use it to check that the crops look like your training images. It's also handy for journal screenshots. In webcam mode the debug labels appear backwards because the live view is mirrored.

## Models and libraries

| What | Source |
|---|---|
| Face classifier (Teachable Machine image) | https://teachablemachine.withgoogle.com/models/q5b7GQQSp/ (`cassius_happy`, `cassius_neutral`, `not_cassius`) |
| Pose classifier (Teachable Machine pose) | https://teachablemachine.withgoogle.com/models/oMQMyNhPT/ (`arms_up`, `arms_down`) |
| Face landmarks | MediaPipe Tasks Vision `FaceLandmarker` 1.0.1, `face_landmarker.task` (float16) |
| TensorFlow.js | 1.3.1, with `@teachablemachine/image@0.8` and `@teachablemachine/pose@0.8` |

## Decisions

Each entry says whether the choice was familiar (what this project or Teachable Machine already does) or new.

- **Library versions pinned** (familiar). TF.js 1.3.1 and TM 0.8 are what Teachable Machine's own export snippets use. MediaPipe is pinned to 1.0.1 instead of "latest" so a future release can't break the app. A test page confirmed all three run together on one page before the app was built.
- **One MediaPipe instance, GPU with CPU fallback** (new). Some phones and older laptops can't run the GPU path, so the app falls back to CPU instead of failing.
- **Label names repaired** (new). The hosted model saves its labels cut off as `cassius_happ...` and `cassius_neut...`. The app matches the start of each name and maps it back to the full name. Without this, no crown would ever appear.
- **Crown rule: combined Cassius score ≥ 0.8** (new, changed from the original spec). The original rule was "top class ≥ 0.6". Now `happy + neutral` decides whether it's Cassius, and the larger of the two picks the color. A face scored 45% happy / 45% neutral is clearly Cassius, but the original rule would have given it no crown. The threshold was raised from 0.6 to 0.8 after testing showed strangers being crowned (see "Crop fidelity").
- **Crop: landmark bounding box + 40% padding** (changed from the spec's 15%). The bounding box of all 478 face landmarks runs forehead to chin and ear to ear. It's made square, padded, and resized to 224×224. Testing showed 40% matches the training images better than 15% (see "Crop fidelity"). This is a stopgap until the model is retrained.
- **Crown placement** (from the spec). Anchor at landmark 10, width 1.2× the distance from 234 to 454, rotated by the eye-corner angle (33 → 263), lifted slightly above the forehead. Drawn procedurally: 5 points, a band, and jewels.
- **Colors: gold = happy, green = neutral** (changed during planning; the original spec had them the other way around).
- **Pose "no body" check** (new). PoseNet always returns a pose, even for a face-only close-up (a selfie scored 0.40 overall). So the app only trusts the pose when both shoulders are at least 30% visible.
- **Photos capped at 1600 px on the long edge** (new). This keeps large phone photos fast. Downloads use that resolution.
- **Webcam: one pipeline, throttled** (new). Face landmarks run every frame so crowns track smoothly. The face classifier runs about every 400 ms and the pose model about every second, so slower laptops and phones stay responsive. Each face is matched to the nearest face from the previous frame, so its predictions follow it.
- **Webcam: smoothing and hysteresis** (new). Each face's probabilities are averaged over time. A crown turns on at 0.8 and only turns off below 0.7, so it doesn't flicker when confidence hovers near the threshold.
- **Webcam: mirrored with CSS only** (new). Everything is computed on the real, unmirrored frame. Only the on-screen canvas is flipped with `transform: scaleX(-1)`. This avoids juggling two coordinate systems, where the crown tilt would flip sign.
- **No storage, no backend** (from the spec). Nothing is saved between visits.

## Crop fidelity (tested)

All 330 image-model training photos were run through the app's face-crop step. They're training images, so these numbers are optimistic, but they show how well the crop framing matches what the model learned:

| Framing | Cassius recognized | not_cassius correctly rejected |
|---|---|---|
| Whole photo (how the model was trained) | 198 / 201 | 126 / 129 |
| Face crop, 15% padding, threshold 0.6 (original spec) | 198 / 201 | 108 / 126 |
| Face crop, 40% padding, threshold 0.6 | 199 / 201 | 115 / 126 |
| **Face crop, 40% padding, threshold 0.8 (current)** | **199 / 201** | **116 / 126** |

The training images are full screenshots rather than tight face crops, so the model learned the scenes and framing of Cassius photos more than his face. Most of the strangers it wrongly crowns get 97–100% confidence, so no threshold filters them all out. Example: a stranger from the pose dataset was crowned at 61% under the original settings; with the current settings he's rejected at 100% not_cassius. The most reliable fix is to retrain the model on crops made the same way the app makes them. The `?debug=1` thumbnails show exactly what those crops look like.

## Known limitations

- **Strangers can still get a crown** (about 1 in 12 in testing). The one image model decides both identity and expression, from 330 photos. Small faces in full-body shots are the most likely to be misread.
- **Small or distant faces may be missed.** MediaPipe's face landmarker is built for faces within a couple of meters of the camera. It finds at most 5 faces.
- **Pose uses one person.** PoseNet reads a single body. In a group photo, someone else's raised arms can trigger your sparkles.
- **HEIC photos** (iPhone) may not open in desktop Chrome. Convert them to JPG first. Phone browsers usually convert them automatically.
- **First load is about 15–20 MB** of models, so it takes a moment on mobile data.
