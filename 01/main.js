// Vanilla JS MVP: 相機 + MediaPipe Hands 整合 + Canvas 圓形覆蓋
// 依賴：Facebook Hands透過 CDN 直接掛載成全域物件 Hands

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const status = document.getElementById('status');

let hands = null;
let streaming = false;

function toggleVideoVisibility() {
  const btn = document.getElementById('toggleVideoBtn');
  if (!video) return;
  if (video.style.display === 'none' || video.style.display === '') {
    video.style.display = 'block';
    if (btn) btn.textContent = '隱藏影像';
  } else {
    video.style.display = 'none';
    if (btn) btn.textContent = '顯示影像';
  }
}

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();
    streaming = true;
  } catch (e) {
    console.error('Camera error', e);
    status.textContent = '找不到相機或權限被拒絕。';
  }
}

function resizeCanvasToVideo() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  overlay.width = w;
  overlay.height = h;
}

function renderOverlay(results, vid, canvasCtx) {
  const w = canvasCtx.canvas.width;
  const h = canvasCtx.canvas.height;
  canvasCtx.clearRect(0, 0, w, h);

  if (!results || !results.multiHandLandmarks) return;

  // results.multiHandedness 與 multiHandLandmarks 的順序對應同索引
  const handsCount = results.multiHandLandmarks.length;
  for (let i = 0; i < handsCount; i++) {
    const landmarks = results.multiHandLandmarks[i];
    const handedness = results.multiHandedness && results.multiHandedness[i] ? results.multiHandedness[i].label : 'Left';

    // 計算手掌中心點（21 點的幾何中心）
    let cx = 0, cy = 0;
    for (let j = 0; j < landmarks.length; j++) {
      cx += landmarks[j].x;
      cy += landmarks[j].y;
    }
    cx = (cx / landmarks.length) * w;
    cy = (cy / landmarks.length) * h;

    // 圓形直徑為畫面高度的 1/10，因此半徑為高度的 1/20
    const radius = Math.max(3, canvasCtx.canvas.height / 20); // 高度的 1/20
    // 圓心位置：左右顛倒並往上方偏移
    const centerX = w - cx;
    const centerY = cy - radius * 0.5; // 往上方偏移

    // 顏色：左右顛倒顯示（Left 轉為黃，Right 轉為紅）
    const color = handedness === 'Left' ? 'rgba(255, 204, 0, 0.65)' : 'rgba(255, 0, 0, 0.65)';
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function initHands() {
  // Hands 綁定於全域物件，透過 CDN 載入
  // 參數可根據需要調整：maxNumHands, minDetectionConfidence, minTrackingConfidence
  hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
  hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.8, minTrackingConfidence: 0.5 });
  hands.onResults((results) => renderOverlay(results, video, overlay.getContext('2d')));
}

async function frameLoop() {
  if (!streaming) {
    requestAnimationFrame(frameLoop);
    return;
  }
  if (video.readyState >= 2 && hands) {
    await hands.send({ image: video });
  }
  requestAnimationFrame(frameLoop);
}

async function boot() {
  await initCamera();
  resizeCanvasToVideo();
  await initHands();
  status.textContent = '就緒：可進行手掌偵測與圓形覆蓋';
  frameLoop();
  // 控制面板初始化
  const btn = document.getElementById('toggleVideoBtn');
  if (btn) {
    btn.addEventListener('click', toggleVideoVisibility);
    btn.textContent = '顯示影像';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  boot();
  // 窗口大小改變時同步調整 canvas
  window.addEventListener('resize', resizeCanvasToVideo);
});
