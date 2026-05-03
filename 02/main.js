// Vanilla JS MVP: 相機 + MediaPipe Hands 整合 + Canvas 圓形覆蓋
// 功能：綠色四圓半徑為螢幕高度的 1/10，碰觸時外圈顯示 0→3s 的進度條，如果觸碰持續 3s，整個圓形變成觸碰顏色，3s 後復原。

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const status = document.getElementById('status');

let hands = null;
let streaming = false;

// Per-circle state for A-D (indices 0-3)
let touchStartTimes = [null, null, null, null];
let burstActive = [false, false, false, false];
let burstStart = [0, 0, 0, 0];
let ringColorForTouch = [null, null, null, null];

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

// Helper: render four static green circles with A-D labels (fallback when hands not ready)
function drawStaticGreens(ctx, centers, radius) {
  for (let i = 0; i < centers.length; i++) {
    ctx.fillStyle = 'rgba(0,255,0,0.9)';
    ctx.beginPath();
    ctx.arc(centers[i].x, centers[i].y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.font = `${radius * 0.6}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(['A','B','C','D'][i], centers[i].x, centers[i].y);
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

  // 說明性安全：若尚未載入 Hands，使用靜態四綠圈作為占位，方便測試與排錯
  if (!results || !results.multiHandLandmarks) {
    const greenRadius = Math.max(3, canvasCtx.canvas.height / 10);
    const greenCenters = [
      { x: w * 0.25, y: h * 0.25 }, // A
      { x: w * 0.75, y: h * 0.25 }, // B
      { x: w * 0.25, y: h * 0.75 }, // C
      { x: w * 0.75, y: h * 0.75 }, // D
    ];
    drawStaticGreens(ctx, greenCenters, greenRadius);
    return;
  }

  // 參數與佈局設定
  const greenRadius = Math.max(3, canvasCtx.canvas.height / 10); // 視窗高度的 1/10
  const greenCenters = [
    { x: w * 0.25, y: h * 0.25 }, // A
    { x: w * 0.75, y: h * 0.25 }, // B
    { x: w * 0.25, y: h * 0.75 }, // C
    { x: w * 0.75, y: h * 0.75 }, // D
  ];
  const handRadius = Math.max(3, canvasCtx.canvas.height / 20);

  // 收集手部資料
  const handCircles = [];
  if (results && results.multiHandLandmarks) {
    const handsCount = results.multiHandLandmarks.length;
    for (let hi = 0; hi < handsCount; hi++) {
      const landmarks = results.multiHandLandmarks[hi];
      const handedness = results.multiHandedness && results.multiHandedness[hi] ? results.multiHandedness[hi].label : 'Left';
      let sumX = 0, sumY = 0;
      for (let j = 0; j < landmarks.length; j++) {
        sumX += landmarks[j].x;
        sumY += landmarks[j].y;
      }
      const cx = (sumX / landmarks.length) * w;
      const cy = (sumY / landmarks.length) * h;
      const handX = w - cx; // 左右翻轉
      const handY = cy - handRadius * 0.5; // 往上方偏移
      const color = handedness === 'Left' ? 'rgba(255, 204, 0, 0.65)' : 'rgba(255, 0, 0, 0.65)';
      handCircles.push({ x: handX, y: handY, color });
    }
  }

  // 碰觸判定：找出每個綠色圓最近的手部顏色
  const nearestColor = [null, null, null, null];
  const minDist = [Infinity, Infinity, Infinity, Infinity];
  for (let hi = 0; hi < handCircles.length; hi++) {
    const h = handCircles[hi];
    for (let gi = 0; gi < greenCenters.length; gi++) {
      const gx = greenCenters[gi].x;
      const gy = greenCenters[gi].y;
      const dx = h.x - gx;
      const dy = h.y - gy;
      const dist = Math.hypot(dx, dy);
      if (dist <= (handRadius + greenRadius) && dist < minDist[gi]) {
        minDist[gi] = dist;
        nearestColor[gi] = h.color;
      }
    }
  }

  // 更新觸碰與爆裂狀態：3 秒後改變顏色，3 秒後恢復綠色
  const now = Date.now();
  for (let gi = 0; gi < 4; gi++) {
    if (nearestColor[gi]) {
      if (touchStartTimes[gi] == null) touchStartTimes[gi] = now;
      if (!burstActive[gi]) {
        const elapsed = (now - touchStartTimes[gi]) / 1000;
        if (elapsed >= 3) {
          burstActive[gi] = true;
          burstStart[gi] = now;
          ringColorForTouch[gi] = nearestColor[gi];
        }
      }
      if (burstActive[gi] && now - burstStart[gi] >= 3000) {
        burstActive[gi] = false;
        touchStartTimes[gi] = null;
        ringColorForTouch[gi] = null;
      }
    } else {
      if (touchStartTimes[gi] != null) touchStartTimes[gi] = null;
      burstActive[gi] = false;
      ringColorForTouch[gi] = null;
    }
  }

  // 繪製綠色底色與標籤
  for (let gi = 0; gi < 4; gi++) {
    const fillColor = burstActive[gi] ? ringColorForTouch[gi] : 'rgba(0,255,0,0.9)';
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.arc(greenCenters[gi].x, greenCenters[gi].y, greenRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.font = `${greenRadius * 0.6}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const label = ['A','B','C','D'][gi] || '';
    ctx.fillText(label, greenCenters[gi].x, greenCenters[gi].y);
  }

  // 外圍進度條：僅在被碰觸且尚未完成 3s 時顯示
  for (let gi = 0; gi < 4; gi++) {
    if (nearestColor[gi] && !burstActive[gi] && touchStartTimes[gi] != null) {
      const elapsed = Math.min(3, (now - touchStartTimes[gi]) / 1000);
      const progress = elapsed / 3;
      const ringRadius = greenRadius + 6;
      ctx.lineWidth = 6;
      ctx.strokeStyle = nearestColor[gi];
      ctx.beginPath();
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + progress * Math.PI * 2;
      ctx.arc(greenCenters[gi].x, greenCenters[gi].y, ringRadius, startAngle, endAngle);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  // 繪製手部圓在前景
  for (let hi = 0; hi < handCircles.length; hi++) {
    const h = handCircles[hi];
    ctx.fillStyle = h.color;
    ctx.beginPath();
    ctx.arc(h.x, h.y, handRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function initHands() {
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
  const btn = document.getElementById('toggleVideoBtn');
  if (btn) {
    btn.addEventListener('click', toggleVideoVisibility);
    btn.textContent = '顯示影像';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  boot();
  window.addEventListener('resize', resizeCanvasToVideo);
});
