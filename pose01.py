"""
手部位置偵測器 - MediaPipe 0.10+ 新版 Tasks API
偵測左右手位置（上/下），畫面顯示 右上、右下、左上、左下

安裝：
    pip install opencv-python mediapipe

第一次執行會自動下載模型（約 4MB），之後快取在本地。
執行：
    python hand_position_detector.py
按 Q 離開
"""

import cv2
import urllib.request
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── 模型下載 ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.getcwd(), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("📥 正在下載 MediaPipe 手部模型（約 4MB）...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ 模型已儲存至 {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 下載失敗：{e}")
        print("請手動下載：")
        print(f"  {MODEL_URL}")
        print(f"  存放到：{MODEL_PATH}")
        sys.exit(1)

# ── 顏色 (BGR) ───────────────────────────────────────────────
C_RIGHT  = (0, 200, 255)    # 橘黃：右手
C_LEFT   = (80, 220, 120)   # 綠：左手
C_WRIST  = (80,  80, 255)

# 手部骨架連線
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── 判斷上/下 ────────────────────────────────────────────────
def is_upper(landmarks, threshold=0.45):
    """手腕 y < threshold → 上方（正規化 y，越小越靠頂部）"""
    return landmarks[0].y < threshold

# ── 繪製骨架 ─────────────────────────────────────────────────
def draw_skeleton(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 8 if i == 0 else 5
        c = C_WRIST if i == 0 else color
        cv2.circle(frame, (x, y), r, c, -1, cv2.LINE_AA)

# ── 繪製角落標籤 ─────────────────────────────────────────────
def draw_corners(frame, detected: set):
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(w, h) / 500)
    thick = max(2, int(scale * 2.5))

    labels = {
        "右上": (int(w * 0.04), int(h * 0.13)),
        "右下": (int(w * 0.04), int(h * 0.87)),
        "左上": (int(w * 0.74), int(h * 0.13)),
        "左下": (int(w * 0.74), int(h * 0.87)),
    }

    for label, (x, y) in labels.items():
        active = label in detected
        color  = C_RIGHT if label.startswith("右") else C_LEFT

        (tw, th), _ = cv2.getTextSize(label, font, scale * 1.6, thick)
        pad = 14
        x1, y1 = x - pad, y - th - pad
        x2, y2 = x + tw + pad, y + pad

        if active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x, y), font, scale * 1.6, (15, 15, 15), thick, cv2.LINE_AA)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
            cv2.putText(frame, label, (x, y), font, scale * 1.6, (90, 90, 90), thick, cv2.LINE_AA)

# ── 主程式 ───────────────────────────────────────────────────
def main():
    download_model()

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ 偵測啟動！舉起雙手試試看。按 Q 離開。")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO 模式需要毫秒時間戳
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_idx * 33   # fallback ~30fps

        result = detector.detect_for_video(mp_img, timestamp_ms)

        detected = set()

        if result.hand_landmarks and result.handedness:
            for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                raw_label = handedness_list[0].category_name  # "Left" or "Right"
                # MediaPipe 是對原始影像判斷，flip後 Right=右手、Left=左手（不需對調）
                side     = "右" if raw_label == "Right" else "左"
                vertical = "上" if is_upper(lm_list) else "下"
                label    = side + vertical
                detected.add(label)

                color = C_RIGHT if side == "右" else C_LEFT
                draw_skeleton(frame, lm_list, color)

                wx = int(lm_list[0].x * w)
                wy = int(lm_list[0].y * h)
                cv2.putText(frame, label, (wx + 14, wy - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        draw_corners(frame, detected)

        status = "偵測到：" + "、".join(sorted(detected)) if detected else "請將手部移入畫面"
        cv2.putText(frame, status, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Hand Position Detector", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 程式結束")


if __name__ == "__main__":
    main()