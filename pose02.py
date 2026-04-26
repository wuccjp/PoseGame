"""
手部位置偵測器 - MediaPipe 0.10+ Tasks API
偵測左右手位置（上/下），畫面顯示 右上、右下、左上、左下

安裝：
    pip install opencv-python mediapipe Pillow

執行：
    python hand_position_detector.py
按 Q 離開
"""

import cv2
import urllib.request
import os
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from PIL import ImageFont, ImageDraw, Image

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
        print(f"請手動下載：\n  {MODEL_URL}\n  存放到：{MODEL_PATH}")
        sys.exit(1)

# ── 字型載入 ─────────────────────────────────────────────────
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",                    # macOS
    "C:/Windows/Fonts/msjh.ttc",                             # Windows 微軟正黑
    "C:/Windows/Fonts/mingliu.ttc",                          # Windows 細明體
]

def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

# ── 顏色 (BGR / RGB) ─────────────────────────────────────────
C_RIGHT_BGR  = (0, 200, 255)
C_LEFT_BGR   = (80, 220, 120)
C_RIGHT_RGB  = (255, 200, 0)
C_LEFT_RGB   = (120, 220, 80)
C_WRIST_BGR  = (80,  80, 255)

# ── 手部骨架連線 ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── PIL 繪製中文文字 ─────────────────────────────────────────
def put_chinese_text(frame_bgr, text, pos, font, color_rgb, bg_color_rgb=None, padding=8):
    """在 OpenCV frame 上繪製中文字（透過 PIL）"""
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    # 計算文字大小
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = pos

    # 背景方塊
    if bg_color_rgb is not None:
        draw.rectangle(
            [x - padding, y - padding, x + tw + padding, y + th + padding],
            fill=bg_color_rgb + (190,) if len(bg_color_rgb) == 3 else bg_color_rgb,
        )

    draw.text((x, y), text, font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── 繪製角落標籤（全部用 PIL）────────────────────────────────
def draw_corners(frame, detected: set, font_large):
    h, w = frame.shape[:2]

    label_positions = {
        "右上": (int(w * 0.04), int(h * 0.04)),
        "右下": (int(w * 0.04), int(h * 0.80)),
        "左上": (int(w * 0.76), int(h * 0.04)),
        "左下": (int(w * 0.76), int(h * 0.80)),
    }

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for label, (x, y) in label_positions.items():
        active     = label in detected
        color_rgb  = C_RIGHT_RGB if label.startswith("右") else C_LEFT_RGB

        bbox = draw.textbbox((0, 0), label, font=font_large)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        pad  = 14

        rx1, ry1 = x - pad, y - pad
        rx2, ry2 = x + tw + pad, y + th + pad

        if active:
            draw.rectangle([rx1, ry1, rx2, ry2], fill=color_rgb + (200,))
            draw.rectangle([rx1, ry1, rx2, ry2], outline=color_rgb + (255,), width=3)
            draw.text((x, y), label, font=font_large, fill=(15, 15, 15, 255))
        else:
            draw.rectangle([rx1, ry1, rx2, ry2], fill=(50, 50, 50, 120))
            draw.rectangle([rx1, ry1, rx2, ry2], outline=(100, 100, 100, 180), width=1)
            draw.text((x, y), label, font=font_large, fill=(100, 100, 100, 200))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 繪製骨架 ─────────────────────────────────────────────────
def draw_skeleton(frame, landmarks, color_bgr):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 8 if i == 0 else 5
        c = C_WRIST_BGR if i == 0 else color_bgr
        cv2.circle(frame, (x, y), r, c, -1, cv2.LINE_AA)

# ── 主程式 ───────────────────────────────────────────────────
def main():
    download_model()

    h_frame, w_frame = 720, 1280  # 初始值，後面更新
    font_large  = load_font(max(36, w_frame // 22))
    font_small  = load_font(max(20, w_frame // 40))

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
        h_frame, w_frame = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_idx * 33

        result = detector.detect_for_video(mp_img, timestamp_ms)

        detected = set()

        if result.hand_landmarks and result.handedness:
            for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                raw_label = handedness_list[0].category_name  # "Left" or "Right"
                # flip 後 Right=右手，Left=左手，不需對調
                side     = "右" if raw_label == "Right" else "左"
                vertical = "上" if lm_list[0].y < 0.45 else "下"
                label    = side + vertical
                detected.add(label)

                color_bgr = C_RIGHT_BGR if side == "右" else C_LEFT_BGR
                color_rgb = C_RIGHT_RGB if side == "右" else C_LEFT_RGB
                draw_skeleton(frame, lm_list, color_bgr)

                # 手腕旁的中文標籤
                wx = int(lm_list[0].x * w_frame)
                wy = int(lm_list[0].y * h_frame)
                frame = put_chinese_text(
                    frame, label,
                    (wx + 14, wy - 40),
                    font_small, color_rgb,
                    bg_color_rgb=(20, 20, 20),
                    padding=6,
                )

        # 角落四個標籤
        frame = draw_corners(frame, detected, font_large)

        # 底部狀態列
        status = "偵測到：" + "、".join(sorted(detected)) if detected else "請將手部移入畫面"
        frame = put_chinese_text(
            frame, status,
            (20, h_frame - 45),
            font_small, (160, 160, 160),
        )

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