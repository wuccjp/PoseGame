"""
手部位置偵測器 - MediaPipe 0.10+ Tasks API
偵測左右手位置（上/下），畫面顯示 右上、右下、左上、左下

安裝：
    pip install opencv-python mediapipe Pillow

執行：
    python hand_position_detector.py

操作：
    V 鍵 或 點擊畫面按鈕 → 切換顯示/隱藏影像
    Q 鍵 → 離開
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
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
]

def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

# ── 顏色 ─────────────────────────────────────────────────────
C_RIGHT_BGR = (0, 200, 255)
C_LEFT_BGR  = (80, 220, 120)
C_RIGHT_RGB = (255, 200, 0)
C_LEFT_RGB  = (120, 220, 80)
C_WRIST_BGR = (80,  80, 255)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── 按鈕狀態（全域，供 mouse callback 使用）──────────────────
btn_rect   = [0, 0, 0, 0]   # [x1, y1, x2, y2]，執行時動態設定
show_video = True            # 目前影像顯示狀態

def mouse_callback(event, x, y, flags, param):
    global show_video
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = btn_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            show_video = not show_video

# ── PIL 繪製中文字 ────────────────────────────────────────────
def put_chinese_text(frame_bgr, text, pos, font, color_rgb,
                     bg_color_rgb=None, padding=8):
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    bbox    = draw.textbbox((0, 0), text, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y    = pos
    if bg_color_rgb is not None:
        draw.rectangle(
            [x - padding, y - padding, x + tw + padding, y + th + padding],
            fill=bg_color_rgb + (190,),
        )
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── 角落四個位置標籤 ─────────────────────────────────────────
def draw_corners(frame, detected: set):
    """每個標籤區塊高度 = 視窗高度 1/5，寬度 1/4，緊貼四個角落"""
    h, w = frame.shape[:2]
    box_h = h // 5
    box_w = w // 4

    font_size = max(20, box_h // 2)
    font = load_font(font_size)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    corners = {
        "右上": (0,       0,       box_w, box_h),
        "右下": (0,       h-box_h, box_w, h),
        "左上": (w-box_w, 0,       w,     box_h),
        "左下": (w-box_w, h-box_h, w,     h),
    }

    for label, (rx1, ry1, rx2, ry2) in corners.items():
        active    = label in detected
        color_rgb = C_RIGHT_RGB if label.startswith("右") else C_LEFT_RGB

        # 藍底 → 感應到時紅底，文字永遠白色
        bg_color = (220, 60, 60) if active else (50, 100, 200)
        draw.rectangle([rx1, ry1, rx2, ry2], fill=bg_color + (230,))
        outline_color = (255, 180, 180, 255) if active else (120, 160, 255, 200)
        draw.rectangle([rx1, ry1, rx2, ry2], outline=outline_color, width=4)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = rx1 + (box_w - tw) // 2
        ty = ry1 + (box_h - th) // 2

        draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 切換按鈕 ─────────────────────────────────────────────────
def draw_toggle_button(frame, font_btn):
    """在右上角畫切換按鈕，更新 btn_rect，回傳 frame"""
    global btn_rect
    h, w = frame.shape[:2]

    label    = "📷 隱藏影像" if show_video else "📷 顯示影像"
    on_color = (60, 180, 60)    # 綠：目前顯示中
    off_color= (60, 60, 180)    # 藍：目前隱藏中
    btn_rgb  = on_color if show_video else off_color

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    bbox    = draw.textbbox((0, 0), label, font=font_btn)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad     = 14
    margin  = 16

    # 按鈕置於下方中間
    bx1 = (w - tw - pad * 2) // 2
    bx2 = bx1 + tw + pad * 2
    by2 = h - margin
    by1 = by2 - th - pad * 2

    btn_rect[:] = [bx1, by1, bx2, by2]

    # 陰影
    draw.rectangle([bx1+3, by1+3, bx2+3, by2+3], fill=(0, 0, 0, 100))
    # 按鈕本體
    draw.rectangle([bx1, by1, bx2, by2], fill=btn_rgb + (220,))
    draw.rectangle([bx1, by1, bx2, by2], outline=(255, 255, 255, 160), width=2)
    # 文字
    draw.text((bx1 + pad, by1 + pad), label, font=font_btn, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 繪製骨架 ─────────────────────────────────────────────────
def draw_skeleton(frame, landmarks, color_bgr):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 8 if i == 0 else 5
        c = C_WRIST_BGR if i == 0 else color_bgr
        cv2.circle(frame, (x, y), r, c, -1, cv2.LINE_AA)

# ── 主程式 ───────────────────────────────────────────────────
def main():
    global show_video

    download_model()

    h_frame, w_frame = 720, 1280
    font_large = load_font(max(36, w_frame // 22))
    font_small = load_font(max(20, w_frame // 40))
    font_btn   = load_font(max(22, w_frame // 36))

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

    win_name = "Hand Position Detector"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("✅ 偵測啟動！")
    print("   V 鍵 或 點擊右上角按鈕 → 切換顯示/隱藏影像")
    print("   Q 鍵 → 離開")

    frame_idx = 0
    last_frame = None      # 保留最後一幀，隱藏時用來做偵測

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        raw = cv2.flip(raw, 1)
        h_frame, w_frame = raw.shape[:2]

        # ── 永遠對原始畫面做偵測 ──
        rgb    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_idx * 33
        result = detector.detect_for_video(mp_img, timestamp_ms)

        detected = set()

        # ── 決定顯示畫布 ──
        if show_video:
            canvas = raw.copy()
        else:
            # 純黑背景
            canvas = np.zeros_like(raw)

        if result.hand_landmarks and result.handedness:
            for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                raw_label = handedness_list[0].category_name
                side      = "右" if raw_label == "Right" else "左"
                vertical  = "上" if lm_list[0].y < 0.45 else "下"
                label     = side + vertical
                detected.add(label)

                color_bgr = C_RIGHT_BGR if side == "右" else C_LEFT_BGR
                color_rgb = C_RIGHT_RGB if side == "右" else C_LEFT_RGB

        # 角落標籤
        canvas = draw_corners(canvas, detected)

        # 先畫切換按鈕（需要先知道 btn_rect 座標）
        canvas = draw_toggle_button(canvas, font_btn)

        # 底部狀態：置於按鈕上方，間隔約 0.5cm (19px)
        gap = 19
        status = "偵測到：" + "、".join(sorted(detected)) if detected else "請將手部移入畫面"
        _, s_th, _ = font_small.getbbox(status)[1:4] if hasattr(font_small, 'getbbox') else (0, 20, 0)
        try:
            s_bbox = ImageDraw.Draw(Image.new("RGBA", (1,1))).textbbox((0,0), status, font=font_small)
            s_th = s_bbox[3] - s_bbox[1]
        except Exception:
            s_th = 24
        status_y = btn_rect[1] - gap - s_th
        # 水平置中
        try:
            s_bbox = ImageDraw.Draw(Image.new("RGBA", (1,1))).textbbox((0,0), status, font=font_small)
            s_tw = s_bbox[2] - s_bbox[0]
        except Exception:
            s_tw = len(status) * 14
        status_x = (w_frame - s_tw) // 2
        canvas = put_chinese_text(
            canvas, status,
            (status_x, status_y),
            font_small, (220, 220, 220),
            bg_color_rgb=(30, 30, 30),
            padding=6,
        )

        cv2.imshow(win_name, canvas)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("v"):
            show_video = not show_video

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 程式結束")


if __name__ == "__main__":
    main()