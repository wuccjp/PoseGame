"""
手部位置偵測器 - MediaPipe 0.10+ Tasks API

狀態機（每個角落獨立）：
  IDLE      → 未感應
  HOLDING   → 感應中（藍底＋紅色閃光邊框），持續 2 秒後 → TRIGGERED
  TRIGGERED → 已觸發（紅底），鎖定 5 秒後 → IDLE

安裝：  pip install opencv-python mediapipe Pillow
執行：  python hand_position_detector.py
操作：  V 鍵 或 點擊按鈕 → 切換顯示/隱藏影像；Q 鍵 → 離開
"""

import cv2, urllib.request, os, sys, time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from PIL import ImageFont, ImageDraw, Image

# ── 模型 ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.getcwd(), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("📥 正在下載 MediaPipe 手部模型（約 4MB）...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ 模型已儲存至 {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 下載失敗：{e}\n請手動下載：\n  {MODEL_URL}\n  存放到：{MODEL_PATH}")
        sys.exit(1)

# ── 字型 ──────────────────────────────────────────────────────
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

# ── 顏色 ──────────────────────────────────────────────────────
C_RIGHT_BGR = (0, 200, 255)
C_LEFT_BGR  = (80, 220, 120)
C_RIGHT_RGB = (255, 200, 0)
C_LEFT_RGB  = (120, 220, 80)

# ── 狀態常數 ──────────────────────────────────────────────────
IDLE      = "idle"
HOLDING   = "holding"    # 感應中，等待 2 秒
TRIGGERED = "triggered"  # 已觸發，鎖定 5 秒

HOLD_SEC    = 2.0   # 持續感應幾秒後觸發
LOCKOUT_SEC = 5.0   # 觸發後鎖定幾秒

# 每個角落獨立的狀態字典
# { label: {"state": IDLE/HOLDING/TRIGGERED, "ts": float} }
corner_states = {k: {"state": IDLE, "ts": 0.0} for k in ("右上", "右下", "左上", "左下")}

# 全域鎖定時間戳：任一角落觸發後，全部角落共用同一個倒數
global_lockout_ts = 0.0   # 0.0 表示未鎖定

# ── 按鈕全域 ──────────────────────────────────────────────────
btn_rect   = [0, 0, 0, 0]
show_video = True

def mouse_callback(event, x, y, flags, param):
    global show_video
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = btn_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            show_video = not show_video

# ── 狀態機更新 ────────────────────────────────────────────────
def update_states(detected: set, now: float):
    """依據本幀偵測結果更新每個角落的狀態機（全域鎖定版）"""
    global global_lockout_ts

    # 若目前全域鎖定中
    in_lockout = (global_lockout_ts > 0.0)

    if in_lockout:
        if now - global_lockout_ts >= LOCKOUT_SEC:
            # 鎖定結束，全部角落回 IDLE
            global_lockout_ts = 0.0
            for st in corner_states.values():
                st["state"] = IDLE
                st["ts"]    = 0.0
        return   # 鎖定期間不處理任何感應

    # 正常狀態更新
    triggered_now = False
    for label, st in corner_states.items():
        sensed = label in detected

        if st["state"] == IDLE:
            if sensed:
                st["state"] = HOLDING
                st["ts"]    = now

        elif st["state"] == HOLDING:
            if not sensed:
                st["state"] = IDLE
                st["ts"]    = 0.0
            elif now - st["ts"] >= HOLD_SEC:
                triggered_now = True   # 有角落觸發

    # 任一角落達到 2 秒 → 全部角落一起進入 TRIGGERED
    if triggered_now:
        global_lockout_ts = now
        for st in corner_states.values():
            st["state"] = TRIGGERED
            st["ts"]    = now

# ── PIL 中文字 ────────────────────────────────────────────────
def put_chinese_text(frame_bgr, text, pos, font, color_rgb,
                     bg_color_rgb=None, padding=8):
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    bbox    = draw.textbbox((0, 0), text, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y    = pos
    if bg_color_rgb is not None:
        draw.rectangle([x-padding, y-padding, x+tw+padding, y+th+padding],
                       fill=bg_color_rgb + (190,))
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── 角落標籤 ──────────────────────────────────────────────────
def draw_corners(frame, now: float):
    h, w  = frame.shape[:2]
    box_h = h // 5
    box_w = w // 4

    font_size = max(20, box_h // 2)
    font      = load_font(font_size)

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
        st    = corner_states[label]
        state = st["state"]

        # ── 底色 ──
        if state == TRIGGERED:
            bg = (220, 60, 60, 230)      # 紅底
        else:
            bg = (50, 100, 200, 230)     # 藍底（IDLE / HOLDING 都是藍）

        draw.rectangle([rx1, ry1, rx2, ry2], fill=bg)

        # ── 外框／閃光 ──
        if state == HOLDING:
            # 閃光：以 0.25 秒為週期交替紅框
            flash = int((now * 4) % 2) == 0   # True/False 交替
            outline_color = (255, 50, 50, 255) if flash else (255, 160, 160, 180)
            outline_w = 8 if flash else 5
            draw.rectangle([rx1, ry1, rx2, ry2], outline=outline_color, width=outline_w)

            # 進度條（下方）：已等待比例
            elapsed  = now - st["ts"]
            progress = min(elapsed / HOLD_SEC, 1.0)
            bar_w    = int((rx2 - rx1) * progress)
            bar_y    = ry2 - 12
            draw.rectangle([rx1, bar_y, rx1 + bar_w, ry2],
                           fill=(255, 220, 0, 230))

        elif state == TRIGGERED:
            draw.rectangle([rx1, ry1, rx2, ry2],
                           outline=(255, 200, 200, 255), width=4)

            # 倒數進度條（已鎖定比例）
            elapsed  = now - st["ts"]
            progress = min(elapsed / LOCKOUT_SEC, 1.0)
            bar_w    = int((rx2 - rx1) * (1 - progress))   # 從右往左縮短
            bar_y    = ry2 - 12
            draw.rectangle([rx1, bar_y, rx1 + bar_w, ry2],
                           fill=(60, 200, 80, 230))
        else:
            draw.rectangle([rx1, ry1, rx2, ry2],
                           outline=(120, 160, 255, 200), width=2)

        # ── 文字置中 ──
        bbox   = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx     = rx1 + (box_w - tw) // 2
        ty     = ry1 + (box_h - th) // 2
        draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 切換按鈕 ──────────────────────────────────────────────────
def draw_toggle_button(frame, font_btn):
    global btn_rect
    h, w    = frame.shape[:2]
    label   = "📷 隱藏影像" if show_video else "📷 顯示影像"
    btn_rgb = (60, 180, 60) if show_video else (60, 60, 180)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    bbox   = draw.textbbox((0, 0), label, font=font_btn)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad, margin = 14, 16

    bx1 = (w - tw - pad * 2) // 2
    bx2 = bx1 + tw + pad * 2
    by2 = h - margin
    by1 = by2 - th - pad * 2
    btn_rect[:] = [bx1, by1, bx2, by2]

    draw.rectangle([bx1+3, by1+3, bx2+3, by2+3], fill=(0, 0, 0, 100))
    draw.rectangle([bx1, by1, bx2, by2], fill=btn_rgb + (220,))
    draw.rectangle([bx1, by1, bx2, by2], outline=(255, 255, 255, 160), width=2)
    draw.text((bx1 + pad, by1 + pad), label, font=font_btn, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 主程式 ────────────────────────────────────────────────────
def main():
    global show_video

    download_model()

    h_frame, w_frame = 720, 1280
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

    print("✅ 偵測啟動！V 鍵切換影像；Q 鍵離開")

    frame_idx = 0

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        raw = cv2.flip(raw, 1)
        h_frame, w_frame = raw.shape[:2]
        now = time.time()

        # ── 偵測 ──
        rgb    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = frame_idx * 33
        result = detector.detect_for_video(mp_img, timestamp_ms)

        # 本幀感應到的位置（TRIGGERED 狀態的角落不計入，避免重複觸發）
        raw_detected = set()
        if result.hand_landmarks and result.handedness:
            for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                raw_label = handedness_list[0].category_name
                side      = "右" if raw_label == "Right" else "左"
                vertical  = "上" if lm_list[0].y < 0.45 else "下"
                raw_detected.add(side + vertical)

        # 全域鎖定期間所有角落停止感應
        in_lockout = (global_lockout_ts > 0.0 and
                      now - global_lockout_ts < LOCKOUT_SEC)
        detected = set() if in_lockout else raw_detected

        update_states(detected, now)

        # ── 畫布 ──
        canvas = raw.copy() if show_video else np.zeros_like(raw)

        # 角落標籤
        canvas = draw_corners(canvas, now)

        # 切換按鈕
        canvas = draw_toggle_button(canvas, font_btn)

        # 狀態文字（置於按鈕上方 0.5cm）
        triggered_labels = [l for l, s in corner_states.items() if s["state"] == TRIGGERED]
        holding_labels   = [l for l, s in corner_states.items() if s["state"] == HOLDING]
        if triggered_labels:
            status = "已觸發：" + "、".join(sorted(triggered_labels))
        elif holding_labels:
            status = "感應中：" + "、".join(sorted(holding_labels))
        elif raw_detected:
            status = "偵測到：" + "、".join(sorted(raw_detected))
        else:
            status = "請將手部移入畫面"

        gap = 19
        tmp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        try:
            sb   = tmp_draw.textbbox((0, 0), status, font=font_small)
            s_tw = sb[2] - sb[0]
            s_th = sb[3] - sb[1]
        except Exception:
            s_tw, s_th = len(status) * 14, 24

        status_x = (w_frame - s_tw) // 2
        status_y = btn_rect[1] - gap - s_th
        canvas = put_chinese_text(canvas, status,
                                  (status_x, status_y),
                                  font_small, (220, 220, 220),
                                  bg_color_rgb=(30, 30, 30), padding=6)

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