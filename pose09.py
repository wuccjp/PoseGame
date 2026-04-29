"""
4選1手勢測驗 + 手部位置偵測器
MediaPipe 0.10+ Tasks API

題庫格式 test.csv（與程式同目錄）：
  題號,題目,選項1,選項2,選項3,選項4,答案選項序號

狀態機（每個角落獨立）：
  IDLE      → 藍底，未感應
  HOLDING   → 藍底＋紅色閃爍框＋黃色進度條，持續 2 秒 → TRIGGERED
  TRIGGERED → 紅底＋綠色倒數條，鎖定 5 秒後全部恢復

安裝：  pip install opencv-python mediapipe Pillow
執行：  python hand_position_detector.py
操作：  V 鍵 / 點擊按鈕 → 切換影像；Q 鍵 → 離開
"""

import cv2, urllib.request, os, sys, time, csv, random
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

# ── 狀態常數 ──────────────────────────────────────────────────
IDLE      = "idle"
HOLDING   = "holding"
TRIGGERED = "triggered"

HOLD_SEC    = 2.0
LOCKOUT_SEC = 5.0

corner_states     = {k: {"state": IDLE, "ts": 0.0} for k in ("右上", "右下", "左上", "左下")}
global_lockout_ts = 0.0

# ── 題庫 ──────────────────────────────────────────────────────
CORNER_ORDER = ["右上", "右下", "左上", "左下"]   # 固定的4個角落名
CORNER_NUM = {"左上": "1", "右上": "2", "左下": "3", "右下": "4"}  # 角落顯示數字

class Quiz:
    def __init__(self, csv_path="test.csv"):
        self.questions = []
        self.current   = None          # 當前題目 dict
        self.corner_map = {}           # { corner_label: option_text }
        self.answer_corner = None      # 正確答案所在角落
        self.result_text  = ""         # "答對了" / "答錯了" / ""
        self.result_ts    = 0.0        # 結果顯示時間戳
        self.result_color = (100, 220, 100)   # RGB

        self.has_csv = os.path.exists(csv_path)
        if self.has_csv:
            self._load(csv_path)
        else:
            print(f"⚠️  找不到 {csv_path}，純感應模式")

    def _load(self, path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                try:
                    ans = int(row[6].strip()) - 1   # 0-based
                except ValueError:
                    continue
                self.questions.append({
                    "no":      row[0].strip(),
                    "q":       row[1].strip(),
                    "options": [row[2].strip(), row[3].strip(),
                                row[4].strip(), row[5].strip()],
                    "ans":     ans,
                })
        print(f"✅ 載入 {len(self.questions)} 題")

    def next_question(self):
        """隨機抽一題，隨機排列選項到4個角落"""
        if not self.questions:
            return
        self.current     = random.choice(self.questions)
        self.result_text = ""
        options = self.current["options"]   # [opt1, opt2, opt3, opt4]
        ans_idx = self.current["ans"]       # 正確選項 0-based index
        # 角落隨機排列，選項照原本順序對應
        corners = CORNER_ORDER[:]
        random.shuffle(corners)
        self.corner_map    = {corners[i]: options[i] for i in range(4)}
        self.answer_corner = corners[ans_idx]

    def judge(self, triggered_corner: str):
        """判斷觸發角落是否正確"""
        if not self.current:
            return
        if triggered_corner == self.answer_corner:
            self.result_text  = "答對了！✓"
            self.result_color = (80, 220, 80)
        else:
            self.result_text  = "答錯了！✗"
            self.result_color = (220, 80, 80)
        self.result_ts = time.time()

    def win_title(self):
        if self.current:
            return f"Quiz - Q{self.current['no']}"
        return "Quiz"

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
def update_states(detected: set, now: float, quiz: Quiz):
    global global_lockout_ts

    if global_lockout_ts > 0.0:
        still = any(
            st["state"] == TRIGGERED and now - st["ts"] < LOCKOUT_SEC
            for st in corner_states.values()
        )
        if not still:
            global_lockout_ts = 0.0
            for st in corner_states.values():
                st["state"] = IDLE
                st["ts"]    = 0.0
            # 鎖定結束 → 下一題
            if quiz.has_csv:
                quiz.next_question()
        return

    any_triggered = False
    triggered_corner = None
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
                st["state"]      = TRIGGERED
                st["ts"]         = now
                any_triggered    = True
                triggered_corner = label

    if any_triggered:
        global_lockout_ts = now
        for st in corner_states.values():
            if st["state"] == HOLDING:
                st["state"] = IDLE
                st["ts"]    = 0.0
        # 判斷答題
        if quiz.has_csv and triggered_corner:
            quiz.judge(triggered_corner)

# ── PIL 工具 ──────────────────────────────────────────────────
def text_size(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]

def put_chinese_text(frame_bgr, text, pos, font, color_rgb,
                     bg_color_rgb=None, padding=8):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(img)
    tw, th = text_size(d, text, font)
    x, y   = pos
    if bg_color_rgb is not None:
        d.rectangle([x-padding, y-padding, x+tw+padding, y+th+padding],
                    fill=bg_color_rgb + (190,))
    d.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── 角落區塊 ──────────────────────────────────────────────────
def draw_corners(frame, now: float, quiz: Quiz, font):
    h, w  = frame.shape[:2]
    box_h = h // 5
    box_w = w // 4

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

        # 底色
        if state == TRIGGERED:
            bg = (220, 60, 60, 230)
        else:
            bg = (50, 100, 200, 230)
        draw.rectangle([rx1, ry1, rx2, ry2], fill=bg)

        # 外框 / 閃光
        if state == HOLDING:
            flash = int((now * 4) % 2) == 0
            draw.rectangle([rx1, ry1, rx2, ry2],
                           outline=(255, 50, 50, 255) if flash else (255, 160, 160, 180),
                           width=8 if flash else 5)
            elapsed  = now - st["ts"]
            bar_w    = int((rx2 - rx1) * min(elapsed / HOLD_SEC, 1.0))
            draw.rectangle([rx1, ry2-12, rx1+bar_w, ry2], fill=(255, 220, 0, 230))
        elif state == TRIGGERED:
            draw.rectangle([rx1, ry1, rx2, ry2],
                           outline=(255, 200, 200, 255), width=4)
            elapsed  = now - st["ts"]
            bar_w    = int((rx2 - rx1) * (1 - min(elapsed / LOCKOUT_SEC, 1.0)))
            draw.rectangle([rx1, ry2-12, rx1+bar_w, ry2], fill=(60, 200, 80, 230))
        else:
            draw.rectangle([rx1, ry1, rx2, ry2],
                           outline=(120, 160, 255, 200), width=2)

        # 文字：測驗模式顯示選項內容，否則顯示角落名
        if quiz.has_csv and quiz.current and label in quiz.corner_map:
            display = quiz.corner_map[label]
        else:
            display = CORNER_NUM.get(label, label)

        # 自動換行（超過區塊寬度）
        words   = list(display)
        lines   = []
        line    = ""
        max_w   = box_w - 16
        for ch in words:
            tw, _ = text_size(draw, line + ch, font)
            if tw > max_w and line:
                lines.append(line)
                line = ch
            else:
                line += ch
        if line:
            lines.append(line)

        total_h = sum(text_size(draw, l, font)[1] for l in lines) + 4 * (len(lines)-1)
        ty = ry1 + (box_h - total_h) // 2
        for ln in lines:
            lw, lh = text_size(draw, ln, font)
            tx = rx1 + (box_w - lw) // 2
            draw.text((tx, ty), ln, font=font, fill=(255, 255, 255, 255))
            ty += lh + 4

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 題目顯示（畫面中央）─────────────────────────────────────
def draw_question(frame, quiz: Quiz, font_q, font_no):
    if not quiz.has_csv or not quiz.current:
        return frame
    h, w = frame.shape[:2]
    text = quiz.current["q"]
    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    # 自動換行
    max_w = int(w * 0.55)
    words = list(text)
    lines, line = [], ""
    for ch in words:
        tw, _ = text_size(draw, line + ch, font_q)
        if tw > max_w and line:
            lines.append(line); line = ch
        else:
            line += ch
    if line:
        lines.append(line)

    line_h    = text_size(draw, "測", font_q)[1]
    total_h   = line_h * len(lines) + 8 * (len(lines) - 1)
    pad       = 20
    block_w   = max(text_size(draw, l, font_q)[0] for l in lines) + pad*2
    block_h   = total_h + pad*2
    bx        = (w - block_w) // 2
    by        = (h - block_h) // 2

    draw.rectangle([bx, by, bx+block_w, by+block_h], fill=(20, 20, 60, 200))
    draw.rectangle([bx, by, bx+block_w, by+block_h], outline=(100, 140, 255, 200), width=2)

    # 題號顯示在題目框左上角
    no_text = f"第 {quiz.current['no']} 題"
    draw.text((bx + pad, by + 6), no_text, font=font_no, fill=(180, 200, 255, 200))

    ty = by + pad
    for ln in lines:
        lw, lh = text_size(draw, ln, font_q)
        draw.text((bx + (block_w - lw)//2, ty), ln, font=font_q, fill=(255, 255, 200, 255))
        ty += lh + 8

    combined = Image.alpha_composite(img, ov).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 答題結果浮層 ──────────────────────────────────────────────
RESULT_SHOW_SEC = 3.0

def draw_result(frame, quiz: Quiz, font_result):
    if not quiz.result_text:
        return frame
    if time.time() - quiz.result_ts > RESULT_SHOW_SEC:
        quiz.result_text = ""
        return frame
    h, w = frame.shape[:2]
    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    tw, th = text_size(draw, quiz.result_text, font_result)
    pad    = 30
    bx     = (w - tw - pad*2) // 2
    by     = (h - th - pad*2) // 2
    r, g, b = quiz.result_color
    draw.rectangle([bx, by, bx+tw+pad*2, by+th+pad*2], fill=(r, g, b, 210))
    draw.rectangle([bx, by, bx+tw+pad*2, by+th+pad*2], outline=(255,255,255,200), width=3)
    draw.text((bx+pad, by+pad), quiz.result_text, font=font_result,
              fill=(255, 255, 255, 255))
    combined = Image.alpha_composite(img, ov).convert("RGB")
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
    tw, th  = text_size(draw, label, font_btn)
    pad, margin = 14, 16
    bx1 = (w - tw - pad*2) // 2
    bx2 = bx1 + tw + pad*2
    by2 = h - margin
    by1 = by2 - th - pad*2
    btn_rect[:] = [bx1, by1, bx2, by2]
    draw.rectangle([bx1+3, by1+3, bx2+3, by2+3], fill=(0,0,0,100))
    draw.rectangle([bx1, by1, bx2, by2], fill=btn_rgb+(220,))
    draw.rectangle([bx1, by1, bx2, by2], outline=(255,255,255,160), width=2)
    draw.text((bx1+pad, by1+pad), label, font=font_btn, fill=(255,255,255,255))
    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

# ── 主程式 ────────────────────────────────────────────────────
def main():
    global show_video

    download_model()

    quiz = Quiz("test.csv")
    if quiz.has_csv:
        quiz.next_question()

    h_frame, w_frame = 720, 1280
    font_small  = load_font(max(20, w_frame // 40))
    font_btn    = load_font(max(22, w_frame // 36))
    font_q      = load_font(max(56, w_frame // 14))   # 題目字型（原2倍）
    font_no     = load_font(max(18, w_frame // 60))        # 題號小字
    font_result = load_font(max(60, w_frame // 10))
    font_corner = load_font(max(20, (h_frame // 5) // 3))   # 角落選項字型，快取

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

    win_name = quiz.win_title() if quiz.has_csv else "Hand Position Detector"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("✅ 啟動！V 鍵切換影像；Q 鍵離開")
    frame_idx = 0

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        raw = cv2.flip(raw, 1)
        h_frame, w_frame = raw.shape[:2]
        now = time.time()

        # 偵測
        rgb    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or frame_idx * 33
        result = detector.detect_for_video(mp_img, ts_ms)

        raw_detected = set()
        if result.hand_landmarks and result.handedness:
            for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                side     = "右" if handedness_list[0].category_name == "Right" else "左"
                vertical = "上" if lm_list[0].y < 0.45 else "下"
                raw_detected.add(side + vertical)

        in_lockout = (global_lockout_ts > 0.0 and
                      now - global_lockout_ts < LOCKOUT_SEC)
        detected = set() if in_lockout else raw_detected

        update_states(detected, now, quiz)

        # 更新視窗標題（題號可能變）
        new_title = quiz.win_title() if quiz.has_csv else "Hand Position Detector"
        if new_title != win_name:
            cv2.setWindowTitle(win_name, new_title)

        # 畫面合成
        canvas = raw.copy() if show_video else np.zeros_like(raw)
        canvas = draw_corners(canvas, now, quiz, font_corner)
        canvas = draw_question(canvas, quiz, font_q, font_no)
        canvas = draw_result(canvas, quiz, font_result)
        canvas = draw_toggle_button(canvas, font_btn)

        # 狀態列（顯示數字）
        triggered_labels = [l for l, s in corner_states.items() if s["state"] == TRIGGERED]
        holding_labels   = [l for l, s in corner_states.items() if s["state"] == HOLDING]
        if triggered_labels:
            nums = "、".join(CORNER_NUM[l] for l in sorted(triggered_labels))
            status = "已選擇：" + nums
        elif holding_labels:
            nums = "、".join(CORNER_NUM[l] for l in sorted(holding_labels))
            status = "感應中：" + nums
        elif raw_detected:
            nums = "、".join(CORNER_NUM[l] for l in sorted(raw_detected))
            status = "偵測到：" + nums
        else:
            status = "請將手部移入畫面選擇答案"

        tmp  = ImageDraw.Draw(Image.new("RGBA", (1,1)))
        try:
            sb   = tmp.textbbox((0,0), status, font=font_small)
            s_tw = sb[2]-sb[0]; s_th = sb[3]-sb[1]
        except Exception:
            s_tw, s_th = len(status)*14, 24

        canvas = put_chinese_text(canvas, status,
                                  ((w_frame-s_tw)//2, btn_rect[1]-19-s_th),
                                  font_small, (220,220,220),
                                  bg_color_rgb=(30,30,30), padding=6)

        cv2.imshow(win_name, canvas)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        # 按 Q 或點擊視窗關閉按鈕都能結束
        if key == ord("q"):
            break
        elif key == 27:          # ESC
            break
        elif cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("v"):
            show_video = not show_video

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 程式結束")


if __name__ == "__main__":
    main()
