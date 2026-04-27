# 建立虛擬環境
## 1. 建立虛擬環境（在專案資料夾內）
python -m venv venv

## 2. 啟動虛擬環境
### Windows：
venv\Scripts\activate

### macOS / Linux：
source venv/bin/activate

## 3. 確認已啟動（提示符會出現 (venv)）


# 安裝套件
pip install opencv-python mediapipe Pillow

# 建立 requirements.txt（方便日後重建環境）
pip freeze > requirements.txt

## 或者手動建立內容更簡潔的版本：
opencv-python
mediapipe
Pillow

## 日後重建環境
python -m venv venv
source venv/bin/activate      # Windows 用 venv\Scripts\activate
pip install -r requirements.txt

# 退出虛擬環境
deactivate

# 本專案用到的套件說明：
以下是完整的建立步驟：

## 建立虛擬環境

```bash
# 1. 建立虛擬環境（在專案資料夾內）
python -m venv venv

# 2. 啟動虛擬環境
# Windows：
venv\Scripts\activate

# macOS / Linux：
source venv/bin/activate

# 3. 確認已啟動（提示符會出現 (venv)）
```

## 安裝套件

```bash
pip install opencv-python mediapipe Pillow
```

## 建立 requirements.txt（方便日後重建環境）

```bash
pip freeze > requirements.txt
```

或者手動建立內容更簡潔的版本：

```
opencv-python
mediapipe
Pillow
```

## 日後重建環境

```bash
python -m venv venv
source venv/bin/activate      # Windows 用 venv\Scripts\activate
pip install -r requirements.txt
```

## 退出虛擬環境

```bash
deactivate
```

---

**本專案用到的套件說明：**

| 套件 | 用途 |
|---|---|
| `opencv-python` | 攝影機擷取、視窗顯示、影像處理 |
| `mediapipe` | 手部關節點偵測（Google MediaPipe 0.10+ Tasks API） |
| `Pillow` | 繪製繁體中文字型（OpenCV 不支援中文）|

> `csv`、`random`、`os`、`sys`、`time` 都是 Python 內建模組，不需額外安裝。
