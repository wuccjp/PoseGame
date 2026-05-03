# 手掌偵測與實心綠圓進度條

此專案使用 Vanilla JS 與 MediaPipe Hands 透過 CDN 將攝影機畫面與手部偵測結果繪製於畫布上。畫面上有四個實心綠色圓（A、B、C、D），當手部靠近某個綠色圓時，外圈出現 0–3 秒進度條；若觸碰持續滿 3 秒，該綠色圓會變為該手的顏色（左手黃，右手紅），並在 3 秒後回復為綠色。

專案結構
- C:\test01\index.html  – 主頁 HTML
- C:\test01\styles.css  – 基礎樣式
- C:\test01\main.js      – 主要邏輯與繪圖
- C:\test01\README.md    – 本檔（本說明）

功能要點
- 四個綠色圓：A、B、C、D，大小以螢幕高度的 1/10 作為半徑
- 左手出現黃色外圈，右手出現紅色外圈，並在圓周圍繪製進度條
- 觸碰時，進度條為實時更新；達成 3 秒後，綠色圓改為碰觸顏色，3 秒後回復綠色
- 圓圈內保留字母 A、B、C、D，方便辨識與對齊

快速開始
1) 啟動本機伺服器（建議使用本地靜態伺服器，避免 file:// 啟動限制）
- 使用 Python：
  - 進入專案資料夾
  - python -m http.server 8000
- 使用 http-server（Node.js）
  - 安裝：npm i -g http-server
  - 進入專案資料夾
  - http-server -p 8000
2) 開啟瀏覽器，訪問 http://localhost:8000/index.html
3) 允許相機權限，測試手掌偵測與圓形效果

可自訂與進階
- main.js 中的綠色圓半徑為 canvas.height / 10，若要調整大小，修改此值
- 觸碰條件與時長為固定的 3 秒；可依需求修改延遲時間與動畫效果
- 進度條樣式、顏色或動畫可於 main.js 的繪製邏輯中調整

疑難排解
- 若頁面無法顯示圓形，請先檢查瀏覽器控制台是否有錯誤訊息，特別是關於 Hands 載入或 getUserMedia 權限的訊息
- 確認網路可以載入 Mediapipe Hands 的 CDN（https://cdn.jsdelivr.net/npm/@mediapipe/hands/）
- 建議在本機伺服器執行，而非直接打開檔案

參考與引用
- Mediapipe Hands via CDN
- 本專案僅使用前端的畫布繪製，不包含後端服務
