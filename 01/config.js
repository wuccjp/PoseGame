// 基本設定，未來可在 UI 中調整並儲存至 localStorage
export const CONFIG = {
  maxNumHands: 2,
  minDetectionConfidence: 0.8,
  minTrackingConfidence: 0.5,
  palmToRadiusScale: 0.9, // 半徑與 palmWidth 的乘數 (保留以便日後調整)
  leftColor: 'rgba(255,0,0,0.65)', // 左手顏色
  rightColor: 'rgba(255,204,0,0.65)', // 右手顏色
  circleOpacity: 0.65
};
