"""
パチスロリール領域の自動キャリブレーション機能を提供するモジュール。
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReelCalibrator:
    """
    パチスロリール領域を自動的にキャリブレーションするクラス。
    画像処理技術を使用してリール位置を特定する。
    
    Attributes:
        min_reel_height_ratio (float): 画面高さに対するリール高さの最小比率
        max_reel_height_ratio (float): 画面高さに対するリール高さの最大比率
        min_reel_width_ratio (float): 画面幅に対するリール幅の最小比率
        max_reel_width_ratio (float): 画面幅に対するリール幅の最大比率
        min_reel_aspect_ratio (float): リールの最小アスペクト比（高さ/幅）
        max_reel_aspect_ratio (float): リールの最大アスペクト比（高さ/幅）
        history_size (int): 検出結果の履歴サイズ
        position_history (List): 検出されたリール位置の履歴
    """
    
    def __init__(self,
                 min_reel_height_ratio: float = 0.5,
                 max_reel_height_ratio: float = 0.9,
                 min_reel_width_ratio: float = 0.6,
                 max_reel_width_ratio: float = 0.95,
                 min_reel_aspect_ratio: float = 1.0,
                 max_reel_aspect_ratio: float = 3.0,
                 history_size: int = 10):
        """
        ReelCalibratorクラスの初期化。
        
        Args:
            min_reel_height_ratio (float): 画面高さに対するリール高さの最小比率
            max_reel_height_ratio (float): 画面高さに対するリール高さの最大比率
            min_reel_width_ratio (float): 画面幅に対するリール幅の最小比率
            max_reel_width_ratio (float): 画面幅に対するリール幅の最大比率
            min_reel_aspect_ratio (float): リールの最小アスペクト比（高さ/幅）
            max_reel_aspect_ratio (float): リールの最大アスペクト比（高さ/幅）
            history_size (int): 検出結果の履歴サイズ
        """
        self.min_reel_height_ratio = min_reel_height_ratio
        self.max_reel_height_ratio = max_reel_height_ratio
        self.min_reel_width_ratio = min_reel_width_ratio
        self.max_reel_width_ratio = max_reel_width_ratio
        self.min_reel_aspect_ratio = min_reel_aspect_ratio
        self.max_reel_aspect_ratio = max_reel_aspect_ratio
        self.history_size = history_size
        self.position_history = []
    
    def detect_reel_area(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        フレーム内のリール領域を検出する。
        
        Args:
            frame (np.ndarray): 検出対象のフレーム画像
        
        Returns:
            Optional[Tuple[int, int, int, int]]: 検出されたリール領域 (x, y, width, height) または None
        """
        # グレースケールに変換
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 画像の前処理
        # ノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # エッジ検出（Cannyエッジ検出アルゴリズム）
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨張処理でエッジを強調
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 輪郭検出
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 画像サイズを取得
        height, width = frame.shape[:2]
        
        # 条件に合う矩形を探す
        candidates = []
        
        for contour in contours:
            # 輪郭を矩形に近似
            x, y, w, h = cv2.boundingRect(contour)
            
            # リールらしい矩形の条件
            aspect_ratio = h / w if w > 0 else 0
            height_ratio = h / height
            width_ratio = w / width
            
            # 条件チェック
            if (self.min_reel_height_ratio <= height_ratio <= self.max_reel_height_ratio and
                self.min_reel_width_ratio <= width_ratio <= self.max_reel_width_ratio and
                self.min_reel_aspect_ratio <= aspect_ratio <= self.max_reel_aspect_ratio):
                
                # スコア計算（サイズと縦横比から）
                size_score = (h * w) / (height * width)  # 相対的なサイズ
                aspect_score = min(1.0, aspect_ratio / 2.0)  # 縦長ほど高いスコア
                score = size_score * aspect_score
                
                candidates.append((x, y, w, h, score))
        
        # スコア順にソート
        candidates.sort(key=lambda c: c[4], reverse=True)
        
        # 最もスコアの高い候補を選択
        if candidates:
            x, y, w, h, _ = candidates[0]
            detected_area = (x, y, w, h)
            
            # 検出結果を履歴に追加
            self.position_history.append(detected_area)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)
            
            return detected_area
        
        return None
    
    def detect_reel_divisions(self, 
                              frame: np.ndarray, 
                              reel_area: Tuple[int, int, int, int], 
                              reel_count: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        リール領域内の個別リールの分割位置を検出する。
        
        Args:
            frame (np.ndarray): フレーム画像
            reel_area (Tuple[int, int, int, int]): リール全体の領域 (x, y, width, height)
            reel_count (int): リールの数（デフォルトは3）
        
        Returns:
            List[Tuple[int, int, int, int]]: 各リールの位置 [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
        """
        x, y, w, h = reel_area
        
        # リール領域を切り出し
        reel_region = frame[y:y+h, x:x+w]
        
        # リール間の区切りを検出する（グレースケールで）
        if len(reel_region.shape) == 3:
            gray_region = cv2.cvtColor(reel_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = reel_region
        
        # 垂直方向の輝度プロファイルを計算
        # （横方向に平均をとり、垂直方向のプロファイルを得る）
        v_profile = np.mean(gray_region, axis=0)
        
        # プロファイルを平滑化
        v_profile_smooth = cv2.GaussianBlur(v_profile, (5, 1), 0)
        
        # プロファイルの微分を計算（変化点を見つけるため）
        v_profile_diff = np.diff(v_profile_smooth)
        
        # 閾値を計算（適応的な閾値）
        threshold = np.std(v_profile_diff) * 1.5
        
        # 変化点を検出
        change_points = []
        for i in range(1, len(v_profile_diff)):
            if abs(v_profile_diff[i] - v_profile_diff[i-1]) > threshold:
                change_points.append(i)
        
        # 区切り位置のグループ化（近い位置の統合）
        grouped_points = []
        if change_points:
            current_group = [change_points[0]]
            
            for point in change_points[1:]:
                if point - current_group[-1] < 10:  # 閾値（近傍とみなす距離）
                    current_group.append(point)
                else:
                    # グループの代表点（中央値）を追加
                    grouped_points.append(int(np.median(current_group)))
                    current_group = [point]
            
            # 最後のグループを追加
            if current_group:
                grouped_points.append(int(np.median(current_group)))
        
        # リール数で均等に分割（検出に失敗した場合のフォールバック）
        if len(grouped_points) < reel_count - 1:
            # 均等分割
            grouped_points = [int(w * (i + 1) / reel_count) for i in range(reel_count - 1)]
        elif len(grouped_points) > reel_count - 1:
            # 余分な区切りを除外（端から等間隔に選択）
            step = len(grouped_points) / (reel_count - 1)
            grouped_points = [grouped_points[int(i * step)] for i in range(reel_count - 1)]
        
        # 個別リールの領域を計算
        reel_divisions = []
        
        # 最初のリール
        first_reel = (x, y, grouped_points[0], h)
        reel_divisions.append(first_reel)
        
        # 中間のリール
        for i in range(len(grouped_points) - 1):
            reel_x = x + grouped_points[i]
            reel_w = grouped_points[i+1] - grouped_points[i]
            reel_divisions.append((reel_x, y, reel_w, h))
        
        # 最後のリール
        last_x = x + grouped_points[-1]
        last_w = w - grouped_points[-1]
        reel_divisions.append((last_x, y, last_w, h))
        
        return reel_divisions
    
    def get_stable_reel_area(self) -> Optional[Tuple[int, int, int, int]]:
        """
        履歴から安定したリール領域を取得する。
        
        Returns:
            Optional[Tuple[int, int, int, int]]: 安定したリール領域 (x, y, width, height) または None
        """
        if not self.position_history:
            return None
        
        # 各座標の中央値を計算
        x_values = [pos[0] for pos in self.position_history]
        y_values = [pos[1] for pos in self.position_history]
        w_values = [pos[2] for pos in self.position_history]
        h_values = [pos[3] for pos in self.position_history]
        
        stable_x = int(np.median(x_values))
        stable_y = int(np.median(y_values))
        stable_w = int(np.median(w_values))
        stable_h = int(np.median(h_values))
        
        return (stable_x, stable_y, stable_w, stable_h)
    
    def calibrate(self, 
                  frames: List[np.ndarray], 
                  reel_count: int = 3) -> Dict[str, Any]:
        """
        複数フレームを使用してリール領域をキャリブレーションする。
        
        Args:
            frames (List[np.ndarray]): キャリブレーションに使用するフレーム画像のリスト
            reel_count (int): リールの数（デフォルトは3）
        
        Returns:
            Dict[str, Any]: キャリブレーション結果
                {
                    'reel_area': (x, y, width, height),  # リール全体の領域
                    'reel_divisions': [(x1, y1, w1, h1), ...],  # 各リールの領域
                    'confidence': float  # 検出結果の信頼度（0.0〜1.0）
                }
        """
        # 履歴をクリア
        self.position_history = []
        
        # 各フレームでリール領域を検出
        for frame in frames:
            self.detect_reel_area(frame)
        
        # 安定したリール領域を取得
        stable_area = self.get_stable_reel_area()
        
        if stable_area is None:
            logger.warning("リール領域の検出に失敗しました")
            return {
                'reel_area': None,
                'reel_divisions': None,
                'confidence': 0.0
            }
        
        # 最後のフレームでリール分割を検出
        last_frame = frames[-1]
        reel_divisions = self.detect_reel_divisions(last_frame, stable_area, reel_count)
        
        # 検出結果の信頼度を計算
        # 安定性（分散）から信頼度を計算
        if len(self.position_history) > 1:
            x_std = np.std([pos[0] for pos in self.position_history])
            y_std = np.std([pos[1] for pos in self.position_history])
            w_std = np.std([pos[2] for pos in self.position_history])
            h_std = np.std([pos[3] for pos in self.position_history])
            
            # 標準偏差が小さいほど安定している
            normalized_std = (x_std + y_std + w_std + h_std) / (stable_area[2] + stable_area[3])
            confidence = max(0.0, min(1.0, 1.0 - normalized_std))
        else:
            confidence = 0.5  # デフォルト値
        
        logger.info(f"リール領域をキャリブレーションしました: {stable_area}, 信頼度: {confidence:.2f}")
        
        return {
            'reel_area': stable_area,
            'reel_divisions': reel_divisions,
            'confidence': confidence
        }


class AutoCalibrationDialog:
    """
    自動キャリブレーションを実行するためのダイアログ。
    キャプチャとキャリブレーションのロジックを提供する。
    
    Attributes:
        screen_capture: スクリーンキャプチャオブジェクト
        calibrator (ReelCalibrator): リールキャリブレーターオブジェクト
        frames (List[np.ndarray]): キャプチャされたフレームのリスト
        calibration_result (Dict[str, Any]): キャリブレーション結果
    """
    
    def __init__(self, screen_capture, max_frames: int = 10):
        """
        AutoCalibrationDialogクラスの初期化。
        
        Args:
            screen_capture: スクリーンキャプチャオブジェクト
            max_frames (int): キャリブレーションに使用する最大フレーム数
        """
        self.screen_capture = screen_capture
        self.calibrator = ReelCalibrator()
        self.frames = []
        self.max_frames = max_frames
        self.calibration_result = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        1フレームをキャプチャする。
        
        Returns:
            Optional[np.ndarray]: キャプチャしたフレーム画像または None
        """
        try:
            frame, _ = self.screen_capture.capture_frame()
            return frame
        except Exception as e:
            logger.error(f"フレームのキャプチャに失敗しました: {str(e)}")
            return None
    
    def start_calibration(self) -> None:
        """
        キャリブレーションプロセスを開始する。
        """
        # フレームリストをクリア
        self.frames = []
        self.calibration_result = None
        
        logger.info("自動キャリブレーションを開始します")
    
    def add_calibration_frame(self) -> bool:
        """
        キャリブレーション用のフレームを追加する。
        
        Returns:
            bool: フレーム追加が成功したかどうか
        """
        frame = self.capture_frame()
        
        if frame is not None:
            self.frames.append(frame)
            logger.debug(f"キャリブレーションフレームを追加しました: {len(self.frames)}/{self.max_frames}")
            return True
        
        return False
    
    def is_calibration_complete(self) -> bool:
        """
        キャリブレーションプロセスが完了したかどうかをチェックする。
        
        Returns:
            bool: キャリブレーションが完了したかどうか
        """
        return len(self.frames) >= self.max_frames
    
    def finish_calibration(self, reel_count: int = 3) -> Dict[str, Any]:
        """
        キャリブレーションを完了し、結果を取得する。
        
        Args:
            reel_count (int): リールの数（デフォルトは3）
        
        Returns:
            Dict[str, Any]: キャリブレーション結果
        """
        if not self.frames:
            logger.warning("キャリブレーションフレームがありません")
            return {
                'reel_area': None,
                'reel_divisions': None,
                'confidence': 0.0
            }
        
        # キャリブレーションを実行
        result = self.calibrator.calibrate(self.frames, reel_count)
        self.calibration_result = result
        
        logger.info(f"キャリブレーションが完了しました: {result['confidence']:.2f}")
        
        return result
    
    def get_last_frame_with_overlay(self) -> Optional[np.ndarray]:
        """
        最後のフレームにキャリブレーション結果をオーバーレイして返す。
        
        Returns:
            Optional[np.ndarray]: オーバーレイされたフレーム画像または None
        """
        if not self.frames or self.calibration_result is None:
            return None
        
        last_frame = self.frames[-1].copy()
        
        # リール全体の領域を描画
        reel_area = self.calibration_result['reel_area']
        if reel_area:
            x, y, w, h = reel_area
            cv2.rectangle(last_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(last_frame, "Reel Area", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 各リールの区切りを描画
        reel_divisions = self.calibration_result['reel_divisions']
        if reel_divisions:
            for i, (rx, ry, rw, rh) in enumerate(reel_divisions):
                cv2.rectangle(last_frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 1)
                cv2.putText(last_frame, f"Reel {i+1}", (rx, ry-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 信頼度を表示
        confidence = self.calibration_result['confidence']
        cv2.putText(last_frame, f"Confidence: {confidence:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return last_frame
