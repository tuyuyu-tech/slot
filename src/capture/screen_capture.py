"""
スクリーンキャプチャ機能を提供するモジュール。
"""
import time
import numpy as np
import cv2
from typing import Tuple, Optional
import mss
import mss.tools


class ScreenCapture:
    """
    スクリーンキャプチャの基本クラス。
    指定された領域の画面をキャプチャする機能を提供する。
    
    Attributes:
        capture_area (dict): キャプチャする画面領域の座標情報
        last_frame_time (float): 最後にフレームをキャプチャした時間
        frame_rate (float): キャプチャのフレームレート
    """
    
    def __init__(self, x: int = 0, y: int = 0, width: int = 1920, height: int = 1080):
        """
        ScreenCaptureクラスの初期化。
        
        Args:
            x (int, optional): キャプチャ開始位置のX座標。デフォルトは0。
            y (int, optional): キャプチャ開始位置のY座標。デフォルトは0。
            width (int, optional): キャプチャ領域の幅。デフォルトは1920。
            height (int, optional): キャプチャ領域の高さ。デフォルトは1080。
        """
        self.set_capture_area(x, y, width, height)
        self.last_frame_time = 0
        self.frame_rate = 0
        self.sct = mss.mss()
    
    def set_capture_area(self, x: int, y: int, width: int, height: int) -> None:
        """
        キャプチャする画面領域を設定する。
        
        Args:
            x (int): キャプチャ開始位置のX座標
            y (int): キャプチャ開始位置のY座標
            width (int): キャプチャ領域の幅
            height (int): キャプチャ領域の高さ
        """
        self.capture_area = {
            'left': x,
            'top': y,
            'width': width,
            'height': height
        }
    
    def capture_frame(self) -> Tuple[np.ndarray, float]:
        """
        画面から1フレームをキャプチャする。
        
        Returns:
            Tuple[np.ndarray, float]: キャプチャした画像とフレームレート
        """
        # フレーム間の時間を計算してフレームレートを更新
        current_time = time.time()
        if self.last_frame_time > 0:
            time_diff = current_time - self.last_frame_time
            self.frame_rate = 1.0 / time_diff if time_diff > 0 else 0
        self.last_frame_time = current_time
        
        # スクリーンキャプチャを実行
        sct_img = self.sct.grab(self.capture_area)
        
        # numpy配列に変換（BGRA形式）
        frame = np.array(sct_img)
        
        # BGRに変換（OpenCV形式）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        return frame, self.frame_rate
    
    def get_frame_rate(self) -> float:
        """
        現在のフレームレートを取得する。
        
        Returns:
            float: 現在のフレームレート（FPS）
        """
        return self.frame_rate


class VideoProcessor:
    """
    キャプチャした画像の前処理を行うクラス。
    リール領域の検出や画像の前処理などを提供する。
    
    Attributes:
        reel_area (Optional[Tuple[int, int, int, int]]): 検出されたリール領域（x, y, width, height）
        last_detection_time (float): 最後にリール領域を検出した時間
        detection_timeout (float): リール領域検出のタイムアウト時間（秒）
    """
    
    def __init__(self):
        """VideoProcessorクラスの初期化。"""
        self.reel_area = None
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # 検出タイムアウト時間（秒）
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        キャプチャした画像の前処理を行う。
        ノイズ除去や明度調整などを実施。
        
        Args:
            frame (np.ndarray): 前処理を行う画像
            
        Returns:
            np.ndarray: 前処理後の画像
        """
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去（ガウシアンブラー）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_reel_area(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        画像内のパチスロリール領域を自動検出する。
        エッジ検出とコンター解析を使用。
        
        Args:
            frame (np.ndarray): リール領域を検出する画像
            
        Returns:
            Optional[Tuple[int, int, int, int]]: 検出されたリール領域 (x, y, width, height) または None
        """
        # 前処理
        preprocessed = self.preprocess(frame)
        
        # エッジ検出
        edges = cv2.Canny(preprocessed, 50, 150)
        
        # 膨張・収縮処理でエッジを強調
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # 輪郭検出
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 面積が大きい矩形を探す
        max_area = 0
        max_rect = None
        
        for contour in contours:
            # 輪郭を矩形に近似
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 一定の面積と縦横比を持つものだけを対象にする
            aspect_ratio = w / h if h > 0 else 0
            
            # リールらしい矩形の条件
            # 1. ある程度の大きさ（画面の5%以上）
            # 2. 縦長の矩形（縦横比が0.5以下）
            min_area = frame.shape[0] * frame.shape[1] * 0.05
            if area > min_area and aspect_ratio < 0.5 and area > max_area:
                max_area = area
                max_rect = (x, y, w, h)
        
        self.reel_area = max_rect
        return max_rect
    
    def extract_reel_area(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        フレームからリール領域を抽出する。
        リール領域が未検出の場合は自動検出を試みる。
        前回の検出から一定時間内であれば、検出に失敗しても前回の領域を使用する。
        
        Args:
            frame (np.ndarray): 元の画像フレーム
            
        Returns:
            Optional[np.ndarray]: 抽出されたリール領域の画像、検出できない場合はNone
        """
        current_time = time.time()
        
        # リール領域が未検出の場合は自動検出を試みる
        if self.reel_area is None:
            if self.detect_reel_area(frame) is not None:
                self.last_detection_time = current_time
        else:
            # 一定間隔で再検出を試みる（現在の検出を維持しながら）
            if current_time - self.last_detection_time > self.detection_timeout:
                new_area = self.detect_reel_area(frame)
                if new_area is not None:
                    self.last_detection_time = current_time
        
        # リール領域が存在し、かつタイムアウト時間内であれば領域を抽出
        if self.reel_area is not None and (current_time - self.last_detection_time <= self.detection_timeout):
            x, y, w, h = self.reel_area
            return frame[y:y+h, x:x+w]
        
        # 前回の検出から時間が経過し過ぎている場合は検出結果をリセット
        if self.reel_area is not None and (current_time - self.last_detection_time > self.detection_timeout):
            self.reel_area = None
        
        return None
