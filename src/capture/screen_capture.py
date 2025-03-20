"""
スクリーンキャプチャ機能を提供するモジュール。
"""
import time
import numpy as np
import cv2
from typing import Tuple, Optional
import mss
import mss.tools


import threading
from queue import Queue, Empty

class ScreenCapture:
    """
    スクリーンキャプチャの基本クラス。
    指定された領域の画面をキャプチャする機能を提供する。
    パフォーマンス向上のため、バックグラウンドスレッドでのキャプチャをサポート。
    
    Attributes:
        capture_area (dict): キャプチャする画面領域の座標情報
        last_frame_time (float): 最後にフレームをキャプチャした時間
        frame_rate (float): キャプチャのフレームレート
        use_threading (bool): スレッド処理を使用するかどうか
        _thread (threading.Thread): キャプチャ用のバックグラウンドスレッド
        _queue (Queue): キャプチャしたフレームを保存するキュー
        _running (bool): スレッドの実行状態
        _last_frame (np.ndarray): 最後にキャプチャしたフレーム
        _downsample (bool): ダウンサンプリングを行うかどうか
        _downsample_factor (float): ダウンサンプリングの倍率
    """
    
    def __init__(self, x: int = 0, y: int = 0, width: int = 1920, height: int = 1080,
                 use_threading: bool = True, queue_size: int = 2, downsample: bool = False,
                 downsample_factor: float = 0.5):
        """
        ScreenCaptureクラスの初期化。
        
        Args:
            x (int, optional): キャプチャ開始位置のX座標。デフォルトは0。
            y (int, optional): キャプチャ開始位置のY座標。デフォルトは0。
            width (int, optional): キャプチャ領域の幅。デフォルトは1920。
            height (int, optional): キャプチャ領域の高さ。デフォルトは1080。
            use_threading (bool, optional): スレッド処理を使用するかどうか。デフォルトはTrue。
            queue_size (int, optional): キャプチャキューのサイズ。デフォルトは2。
            downsample (bool, optional): ダウンサンプリングを行うかどうか。デフォルトはFalse。
            downsample_factor (float, optional): ダウンサンプリングの倍率。デフォルトは0.5。
        """
        self.set_capture_area(x, y, width, height)
        self.last_frame_time = 0
        self.frame_rate = 0
        self.sct = mss.mss()
        
        # スレッド関連の設定
        self.use_threading = use_threading
        self._thread = None
        self._queue = Queue(maxsize=queue_size)
        self._running = False
        self._last_frame = None
        
        # ダウンサンプリング設定
        self._downsample = downsample
        self._downsample_factor = downsample_factor
        
        # スレッド処理が有効な場合、キャプチャスレッドを開始
        if self.use_threading:
            self._start_capture_thread()
    
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
    
    def _start_capture_thread(self) -> None:
        """キャプチャ用のバックグラウンドスレッドを開始する。"""
        if self._thread is not None and self._thread.is_alive():
            self._running = False
            self._thread.join(timeout=1.0)
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()
    
    def _capture_thread(self) -> None:
        """
        バックグラウンドでフレームをキャプチャし続けるスレッド関数。
        キャプチャしたフレームをキューに保存する。
        """
        last_time = 0
        
        # スレッド固有のmssインスタンスを作成
        thread_sct = mss.mss()
        
        while self._running:
            try:
                # スレッド固有のインスタンスを使用
                sct_img = thread_sct.grab(self.capture_area)
                
                # numpy配列に変換（BGRA形式）
                frame = np.array(sct_img)
                
                # BGRに変換（OpenCV形式）
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # ダウンサンプリングが有効な場合は縮小
                if self._downsample:
                    new_width = int(frame.shape[1] * self._downsample_factor)
                    new_height = int(frame.shape[0] * self._downsample_factor)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # フレームレート計算
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if last_time > 0 else 0
                last_time = current_time
                
                # キューが満杯の場合は古いフレームを破棄
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                
                # 新しいフレームをキューに追加
                self._queue.put((frame, fps), block=False)
                
                # CPUリソースを浪費しないよう少し待機
                time.sleep(0.001)
                
            except Exception as e:
                print(f"キャプチャスレッドでエラーが発生: {str(e)}")
                time.sleep(0.1)  # エラー時はより長く待機
        
        # スレッド終了時にmssインスタンスをクリーンアップ
        thread_sct.close()
    
    def capture_frame(self) -> Tuple[np.ndarray, float]:
        """
        画面から1フレームをキャプチャする。
        スレッド処理が有効な場合はキューからフレームを取得し、
        そうでない場合は直接キャプチャを実行する。
        
        Returns:
            Tuple[np.ndarray, float]: キャプチャした画像とフレームレート
        """
        if self.use_threading:
            try:
                # キューからフレームを取得（ブロックしない）
                frame, fps = self._queue.get_nowait()
                self._last_frame = frame
                self.frame_rate = fps
                return frame, fps
            except Empty:
                # キューが空の場合は最後のフレームを再利用
                if self._last_frame is not None:
                    return self._last_frame, self.frame_rate
        
        # スレッド処理が無効または最初のフレームの場合は直接キャプチャ
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
        
        # ダウンサンプリングが有効な場合は縮小
        if self._downsample:
            new_width = int(frame.shape[1] * self._downsample_factor)
            new_height = int(frame.shape[0] * self._downsample_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        self._last_frame = frame
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
    パフォーマンス向上のためキャッシュや最適化機能を追加。
    
    Attributes:
        reel_area (Optional[Tuple[int, int, int, int]]): 検出されたリール領域（x, y, width, height）
        last_detection_time (float): 最後にリール領域を検出した時間
        detection_timeout (float): リール領域検出のタイムアウト時間（秒）
        detection_interval (float): リール検出を実行する間隔（秒）
        last_frame_size (Tuple[int, int]): 最後に処理したフレームのサイズ
        clahe (cv2.CLAHE): コントラスト強調のためのCLAHEオブジェクト
        _preprocess_cache (dict): 前処理結果のキャッシュ
        _frame_hash (int): 最後に処理したフレームのハッシュ値
        _detection_count (int): 検出試行回数のカウンター
        _total_detection_time (float): 検出処理の合計時間
    """
    
    def __init__(self, detection_interval: float = 5.0):
        """
        VideoProcessorクラスの初期化。
        
        Args:
            detection_interval (float, optional): リール検出を実行する間隔（秒）。デフォルトは5.0。
        """
        self.reel_area = None
        self.last_detection_time = 0
        self.detection_timeout = 3.0  # 検出タイムアウト時間（秒）
        self.detection_interval = detection_interval
        self.last_frame_size = None
        
        # 前処理用のCLAHEオブジェクトを事前に作成
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # キャッシュと統計情報
        self._preprocess_cache = {}
        self._frame_hash = 0
        self._detection_count = 0
        self._total_detection_time = 0
    
    def preprocess(self, frame: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        キャプチャした画像の前処理を行う。
        ノイズ除去や明度調整などを実施。
        パフォーマンス向上のためキャッシュを使用。
        
        Args:
            frame (np.ndarray): 前処理を行う画像
            use_cache (bool, optional): キャッシュを使用するかどうか。デフォルトはTrue。
            
        Returns:
            np.ndarray: 前処理後の画像
        """
        # フレームサイズをチェック
        if self.last_frame_size != frame.shape[:2]:
            self.last_frame_size = frame.shape[:2]
            self._preprocess_cache = {}  # サイズ変更時はキャッシュをクリア
        
        # キャッシュ使用時、同一フレームならキャッシュを返す
        if use_cache:
            # フレームデータからハッシュ値を計算（シンプルなハッシング）
            frame_hash = hash(frame.tobytes()[:1000]) # 先頭部分だけを使用
            
            if frame_hash == self._frame_hash and frame_hash in self._preprocess_cache:
                return self._preprocess_cache[frame_hash]
            
            self._frame_hash = frame_hash
        
        # グレースケール変換（cvtColorは計算コストが高いため、チャンネル数を確認）
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # ノイズ除去（高速化のため小さいカーネルサイズを使用）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # コントラスト強調（事前初期化したCLAHEを使用）
        enhanced = self.clahe.apply(blurred)
        
        # キャッシュに保存
        if use_cache:
            self._preprocess_cache[self._frame_hash] = enhanced
            
            # キャッシュサイズの制限（5フレーム以上は保持しない）
            if len(self._preprocess_cache) > 5:
                oldest_key = next(iter(self._preprocess_cache))
                del self._preprocess_cache[oldest_key]
        
        return enhanced
    
    def detect_reel_area(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        画像内のパチスロリール領域を自動検出する。
        エッジ検出とコンター解析を使用。
        パフォーマンス計測と最適化機能を追加。
        
        Args:
            frame (np.ndarray): リール領域を検出する画像
            
        Returns:
            Optional[Tuple[int, int, int, int]]: 検出されたリール領域 (x, y, width, height) または None
        """
        # 検出回数をカウント
        self._detection_count += 1
        start_time = time.time()
        
        # 前処理（低解像度で処理しパフォーマンス向上）
        scale_factor = 0.5
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_AREA)
        
        preprocessed = self.preprocess(small_frame, use_cache=False)
        
        # エッジ検出（低しきい値・高しきい値を調整して効率化）
        edges = cv2.Canny(preprocessed, 30, 120)
        
        # 膨張処理でエッジを強調（反復回数を1回に制限）
        kernel = np.ones((3, 3), np.uint8)  # カーネルサイズを小さく
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 輪郭検出（階層情報を省略して高速化）
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 面積が大きい矩形を探す
        max_area = 0
        max_rect = None
        
        # フレームサイズに基づいて最小面積を計算
        min_area = small_frame.shape[0] * small_frame.shape[1] * 0.05
        
        for contour in contours:
            # 小さい輪郭はスキップして処理を削減
            if cv2.contourArea(contour) < min_area:
                continue
                
            # 輪郭を矩形に近似
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 一定の面積と縦横比を持つものだけを対象にする
            aspect_ratio = w / h if h > 0 else 0
            
            # リールらしい矩形の条件
            # 1. 一定以上の面積
            # 2. 縦長の矩形（縦横比が0.7以下に緩和）
            if area > min_area and aspect_ratio < 0.7 and area > max_area:
                max_area = area
                # 元のサイズに戻す
                max_rect = (int(x / scale_factor), int(y / scale_factor),
                           int(w / scale_factor), int(h / scale_factor))
        
        # パフォーマンス計測
        end_time = time.time()
        detection_time = end_time - start_time
        self._total_detection_time += detection_time
        
        # 平均検出時間を計算（デバッグ用）
        avg_detection_time = self._total_detection_time / self._detection_count
        
        # 検出結果を更新
        self.reel_area = max_rect
        return max_rect
    
    def extract_reel_area(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        フレームからリール領域を抽出する。
        リール領域が未検出の場合は自動検出を試みる。
        前回の検出から一定時間内であれば、検出に失敗しても前回の領域を使用する。
        パフォーマンス向上のため、検出頻度を削減。
        
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
            # 検出間隔を長くして、一定間隔でのみ再検出を試みる
            if current_time - self.last_detection_time > self.detection_interval:
                new_area = self.detect_reel_area(frame)
                if new_area is not None:
                    self.last_detection_time = current_time
        
        # リール領域が存在し、かつタイムアウト時間内であれば領域を抽出
        if self.reel_area is not None and (current_time - self.last_detection_time <= self.detection_timeout):
            x, y, w, h = self.reel_area
            
            # フレームの範囲チェック
            if (x >= 0 and y >= 0 and
                x + w <= frame.shape[1] and y + h <= frame.shape[0] and
                w > 0 and h > 0):
                return frame[y:y+h, x:x+w]
        
        # 前回の検出から時間が経過し過ぎている場合は検出結果をリセット
        if self.reel_area is not None and (current_time - self.last_detection_time > self.detection_timeout):
            self.reel_area = None
        
        return None
