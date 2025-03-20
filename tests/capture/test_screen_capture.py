"""
screen_captureモジュールのテスト。
"""
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from src.capture.screen_capture import ScreenCapture, VideoProcessor

class TestScreenCapture:
    """ScreenCaptureクラスのテスト"""
    
    def test_init(self):
        """初期化処理のテスト"""
        capture = ScreenCapture()
        
        # デフォルト値の確認
        assert capture.capture_area['left'] == 0
        assert capture.capture_area['top'] == 0
        assert capture.capture_area['width'] == 1920
        assert capture.capture_area['height'] == 1080
        assert capture.last_frame_time == 0
        assert capture.frame_rate == 0
    
    def test_set_capture_area(self):
        """キャプチャ領域設定のテスト"""
        capture = ScreenCapture()
        capture.set_capture_area(100, 200, 640, 480)
        
        assert capture.capture_area['left'] == 100
        assert capture.capture_area['top'] == 200
        assert capture.capture_area['width'] == 640
        assert capture.capture_area['height'] == 480
    
    @patch('mss.mss')
    def test_capture_frame(self, mock_mss):
        """フレームキャプチャのテスト"""
        # モックの設定
        mock_sct = MagicMock()
        mock_mss.return_value = mock_sct
        
        # テスト用の画像データを用意
        test_img = np.zeros((100, 100, 4), dtype=np.uint8)  # BGRA形式
        mock_sct.grab.return_value = test_img
        
        # テスト対象のメソッド実行
        capture = ScreenCapture()
        frame, fps = capture.capture_frame()
        
        # 期待される結果の確認
        mock_sct.grab.assert_called_once_with(capture.capture_area)
        assert frame.shape[:2] == test_img.shape[:2]  # サイズが一致
        assert frame.shape[2] == 3  # チャンネル数がBGRA→BGRに変換されていること
        
        # 2回目の呼び出しでfpsが計算されること
        frame, fps = capture.capture_frame()
        assert fps > 0
    
    def test_get_frame_rate(self):
        """フレームレート取得のテスト"""
        capture = ScreenCapture()
        # 初期値は0
        assert capture.get_frame_rate() == 0
        
        # 手動で値を設定
        capture.frame_rate = 30
        assert capture.get_frame_rate() == 30


class TestVideoProcessor:
    """VideoProcessorクラスのテスト"""
    
    def test_init(self):
        """初期化処理のテスト"""
        processor = VideoProcessor()
        assert processor.reel_area is None
    
    def test_preprocess(self, test_image):
        """前処理のテスト"""
        processor = VideoProcessor()
        processed = processor.preprocess(test_image)
        
        # 処理結果がグレースケールであること
        assert len(processed.shape) == 2
        # 元の画像と同じサイズであること
        assert processed.shape[:2] == test_image.shape[:2]
    
    def test_detect_reel_area(self, test_image, reel_image):
        """リール領域検出のテスト"""
        processor = VideoProcessor()
        
        # リール画像でテスト
        result = processor.detect_reel_area(reel_image)
        
        # リール領域が検出されるべき
        assert result is not None
        assert len(result) == 4  # (x, y, width, height)の形式
        assert all(isinstance(val, int) for val in result)
        
        # 検出結果がreel_areaに保存されていること
        assert processor.reel_area == result
    
    def test_extract_reel_area(self, reel_image):
        """リール領域抽出のテスト"""
        processor = VideoProcessor()
        
        # 事前にリール領域を検出
        processor.detect_reel_area(reel_image)
        
        # リール領域を抽出
        extracted = processor.extract_reel_area(reel_image)
        
        # 抽出されたリール領域のサイズ確認
        assert extracted is not None
        assert extracted.shape[0] <= reel_image.shape[0]  # 高さが元画像以下
        assert extracted.shape[1] <= reel_image.shape[1]  # 幅が元画像以下
        
        # リール領域が未検出の場合のテスト
        processor.reel_area = None
        extracted = processor.extract_reel_area(test_image)
        # テスト画像ではリール領域が検出されないはずなのでNoneが返るか確認
        assert extracted is None
