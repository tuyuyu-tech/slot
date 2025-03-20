"""
main_windowモジュールのテスト。
UIモジュールのテストは複雑なため、基本的な部分のみをテスト。
"""
import pytest
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import VideoFrame, CaptureAreaSelector, MainWindow

class TestVideoFrame:
    """VideoFrameクラスのテスト"""
    
    def test_init(self, qapp):
        """初期化処理のテスト"""
        frame = VideoFrame()
        
        assert frame.frame is None
        assert frame.symbols == []
        assert frame.timing_results == {}
        assert frame.show_recognition is True
        assert frame.show_timing is True
    
    def test_update_frame(self, qapp, test_image):
        """フレーム更新のテスト"""
        frame = VideoFrame()
        
        # フレーム更新
        frame.update_frame(test_image)
        
        # フレームが保存されていること
        assert frame.frame is not None
        assert np.array_equal(frame.frame, test_image)
    
    def test_update_symbols(self, qapp):
        """図柄情報更新のテスト"""
        frame = VideoFrame()
        
        # テスト用の図柄情報
        symbols = [
            {"name": "symbol1", "x": 100, "y": 150, "width": 50, "height": 50, "score": 0.9}
        ]
        
        # 図柄情報更新
        frame.update_symbols(symbols)
        
        # 図柄情報が保存されていること
        assert frame.symbols == symbols
    
    def test_update_timing(self, qapp):
        """タイミング情報更新のテスト"""
        frame = VideoFrame()
        
        # テスト用のタイミング情報
        timing = {
            1: {"optimal_frame": 100, "frames_until_push": 10}
        }
        
        # タイミング情報更新
        frame.update_timing(timing)
        
        # タイミング情報が保存されていること
        assert frame.timing_results == timing
    
    def test_set_show_recognition(self, qapp):
        """認識表示設定のテスト"""
        frame = VideoFrame()
        
        # デフォルトはTrue
        assert frame.show_recognition is True
        
        # Falseに設定
        frame.set_show_recognition(False)
        assert frame.show_recognition is False
        
        # Trueに戻す
        frame.set_show_recognition(True)
        assert frame.show_recognition is True
    
    def test_set_show_timing(self, qapp):
        """タイミング表示設定のテスト"""
        frame = VideoFrame()
        
        # デフォルトはTrue
        assert frame.show_timing is True
        
        # Falseに設定
        frame.set_show_timing(False)
        assert frame.show_timing is False
        
        # Trueに戻す
        frame.set_show_timing(True)
        assert frame.show_timing is True


class TestCaptureAreaSelector:
    """CaptureAreaSelectorクラスのテスト"""
    
    def test_init(self, qapp):
        """初期化処理のテスト"""
        selector = CaptureAreaSelector()
        
        # デフォルト値の確認
        assert selector.x_spin.value() == 0
        assert selector.y_spin.value() == 0
        assert selector.width_spin.value() == 640
        assert selector.height_spin.value() == 480
    
    def test_on_apply(self, qapp, qtbot):
        """適用ボタンのテスト"""
        selector = CaptureAreaSelector()
        
        # 値を設定
        selector.x_spin.setValue(100)
        selector.y_spin.setValue(200)
        selector.width_spin.setValue(800)
        selector.height_spin.setValue(600)
        
        # シグナルをモニタリング
        with qtbot.waitSignal(selector.area_selected) as blocker:
            # 適用ボタンをクリック
            qtbot.mouseClick(selector.apply_button, Qt.LeftButton)
        
        # シグナルの引数を確認
        assert blocker.args == [100, 200, 800, 600]


class TestMainWindow:
    """MainWindowクラスの基本的なテスト"""
    
    def test_init(self, qapp):
        """初期化処理のテスト"""
        window = MainWindow()
        
        # 基本的なUIコンポーネントの確認
        assert hasattr(window, 'video_frame')
        assert hasattr(window, 'capture_area_selector')
        assert hasattr(window, 'symbol_registration')
        assert hasattr(window, 'timing_settings')
        assert hasattr(window, 'start_stop_button')
        
        # 初期状態の確認
        assert window.capturing is False
        assert window.processing is False
        assert window.start_stop_button.text() == "キャプチャ開始"
    
    def test_toggle_capture(self, qapp, qtbot):
        """キャプチャ切替のテスト"""
        window = MainWindow()
        
        # キャプチャ開始
        qtbot.mouseClick(window.start_stop_button, Qt.LeftButton)
        
        # 状態確認
        assert window.capturing is True
        assert window.start_stop_button.text() == "キャプチャ停止"
        
        # キャプチャ停止
        qtbot.mouseClick(window.start_stop_button, Qt.LeftButton)
        
        # 状態確認
        assert window.capturing is False
        assert window.start_stop_button.text() == "キャプチャ開始"
    
    def test_set_capture_area(self, qapp):
        """キャプチャ領域設定のテスト"""
        window = MainWindow()
        
        # キャプチャ領域設定
        window.set_capture_area(200, 300, 1280, 720)
        
        # 設定が反映されていること
        area = window.screen_capture.capture_area
        assert area['left'] == 200
        assert area['top'] == 300
        assert area['width'] == 1280
        assert area['height'] == 720
