"""
パチスロリール領域の自動キャリブレーション機能のテスト。
"""
import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import os

# テスト対象のモジュールをインポート
from src.capture.auto_calibration import ReelCalibrator, AutoCalibrationDialog


class TestReelCalibrator(unittest.TestCase):
    """自動キャリブレーション機能のテスト。"""
    
    def setUp(self):
        """各テスト前の準備。"""
        self.calibrator = ReelCalibrator()
        
        # テスト用の画像データを作成
        self.create_test_images()
    
    def create_test_images(self):
        """テスト用の画像データを作成する。"""
        # テスト用フレーム画像（リール領域を模擬）
        self.test_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # リール領域を描画（画面中央に縦長の矩形）
        rect_x, rect_y = 100, 50
        rect_w, rect_h = 200, 200
        cv2.rectangle(self.test_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), 2)
        
        # リール区切りを描画（垂直線）
        div1_x = rect_x + int(rect_w / 3)
        div2_x = rect_x + int(2 * rect_w / 3)
        cv2.line(self.test_frame, (div1_x, rect_y), (div1_x, rect_y + rect_h), (255, 255, 255), 1)
        cv2.line(self.test_frame, (div2_x, rect_y), (div2_x, rect_y + rect_h), (255, 255, 255), 1)
        
        # 図柄を模擬した矩形を描画
        for i in range(3):
            for j in range(3):
                symbol_x = rect_x + i * int(rect_w / 3) + 10
                symbol_y = rect_y + j * int(rect_h / 3) + 10
                symbol_w = int(rect_w / 3) - 20
                symbol_h = int(rect_h / 3) - 20
                cv2.rectangle(self.test_frame, 
                             (symbol_x, symbol_y), 
                             (symbol_x + symbol_w, symbol_y + symbol_h), 
                             (0, 128, 255), -1)
        
        # 複数フレームを作成（少しずつ位置を変えて、安定度のテスト用）
        self.test_frames = [self.test_frame.copy()]
        
        for i in range(5):
            # 位置を少しずつ変えたフレームを作成
            shift_frame = np.zeros((300, 400, 3), dtype=np.uint8)
            # 少しずれたリール領域を描画
            shift_x = rect_x + np.random.randint(-5, 5)
            shift_y = rect_y + np.random.randint(-5, 5)
            cv2.rectangle(shift_frame, (shift_x, shift_y), 
                         (shift_x + rect_w, shift_y + rect_h), 
                         (255, 255, 255), 2)
            
            # 区切り線を描画
            div1_x = shift_x + int(rect_w / 3)
            div2_x = shift_x + int(2 * rect_w / 3)
            cv2.line(shift_frame, (div1_x, shift_y), (div1_x, shift_y + rect_h), (255, 255, 255), 1)
            cv2.line(shift_frame, (div2_x, shift_y), (div2_x, shift_y + rect_h), (255, 255, 255), 1)
            
            # 図柄を模擬した矩形を描画
            for i in range(3):
                for j in range(3):
                    symbol_x = shift_x + i * int(rect_w / 3) + 10
                    symbol_y = shift_y + j * int(rect_h / 3) + 10
                    symbol_w = int(rect_w / 3) - 20
                    symbol_h = int(rect_h / 3) - 20
                    cv2.rectangle(shift_frame, 
                                 (symbol_x, symbol_y), 
                                 (symbol_x + symbol_w, symbol_y + symbol_h), 
                                 (0, 128, 255), -1)
            
            self.test_frames.append(shift_frame)
    
    def test_detect_reel_area(self):
        """リール領域検出のテスト。"""
        # リール領域を検出
        reel_area = self.calibrator.detect_reel_area(self.test_frame)
        
        # 検出結果が正しい形式（x, y, width, height）であることを確認
        self.assertIsNotNone(reel_area)
        self.assertEqual(len(reel_area), 4)
        
        # 検出された領域が妥当であることを確認
        x, y, w, h = reel_area
        self.assertTrue(50 <= x <= 150)  # 想定されるx座標の範囲
        self.assertTrue(30 <= y <= 70)   # 想定されるy座標の範囲
        self.assertTrue(150 <= w <= 250)  # 想定される幅の範囲
        self.assertTrue(150 <= h <= 250)  # 想定される高さの範囲
        
        # アスペクト比が妥当であることを確認
        aspect_ratio = h / w
        self.assertTrue(0.8 <= aspect_ratio <= 1.2)  # ほぼ正方形か少し縦長
    
    def test_detect_reel_divisions(self):
        """リール区切り検出のテスト。"""
        # 仮のリール領域を指定
        reel_area = (100, 50, 200, 200)
        
        # リール区切りを検出
        divisions = self.calibrator.detect_reel_divisions(self.test_frame, reel_area)
        
        # 検出結果が正しい形式であることを確認
        self.assertIsNotNone(divisions)
        self.assertEqual(len(divisions), 3)  # 3つのリール
        
        # 各区切りが正しい形式（x, y, width, height）であることを確認
        for division in divisions:
            self.assertEqual(len(division), 4)
        
        # 区切りの合計幅がリール全体の幅とほぼ一致することを確認
        total_width = sum(div[2] for div in divisions)
        self.assertAlmostEqual(total_width, reel_area[2], delta=5)
        
        # 各区切りの高さがリール全体の高さと一致することを確認
        for division in divisions:
            self.assertEqual(division[3], reel_area[3])
    
    def test_get_stable_reel_area(self):
        """安定したリール領域取得のテスト。"""
        # 複数のフレームでリール領域を検出
        for frame in self.test_frames:
            self.calibrator.detect_reel_area(frame)
        
        # 安定したリール領域を取得
        stable_area = self.calibrator.get_stable_reel_area()
        
        # 検出結果が正しい形式であることを確認
        self.assertIsNotNone(stable_area)
        self.assertEqual(len(stable_area), 4)
        
        # 検出された領域が妥当であることを確認
        x, y, w, h = stable_area
        self.assertTrue(50 <= x <= 150)
        self.assertTrue(30 <= y <= 70)
        self.assertTrue(150 <= w <= 250)
        self.assertTrue(150 <= h <= 250)
    
    def test_calibrate(self):
        """キャリブレーション全体のテスト。"""
        # キャリブレーションを実行
        result = self.calibrator.calibrate(self.test_frames)
        
        # 検出結果が正しい形式であることを確認
        self.assertIsNotNone(result)
        self.assertIn('reel_area', result)
        self.assertIn('reel_divisions', result)
        self.assertIn('confidence', result)
        
        # リール領域が検出されたことを確認
        self.assertIsNotNone(result['reel_area'])
        
        # リール区切りが検出されたことを確認
        self.assertIsNotNone(result['reel_divisions'])
        self.assertEqual(len(result['reel_divisions']), 3)
        
        # 信頼度が正しい範囲であることを確認
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)


class TestAutoCalibrationDialog(unittest.TestCase):
    """自動キャリブレーションダイアログのテスト。"""
    
    def setUp(self):
        """各テスト前の準備。"""
        # スクリーンキャプチャのモックを作成
        self.mock_screen_capture = MagicMock()
        
        # テスト用の画像を設定
        self.test_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # リール領域を描画
        cv2.rectangle(self.test_frame, (100, 50), (300, 250), (255, 255, 255), 2)
        
        # モックのcapture_frameメソッドが画像を返すように設定
        self.mock_screen_capture.capture_frame.return_value = (self.test_frame, 30.0)
        
        # テスト対象のインスタンスを作成
        self.dialog = AutoCalibrationDialog(self.mock_screen_capture)
    
    def test_init(self):
        """初期化のテスト。"""
        self.assertEqual(self.dialog.screen_capture, self.mock_screen_capture)
        self.assertIsNotNone(self.dialog.calibrator)
        self.assertEqual(self.dialog.frames, [])
        self.assertIsNone(self.dialog.calibration_result)
    
    def test_capture_frame(self):
        """フレームキャプチャのテスト。"""
        # フレームをキャプチャ
        frame = self.dialog.capture_frame()
        
        # 正しいフレームが返されたことを確認
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, self.test_frame.shape)
        
        # スクリーンキャプチャのメソッドが呼ばれたことを確認
        self.mock_screen_capture.capture_frame.assert_called_once()
    
    def test_start_calibration(self):
        """キャリブレーション開始のテスト。"""
        # 事前にフレームを追加
        self.dialog.frames = [self.test_frame]
        self.dialog.calibration_result = {'dummy': 'data'}
        
        # キャリブレーションを開始
        self.dialog.start_calibration()
        
        # フレームリストとキャリブレーション結果がクリアされたことを確認
        self.assertEqual(self.dialog.frames, [])
        self.assertIsNone(self.dialog.calibration_result)
    
    def test_add_calibration_frame(self):
        """キャリブレーションフレーム追加のテスト。"""
        # フレームを追加
        result = self.dialog.add_calibration_frame()
        
        # 追加が成功したことを確認
        self.assertTrue(result)
        
        # フレームが追加されたことを確認
        self.assertEqual(len(self.dialog.frames), 1)
        self.assertEqual(self.dialog.frames[0].shape, self.test_frame.shape)
    
    def test_is_calibration_complete(self):
        """キャリブレーション完了チェックのテスト。"""
        # 初期状態では完了していないことを確認
        self.assertFalse(self.dialog.is_calibration_complete())
        
        # 最大フレーム数まで追加
        for _ in range(self.dialog.max_frames):
            self.dialog.frames.append(self.test_frame)
        
        # 完了していることを確認
        self.assertTrue(self.dialog.is_calibration_complete())
    
    @patch.object(ReelCalibrator, 'calibrate')
    def test_finish_calibration(self, mock_calibrate):
        """キャリブレーション完了のテスト。"""
        # モックの返り値を設定
        expected_result = {
            'reel_area': (100, 50, 200, 200),
            'reel_divisions': [(100, 50, 66, 200), (166, 50, 67, 200), (233, 50, 67, 200)],
            'confidence': 0.9
        }
        mock_calibrate.return_value = expected_result
        
        # フレームを追加
        self.dialog.frames = [self.test_frame]
        
        # キャリブレーションを完了
        result = self.dialog.finish_calibration()
        
        # 正しい結果が返されたことを確認
        self.assertEqual(result, expected_result)
        
        # キャリブレーション結果が保存されたことを確認
        self.assertEqual(self.dialog.calibration_result, expected_result)
    
    def test_get_last_frame_with_overlay(self):
        """オーバーレイ付きフレーム取得のテスト。"""
        # キャリブレーション結果を設定
        self.dialog.frames = [self.test_frame]
        self.dialog.calibration_result = {
            'reel_area': (100, 50, 200, 200),
            'reel_divisions': [(100, 50, 66, 200), (166, 50, 67, 200), (233, 50, 67, 200)],
            'confidence': 0.9
        }
        
        # オーバーレイ付きフレームを取得
        overlay_frame = self.dialog.get_last_frame_with_overlay()
        
        # フレームが返されたことを確認
        self.assertIsNotNone(overlay_frame)
        self.assertEqual(overlay_frame.shape, self.test_frame.shape)
        
        # オーバーレイが追加されたことを確認（完全に確認するのは難しいので、形状が変わっていないことだけ確認）
        # 実際にはdrawしたピクセルがあるかどうかを確認する方法もありますが、ここでは省略


if __name__ == '__main__':
    unittest.main()
