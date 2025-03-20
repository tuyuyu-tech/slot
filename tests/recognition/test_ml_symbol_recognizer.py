"""
機械学習を用いた図柄認識クラスのテスト。
"""
import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# テスト対象のモジュールをインポート
from src.recognition.ml_symbol_recognizer import MLSymbolRecognizer, HybridSymbolRecognizer


class TestMLSymbolRecognizer(unittest.TestCase):
    """機械学習を用いた図柄認識クラスのテスト。"""
    
    def setUp(self):
        """各テスト前の準備。"""
        # テスト用の一時ディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = os.path.join(self.test_dir, 'templates')
        self.models_dir = os.path.join(self.test_dir, 'models')
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # テスト用の画像データを作成
        self.create_test_images()
        
        # テスト対象のインスタンスを作成
        self.recognizer = MLSymbolRecognizer(
            templates_dir=self.templates_dir,
            models_dir=self.models_dir
        )
    
    def tearDown(self):
        """各テスト後の後処理。"""
        # テスト用ディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def create_test_images(self):
        """テスト用の画像データを作成する。"""
        # 7図柄のテスト画像
        seven_img = np.zeros((64, 64), dtype=np.uint8)
        cv2.putText(seven_img, "7", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        self.seven_path = os.path.join(self.templates_dir, "seven.png")
        cv2.imwrite(self.seven_path, seven_img)
        
        # BAR図柄のテスト画像
        bar_img = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(bar_img, (10, 20), (54, 30), 255, -1)
        cv2.rectangle(bar_img, (10, 35), (54, 45), 255, -1)
        self.bar_path = os.path.join(self.templates_dir, "bar.png")
        cv2.imwrite(self.bar_path, bar_img)
        
        # テスト用フレーム画像（複数の図柄を含む）
        self.test_frame = np.zeros((200, 200), dtype=np.uint8)
        # 7を配置
        cv2.putText(self.test_frame, "7", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        # BARを配置
        cv2.rectangle(self.test_frame, (120, 130), (164, 140), 255, -1)
        cv2.rectangle(self.test_frame, (120, 145), (164, 155), 255, -1)
    
    def test_init(self):
        """初期化のテスト。"""
        self.assertEqual(self.recognizer.templates_dir, self.templates_dir)
        self.assertEqual(self.recognizer.model_path, os.path.join(self.models_dir, 'symbol_classifier.joblib'))
        self.assertIsNone(self.recognizer.model)
    
    def test_register_template(self):
        """テンプレート登録のテスト。"""
        # 7図柄を登録
        seven_img = cv2.imread(self.seven_path, cv2.IMREAD_GRAYSCALE)
        result = self.recognizer.register_template("7", seven_img, 0.7)
        self.assertTrue(result)
        self.assertIn("7", self.recognizer.symbols)
        
        # BAR図柄を登録
        bar_img = cv2.imread(self.bar_path, cv2.IMREAD_GRAYSCALE)
        result = self.recognizer.register_template("BAR", bar_img, 0.7)
        self.assertTrue(result)
        self.assertIn("BAR", self.recognizer.symbols)
    
    @patch('joblib.dump')
    def test_train_model(self, mock_dump):
        """モデル学習のテスト。"""
        # テンプレートを登録
        seven_img = cv2.imread(self.seven_path, cv2.IMREAD_GRAYSCALE)
        bar_img = cv2.imread(self.bar_path, cv2.IMREAD_GRAYSCALE)
        self.recognizer.register_template("7", seven_img, 0.7)
        self.recognizer.register_template("BAR", bar_img, 0.7)
        
        # モデル学習を実行
        result = self.recognizer.train_model()
        self.assertTrue(result)
        self.assertIsNotNone(self.recognizer.model)
        mock_dump.assert_called_once()
    
    @patch('joblib.load')
    def test_load_model(self, mock_load):
        """モデル読み込みのテスト。"""
        # モックのモデルを作成
        mock_model = {'classifier': MagicMock(), 'scaler': MagicMock()}
        mock_load.return_value = mock_model
        
        # モデルパスに空ファイルを作成
        with open(self.recognizer.model_path, 'w') as f:
            f.write('')
        
        # モデルを読み込み
        self.recognizer._load_model()
        self.assertEqual(self.recognizer.model, mock_model)
    
    def test_extract_features(self):
        """特徴量抽出のテスト。"""
        # テスト画像を用意
        test_img = np.zeros((64, 64), dtype=np.uint8)
        cv2.putText(test_img, "T", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        
        # 特徴量を抽出
        features = self.recognizer._extract_features(test_img)
        
        # 特徴量の形状を確認
        self.assertTrue(isinstance(features, np.ndarray))
        self.assertTrue(features.ndim == 1)  # 1次元配列であること
    
    @patch.object(MLSymbolRecognizer, '_non_max_suppression')
    def test_sliding_window(self, mock_nms):
        """スライディングウィンドウのテスト。"""
        # モックの返り値を設定
        mock_nms.return_value = [{'x': 30, 'y': 30, 'width': 64, 'height': 64, 'name': '7', 'score': 0.9}]
        
        # モックのモデルを設定
        mock_classifier = MagicMock()
        mock_classifier.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_classifier.classes_ = np.array(['BAR', '7'])
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1, 2, 3]])
        
        self.recognizer.model = {
            'classifier': mock_classifier,
            'scaler': mock_scaler
        }
        
        # スライディングウィンドウを実行
        results = self.recognizer.sliding_window(self.test_frame)
        
        # NMSが呼ばれたことを確認
        mock_nms.assert_called_once()
        
        # 結果を確認
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], '7')
        self.assertEqual(results[0]['score'], 0.9)
    
    def test_non_max_suppression(self):
        """Non-Maximum Suppressionのテスト。"""
        # テスト用の検出結果
        detections = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'name': '7', 'score': 0.9},
            {'x': 15, 'y': 15, 'width': 50, 'height': 50, 'name': '7', 'score': 0.8},  # 重複
            {'x': 100, 'y': 100, 'width': 50, 'height': 50, 'name': 'BAR', 'score': 0.85}
        ]
        
        # NMSを実行
        results = self.recognizer._non_max_suppression(detections)
        
        # 結果を確認
        self.assertEqual(len(results), 2)  # 重複が除外されて2つになる
        self.assertEqual(results[0]['name'], '7')
        self.assertEqual(results[0]['score'], 0.9)
        self.assertEqual(results[1]['name'], 'BAR')
        self.assertEqual(results[1]['score'], 0.85)
    
    @patch.object(MLSymbolRecognizer, 'sliding_window')
    def test_recognize_symbols_with_model(self, mock_sliding_window):
        """モデルがある場合の図柄認識テスト。"""
        # モックの返り値を設定
        expected_results = [{'x': 30, 'y': 30, 'width': 64, 'height': 64, 'name': '7', 'score': 0.9}]
        mock_sliding_window.return_value = expected_results
        
        # モデルを設定
        self.recognizer.model = {'classifier': MagicMock(), 'scaler': MagicMock()}
        
        # 図柄認識を実行
        results = self.recognizer.recognize_symbols(self.test_frame)
        
        # スライディングウィンドウが呼ばれたことを確認
        mock_sliding_window.assert_called_once_with(self.test_frame)
        
        # 結果を確認
        self.assertEqual(results, expected_results)
    
    def test_recognize_symbols_without_model(self):
        """モデルがない場合の図柄認識テスト。"""
        # モデルを設定しない
        self.recognizer.model = None
        
        # 図柄認識を実行
        results = self.recognizer.recognize_symbols(self.test_frame)
        
        # 結果を確認
        self.assertEqual(results, [])


class TestHybridSymbolRecognizer(unittest.TestCase):
    """ハイブリッド図柄認識クラスのテスト。"""
    
    def setUp(self):
        """各テスト前の準備。"""
        # テスト用の一時ディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = os.path.join(self.test_dir, 'templates')
        self.models_dir = os.path.join(self.test_dir, 'models')
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # テスト対象のインスタンスを作成
        with patch('src.recognition.ml_symbol_recognizer.TemplateMatching') as mock_tm:
            with patch('src.recognition.ml_symbol_recognizer.MLSymbolRecognizer') as mock_ml:
                # モックの設定
                mock_tm.return_value = MagicMock()
                mock_ml.return_value = MagicMock()
                mock_ml.return_value.model = None  # 最初はモデルなし
                
                self.recognizer = HybridSymbolRecognizer(
                    templates_dir=self.templates_dir,
                    models_dir=self.models_dir
                )
                
                # モックを保存
                self.mock_template_matcher = mock_tm.return_value
                self.mock_ml_recognizer = mock_ml.return_value
    
    def tearDown(self):
        """各テスト後の後処理。"""
        # テスト用ディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """初期化のテスト。"""
        self.assertEqual(self.recognizer.templates_dir, self.templates_dir)
        self.assertEqual(self.recognizer.template_matcher, self.mock_template_matcher)
        self.assertEqual(self.recognizer.ml_recognizer, self.mock_ml_recognizer)
        self.assertFalse(self.recognizer.use_ml)  # モデルがないのでFalse
    
    def test_register_template(self):
        """テンプレート登録のテスト。"""
        # モックの返り値を設定
        self.mock_template_matcher.register_template.return_value = True
        self.mock_ml_recognizer.register_template.return_value = True
        self.mock_ml_recognizer.train_model.return_value = True
        
        # テスト用の画像
        test_img = np.zeros((64, 64), dtype=np.uint8)
        
        # テンプレート登録を実行
        result = self.recognizer.register_template("TEST", test_img, 0.7)
        
        # 両方の認識器にテンプレート登録が呼ばれたことを確認
        self.mock_template_matcher.register_template.assert_called_once()
        self.mock_ml_recognizer.register_template.assert_called_once()
        
        # モデル学習が呼ばれたことを確認
        self.mock_ml_recognizer.train_model.assert_called_once()
        
        # 結果を確認
        self.assertTrue(result)
    
    def test_recognize_symbols_without_ml(self):
        """機械学習なしでの図柄認識テスト。"""
        # モックの返り値を設定
        expected_results = [{'x': 30, 'y': 30, 'width': 64, 'height': 64, 'name': '7', 'score': 0.9}]
        self.mock_template_matcher.recognize_symbols.return_value = expected_results
        
        # MLモデルなし
        self.recognizer.use_ml = False
        
        # テスト用フレーム
        test_frame = np.zeros((200, 200), dtype=np.uint8)
        
        # 図柄認識を実行
        with patch.object(self.recognizer, '_merge_results') as mock_merge:
            mock_merge.return_value = expected_results
            results = self.recognizer.recognize_symbols(test_frame)
        
        # テンプレートマッチングが呼ばれたことを確認
        self.mock_template_matcher.recognize_symbols.assert_called_once_with(test_frame)
        
        # 機械学習認識が呼ばれていないことを確認
        self.mock_ml_recognizer.recognize_symbols.assert_not_called()
        
        # 結果を確認
        self.assertEqual(results, expected_results)
    
    def test_recognize_symbols_with_ml(self):
        """機械学習ありでの図柄認識テスト。"""
        # モックの返り値を設定
        template_results = [{'x': 30, 'y': 30, 'width': 64, 'height': 64, 'name': '7', 'score': 0.9}]
        ml_results = [{'x': 100, 'y': 100, 'width': 64, 'height': 64, 'name': 'BAR', 'score': 0.85}]
        merged_results = template_results + ml_results
        
        self.mock_template_matcher.recognize_symbols.return_value = template_results
        self.mock_ml_recognizer.recognize_symbols.return_value = ml_results
        
        # MLモデルあり
        self.recognizer.use_ml = True
        
        # テスト用フレーム
        test_frame = np.zeros((200, 200), dtype=np.uint8)
        
        # 図柄認識を実行
        with patch.object(self.recognizer, '_merge_results') as mock_merge:
            mock_merge.return_value = merged_results
            results = self.recognizer.recognize_symbols(test_frame)
        
        # テンプレートマッチングと機械学習認識の両方が呼ばれたことを確認
        self.mock_template_matcher.recognize_symbols.assert_called_once_with(test_frame)
        self.mock_ml_recognizer.recognize_symbols.assert_called_once_with(test_frame)
        
        # 結果を確認
        self.assertEqual(results, merged_results)
    
    def test_merge_results(self):
        """認識結果の統合テスト。"""
        # テスト用の検出結果
        template_results = [
            {'x': 10, 'y': 10, 'width': 50, 'height': 50, 'name': '7', 'score': 0.9}
        ]
        ml_results = [
            {'x': 15, 'y': 15, 'width': 50, 'height': 50, 'name': '7', 'score': 0.8},  # 重複
            {'x': 100, 'y': 100, 'width': 50, 'height': 50, 'name': 'BAR', 'score': 0.85}
        ]
        
        # 結果を統合
        results = self.recognizer._merge_results(template_results, ml_results)
        
        # 結果を確認
        self.assertEqual(len(results), 2)  # 重複が除外されて2つになる
        self.assertEqual(results[0]['name'], '7')
        self.assertEqual(results[0]['score'], 0.9)
        self.assertEqual(results[1]['name'], 'BAR')
        self.assertEqual(results[1]['score'], 0.85)


if __name__ == '__main__':
    unittest.main()
