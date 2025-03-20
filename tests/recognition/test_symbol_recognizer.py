"""
symbol_recognizerモジュールのテスト。
"""
import pytest
import numpy as np
import cv2
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from src.recognition.symbol_recognizer import Symbol, SymbolRecognizer, TemplateMatching, SymbolTracker

class TestSymbol:
    """Symbolクラスのテスト"""
    
    def test_init(self):
        """初期化処理のテスト"""
        template = np.zeros((50, 50), dtype=np.uint8)
        symbol = Symbol("test_symbol", template, 0.8, {"test": "data"})
        
        assert symbol.name == "test_symbol"
        assert np.array_equal(symbol.template, template)
        assert symbol.threshold == 0.8
        assert symbol.metadata == {"test": "data"}
    
    def test_to_dict(self):
        """辞書変換のテスト"""
        template = np.zeros((50, 50), dtype=np.uint8)
        metadata = {"test": "data", "value": 42}
        symbol = Symbol("test_symbol", template, 0.8, metadata)
        
        result = symbol.to_dict()
        
        assert result["name"] == "test_symbol"
        assert result["threshold"] == 0.8
        assert result["metadata"] == metadata


class TestSymbolRecognizer:
    """SymbolRecognizerクラスのテスト"""
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリを作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_init(self, temp_dir):
        """初期化処理のテスト"""
        recognizer = SymbolRecognizer(templates_dir=temp_dir)
        
        assert recognizer.symbols == {}
        assert recognizer.templates_dir == temp_dir
        
        # テンプレートディレクトリが作成されていること
        assert os.path.exists(temp_dir)
    
    def test_load_templates_no_index(self, temp_dir):
        """インデックスファイルがない場合のテスト"""
        recognizer = SymbolRecognizer(templates_dir=temp_dir)
        # インデックスファイルがないので空のままのはず
        assert recognizer.symbols == {}
    
    def test_load_templates(self, temp_dir, symbol_template):
        """テンプレート読み込みのテスト"""
        # テスト用のインデックスファイルを作成
        index_data = [
            {
                "name": "test_symbol",
                "threshold": 0.75,
                "metadata": {"test": "data"}
            }
        ]
        
        with open(os.path.join(temp_dir, "index.json"), "w") as f:
            json.dump(index_data, f)
        
        # テスト用のテンプレート画像を保存
        cv2.imwrite(os.path.join(temp_dir, "test_symbol.png"), symbol_template)
        
        # テスト実行
        recognizer = SymbolRecognizer(templates_dir=temp_dir)
        
        # テンプレートが読み込まれていること
        assert "test_symbol" in recognizer.symbols
        assert recognizer.symbols["test_symbol"].name == "test_symbol"
        assert recognizer.symbols["test_symbol"].threshold == 0.75
        assert recognizer.symbols["test_symbol"].metadata == {"test": "data"}
    
    def test_register_template(self, temp_dir, symbol_template):
        """テンプレート登録のテスト"""
        recognizer = SymbolRecognizer(templates_dir=temp_dir)
        
        # テンプレート登録
        result = recognizer.register_template(
            "new_symbol", 
            symbol_template, 
            0.85, 
            {"category": "special"}
        )
        
        # 登録成功
        assert result is True
        assert "new_symbol" in recognizer.symbols
        
        # 登録された内容確認
        symbol = recognizer.symbols["new_symbol"]
        assert symbol.name == "new_symbol"
        assert np.array_equal(symbol.template, cv2.cvtColor(symbol_template, cv2.COLOR_BGR2GRAY))
        assert symbol.threshold == 0.85
        assert symbol.metadata == {"category": "special"}
        
        # ファイルが保存されていること
        assert os.path.exists(os.path.join(temp_dir, "index.json"))
        assert os.path.exists(os.path.join(temp_dir, "new_symbol.png"))
        
        # インデックスファイルの内容確認
        with open(os.path.join(temp_dir, "index.json"), "r") as f:
            index_data = json.load(f)
        
        assert len(index_data) == 1
        assert index_data[0]["name"] == "new_symbol"
        assert index_data[0]["threshold"] == 0.85
        assert index_data[0]["metadata"] == {"category": "special"}


class TestTemplateMatching:
    """TemplateMatchingクラスのテスト"""
    
    @pytest.fixture
    def template_matcher(self, temp_dir):
        """テンプレートマッチングのインスタンス"""
        return TemplateMatching(templates_dir=temp_dir)
    
    def test_init(self, template_matcher):
        """初期化処理のテスト"""
        assert template_matcher.method == cv2.TM_CCOEFF_NORMED
    
    def test_match_template(self, template_matcher, test_image, symbol_template):
        """テンプレートマッチングのテスト"""
        # テスト用画像にテンプレートを埋め込む
        x, y = 100, 80
        h, w = symbol_template.shape[:2]
        test_image[y:y+h, x:x+w] = symbol_template
        
        # グレースケール変換
        gray_template = cv2.cvtColor(symbol_template, cv2.COLOR_BGR2GRAY)
        
        # マッチング実行
        matches = template_matcher.match_template(test_image, gray_template, 0.7)
        
        # 少なくとも1つのマッチが見つかるはず
        assert len(matches) > 0
        
        # 最初のマッチがおおよそ正しい位置にあること
        first_match = matches[0]
        assert abs(first_match["x"] - x) < 10
        assert abs(first_match["y"] - y) < 10
        assert first_match["width"] == w
        assert first_match["height"] == h
        assert first_match["score"] > 0.7
    
    def test_recognize_symbols(self, template_matcher, test_image, symbol_template):
        """図柄認識のテスト"""
        # テスト用の図柄を登録
        template_matcher.register_template("test_symbol", symbol_template, 0.7)
        
        # テスト用画像にテンプレートを埋め込む
        x, y = 100, 80
        h, w = symbol_template.shape[:2]
        test_image[y:y+h, x:x+w] = symbol_template
        
        # 図柄認識実行
        results = template_matcher.recognize_symbols(test_image)
        
        # 少なくとも1つの図柄が認識されるはず
        assert len(results) > 0
        
        # 認識結果確認
        first_result = results[0]
        assert first_result["name"] == "test_symbol"
        assert "metadata" in first_result
        assert abs(first_result["x"] - x) < 10
        assert abs(first_result["y"] - y) < 10


class TestSymbolTracker:
    """SymbolTrackerクラスのテスト"""
    
    def test_init(self):
        """初期化処理のテスト"""
        tracker = SymbolTracker(max_tracking_frames=15)
        
        assert tracker.tracked_symbols == []
        assert tracker.max_tracking_frames == 15
    
    def test_update_initial(self, test_image):
        """初期更新のテスト"""
        tracker = SymbolTracker()
        
        # テスト用の認識結果
        recognized_symbols = [
            {"name": "symbol1", "x": 100, "y": 150, "width": 50, "height": 50, "score": 0.9},
            {"name": "symbol2", "x": 200, "y": 250, "width": 50, "height": 50, "score": 0.8}
        ]
        
        # 更新実行
        result = tracker.update(recognized_symbols, test_image)
        
        # 結果確認
        assert len(result) == 2
        assert result[0]["name"] == "symbol1"
        assert result[0]["tracking_frames"] == 0
        assert len(result[0]["prev_positions"]) == 1
        assert result[0]["prev_positions"][0] == (100, 150)
        assert result[0]["velocity"] == (0, 0)
    
    def test_update_tracking(self, test_image):
        """追跡更新のテスト"""
        tracker = SymbolTracker()
        
        # 初期状態の認識結果
        initial_symbols = [
            {"name": "symbol1", "x": 100, "y": 150, "width": 50, "height": 50, "score": 0.9}
        ]
        
        # 更新1回目
        tracker.update(initial_symbols, test_image)
        
        # 次のフレームの認識結果（少し移動）
        next_symbols = [
            {"name": "symbol1", "x": 105, "y": 160, "width": 50, "height": 50, "score": 0.9}
        ]
        
        # 更新2回目
        result = tracker.update(next_symbols, test_image)
        
        # 結果確認
        assert len(result) == 1
        assert result[0]["name"] == "symbol1"
        assert result[0]["tracking_frames"] == 1
        assert len(result[0]["prev_positions"]) == 2
        assert result[0]["prev_positions"][0] == (100, 150)
        assert result[0]["prev_positions"][1] == (105, 160)
        assert result[0]["velocity"] == (5, 10)  # 移動差分
    
    def test_predict_movement(self, test_image):
        """移動予測のテスト"""
        tracker = SymbolTracker()
        
        # テスト用の追跡中図柄
        symbol = {
            "name": "test",
            "x": 100,
            "y": 150,
            "prev_positions": [(90, 130), (95, 140), (100, 150)],
            "velocity": (5, 10)
        }
        
        # 予測実行
        pred_x, pred_y = tracker.predict_movement(symbol, frames_ahead=2)
        
        # 予測位置は現在位置 + 速度*フレーム数
        assert pred_x == 100 + 5*2
        assert pred_y == 150 + 10*2
    
    def test_get_tracked_symbols(self):
        """追跡中図柄取得のテスト"""
        tracker = SymbolTracker()
        
        # テスト用のデータをセット
        test_symbols = [{"name": "test1"}, {"name": "test2"}]
        tracker.tracked_symbols = test_symbols
        
        # 取得実行
        result = tracker.get_tracked_symbols()
        
        # 結果確認
        assert result == test_symbols
