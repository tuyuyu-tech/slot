"""
reel_analyzerモジュールのテスト。
"""
import pytest
import numpy as np
from collections import deque
from src.timing.reel_analyzer import ReelAnalyzer, TimingPredictor, TimingManager

class TestReelAnalyzer:
    """ReelAnalyzerクラスのテスト"""
    
    def test_init(self):
        """初期化処理のテスト"""
        analyzer = ReelAnalyzer(history_size=20)
        
        assert isinstance(analyzer.position_history, deque)
        assert analyzer.position_history.maxlen == 20
        assert isinstance(analyzer.time_history, deque)
        assert analyzer.time_history.maxlen == 20
        assert analyzer.rotation_speed == 0.0
        assert analyzer.cycle_frames == 0
        assert analyzer.is_stable is False
    
    def test_add_position(self):
        """位置追加のテスト"""
        analyzer = ReelAnalyzer()
        
        # 位置を追加
        analyzer.add_position((100, 150), 10)
        
        # 履歴に追加されていることを確認
        assert len(analyzer.position_history) == 1
        assert analyzer.position_history[0] == (100, 150)
        assert len(analyzer.time_history) == 1
        assert analyzer.time_history[0] == 10
    
    def test_calculate_speed(self):
        """速度計算のテスト"""
        analyzer = ReelAnalyzer()
        
        # 複数の位置を追加（等速下降）
        positions = [(100, 100), (100, 110), (100, 120), (100, 130)]
        for i, pos in enumerate(positions):
            analyzer.add_position(pos, i+1)
        
        # 内部メソッド呼び出し
        analyzer._calculate_speed()
        
        # 速度は約10px/frameのはず
        assert abs(analyzer.rotation_speed - 10.0) < 0.1
    
    def test_detect_cycle(self):
        """周期検出のテスト"""
        analyzer = ReelAnalyzer()
        
        # 周期的な動きをシミュレート（100pxごとにパターン繰り返し）
        frame = 0
        for i in range(3):  # 3周期分
            # 下降フェーズ
            for y in range(0, 100, 10):
                analyzer.add_position((100, y), frame)
                frame += 1
        
        # 周期検出を実行
        analyzer._detect_cycle()
        
        # 10フレームで1周期のはず
        assert abs(analyzer.cycle_frames - 10) <= 1
        # 安定した周期と判断されるべき
        assert analyzer.is_stable is True
    
    def test_calculate_speed_getter(self):
        """速度取得のテスト"""
        analyzer = ReelAnalyzer()
        analyzer.rotation_speed = 15.5
        
        assert analyzer.calculate_speed() == 15.5
    
    def test_detect_cycle_getter(self):
        """周期取得のテスト"""
        analyzer = ReelAnalyzer()
        analyzer.cycle_frames = 20
        
        assert analyzer.detect_cycle() == 20
    
    def test_is_rotation_stable(self):
        """安定判定のテスト"""
        analyzer = ReelAnalyzer()
        
        # デフォルトは安定していない
        assert analyzer.is_rotation_stable() is False
        
        # 安定状態に設定
        analyzer.is_stable = True
        assert analyzer.is_rotation_stable() is True


class TestTimingPredictor:
    """TimingPredictorクラスのテスト"""
    
    @pytest.fixture
    def reel_analyzer(self):
        """ReelAnalyzerのモックインスタンス"""
        analyzer = ReelAnalyzer()
        analyzer.rotation_speed = 10.0
        analyzer.cycle_frames = 30
        analyzer.is_stable = True
        return analyzer
    
    def test_init(self, reel_analyzer):
        """初期化処理のテスト"""
        predictor = TimingPredictor(reel_analyzer, human_delay=4)
        
        assert predictor.reel_analyzer == reel_analyzer
        assert len(predictor.machine_profiles) > 0
        assert "default" in predictor.machine_profiles
        assert predictor.current_machine == "default"
        assert predictor.human_delay == 4
    
    def test_register_machine_profile(self, reel_analyzer):
        """機種プロファイル登録のテスト"""
        predictor = TimingPredictor(reel_analyzer)
        
        predictor.register_machine_profile(
            "test_machine", 
            slip_frames=2, 
            pull_in_range=4, 
            button_to_stop_frames=3
        )
        
        assert "test_machine" in predictor.machine_profiles
        profile = predictor.machine_profiles["test_machine"]
        assert profile["slip_frames"] == 2
        assert profile["pull_in_range"] == 4
        assert profile["button_to_stop_frames"] == 3
    
    def test_set_current_machine(self, reel_analyzer):
        """機種設定のテスト"""
        predictor = TimingPredictor(reel_analyzer)
        
        # 有効な機種名
        predictor.register_machine_profile("machine1", 1, 1, 1)
        result = predictor.set_current_machine("machine1")
        
        assert result is True
        assert predictor.current_machine == "machine1"
        
        # 無効な機種名
        result = predictor.set_current_machine("nonexistent")
        
        assert result is False
        assert predictor.current_machine == "machine1"  # 変わらない
    
    def test_set_human_delay(self, reel_analyzer):
        """反応遅延設定のテスト"""
        predictor = TimingPredictor(reel_analyzer)
        
        predictor.set_human_delay(7)
        
        assert predictor.human_delay == 7
    
    def test_predict_timing(self, reel_analyzer):
        """タイミング予測のテスト"""
        predictor = TimingPredictor(reel_analyzer)
        
        # テスト用の目標図柄
        target_symbol = {
            "name": "test_symbol",
            "x": 100,
            "y": 150,
            "width": 50,
            "height": 50
        }
        
        # 目標位置
        target_position = 200
        
        # 現在フレーム
        current_frame = 100
        
        # タイミング予測実行
        result = predictor.predict_timing(target_symbol, target_position, current_frame)
        
        # 結果確認
        assert "optimal_frame" in result
        assert "frames_until_push" in result
        assert "accuracy" in result
        assert "is_reliable" in result
        assert "is_in_pull_in_range" in result
        assert "distance_pixels" in result
        assert "rotation_speed" in result
        assert "cycle_frames" in result
        
        # 距離 = 目標位置 - 現在位置 = 50px
        assert abs(result["distance_pixels"] - 50) < 0.1
        
        # 距離 / 速度 = フレーム数 = 50 / 10 = 5フレーム
        # 補正 = ボタン〜停止 + すべり - 人間遅延 = 2 + 1 - 5 = -2
        # 最適フレーム = 現在 + 距離フレーム - 補正 = 100 + 5 - (-2) = 107
        expected_optimal_frame = current_frame + int(50 / 10) + 2
        assert abs(result["optimal_frame"] - expected_optimal_frame) <= 1
        
        # 回転速度と周期がReelAnalyzerから取得されていること
        assert result["rotation_speed"] == 10.0
        assert result["cycle_frames"] == 30
        
        # 安定状態が反映されていること
        assert result["is_reliable"] is True
        assert result["accuracy"] == 1.0


class TestTimingManager:
    """TimingManagerクラスのテスト"""
    
    def test_init(self):
        """初期化処理のテスト"""
        manager = TimingManager(reel_count=3)
        
        assert len(manager.reel_analyzers) == 3
        assert all(isinstance(analyzer, ReelAnalyzer) for analyzer in manager.reel_analyzers.values())
        assert isinstance(manager.timing_predictor, TimingPredictor)
        assert manager.target_symbols == {}
        assert manager.frame_counter == 0
    
    def test_update(self):
        """更新処理のテスト"""
        manager = TimingManager(reel_count=1)
        
        # テスト用のリール図柄
        reel_symbols = {
            1: [
                {"name": "symbol1", "x": 100, "y": 150, "width": 50, "height": 50}
            ]
        }
        
        # 目標図柄設定
        manager.set_target_symbol(1, "symbol1", 200)
        
        # 更新実行
        result = manager.update(reel_symbols)
        
        # フレームカウンターが更新されること
        assert manager.frame_counter == 1
        
        # リール1のタイミング結果があること
        assert 1 in result
        assert "optimal_frame" in result[1]
        assert "frames_until_push" in result[1]
    
    def test_set_target_symbol(self):
        """目標図柄設定のテスト"""
        manager = TimingManager()
        
        manager.set_target_symbol(2, "target_symbol", 150)
        
        assert 2 in manager.target_symbols
        assert manager.target_symbols[2]["name"] == "target_symbol"
        assert manager.target_symbols[2]["position"] == 150
    
    def test_clear_target_symbol(self):
        """目標図柄クリアのテスト"""
        manager = TimingManager()
        
        # 目標図柄を設定
        manager.set_target_symbol(1, "symbol1", 100)
        assert 1 in manager.target_symbols
        
        # クリア実行
        manager.clear_target_symbol(1)
        
        # 削除されていることを確認
        assert 1 not in manager.target_symbols
    
    def test_reset_frame_counter(self):
        """フレームカウンターリセットのテスト"""
        manager = TimingManager()
        
        # フレームカウンターを進める
        manager.frame_counter = 42
        
        # リセット実行
        manager.reset_frame_counter()
        
        # 0にリセットされていること
        assert manager.frame_counter == 0
