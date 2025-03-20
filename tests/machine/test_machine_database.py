"""
MachineDatabase クラスのテスト。
"""
import os
import sys
import shutil
import pytest
import numpy as np
import cv2
import json
import tempfile
from unittest.mock import patch, MagicMock

# テスト対象のモジュールを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.machine.machine_database import MachineDatabase, MachineDetector


class TestMachineDatabase:
    """MachineDatabase クラスのテスト。"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用の一時ディレクトリを作成する。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def machine_db(self, temp_dir):
        """テスト用の MachineDatabase インスタンスを作成する。"""
        return MachineDatabase(data_dir=temp_dir)
    
    def test_init(self, machine_db, temp_dir):
        """初期化のテスト。"""
        assert machine_db.current_machine == "default"
        assert "default" in machine_db.machines
        assert os.path.exists(os.path.join(temp_dir, "machines.json"))
    
    def test_add_machine(self, machine_db):
        """機種追加のテスト。"""
        # テスト用の機種データ
        machine_data = {
            "slip_frames": 2,
            "pull_in_range": 4,
            "button_to_stop_frames": 3
        }
        
        # 機種を追加
        result = machine_db.add_machine("test_machine", machine_data)
        
        # 追加結果の確認
        assert result is True
        assert "test_machine" in machine_db.machines
        assert machine_db.machines["test_machine"]["slip_frames"] == 2
        assert machine_db.machines["test_machine"]["pull_in_range"] == 4
        assert machine_db.machines["test_machine"]["button_to_stop_frames"] == 3
        assert "added_date" in machine_db.machines["test_machine"]
    
    def test_get_machine(self, machine_db):
        """機種取得のテスト。"""
        # テスト用の機種データ
        machine_data = {
            "slip_frames": 2,
            "pull_in_range": 4,
            "button_to_stop_frames": 3
        }
        
        # 機種を追加
        machine_db.add_machine("test_machine", machine_data)
        
        # 機種取得
        machine = machine_db.get_machine("test_machine")
        
        # 取得結果の確認
        assert machine is not None
        assert machine["slip_frames"] == 2
        assert machine["pull_in_range"] == 4
        assert machine["button_to_stop_frames"] == 3
        
        # 存在しない機種
        machine = machine_db.get_machine("non_existent")
        assert machine is None
    
    def test_remove_machine(self, machine_db):
        """機種削除のテスト。"""
        # テスト用の機種データ
        machine_data = {
            "slip_frames": 2,
            "pull_in_range": 4,
            "button_to_stop_frames": 3
        }
        
        # 機種を追加
        machine_db.add_machine("test_machine", machine_data)
        
        # 機種削除
        result = machine_db.remove_machine("test_machine")
        
        # 削除結果の確認
        assert result is True
        assert "test_machine" not in machine_db.machines
        
        # デフォルト機種は削除できない
        result = machine_db.remove_machine("default")
        assert result is False
        assert "default" in machine_db.machines
        
        # 存在しない機種
        result = machine_db.remove_machine("non_existent")
        assert result is False
    
    def test_set_current_machine(self, machine_db):
        """現在の機種設定のテスト。"""
        # テスト用の機種データ
        machine_data = {
            "slip_frames": 2,
            "pull_in_range": 4,
            "button_to_stop_frames": 3
        }
        
        # 機種を追加
        machine_db.add_machine("test_machine", machine_data)
        
        # 現在の機種を設定
        result = machine_db.set_current_machine("test_machine")
        
        # 設定結果の確認
        assert result is True
        assert machine_db.current_machine == "test_machine"
        
        # 存在しない機種
        result = machine_db.set_current_machine("non_existent")
        assert result is False
        assert machine_db.current_machine == "test_machine"  # 変更されていない
    
    def test_get_all_machine_names(self, machine_db):
        """全機種名取得のテスト。"""
        # 初期状態ではdefaultのみ
        names = machine_db.get_all_machine_names()
        assert names == ["default"]
        
        # 機種を追加
        machine_db.add_machine("test_machine1", {"slip_frames": 2, "pull_in_range": 4, "button_to_stop_frames": 3})
        machine_db.add_machine("test_machine2", {"slip_frames": 1, "pull_in_range": 5, "button_to_stop_frames": 2})
        
        # 全機種名を取得
        names = machine_db.get_all_machine_names()
        
        # 結果の確認
        assert len(names) == 3
        assert "default" in names
        assert "test_machine1" in names
        assert "test_machine2" in names
    
    def test_store_reel_array(self, machine_db):
        """リール配列保存のテスト。"""
        # テスト用の機種データ
        machine_data = {
            "slip_frames": 2,
            "pull_in_range": 4,
            "button_to_stop_frames": 3
        }
        
        # 機種を追加
        machine_db.add_machine("test_machine", machine_data)
        
        # リール配列
        reel_array = [
            ["7", "BAR", "ベル", "スイカ"],
            ["BAR", "7", "スイカ", "ベル"],
            ["ベル", "スイカ", "7", "BAR"]
        ]
        
        # リール配列を保存
        result = machine_db.store_reel_array("test_machine", reel_array)
        
        # 保存結果の確認
        assert result is True
        assert machine_db.machines["test_machine"]["reel_array"] == reel_array
        
        # 存在しない機種
        result = machine_db.store_reel_array("non_existent", reel_array)
        assert result is False
        
        # 不正なリール配列
        result = machine_db.store_reel_array("test_machine", "not_an_array")
        assert result is False
        
        result = machine_db.store_reel_array("test_machine", [])
        assert result is False
        
        result = machine_db.store_reel_array("test_machine", ["not_array"])
        assert result is False


class TestMachineDetector:
    """MachineDetector クラスのテスト。"""
    
    @pytest.fixture
    def temp_dir(self):
        """テスト用の一時ディレクトリを作成する。"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def machines(self):
        """テスト用の機種データ。"""
        return {
            "default": {
                "name": "デフォルト",
                "slip_frames": 1,
                "pull_in_range": 3,
                "button_to_stop_frames": 2,
                "visual_features": {}
            },
            "machine1": {
                "name": "機種1",
                "slip_frames": 2,
                "pull_in_range": 4,
                "button_to_stop_frames": 3,
                "visual_features": {
                    "general": [
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4]
                    ]
                }
            },
            "machine2": {
                "name": "機種2",
                "slip_frames": 1,
                "pull_in_range": 5,
                "button_to_stop_frames": 2,
                "visual_features": {
                    "general": [
                        [0.5, 0.6, 0.7],
                        [0.6, 0.7, 0.8]
                    ]
                }
            }
        }
    
    @pytest.fixture
    def detector(self, temp_dir, machines):
        """テスト用の MachineDetector インスタンスを作成する。"""
        return MachineDetector(machines, model_dir=temp_dir)
    
    def test_train(self, detector):
        """モデル学習のテスト。"""
        # モックモデルを作成
        mock_svc = MagicMock()
        mock_scaler = MagicMock()
        
        # SVCとStandardScalerをモック化
        with patch('src.machine.machine_database.SVC', return_value=mock_svc):
            with patch('src.machine.machine_database.StandardScaler', return_value=mock_scaler):
                # モデル学習
                result = detector.train()
                
                # 学習結果の確認
                assert result is True
                assert detector.model_trained is True
                assert detector.model is mock_svc
                assert detector.scaler is mock_scaler
                
                # fitメソッドが呼び出されたことを確認
                mock_svc.fit.assert_called_once()
                mock_scaler.fit_transform.assert_called_once()
    
    def test_detect(self, detector):
        """機種判別のテスト。"""
        # テスト用のフレーム
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # モックモデルを作成
        mock_svc = MagicMock()
        mock_svc.predict.return_value = ["machine1"]
        mock_svc.predict_proba.return_value = [[0.2, 0.7, 0.1]]  # 確率70%でmachine1
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]
        
        # モデルをモック化
        detector.model = mock_svc
        detector.scaler = mock_scaler
        detector.model_trained = True
        
        # 機種判別
        result = detector.detect(frame)
        
        # 判別結果の確認
        assert result == "machine1"
        
        # モデルが学習されていない場合
        detector.model_trained = False
        result = detector.detect(frame)
        assert result == "default"
        
        # 確率が低い場合
        detector.model_trained = True
        mock_svc.predict_proba.return_value = [[0.4, 0.5, 0.1]]  # 確率50%でmachine1
        result = detector.detect(frame)
        assert result == "default"
