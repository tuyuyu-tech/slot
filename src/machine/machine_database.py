"""
パチスロ機種データベースを管理するモジュール。
機種ごとの特性を保存・読み込みし、機種自動判別機能を提供する。
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MachineDatabase:
    """
    パチスロ機種データベースを管理するクラス。
    
    Attributes:
        machines (Dict[str, Dict]): 機種データの辞書
        machine_detector (Optional[MachineDetector]): 機種自動判別クラス
        current_machine (str): 現在選択されている機種
        data_dir (str): データの保存ディレクトリ
    """
    
    def __init__(self, data_dir: str = "../data/machines"):
        """
        MachineDatabase クラスの初期化。
        
        Args:
            data_dir (str): 機種データの保存ディレクトリ
        """
        self.machines = {}
        self.machine_detector = None
        self.current_machine = "default"
        self.data_dir = data_dir
        
        # データディレクトリの作成
        os.makedirs(data_dir, exist_ok=True)
        
        # 保存済みの機種データを読み込み
        self.load_machines()
        
        # 機種判別器の初期化
        self.machine_detector = MachineDetector(self.machines)
    
    def load_machines(self) -> None:
        """保存済みの機種データを読み込む。"""
        machines_file = os.path.join(self.data_dir, "machines.json")
        
        if os.path.exists(machines_file):
            try:
                with open(machines_file, 'r', encoding='utf-8') as f:
                    self.machines = json.load(f)
                logging.info(f"{len(self.machines)}機種のデータを読み込みました")
            except Exception as e:
                logging.error(f"機種データの読み込みに失敗しました: {str(e)}")
                # デフォルト機種を作成
                self._create_default_machine()
        else:
            logging.info("機種データファイルが見つかりません。デフォルト機種を作成します。")
            self._create_default_machine()
    
    def _create_default_machine(self) -> None:
        """デフォルト機種データを作成する。"""
        self.machines = {
            "default": {
                "name": "デフォルト",
                "reel_array": [],  # リール配列情報
                "slip_frames": 1,  # すべりフレーム数
                "pull_in_range": 3,  # 引き込み範囲
                "button_to_stop_frames": 2,  # ボタン押下から停止までのフレーム数
                "visual_features": {},  # 視覚的特徴
                "added_date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.save_machines()
    
    def save_machines(self) -> bool:
        """
        機種データを保存する。
        
        Returns:
            bool: 保存に成功した場合はTrue
        """
        machines_file = os.path.join(self.data_dir, "machines.json")
        
        try:
            with open(machines_file, 'w', encoding='utf-8') as f:
                json.dump(self.machines, f, ensure_ascii=False, indent=2)
            logging.info(f"{len(self.machines)}機種のデータを保存しました")
            return True
        except Exception as e:
            logging.error(f"機種データの保存に失敗しました: {str(e)}")
            return False
    
    def add_machine(self, name: str, machine_data: Dict) -> bool:
        """
        機種データを追加する。
        
        Args:
            name (str): 機種名
            machine_data (Dict): 機種データ
        
        Returns:
            bool: 追加に成功した場合はTrue
        """
        if name in self.machines and name != "default":
            logging.warning(f"機種「{name}」は既に存在します。上書きします。")
        
        # 機種データの検証
        required_keys = ["slip_frames", "pull_in_range", "button_to_stop_frames"]
        for key in required_keys:
            if key not in machine_data:
                logging.error(f"機種データに必須キー「{key}」が含まれていません")
                return False
        
        # 現在の日時を追加
        machine_data["added_date"] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 機種名を追加
        machine_data["name"] = name
        
        # 機種データを登録
        self.machines[name] = machine_data
        
        # データを保存
        success = self.save_machines()
        
        # 機種判別器を更新
        if success and self.machine_detector is not None:
            self.machine_detector.update_machines(self.machines)
        
        return success
    
    def remove_machine(self, name: str) -> bool:
        """
        機種データを削除する。
        
        Args:
            name (str): 機種名
        
        Returns:
            bool: 削除に成功した場合はTrue
        """
        if name == "default":
            logging.error("デフォルト機種は削除できません")
            return False
        
        if name not in self.machines:
            logging.error(f"機種「{name}」は存在しません")
            return False
        
        # 機種データを削除
        del self.machines[name]
        
        # 現在の機種が削除された場合、デフォルトに戻す
        if self.current_machine == name:
            self.current_machine = "default"
        
        # データを保存
        success = self.save_machines()
        
        # 機種判別器を更新
        if success and self.machine_detector is not None:
            self.machine_detector.update_machines(self.machines)
        
        return success
    
    def get_machine(self, name: str) -> Optional[Dict]:
        """
        指定された機種のデータを取得する。
        
        Args:
            name (str): 機種名
        
        Returns:
            Optional[Dict]: 機種データ、存在しない場合はNone
        """
        return self.machines.get(name)
    
    def set_current_machine(self, name: str) -> bool:
        """
        現在の機種を設定する。
        
        Args:
            name (str): 機種名
        
        Returns:
            bool: 設定に成功した場合はTrue
        """
        if name not in self.machines:
            logging.error(f"機種「{name}」は存在しません")
            return False
        
        self.current_machine = name
        logging.info(f"現在の機種を「{name}」に設定しました")
        return True
    
    def get_all_machine_names(self) -> List[str]:
        """
        登録されている全機種名を取得する。
        
        Returns:
            List[str]: 機種名のリスト
        """
        return list(self.machines.keys())
    
    def detect_machine(self, frame: np.ndarray) -> str:
        """
        フレームから機種を自動判別する。
        
        Args:
            frame (np.ndarray): 判別対象のフレーム
        
        Returns:
            str: 判別された機種名（デフォルトは "default"）
        """
        if self.machine_detector is None or len(self.machines) <= 1:
            return "default"
        
        return self.machine_detector.detect(frame)
    
    def add_visual_feature(self, machine_name: str, frame: np.ndarray, label: str = "general") -> bool:
        """
        機種の視覚的特徴を追加する。
        
        Args:
            machine_name (str): 機種名
            frame (np.ndarray): 視覚的特徴を抽出するフレーム
            label (str): 特徴のラベル
        
        Returns:
            bool: 追加に成功した場合はTrue
        """
        if machine_name not in self.machines:
            logging.error(f"機種「{machine_name}」は存在しません")
            return False
        
        # 機種データを取得
        machine_data = self.machines[machine_name]
        
        # 視覚的特徴がない場合は初期化
        if "visual_features" not in machine_data:
            machine_data["visual_features"] = {}
        
        # 特徴抽出
        feature = self._extract_visual_feature(frame)
        
        # ラベル付けされた特徴リストを取得または初期化
        if label not in machine_data["visual_features"]:
            machine_data["visual_features"][label] = []
        
        # 特徴を追加
        machine_data["visual_features"][label].append(feature.tolist())
        
        # データを保存
        success = self.save_machines()
        
        # 機種判別器を更新
        if success and self.machine_detector is not None:
            self.machine_detector.update_machines(self.machines)
            self.machine_detector.train()
        
        return success
    
    def _extract_visual_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームから視覚的特徴を抽出する。
        
        Args:
            frame (np.ndarray): 特徴を抽出するフレーム
        
        Returns:
            np.ndarray: 抽出された特徴ベクトル
        """
        # フレームが大きい場合はリサイズ
        if frame.shape[0] > 300 or frame.shape[1] > 300:
            # アスペクト比を維持してリサイズ
            scale = min(300 / frame.shape[0], 300 / frame.shape[1])
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (width, height))
        
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ヒストグラム特徴
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # エッジ検出
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [2], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        
        # HOG特徴（オプション）
        # win_size = (64, 64)
        # if gray.shape[0] < win_size[0] or gray.shape[1] < win_size[1]:
        #     gray = cv2.resize(gray, win_size)
        # hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        # hog_features = hog.compute(gray)
        
        # 特徴を結合
        features = np.concatenate([hist, edge_hist])
        
        return features
    
    def store_reel_array(self, machine_name: str, reel_array: List[List[str]]) -> bool:
        """
        機種のリール配列情報を保存する。
        
        Args:
            machine_name (str): 機種名
            reel_array (List[List[str]]): リール配列情報
                [[リール1の図柄順], [リール2の図柄順], [リール3の図柄順]]
        
        Returns:
            bool: 保存に成功した場合はTrue
        """
        if machine_name not in self.machines:
            logging.error(f"機種「{machine_name}」は存在しません")
            return False
        
        # リール配列のバリデーション
        if not isinstance(reel_array, list) or len(reel_array) == 0:
            logging.error("リール配列は空でない配列である必要があります")
            return False
        
        for reel in reel_array:
            if not isinstance(reel, list) or len(reel) == 0:
                logging.error("各リールは空でない図柄配列である必要があります")
                return False
        
        # 機種データを更新
        self.machines[machine_name]["reel_array"] = reel_array
        
        # データを保存
        return self.save_machines()


class MachineDetector:
    """
    機種自動判別を行うクラス。
    
    Attributes:
        machines (Dict[str, Dict]): 機種データの辞書
        model (Optional[SVC]): 機種判別モデル
        scaler (Optional[StandardScaler]): 特徴の標準化
        model_trained (bool): モデルが学習済みかどうか
        model_file (str): モデルの保存ファイル
    """
    
    def __init__(self, machines: Dict[str, Dict], model_dir: str = "../data/models"):
        """
        MachineDetector クラスの初期化。
        
        Args:
            machines (Dict[str, Dict]): 機種データの辞書
            model_dir (str): モデルの保存ディレクトリ
        """
        self.machines = machines
        self.model = None
        self.scaler = None
        self.model_trained = False
        
        # モデル保存ディレクトリの作成
        os.makedirs(model_dir, exist_ok=True)
        
        # モデルファイルのパス
        self.model_file = os.path.join(model_dir, "machine_detector.joblib")
        
        # 保存済みモデルの読み込み
        self.load_model()
    
    def update_machines(self, machines: Dict[str, Dict]) -> None:
        """
        機種データを更新する。
        
        Args:
            machines (Dict[str, Dict]): 機種データの辞書
        """
        self.machines = machines
    
    def load_model(self) -> bool:
        """
        保存済みモデルを読み込む。
        
        Returns:
            bool: 読み込みに成功した場合はTrue
        """
        if os.path.exists(self.model_file):
            try:
                model_data = joblib.load(self.model_file)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
                self.model_trained = True
                logging.info("機種判別モデルを読み込みました")
                return True
            except Exception as e:
                logging.error(f"機種判別モデルの読み込みに失敗しました: {str(e)}")
        
        return False
    
    def save_model(self) -> bool:
        """
        モデルを保存する。
        
        Returns:
            bool: 保存に成功した場合はTrue
        """
        if self.model is None or not self.model_trained:
            logging.error("保存可能なモデルがありません")
            return False
        
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler
            }
            joblib.dump(model_data, self.model_file)
            logging.info("機種判別モデルを保存しました")
            return True
        except Exception as e:
            logging.error(f"機種判別モデルの保存に失敗しました: {str(e)}")
            return False
    
    def train(self) -> bool:
        """
        機種判別モデルを学習する。
        
        Returns:
            bool: 学習に成功した場合はTrue
        """
        # 学習データの収集
        X = []
        y = []
        
        for machine_name, machine_data in self.machines.items():
            if "visual_features" not in machine_data:
                continue
            
            for label, features_list in machine_data["visual_features"].items():
                for feature in features_list:
                    X.append(feature)
                    y.append(machine_name)
        
        # 学習データがない場合
        if len(X) == 0 or len(set(y)) <= 1:
            logging.warning("有効な学習データがありません")
            return False
        
        X = np.array(X)
        
        # 特徴の標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # モデルの学習
        self.model = SVC(probability=True)
        self.model.fit(X_scaled, y)
        
        self.model_trained = True
        
        # モデルの保存
        self.save_model()
        
        logging.info(f"機種判別モデルを{len(X)}サンプル、{len(set(y))}クラスで学習しました")
        return True
    
    def detect(self, frame: np.ndarray) -> str:
        """
        フレームから機種を判別する。
        
        Args:
            frame (np.ndarray): 判別対象のフレーム
        
        Returns:
            str: 判別された機種名（デフォルトは "default"）
        """
        if self.model is None or not self.model_trained:
            return "default"
        
        # 特徴抽出
        feature = self._extract_visual_feature(frame)
        
        # 特徴の標準化
        feature_scaled = self.scaler.transform([feature])
        
        # 予測
        try:
            machine_name = self.model.predict(feature_scaled)[0]
            proba = np.max(self.model.predict_proba(feature_scaled))
            
            # 確率が低い場合はデフォルトを返す
            if proba < 0.6:
                logging.info(f"機種判別の確率が低いため({proba:.2f})、デフォルトを使用します")
                return "default"
            
            logging.info(f"機種「{machine_name}」と判別しました（確率: {proba:.2f}）")
            return machine_name
        except Exception as e:
            logging.error(f"機種判別中にエラーが発生しました: {str(e)}")
            return "default"
    
    def _extract_visual_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームから視覚的特徴を抽出する。
        
        Args:
            frame (np.ndarray): 特徴を抽出するフレーム
        
        Returns:
            np.ndarray: 抽出された特徴ベクトル
        """
        # フレームが大きい場合はリサイズ
        if frame.shape[0] > 300 or frame.shape[1] > 300:
            # アスペクト比を維持してリサイズ
            scale = min(300 / frame.shape[0], 300 / frame.shape[1])
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (width, height))
        
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ヒストグラム特徴
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # エッジ検出
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [2], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        
        # 特徴を結合
        features = np.concatenate([hist, edge_hist])
        
        return features
