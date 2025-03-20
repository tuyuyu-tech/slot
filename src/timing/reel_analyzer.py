"""
パチスロリールの回転特性を解析し、ビタ押しタイミングを計算するモジュール。
"""
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReelAnalyzer:
    """
    リール回転の速度や周期を解析するクラス。
    
    Attributes:
        position_history (deque): 図柄位置の履歴
        time_history (deque): 位置履歴の時間情報
        rotation_speed (float): 計算されたリール回転速度（ピクセル/フレーム）
        cycle_frames (int): 一周期のフレーム数
        is_stable (bool): 回転が安定しているかどうか
    """
    
    def __init__(self, history_size: int = 30):
        """
        ReelAnalyzerクラスの初期化。
        
        Args:
            history_size (int, optional): 保持する位置履歴の最大サイズ。デフォルトは30。
        """
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.rotation_speed = 0.0
        self.cycle_frames = 0
        self.is_stable = False
    
    def add_position(self, position: Tuple[int, int], frame_index: int) -> None:
        """
        図柄位置情報を履歴に追加する。
        
        Args:
            position (Tuple[int, int]): 図柄のx, y座標
            frame_index (int): フレーム番号
        """
        self.position_history.append(position)
        self.time_history.append(frame_index)
        
        # 十分なデータが集まったらリール速度を更新
        if len(self.position_history) >= 3:
            self._calculate_speed()
            self._detect_cycle()
    
    def _calculate_speed(self) -> None:
        """
        位置履歴から回転速度を計算する。
        主にY軸方向（縦）の移動速度を計算。
        """
        speeds = []
        
        # 隣接するフレーム間の移動量を計算
        for i in range(1, len(self.position_history)):
            prev_x, prev_y = self.position_history[i-1]
            curr_x, curr_y = self.position_history[i]
            prev_time = self.time_history[i-1]
            curr_time = self.time_history[i]
            
            # 時間差がある場合のみ速度を計算
            time_diff = curr_time - prev_time
            if time_diff > 0:
                # Y軸方向の速度（ピクセル/フレーム）を計算
                speed_y = (curr_y - prev_y) / time_diff
                speeds.append(speed_y)
        
        # 外れ値を除外するため中央値を使用
        if speeds:
            self.rotation_speed = np.median(speeds)
            logger.debug(f"回転速度計算: {self.rotation_speed:.2f} px/frame")
    
    def _detect_cycle(self) -> None:
        """
        位置履歴から回転周期を検出する。
        周期的なパターンを見つけて安定しているかを判定。
        """
        if len(self.position_history) < 10:
            return
        
        # Y座標の変化を解析
        y_positions = [pos[1] for pos in self.position_history]
        frame_indices = list(self.time_history)
        
        # 連続する下降を見つける（リール回転の特徴）
        descending_segments = []
        start_idx = 0
        
        for i in range(1, len(y_positions)):
            if y_positions[i] < y_positions[i-1]:
                # 下降が継続中
                continue
            elif i - start_idx > 3:
                # 十分な長さの下降セグメントを追加
                descending_segments.append((start_idx, i-1))
                start_idx = i
            else:
                # 短すぎるセグメントはスキップ
                start_idx = i
        
        # 最後のセグメントを追加（必要なら）
        if start_idx < len(y_positions) - 3:
            descending_segments.append((start_idx, len(y_positions) - 1))
        
        # 周期フレーム数の計算（2つ以上のセグメントが必要）
        if len(descending_segments) >= 2:
            cycle_lengths = []
            
            for i in range(1, len(descending_segments)):
                prev_start = descending_segments[i-1][0]
                curr_start = descending_segments[i][0]
                cycle_length = frame_indices[curr_start] - frame_indices[prev_start]
                cycle_lengths.append(cycle_length)
            
            if cycle_lengths:
                # 中央値をとって周期フレーム数とする
                self.cycle_frames = int(np.median(cycle_lengths))
                
                # 周期の安定性を評価
                variation = np.std(cycle_lengths) / np.mean(cycle_lengths) if np.mean(cycle_lengths) > 0 else 1.0
                self.is_stable = variation < 0.1  # 変動係数が10%未満なら安定と判断
                
                logger.debug(f"周期検出: {self.cycle_frames} frames, 安定性: {self.is_stable}")
    
    def calculate_speed(self) -> float:
        """
        現在の回転速度を取得する。
        
        Returns:
            float: 回転速度（ピクセル/フレーム）
        """
        return self.rotation_speed
    
    def detect_cycle(self) -> int:
        """
        検出された回転周期を取得する。
        
        Returns:
            int: 一周期のフレーム数
        """
        return self.cycle_frames
    
    def is_rotation_stable(self) -> bool:
        """
        回転が安定しているかどうかを返す。
        
        Returns:
            bool: 回転が安定している場合はTrue
        """
        return self.is_stable


class TimingPredictor:
    """
    ビタ押しタイミングを予測するクラス。
    リールの回転特性とゲーム機の挙動を考慮してタイミングを算出する。
    
    Attributes:
        reel_analyzer (ReelAnalyzer): リール解析器
        machine_profiles (Dict[str, Any]): 機種別の特性データ
        current_machine (str): 現在使用中の機種プロファイル
        human_delay (int): 人間の反応遅延（フレーム数）
    """
    
    def __init__(self, reel_analyzer: ReelAnalyzer, human_delay: int = 5):
        """
        TimingPredictorクラスの初期化。
        
        Args:
            reel_analyzer (ReelAnalyzer): リール解析器インスタンス
            human_delay (int, optional): 人間の反応遅延（フレーム数）。デフォルトは5。
        """
        self.reel_analyzer = reel_analyzer
        self.machine_profiles = {
            'default': {
                'slip_frames': 1,  # デフォルトのすべりフレーム数
                'pull_in_range': 3,  # 引き込み範囲（コマ数）
                'button_to_stop_frames': 2  # ボタンを押してからリールが停止するまでのフレーム数
            }
        }
        self.current_machine = 'default'
        self.human_delay = human_delay
    
    def register_machine_profile(self, 
                                 name: str, 
                                 slip_frames: int, 
                                 pull_in_range: int, 
                                 button_to_stop_frames: int) -> None:
        """
        新しい機種プロファイルを登録する。
        
        Args:
            name (str): 機種名
            slip_frames (int): すべりフレーム数
            pull_in_range (int): 引き込み範囲（コマ数）
            button_to_stop_frames (int): ボタンを押してからリールが停止するまでのフレーム数
        """
        self.machine_profiles[name] = {
            'slip_frames': slip_frames,
            'pull_in_range': pull_in_range,
            'button_to_stop_frames': button_to_stop_frames
        }
        logger.info(f"機種プロファイルを登録しました: {name}")
    
    def set_current_machine(self, name: str) -> bool:
        """
        使用する機種プロファイルを設定する。
        
        Args:
            name (str): 機種名
            
        Returns:
            bool: 設定が成功したかどうか
        """
        if name in self.machine_profiles:
            self.current_machine = name
            logger.info(f"現在の機種プロファイルを設定: {name}")
            return True
        else:
            logger.warning(f"指定された機種プロファイルが見つかりません: {name}")
            return False
    
    def set_human_delay(self, delay_frames: int) -> None:
        """
        人間の反応遅延を設定する。
        
        Args:
            delay_frames (int): 反応遅延のフレーム数
        """
        self.human_delay = delay_frames
        logger.info(f"人間の反応遅延を設定: {delay_frames} frames")
    
    def predict_timing(self, 
                       target_symbol: Dict[str, Any], 
                       target_position: int, 
                       current_frame: int) -> Dict[str, Any]:
        """
        最適なビタ押しタイミングを予測する。
        
        Args:
            target_symbol (Dict[str, Any]): 目標図柄の情報
            target_position (int): 停止させたい位置（y座標）
            current_frame (int): 現在のフレーム番号
            
        Returns:
            Dict[str, Any]: タイミング予測結果
        """
        # リール回転が安定していない場合は信頼性の低い予測であることを示す
        is_reliable = self.reel_analyzer.is_rotation_stable()
        
        # 回転速度と周期を取得
        rotation_speed = self.reel_analyzer.calculate_speed()
        cycle_frames = self.reel_analyzer.detect_cycle()
        
        # 回転データが利用できない場合はデフォルト値を使用
        if rotation_speed == 0:
            rotation_speed = 10.0  # 仮定値
        
        if cycle_frames == 0:
            cycle_frames = 30  # 仮定値
        
        # 機種プロファイルを取得
        profile = self.machine_profiles[self.current_machine]
        slip_frames = profile['slip_frames']
        pull_in_range = profile['pull_in_range']
        button_to_stop_frames = profile['button_to_stop_frames']
        
        # 現在の図柄位置
        current_position = target_symbol['y']
        
        # 目標位置までのフレーム数を計算
        distance = target_position - current_position
        
        # 距離がマイナスの場合（目標位置が現在位置より上にある場合）
        # 一周分加える必要がある
        if distance < 0:
            # 一周のピクセル数を推定
            cycle_pixels = abs(rotation_speed) * cycle_frames
            distance += cycle_pixels
        
        # 距離をフレーム数に変換
        frames_to_target = distance / abs(rotation_speed) if rotation_speed != 0 else 0
        
        # 補正：機種特性とヒューマンディレイを考慮
        correction_frames = button_to_stop_frames + slip_frames - self.human_delay
        
        # 最終的なビタ押しタイミング（フレーム数）
        optimal_frame = current_frame + int(frames_to_target) - correction_frames
        
        # 予測精度（信頼度スコア）
        accuracy = 1.0 if is_reliable else 0.5
        
        # 引き込み範囲内かどうかを計算
        # 実際の引き込み範囲はリールコマ単位だが、ここではピクセルとフレームで近似
        pull_in_pixels = pull_in_range * (cycle_pixels / 21)  # 一般的なリールは21コマ
        is_in_pull_in_range = distance <= pull_in_pixels
        
        # 結果を返す
        return {
            'optimal_frame': optimal_frame,
            'frames_until_push': optimal_frame - current_frame,
            'accuracy': accuracy,
            'is_reliable': is_reliable,
            'is_in_pull_in_range': is_in_pull_in_range,
            'distance_pixels': distance,
            'rotation_speed': rotation_speed,
            'cycle_frames': cycle_frames
        }


class TimingManager:
    """
    タイミング分析と通知を管理するクラス。
    リール回転解析と目標図柄のタイミング予測を統合する。
    
    Attributes:
        reel_analyzers (Dict[str, ReelAnalyzer]): リール番号ごとのリール解析器
        timing_predictor (TimingPredictor): タイミング予測器
        target_symbols (Dict[int, Dict]): リール番号ごとの目標図柄
        frame_counter (int): 現在のフレームカウンター
    """
    
    def __init__(self, reel_count: int = 3):
        """
        TimingManagerクラスの初期化。
        
        Args:
            reel_count (int, optional): リールの数。デフォルトは3。
        """
        # リール番号ごとにリール解析器を初期化
        self.reel_analyzers = {}
        for i in range(1, reel_count + 1):
            self.reel_analyzers[i] = ReelAnalyzer()
        
        # タイミング予測器を初期化
        self.timing_predictor = TimingPredictor(next(iter(self.reel_analyzers.values())))
        
        # リール番号ごとの目標図柄
        self.target_symbols = {}
        
        # フレームカウンター
        self.frame_counter = 0
    
    def update(self, reel_symbols: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """
        リール解析と予測を更新する。
        
        Args:
            reel_symbols (Dict[int, List[Dict[str, Any]]]): リール番号ごとに認識された図柄のリスト
            
        Returns:
            Dict[int, Dict[str, Any]]: リール番号ごとのタイミング予測結果
        """
        # フレームカウンターを更新
        self.frame_counter += 1
        
        timing_results = {}
        
        # 各リールについて処理
        for reel_id, symbols in reel_symbols.items():
            if reel_id not in self.reel_analyzers:
                continue
            
            reel_analyzer = self.reel_analyzers[reel_id]
            
            # 各図柄の位置を解析器に追加
            for symbol in symbols:
                position = (symbol['x'], symbol['y'])
                reel_analyzer.add_position(position, self.frame_counter)
            
            # 目標図柄が設定されており、リール上に存在する場合
            if reel_id in self.target_symbols and symbols:
                target_name = self.target_symbols[reel_id].get('name')
                target_position = self.target_symbols[reel_id].get('position', 0)
                
                # 目標図柄をリール上で見つける
                target_symbol = None
                for symbol in symbols:
                    if symbol['name'] == target_name:
                        target_symbol = symbol
                        break
                
                # 目標図柄が見つかった場合、タイミングを予測
                if target_symbol:
                    # 現在使用中のリール解析器をタイミング予測器に設定
                    self.timing_predictor.reel_analyzer = reel_analyzer
                    
                    # タイミングを予測
                    timing_result = self.timing_predictor.predict_timing(
                        target_symbol=target_symbol,
                        target_position=target_position,
                        current_frame=self.frame_counter
                    )
                    
                    timing_results[reel_id] = timing_result
        
        return timing_results
    
    def set_target_symbol(self, reel_id: int, symbol_name: str, position: int) -> None:
        """
        リールの目標図柄を設定する。
        
        Args:
            reel_id (int): リール番号
            symbol_name (str): 目標図柄の名前
            position (int): 目標停止位置（y座標）
        """
        self.target_symbols[reel_id] = {
            'name': symbol_name,
            'position': position
        }
        logger.info(f"リール{reel_id}の目標図柄を設定: {symbol_name}, 位置: {position}")
    
    def clear_target_symbol(self, reel_id: int) -> None:
        """
        リールの目標図柄設定をクリアする。
        
        Args:
            reel_id (int): リール番号
        """
        if reel_id in self.target_symbols:
            del self.target_symbols[reel_id]
            logger.info(f"リール{reel_id}の目標図柄をクリアしました")
    
    def reset_frame_counter(self) -> None:
        """
        フレームカウンターをリセットする。
        新しい解析セッションを開始するときに使用。
        """
        self.frame_counter = 0
        logger.info("フレームカウンターをリセットしました")
    
    def reset_timing_data(self) -> None:
        """
        タイミング関連のデータをすべてリセットする。
        図柄表示のリセット時に使用。
        """
        # 目標図柄情報をクリア
        self.target_symbols = {}
        
        # 各リール解析器をリセット
        for reel_id, analyzer in self.reel_analyzers.items():
            analyzer.position_history.clear()
            analyzer.time_history.clear()
            analyzer.rotation_speed = 0.0
            analyzer.cycle_frames = 0
            analyzer.is_stable = False
        
        # フレームカウンターをリセット
        self.reset_frame_counter()
        
        logger.info("タイミングデータをリセットしました")
