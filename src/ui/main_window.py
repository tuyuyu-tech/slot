"""
パチスロビタ押しタイミング認識システムのメインウィンドウ。
"""
import sys
import time
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QGroupBox,
    QSpinBox, QFileDialog, QMessageBox, QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect

# 相対インポートのための処理（直接実行時の対応）
if __name__ == "__main__":
    import os
    import sys
    # 現在のファイルの絶対パスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトのルートディレクトリを取得（src/uiの親の親）
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    # Pythonのパスにプロジェクトのルートを追加
    sys.path.append(root_dir)

# 自作モジュールのインポート
from src.capture.screen_capture import ScreenCapture, VideoProcessor
from src.recognition.symbol_recognizer import TemplateMatching, SymbolTracker
from src.recognition.ml_symbol_recognizer import HybridSymbolRecognizer
from src.timing.reel_analyzer import TimingManager
from src.machine.machine_database import MachineDatabase
from src.ui.machine_dialog import MachineDialog


class VideoFrame(QLabel):
    """
    動画フレームを表示するためのカスタムQLabel。
    図柄認識結果やタイミング情報をオーバーレイ表示する。
    
    Attributes:
        frame (np.ndarray): 表示中のフレーム
        symbols (List[Dict]): 認識された図柄情報
        timing_results (Dict): タイミング予測結果
        show_recognition (bool): 認識結果を表示するかどうか
        show_timing (bool): タイミング情報を表示するかどうか
    """
    
    def __init__(self, parent=None):
        """
        VideoFrameクラスの初期化。
        
        Args:
            parent: 親ウィジェット
        """
        super().__init__(parent)
        self.frame = None
        self.symbols = []
        self.timing_results = {}
        self.show_recognition = True
        self.show_timing = True
        
        # 表示サイズの初期化
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
    
    def update_frame(self, frame: np.ndarray) -> None:
        """
        表示するフレームを更新する。
        
        Args:
            frame (np.ndarray): 新しいフレーム
        """
        self.frame = frame
        self._update_display()
    
    def update_symbols(self, symbols: List[Dict]) -> None:
        """
        認識された図柄情報を更新する。
        
        Args:
            symbols (List[Dict]): 認識された図柄情報
        """
        self.symbols = symbols
        self._update_display()
    
    def update_timing(self, timing_results: Dict) -> None:
        """
        タイミング予測結果を更新する。
        
        Args:
            timing_results (Dict): タイミング予測結果
        """
        self.timing_results = timing_results
        self._update_display()
    
    def set_show_recognition(self, show: bool) -> None:
        """
        認識結果の表示・非表示を設定する。
        
        Args:
            show (bool): 表示する場合はTrue
        """
        self.show_recognition = show
        self._update_display()
    
    def set_show_timing(self, show: bool) -> None:
        """
        タイミング情報の表示・非表示を設定する。
        
        Args:
            show (bool): 表示する場合はTrue
        """
        self.show_timing = show
        self._update_display()
    
    def _update_display(self) -> None:
        """
        ディスプレイ表示を更新する。
        フレームと各種情報をオーバーレイ表示。
        """
        if self.frame is None:
            return
        
        # フレームのコピーを作成
        display_frame = self.frame.copy()
        
        # 認識結果を描画
        if self.show_recognition and self.symbols:
            for symbol in self.symbols:
                x, y = symbol['x'], symbol['y']
                w, h = symbol['width'], symbol['height']
                name = symbol.get('name', 'Unknown')
                score = symbol.get('score', 0)
                
                # 認識枠を描画
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 図柄名とスコアを表示
                text = f"{name} ({score:.2f})"
                cv2.putText(display_frame, text, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # タイミング情報を描画
        if self.show_timing and self.timing_results:
            for reel_id, timing in self.timing_results.items():
                # タイミングインジケータを描画
                frames_until_push = timing.get('frames_until_push', 0)
                accuracy = timing.get('accuracy', 0)
                is_reliable = timing.get('is_reliable', False)
                
                # 信頼性に基づいて色を決定
                color = (0, 255, 0) if is_reliable else (0, 165, 255)
                
                # リール番号ごとに表示位置をずらす
                pos_y = 30 + (reel_id - 1) * 60
                
                # タイミングバーを描画
                bar_width = 200
                bar_height = 20
                bar_x = 20
                bar_y = pos_y
                
                # バー背景
                cv2.rectangle(display_frame, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height), 
                             (50, 50, 50), -1)
                
                # タイミングゲージ
                # フレーム数に基づいてゲージの位置を決定
                if -30 <= frames_until_push <= 30:
                    gauge_pos = int(bar_width/2 + frames_until_push * (bar_width/60))
                    gauge_pos = max(0, min(bar_width, gauge_pos))
                    
                    # ゲージの色（近いほど緑、遠いほど赤）
                    closeness = 1.0 - min(1.0, abs(frames_until_push) / 30)
                    gauge_color = (
                        int(255 * (1.0 - closeness)),  # B
                        int(255 * closeness),          # G
                        0                             # R
                    )
                    
                    # ゲージを描画
                    cv2.rectangle(display_frame, 
                                 (bar_x + gauge_pos - 5, bar_y), 
                                 (bar_x + gauge_pos + 5, bar_y + bar_height), 
                                 gauge_color, -1)
                
                # 中央線
                cv2.line(display_frame, 
                        (bar_x + bar_width//2, bar_y), 
                        (bar_x + bar_width//2, bar_y + bar_height), 
                        (200, 200, 200), 1)
                
                # テキスト情報
                text = f"リール{reel_id}: {frames_until_push:+d}フレーム"
                cv2.putText(display_frame, text, 
                           (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # ビタ押しタイミングに近い場合、大きな通知を表示
                if -2 <= frames_until_push <= 2:
                    cv2.putText(display_frame, "PUSH!", 
                               (bar_x + bar_width + 20, bar_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # OpenCV画像をQPixmapに変換
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        # 表示サイズに合わせて拡大・縮小（スムーズなスケーリングを追加）
        pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 画像を表示
        self.setPixmap(pixmap)


class CaptureAreaSelector(QWidget):
    """
    キャプチャ領域を選択するためのウィジェット。
    """
    area_selected = pyqtSignal(int, int, int, int)
    
    def __init__(self, parent=None):
        """
        CaptureAreaSelectorクラスの初期化。
        
        Args:
            parent: 親ウィジェット
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """UIの初期化。"""
        layout = QVBoxLayout(self)
        
        # キャプチャ領域設定グループ
        area_group = QGroupBox("キャプチャ領域設定")
        area_layout = QGridLayout()
        
        # X座標
        area_layout.addWidget(QLabel("左端 X:"), 0, 0)
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 3840)
        self.x_spin.setValue(0)
        area_layout.addWidget(self.x_spin, 0, 1)
        
        # Y座標
        area_layout.addWidget(QLabel("上端 Y:"), 0, 2)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 2160)
        self.y_spin.setValue(0)
        area_layout.addWidget(self.y_spin, 0, 3)
        
        # 幅
        area_layout.addWidget(QLabel("幅:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(50, 3840)
        self.width_spin.setValue(640)
        area_layout.addWidget(self.width_spin, 1, 1)
        
        # 高さ
        area_layout.addWidget(QLabel("高さ:"), 1, 2)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(50, 2160)
        self.height_spin.setValue(480)
        area_layout.addWidget(self.height_spin, 1, 3)
        
        # 適用ボタン
        self.apply_button = QPushButton("適用")
        self.apply_button.clicked.connect(self.on_apply)
        area_layout.addWidget(self.apply_button, 2, 0, 1, 4)
        
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)
        
        # 余白を追加
        layout.addStretch()
    
    def on_apply(self):
        """適用ボタンが押されたときの処理。"""
        x = self.x_spin.value()
        y = self.y_spin.value()
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.area_selected.emit(x, y, width, height)


class SymbolRegistration(QWidget):
    """
    図柄テンプレートを登録するためのウィジェット。
    """
    symbol_registered = pyqtSignal(str, np.ndarray, float, dict)
    
    def __init__(self, parent=None):
        """
        SymbolRegistrationクラスの初期化。
        
        Args:
            parent: 親ウィジェット
        """
        super().__init__(parent)
        self.capture_frame = None
        self.selection_rect = None
        self.initUI()
    
    def initUI(self):
        """UIの初期化。"""
        layout = QVBoxLayout(self)
        
        # 図柄登録グループ
        reg_group = QGroupBox("図柄テンプレート登録")
        reg_layout = QVBoxLayout()
        
        # フレーム表示ラベル
        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(320, 240)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("border: 1px solid gray;")
        self.frame_label.mousePressEvent = self.start_selection
        self.frame_label.mouseMoveEvent = self.update_selection
        self.frame_label.mouseReleaseEvent = self.end_selection
        reg_layout.addWidget(self.frame_label)
        
        # 名前入力
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("図柄名:"))
        self.name_edit = QComboBox()
        self.name_edit.setEditable(True)
        # 一般的なパチスロ図柄を追加
        common_symbols = ["7", "BAR", "ベル", "スイカ", "チェリー", "リプレイ"]
        self.name_edit.addItems(common_symbols)
        name_layout.addWidget(self.name_edit)
        reg_layout.addLayout(name_layout)
        
        # しきい値
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("マッチング閾値:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(80)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_value = QLabel("0.80")
        threshold_layout.addWidget(self.threshold_value)
        reg_layout.addLayout(threshold_layout)
        
        # スライダーの値が変更されたときの処理
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        
        # 登録ボタン
        self.register_button = QPushButton("テンプレート登録")
        self.register_button.clicked.connect(self.register_template)
        self.register_button.setEnabled(False)
        reg_layout.addWidget(self.register_button)
        
        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)
        
        # 余白を追加
        layout.addStretch()
    
    def update_threshold_value(self):
        """閾値スライダーの値が変更されたときの処理。"""
        value = self.threshold_slider.value() / 100.0
        self.threshold_value.setText(f"{value:.2f}")
    
    def update_frame(self, frame: np.ndarray) -> None:
        """
        表示するフレームを更新する。
        
        Args:
            frame (np.ndarray): 新しいフレーム
        """
        self.capture_frame = frame
        self._update_display()
    
    def _update_display(self) -> None:
        """ディスプレイ表示を更新する。"""
        if self.capture_frame is None:
            return
        
        # フレームのコピーを作成
        display_frame = self.capture_frame.copy()
        
        # 選択領域を描画
        if self.selection_rect is not None:
            x, y, w, h = self.selection_rect
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # OpenCV画像をQPixmapに変換
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        # 表示サイズに合わせて拡大・縮小（スムーズなスケーリングを追加）
        pixmap = pixmap.scaled(self.frame_label.width(), self.frame_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 画像を表示
        self.frame_label.setPixmap(pixmap)
    
    def start_selection(self, event):
        """
        マウス押下で選択開始。
        
        Args:
            event: マウスイベント
        """
        if self.capture_frame is None:
            return
        
        # ラベル上のマウス位置を取得
        pos = event.pos()
        pixmap = self.frame_label.pixmap()
        
        if pixmap:
            # 表示画像の実際のサイズを取得
            img_rect = pixmap.rect()
            img_rect.moveCenter(self.frame_label.rect().center())
            
            if img_rect.contains(pos):
                # 画像座標系に変換
                x_scale = self.capture_frame.shape[1] / img_rect.width()
                y_scale = self.capture_frame.shape[0] / img_rect.height()
                
                x = int((pos.x() - img_rect.left()) * x_scale)
                y = int((pos.y() - img_rect.top()) * y_scale)
                
                # 選択開始点を設定
                self.selection_start = (x, y)
                self.selection_rect = (x, y, 0, 0)
                self._update_display()
    
    def update_selection(self, event):
        """
        マウス移動で選択領域を更新。
        
        Args:
            event: マウスイベント
        """
        if self.capture_frame is None or self.selection_rect is None:
            return
        
        # ラベル上のマウス位置を取得
        pos = event.pos()
        pixmap = self.frame_label.pixmap()
        
        if pixmap:
            # 表示画像の実際のサイズを取得
            img_rect = pixmap.rect()
            img_rect.moveCenter(self.frame_label.rect().center())
            
            if img_rect.contains(pos):
                # 画像座標系に変換
                x_scale = self.capture_frame.shape[1] / img_rect.width()
                y_scale = self.capture_frame.shape[0] / img_rect.height()
                
                x = int((pos.x() - img_rect.left()) * x_scale)
                y = int((pos.y() - img_rect.top()) * y_scale)
                
                # 選択領域を更新
                sx, sy = self.selection_start
                w = x - sx
                h = y - sy
                
                # 負の幅/高さを処理
                if w < 0:
                    sx = x
                    w = abs(w)
                if h < 0:
                    sy = y
                    h = abs(h)
                
                self.selection_rect = (sx, sy, w, h)
                self._update_display()
    
    def end_selection(self, event):
        """
        マウスリリースで選択終了。
        
        Args:
            event: マウスイベント
        """
        if self.selection_rect is not None:
            # 最小サイズを確認
            x, y, w, h = self.selection_rect
            if w > 10 and h > 10:
                self.register_button.setEnabled(True)
            else:
                self.selection_rect = None
                self.register_button.setEnabled(False)
                self._update_display()
    
    def register_template(self):
        """選択領域から図柄テンプレートを登録。"""
        if self.capture_frame is None or self.selection_rect is None:
            return
        
        # 選択領域から図柄画像を切り出し
        x, y, w, h = self.selection_rect
        template = self.capture_frame[y:y+h, x:x+w].copy()
        
        # 図柄名を取得
        name = self.name_edit.currentText()
        if not name:
            QMessageBox.warning(self, "入力エラー", "図柄名を入力してください。")
            return
        
        # マッチング閾値を取得
        threshold = self.threshold_slider.value() / 100.0
        
        # メタデータを設定
        metadata = {
            'registered_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_position': {'x': x, 'y': y, 'width': w, 'height': h}
        }
        
        # 登録シグナルを発行
        self.symbol_registered.emit(name, template, threshold, metadata)
        
        # 選択領域をクリア
        self.selection_rect = None
        self.register_button.setEnabled(False)
        self._update_display()
        
        QMessageBox.information(self, "登録完了", f"図柄「{name}」を登録しました。")


class TimingSettings(QWidget):
    """
    タイミング設定を行うためのウィジェット。
    """
    target_symbol_set = pyqtSignal(int, str, int)
    human_delay_set = pyqtSignal(int)
    machine_profile_registered = pyqtSignal(str, int, int, int)
    machine_profile_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        TimingSettingsクラスの初期化。
        
        Args:
            parent: 親ウィジェット
        """
        super().__init__(parent)
        self.symbol_names = []
        self.initUI()
    
    def initUI(self):
        """UIの初期化。"""
        layout = QVBoxLayout(self)
        
        # 目標図柄設定グループ
        target_group = QGroupBox("目標図柄設定")
        target_layout = QGridLayout()
        
        # リール選択
        target_layout.addWidget(QLabel("リール:"), 0, 0)
        self.reel_combo = QComboBox()
        self.reel_combo.addItems(["リール1", "リール2", "リール3"])
        target_layout.addWidget(self.reel_combo, 0, 1)
        
        # 図柄選択
        target_layout.addWidget(QLabel("図柄:"), 1, 0)
        self.symbol_combo = QComboBox()
        target_layout.addWidget(self.symbol_combo, 1, 1)
        
        # 目標位置
        target_layout.addWidget(QLabel("位置:"), 2, 0)
        self.position_spin = QSpinBox()
        self.position_spin.setRange(0, 1000)
        self.position_spin.setValue(100)
        target_layout.addWidget(self.position_spin, 2, 1)
        
        # 設定ボタン
        self.set_target_button = QPushButton("目標設定")
        self.set_target_button.clicked.connect(self.set_target_symbol)
        target_layout.addWidget(self.set_target_button, 3, 0, 1, 2)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # 機種プロファイル設定グループ
        profile_group = QGroupBox("機種プロファイル設定")
        profile_layout = QGridLayout()
        
        # プロファイル選択
        profile_layout.addWidget(QLabel("機種:"), 0, 0)
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("デフォルト")
        self.profile_combo.currentTextChanged.connect(self.on_profile_selected)
        profile_layout.addWidget(self.profile_combo, 0, 1, 1, 3)
        
        # すべりフレーム数
        profile_layout.addWidget(QLabel("すべりフレーム:"), 1, 0)
        self.slip_spin = QSpinBox()
        self.slip_spin.setRange(0, 10)
        self.slip_spin.setValue(1)
        profile_layout.addWidget(self.slip_spin, 1, 1)
        
        # 引き込み範囲
        profile_layout.addWidget(QLabel("引き込み範囲:"), 1, 2)
        self.pull_in_spin = QSpinBox()
        self.pull_in_spin.setRange(0, 10)
        self.pull_in_spin.setValue(3)
        profile_layout.addWidget(self.pull_in_spin, 1, 3)
        
        # ボタン～停止フレーム数
        profile_layout.addWidget(QLabel("ボタン-停止フレーム:"), 2, 0, 1, 2)
        self.button_to_stop_spin = QSpinBox()
        self.button_to_stop_spin.setRange(0, 10)
        self.button_to_stop_spin.setValue(2)
        profile_layout.addWidget(self.button_to_stop_spin, 2, 2, 1, 2)
        
        # プロファイル名入力
        profile_layout.addWidget(QLabel("新規プロファイル名:"), 3, 0, 1, 2)
        self.profile_name_edit = QComboBox()
        self.profile_name_edit.setEditable(True)
        profile_layout.addWidget(self.profile_name_edit, 3, 2, 1, 2)
        
        # 登録ボタン
        self.register_profile_button = QPushButton("プロファイル登録")
        self.register_profile_button.clicked.connect(self.register_machine_profile)
        profile_layout.addWidget(self.register_profile_button, 4, 0, 1, 4)
        
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)
        
        # ヒューマンディレイ設定グループ
        delay_group = QGroupBox("反応遅延設定")
        delay_layout = QHBoxLayout()
        
        delay_layout.addWidget(QLabel("反応遅延フレーム数:"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 15)
        self.delay_spin.setValue(5)
        self.delay_spin.valueChanged.connect(self.set_human_delay)
        delay_layout.addWidget(self.delay_spin)
        
        delay_group.setLayout(delay_layout)
        layout.addWidget(delay_group)
        
        # 余白を追加
        layout.addStretch()
    
    def update_symbol_names(self, names: List[str]) -> None:
        """
        選択可能な図柄名リストを更新する。
        
        Args:
            names (List[str]): 図柄名リスト
        """
        self.symbol_names = names
        current_text = self.symbol_combo.currentText()
        
        # コンボボックスを更新
        self.symbol_combo.clear()
        self.symbol_combo.addItems(names)
        
        # 以前の選択を復元（可能な場合）
        index = self.symbol_combo.findText(current_text)
        if index >= 0:
            self.symbol_combo.setCurrentIndex(index)
    
    def update_profile_names(self, names: List[str]) -> None:
        """
        選択可能な機種プロファイル名リストを更新する。
        
        Args:
            names (List[str]): プロファイル名リスト
        """
        current_text = self.profile_combo.currentText()
        
        # コンボボックスを更新
        self.profile_combo.clear()
        self.profile_combo.addItems(names)
        
        # 以前の選択を復元（可能な場合）
        index = self.profile_combo.findText(current_text)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
    
    def set_target_symbol(self) -> None:
        """目標図柄を設定するときの処理。"""
        reel_id = self.reel_combo.currentIndex() + 1
        symbol_name = self.symbol_combo.currentText()
        position = self.position_spin.value()
        
        if not symbol_name:
            QMessageBox.warning(self, "選択エラー", "図柄を選択してください。")
            return
        
        self.target_symbol_set.emit(reel_id, symbol_name, position)
    
    def set_human_delay(self) -> None:
        """ヒューマンディレイを設定するときの処理。"""
        delay = self.delay_spin.value()
        self.human_delay_set.emit(delay)
    
    def register_machine_profile(self) -> None:
        """機種プロファイルを登録するときの処理。"""
        name = self.profile_name_edit.currentText()
        if not name:
            QMessageBox.warning(self, "入力エラー", "プロファイル名を入力してください。")
            return
        
        slip_frames = self.slip_spin.value()
        pull_in_range = self.pull_in_spin.value()
        button_to_stop_frames = self.button_to_stop_spin.value()
        
        self.machine_profile_registered.emit(name, slip_frames, pull_in_range, button_to_stop_frames)
        
        # プロファイル名リストに追加
        if self.profile_combo.findText(name) < 0:
            self.profile_combo.addItem(name)
            self.profile_name_edit.addItem(name)
        
        # 新しいプロファイルを選択
        index = self.profile_combo.findText(name)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
    
    def on_profile_selected(self, name: str) -> None:
        """
        機種プロファイルが選択されたときの処理。
        
        Args:
            name (str): 選択されたプロファイル名
        """
        self.machine_profile_selected.emit(name)


class MainWindow(QMainWindow):
    """
    アプリケーションのメインウィンドウ。
    """
    
    def __init__(self):
        """MainWindowクラスの初期化。"""
        super().__init__()
        
        # モジュールの初期化
        self.screen_capture = ScreenCapture()
        self.video_processor = VideoProcessor()
        
        # 機種データベースの初期化
        self.machine_database = MachineDatabase()
        
        # 図柄認識器を初期化（ハイブリッド認識器を使用）
        try:
            self.symbol_recognizer = HybridSymbolRecognizer()
            self.using_hybrid_recognizer = True
            logging.info("ハイブリッド図柄認識器を初期化しました")
        except Exception as e:
            # エラー時はTemplateMatchingを使用
            self.symbol_recognizer = TemplateMatching()
            self.using_hybrid_recognizer = False
            logging.warning(f"ハイブリッド図柄認識器の初期化に失敗しました: {str(e)}、テンプレートマッチングを使用します")
        
        self.symbol_tracker = SymbolTracker()
        self.timing_manager = TimingManager()
        
        # フレーム処理タイマー
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        
        # 処理中フラグ
        self.processing = False
        
        # キャプチャ中フラグ
        self.capturing = False
        
        # 自動機種判別フラグ
        self.auto_detect_machine = False
        self.last_detection_time = 0
        
        # 初期設定
        self.initUI()
        
        # ウィンドウのタイトルとサイズを設定
        self.setWindowTitle("パチスロビタ押しタイミング認識システム")
        self.resize(1280, 720)
        
        # 設定の読み込み
        self.load_settings()
    
    def initUI(self):
        """UIの初期化。"""
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        
        # 左側：ビデオフレーム
        self.video_frame = VideoFrame()
        main_layout.addWidget(self.video_frame, 3)
        
        # 右側：コントロールパネル
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 1)
        
        # コントロールタブ
        tabs = QTabWidget()
        control_layout.addWidget(tabs)
        
        # キャプチャタブ
        capture_tab = QWidget()
        capture_layout = QVBoxLayout(capture_tab)
        
        # キャプチャ領域選択器
        self.capture_area_selector = CaptureAreaSelector()
        self.capture_area_selector.area_selected.connect(self.set_capture_area)
        capture_layout.addWidget(self.capture_area_selector)
        
        # 自動キャリブレーションボタン
        auto_calib_btn = QPushButton("自動キャリブレーション")
        auto_calib_btn.clicked.connect(self.show_auto_calibration_dialog)
        capture_layout.addWidget(auto_calib_btn)
        
        tabs.addTab(capture_tab, "キャプチャ設定")
        
        # 図柄登録タブ
        self.symbol_registration = SymbolRegistration()
        self.symbol_registration.symbol_registered.connect(self.register_symbol)
        tabs.addTab(self.symbol_registration, "図柄登録")
        
        # タイミング設定タブ
        self.timing_settings = TimingSettings()
        self.timing_settings.target_symbol_set.connect(self.set_target_symbol)
        self.timing_settings.human_delay_set.connect(self.set_human_delay)
        self.timing_settings.machine_profile_registered.connect(self.register_machine_profile)
        self.timing_settings.machine_profile_selected.connect(self.select_machine_profile)
        tabs.addTab(self.timing_settings, "タイミング設定")
        
        # 表示設定
        display_group = QGroupBox("表示設定")
        display_layout = QVBoxLayout()
        
        # 認識結果表示
        self.show_recognition_check = QCheckBox("認識結果表示")
        self.show_recognition_check.setChecked(True)
        self.show_recognition_check.stateChanged.connect(
            lambda state: self.video_frame.set_show_recognition(state == Qt.Checked))
        display_layout.addWidget(self.show_recognition_check)
        
        # タイミング情報表示
        self.show_timing_check = QCheckBox("タイミング情報表示")
        self.show_timing_check.setChecked(True)
        self.show_timing_check.stateChanged.connect(
            lambda state: self.video_frame.set_show_timing(state == Qt.Checked))
        display_layout.addWidget(self.show_timing_check)
        
        # 自動機種判別
        self.auto_detect_check = QCheckBox("自動機種判別")
        self.auto_detect_check.setChecked(False)
        self.auto_detect_check.stateChanged.connect(
            lambda state: self.set_auto_detect_machine(state == Qt.Checked))
        display_layout.addWidget(self.auto_detect_check)
        
        display_group.setLayout(display_layout)
        control_layout.addWidget(display_group)
        
        # 操作ボタン
        buttons_layout = QHBoxLayout()
        
        # 開始/停止ボタン
        self.start_stop_button = QPushButton("キャプチャ開始")
        self.start_stop_button.clicked.connect(self.toggle_capture)
        buttons_layout.addWidget(self.start_stop_button)
        
        # リセットボタン - 目立つように赤色で表示
        self.reset_button = QPushButton("リセット")
        self.reset_button.setStyleSheet("background-color: #FF5050; color: white; font-weight: bold;")
        self.reset_button.clicked.connect(self.confirm_reset)
        buttons_layout.addWidget(self.reset_button)
        
        # 設定保存ボタン
        self.save_settings_button = QPushButton("設定保存")
        self.save_settings_button.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.save_settings_button)
        
        # 機種管理ボタン
        self.machine_button = QPushButton("機種管理")
        self.machine_button.clicked.connect(self.show_machine_dialog)
        buttons_layout.addWidget(self.machine_button)
        
        control_layout.addLayout(buttons_layout)
        
        # ステータスバー
        self.statusBar().showMessage("準備完了")
    
    def toggle_capture(self):
        """キャプチャの開始・停止を切り替える。"""
        if self.capturing:
            # キャプチャ停止
            self.timer.stop()
            self.capturing = False
            self.start_stop_button.setText("キャプチャ開始")
            self.statusBar().showMessage("キャプチャ停止")
        else:
            # キャプチャ開始
            self.timer.start(33)  # 約30fpsでフレーム処理
            self.capturing = True
            self.start_stop_button.setText("キャプチャ停止")
            self.statusBar().showMessage("キャプチャ中")
    
    def process_frame(self):
        """
        フレームを処理する。
        キャプチャ、認識、タイミング計算の一連の処理を行う。
        """
        if self.processing:
            return
        
        self.processing = True
        
        try:
            # フレームをキャプチャ
            frame, fps = self.screen_capture.capture_frame()
            
            # 現在の元のフレームを保存
            original_frame = frame.copy()
            
            # リール領域を抽出
            reel_frame = self.video_processor.extract_reel_area(frame)
            
            # リール領域が検出された場合、認識処理を実行
            # ただし、前回のフレームを参照して急激な変化を防止する
            if reel_frame is not None:
                processed_frame = reel_frame
                # リール領域が検出されたときのステータス更新
                if hasattr(self, 'using_original_frame') and self.using_original_frame:
                    self.using_original_frame = False
                    self.statusBar().showMessage(f"リール領域を検出しました - FPS: {fps:.1f}")
            else:
                # リール領域が検出されない場合は元のフレームを使用
                processed_frame = original_frame
                # 初めて検出に失敗した場合のみメッセージ表示
                if not hasattr(self, 'using_original_frame') or not self.using_original_frame:
                    self.using_original_frame = True
                    self.statusBar().showMessage(f"リール領域検出に失敗しました。元のフレームを使用します - FPS: {fps:.1f}")
            
            # フレームを表示
            self.video_frame.update_frame(processed_frame)
            self.symbol_registration.update_frame(processed_frame)
            
            # 図柄認識
            recognized_symbols = self.symbol_recognizer.recognize_symbols(processed_frame)
            
            # 図柄追跡
            tracked_symbols = self.symbol_tracker.update(recognized_symbols, processed_frame)
            
            # トラッキング結果を表示
            self.video_frame.update_symbols(tracked_symbols)
            
            # リール番号ごとに図柄をグループ化
            # 簡易的な実装として、画面を3等分して各領域をリールとみなす
            reel_symbols = {}
            width = processed_frame.shape[1]
            
            for symbol in tracked_symbols:
                x = symbol['x']
                # 画面位置からリール番号を推定
                reel_id = min(3, max(1, int(x * 3 / width) + 1))
                
                if reel_id not in reel_symbols:
                    reel_symbols[reel_id] = []
                
                reel_symbols[reel_id].append(symbol)
            
            # 自動機種判別
            if self.auto_detect_machine and time.time() - self.last_detection_time > 5:  # 5秒ごとに判別
                self.last_detection_time = time.time()
                detected_machine = self.machine_database.detect_machine(processed_frame)
                
                if detected_machine != "default" and detected_machine != self.machine_database.current_machine:
                    # 機種が変更された場合、タイミング予測器に新しい機種を設定
                    self.statusBar().showMessage(f"機種「{detected_machine}」を自動判別しました")
                    self.machine_database.set_current_machine(detected_machine)
                    
                    # 機種プロファイルを取得
                    machine_data = self.machine_database.get_machine(detected_machine)
                    if machine_data:
                        # タイミング予測器に設定
                        self.timing_manager.timing_predictor.register_machine_profile(
                            detected_machine,
                            machine_data.get("slip_frames", 1),
                            machine_data.get("pull_in_range", 3),
                            machine_data.get("button_to_stop_frames", 2)
                        )
                        self.timing_manager.timing_predictor.set_current_machine(detected_machine)
            
            # タイミング計算
            timing_results = self.timing_manager.update(reel_symbols)
            
            # タイミング結果を表示
            self.video_frame.update_timing(timing_results)
            
            # 認識器の種類を表示用に取得
            recognizer_type = "ハイブリッド認識" if getattr(self, 'using_hybrid_recognizer', False) else "テンプレートマッチング"
            
            # ステータスバー更新
            self.statusBar().showMessage(f"キャプチャ中 - FPS: {fps:.1f} - 認識図柄: {len(tracked_symbols)} - 認識方式: {recognizer_type}")
            
            # 図柄名リストを更新
            symbol_names = list(self.symbol_recognizer.symbols.keys())
            self.timing_settings.update_symbol_names(symbol_names)
        
        except Exception as e:
            self.statusBar().showMessage(f"エラー: {str(e)}")
        
        finally:
            self.processing = False
    
    def set_capture_area(self, x: int, y: int, width: int, height: int):
        """
        キャプチャ領域を設定する。
        
        Args:
            x (int): 左端X座標
            y (int): 上端Y座標
            width (int): 幅
            height (int): 高さ
        """
        self.screen_capture.set_capture_area(x, y, width, height)
        self.statusBar().showMessage(f"キャプチャ領域設定: ({x}, {y}, {width}, {height})")
    
    def register_symbol(self, name: str, template: np.ndarray, threshold: float, metadata: Dict):
        """
        図柄テンプレートを登録する。
        
        Args:
            name (str): 図柄名
            template (np.ndarray): テンプレート画像
            threshold (float): マッチング閾値
            metadata (Dict): メタデータ
        """
        success = self.symbol_recognizer.register_template(name, template, threshold, metadata)
        
        if success:
            self.statusBar().showMessage(f"図柄「{name}」を登録しました")
            
            # 図柄名リストを更新
            symbol_names = list(self.symbol_recognizer.symbols.keys())
            self.timing_settings.update_symbol_names(symbol_names)
            
            # ハイブリッド認識器を使用している場合、モデルを学習
            if hasattr(self, 'using_hybrid_recognizer') and self.using_hybrid_recognizer:
                try:
                    # モデル学習（非同期で実行するとよいが、ここではシンプルに同期実行）
                    self.statusBar().showMessage(f"図柄「{name}」を登録しました。機械学習モデルを更新中...")
                    
                    # MLSymbolRecognizerのモデル学習メソッドを呼び出す
                    if hasattr(self.symbol_recognizer, 'ml_recognizer'):
                        model_updated = self.symbol_recognizer.ml_recognizer.train_model()
                        
                        # モデル更新の結果を表示
                        if model_updated:
                            self.statusBar().showMessage(f"図柄「{name}」を登録し、機械学習モデルを更新しました")
                        else:
                            self.statusBar().showMessage(f"図柄「{name}」を登録しましたが、機械学習モデルの更新に失敗しました")
                
                except Exception as e:
                    self.statusBar().showMessage(f"図柄「{name}」を登録しましたが、機械学習モデルの更新中にエラーが発生しました: {str(e)}")
        else:
            self.statusBar().showMessage(f"図柄「{name}」の登録に失敗しました")
    
    def set_target_symbol(self, reel_id: int, symbol_name: str, position: int):
        """
        目標図柄を設定する。
        
        Args:
            reel_id (int): リール番号
            symbol_name (str): 図柄名
            position (int): 目標位置
        """
        self.timing_manager.set_target_symbol(reel_id, symbol_name, position)
        self.statusBar().showMessage(f"リール{reel_id}の目標図柄を「{symbol_name}」に設定しました")
    
    def set_human_delay(self, delay: int):
        """
        人間の反応遅延を設定する。
        
        Args:
            delay (int): 遅延フレーム数
        """
        self.timing_manager.timing_predictor.set_human_delay(delay)
        self.statusBar().showMessage(f"反応遅延を{delay}フレームに設定しました")
    
    def register_machine_profile(self, name: str, slip_frames: int, pull_in_range: int, button_to_stop_frames: int):
        """
        機種プロファイルを登録する。
        
        Args:
            name (str): 機種名
            slip_frames (int): すべりフレーム数
            pull_in_range (int): 引き込み範囲
            button_to_stop_frames (int): ボタンを押してからリールが停止するまでのフレーム数
        """
        # タイミング予測器に登録
        self.timing_manager.timing_predictor.register_machine_profile(
            name, slip_frames, pull_in_range, button_to_stop_frames)
        
        # 機種データベースにも登録
        machine_data = {
            "slip_frames": slip_frames,
            "pull_in_range": pull_in_range,
            "button_to_stop_frames": button_to_stop_frames,
            "reel_array": []
        }
        
        self.machine_database.add_machine(name, machine_data)
        
        self.statusBar().showMessage(f"機種プロファイル「{name}」を登録しました")
        
        # プロファイル名リストを更新
        profile_names = list(self.timing_manager.timing_predictor.machine_profiles.keys())
        self.timing_settings.update_profile_names(profile_names)
    
    def select_machine_profile(self, name: str):
        """
        使用する機種プロファイルを選択する。
        
        Args:
            name (str): 機種名
        """
        success = self.timing_manager.timing_predictor.set_current_machine(name)
        
        # 機種データベースの現在の機種も更新
        self.machine_database.set_current_machine(name)
        
        if success:
            self.statusBar().showMessage(f"機種プロファイル「{name}」を選択しました")
        else:
            self.statusBar().showMessage(f"機種プロファイル「{name}」の選択に失敗しました")
    
    def save_settings(self):
        """設定を保存する。"""
        try:
            # 設定ディレクトリを作成
            settings_dir = "../data/settings"
            os.makedirs(settings_dir, exist_ok=True)
            
            # 設定ファイルのパス
            settings_file = os.path.join(settings_dir, "settings.json")
            
            # キャプチャ領域設定
            capture_area = self.screen_capture.capture_area
            
            # 機種プロファイル設定
            machine_profiles = self.timing_manager.timing_predictor.machine_profiles
            
            # 設定データ
            settings = {
                "capture_area": capture_area,
                "human_delay": self.timing_manager.timing_predictor.human_delay,
                "current_machine": self.timing_manager.timing_predictor.current_machine,
                "show_recognition": self.show_recognition_check.isChecked(),
                "show_timing": self.show_timing_check.isChecked(),
                "auto_detect_machine": self.auto_detect_check.isChecked()
            }
            
            # JSONに変換して保存
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            
            self.statusBar().showMessage("設定を保存しました")
        
        except Exception as e:
            self.statusBar().showMessage(f"設定の保存に失敗しました: {str(e)}")
    
    def show_auto_calibration_dialog(self):
        """自動キャリブレーションダイアログを表示する。"""
        from src.ui.calibration_dialog import CalibrationDialog
        
        # キャプチャ中の場合は一時停止
        was_capturing = self.capturing
        if was_capturing:
            self.toggle_capture()
        
        # ダイアログを作成
        dialog = CalibrationDialog(self.screen_capture, self.video_processor, self)
        
        # キャリブレーション完了時のシグナルを接続
        dialog.calibration_completed.connect(self.apply_calibration_result)
        
        # ダイアログを表示
        result = dialog.exec_()
        
        # キャプチャを再開
        if was_capturing:
            self.toggle_capture()
    
    def apply_calibration_result(self, result):
        """
        キャリブレーション結果を適用する。
        
        Args:
            result (dict): キャリブレーション結果
                {
                    'reel_area': (x, y, width, height),  # リール全体の領域
                    'reel_divisions': [(x1, y1, w1, h1), ...],  # 各リールの領域
                    'confidence': float  # 検出結果の信頼度
                }
        """
        if result is None or result['reel_area'] is None:
            self.statusBar().showMessage("キャリブレーション結果の適用に失敗しました")
            return
        
        # リール領域を適用
        x, y, width, height = result['reel_area']
        self.screen_capture.set_capture_area(x, y, width, height)
        
        # UI更新
        self.capture_area_selector.x_spin.setValue(x)
        self.capture_area_selector.y_spin.setValue(y)
        self.capture_area_selector.width_spin.setValue(width)
        self.capture_area_selector.height_spin.setValue(height)
        
        # リール分割情報をVideoProcessorに設定
        if result['reel_divisions']:
            self.video_processor.reel_area = result['reel_area']
            
            self.statusBar().showMessage(f"キャリブレーション結果を適用しました - 信頼度: {result['confidence']:.2f}")
    
    def load_settings(self):
        """設定を読み込む。"""
        try:
            # 設定ファイルのパス
            settings_file = "../data/settings/settings.json"
            
            # ファイルが存在する場合のみ読み込み
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # キャプチャ領域設定
                if "capture_area" in settings:
                    area = settings["capture_area"]
                    self.screen_capture.set_capture_area(
                        area["left"], area["top"], area["width"], area["height"])
                    
                    # UI更新
                    self.capture_area_selector.x_spin.setValue(area["left"])
                    self.capture_area_selector.y_spin.setValue(area["top"])
                    self.capture_area_selector.width_spin.setValue(area["width"])
                    self.capture_area_selector.height_spin.setValue(area["height"])
                
                # ヒューマンディレイ設定
                if "human_delay" in settings:
                    self.timing_manager.timing_predictor.set_human_delay(settings["human_delay"])
                    self.timing_settings.delay_spin.setValue(settings["human_delay"])
                
                # 機種プロファイル設定
                # 機種データベースから機種情報を取得
                machine_names = self.machine_database.get_all_machine_names()
                for name in machine_names:
                    machine_data = self.machine_database.get_machine(name)
                    if name != "default" and machine_data:  # デフォルト以外の機種を登録
                        self.timing_manager.timing_predictor.register_machine_profile(
                            name, 
                            machine_data.get("slip_frames", 1), 
                            machine_data.get("pull_in_range", 3), 
                            machine_data.get("button_to_stop_frames", 2)
                        )
                
                # プロファイル名リストを更新
                profile_names = list(self.timing_manager.timing_predictor.machine_profiles.keys())
                self.timing_settings.update_profile_names(profile_names)
                
                # 現在の機種設定
                if "current_machine" in settings:
                    current_machine = settings["current_machine"]
                    self.timing_manager.timing_predictor.set_current_machine(current_machine)
                    self.machine_database.set_current_machine(current_machine)
                    
                    # UI更新
                    index = self.timing_settings.profile_combo.findText(current_machine)
                    if index >= 0:
                        self.timing_settings.profile_combo.setCurrentIndex(index)
                
                # 表示設定
                if "show_recognition" in settings:
                    self.show_recognition_check.setChecked(settings["show_recognition"])
                    self.video_frame.set_show_recognition(settings["show_recognition"])
                
                if "show_timing" in settings:
                    self.show_timing_check.setChecked(settings["show_timing"])
                    self.video_frame.set_show_timing(settings["show_timing"])
                
                # 自動機種判別設定
                if "auto_detect_machine" in settings:
                    self.auto_detect_check.setChecked(settings["auto_detect_machine"])
                    self.set_auto_detect_machine(settings["auto_detect_machine"])
                
                self.statusBar().showMessage("設定を読み込みました")
            else:
                self.statusBar().showMessage("設定ファイルが見つかりません。デフォルト設定を使用します。")
        
        except Exception as e:
            self.statusBar().showMessage(f"設定の読み込みに失敗しました: {str(e)}")
            
    def show_machine_dialog(self):
        """機種管理ダイアログを表示する。"""
        # キャプチャ中の場合は現在のフレームを渡す
        current_frame = None
        if self.video_frame.frame is not None:
            current_frame = self.video_frame.frame.copy()
        
        # ダイアログを作成
        dialog = MachineDialog(self.machine_database, self)
        
        # 現在のフレームを設定
        if current_frame is not None:
            dialog.update_frame(current_frame)
        
        # 機種更新時のシグナルを接続
        dialog.machine_updated.connect(self.on_machine_updated)
        
        # ダイアログを表示
        dialog.exec_()
    
    def on_machine_updated(self, machine_name: str):
        """機種が更新されたときの処理。
        
        Args:
            machine_name (str): 更新された機種名
        """
        if not machine_name:
            return
        
        # 機種データベースから機種情報を取得
        machine_data = self.machine_database.get_machine(machine_name)
        if not machine_data:
            return
        
        # タイミング予測器に機種情報を設定
        self.timing_manager.timing_predictor.register_machine_profile(
            machine_name,
            machine_data.get("slip_frames", 1),
            machine_data.get("pull_in_range", 3),
            machine_data.get("button_to_stop_frames", 2)
        )
        
        # 現在の機種を更新
        success = self.timing_manager.timing_predictor.set_current_machine(machine_name)
        
        # プロファイル名リストを更新
        profile_names = list(self.timing_manager.timing_predictor.machine_profiles.keys())
        self.timing_settings.update_profile_names(profile_names)
        
        # UI更新
        index = self.timing_settings.profile_combo.findText(machine_name)
        if index >= 0:
            self.timing_settings.profile_combo.setCurrentIndex(index)
        
        # ステータスバー更新
        if success:
            self.statusBar().showMessage(f"機種「{machine_name}」を選択しました")
    
    def set_auto_detect_machine(self, enable: bool):
        """自動機種判別の有効/無効を設定する。
        
        Args:
            enable (bool): 有効にする場合はTrue
        """
        self.auto_detect_machine = enable
        if enable:
            # 自動機種判別が有効な場合、次回のフレーム処理で判別するようにする
            self.last_detection_time = 0
            self.statusBar().showMessage("自動機種判別を有効にしました")
        else:
            self.statusBar().showMessage("自動機種判別を無効にしました")
    
    def confirm_reset(self):
        """
        リセット前に確認ダイアログを表示する。
        ユーザーが確認した場合のみリセット処理を実行する。
        """
        # 確認ダイアログを表示
        reply = QMessageBox.question(
            self,
            'リセット確認',
            '現在表示されている図柄や画像をすべて消去し、初期状態に戻します。\n\nよろしいですか？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        # ユーザーが「はい」を選択した場合のみリセット処理を実行
        if reply == QMessageBox.Yes:
            self.reset_display()
    
    def reset_display(self):
        """
        表示されている図柄やタイミング情報をすべてリセットする。
        """
        # VideoFrameの図柄情報とタイミング情報をリセット
        self.video_frame.symbols = []
        self.video_frame.timing_results = {}
        self.video_frame._update_display()
        
        # SymbolTrackerの追跡情報をリセット
        self.symbol_tracker.reset()
        
        # TimingManagerの情報をリセット
        self.timing_manager.reset_timing_data()
        
        # キャプチャ中の場合、次のフレーム更新で画面もクリアされる
        if not self.capturing:
            # キャプチャ中でない場合は、現在のフレームを表示したまま情報だけリセット
            if self.video_frame.frame is not None:
                self.video_frame.update_frame(self.video_frame.frame.copy())
        
        # ステータスバーを更新
        self.statusBar().showMessage("表示情報をリセットしました")
