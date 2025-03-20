"""
パチスロビタ押しタイミング認識システムのメインウィンドウ。
"""
import sys
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QGroupBox,
    QSpinBox, QFileDialog, QMessageBox, QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect

# 自作モジュールのインポート
from src.capture.screen_capture import ScreenCapture, VideoProcessor
from src.recognition.symbol_recognizer import TemplateMatching, SymbolTracker
from src.timing.reel_analyzer import TimingManager


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
        
        # 表示サイズに合わせて拡大・縮小
        pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        
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
        
        # 表示サイズに合わせて拡大・縮小
        pixmap = pixmap.scaled(self.frame_label.width(), self.frame_label.height(), Qt.KeepAspectRatio)
        
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
        self.symbol_recognizer = TemplateMatching()
        self.symbol_tracker = SymbolTracker()
        self.timing_manager = TimingManager()
        
        # フレーム処理タイマー
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        
        # 処理中フラグ
        self.processing = False
        
        # キャプチャ中フラグ
        self.capturing = False
        
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
        self.capture_area_selector = CaptureAreaSelector()
        self.capture_area_selector.area_selected.connect(self.set_capture_area)
        tabs.addTab(self.capture_area_selector, "キャプチャ設定")
        
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
        
        display_group.setLayout(display_layout)
        control_layout.addWidget(display_group)
        
        # 操作ボタン
        buttons_layout = QHBoxLayout()
        
        # 開始/停止ボタン
        self.start_stop_button = QPushButton("キャプチャ開始")
        self.start_stop_button.clicked.connect(self.toggle_capture)
        buttons_layout.addWidget(self.start_stop_button)
        
        # 設定保存ボタン
        self.save_settings_button = QPushButton("設定保存")
        self.save_settings_button.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.save_settings_button)
        
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
            
            # リール領域を抽出
            reel_frame = self.video_processor.extract_reel_area(frame)
            
            # リール領域が検出された場合、認識処理を実行
            if reel_frame is not None:
                processed_frame = reel_frame
            else:
                processed_frame = frame
            
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
            
            # タイミング計算
            timing_results = self.timing_manager.update(reel_symbols)
            
            # タイミング結果を表示
            self.video_frame.update_timing(timing_results)
            
            # ステータスバー更新
            self.statusBar().showMessage(f"キャプチャ中 - FPS: {fps:.1f} - 認識図柄: {len(tracked_symbols)}")
            
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
        self.timing_manager.timing_predictor.register_machine_profile(
            name, slip_frames, pull_in_range, button_to_stop_frames)
        
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
                "machine_profiles": machine_profiles,
                "show_recognition": self.show_recognition_check.isChecked(),
                "show_timing": self.show_timing_check.isChecked()
            }
            
            # JSONに変換して保存
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            
            self.statusBar().showMessage("設定を保存しました")
        
        except Exception as e:
            self.statusBar().showMessage(f"設定の保存に失敗しました: {str(e)}")
    
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
                if "machine_profiles" in settings:
                    for name, profile in settings["machine_profiles"].items():
                        if name != "default":  # デフォルトプロファイル以外を登録
                            self.timing_manager.timing_predictor.register_machine_profile(
                                name, 
                                profile["slip_frames"], 
                                profile["pull_in_range"], 
                                profile["button_to_stop_frames"]
                            )
                    
                    # プロファイル名リストを更新
                    profile_names = list(self.timing_manager.timing_predictor.machine_profiles.keys())
                    self.timing_settings.update_profile_names(profile_names)
                
                # 現在の機種設定
                if "current_machine" in settings:
                    self.timing_manager.timing_predictor.set_current_machine(settings["current_machine"])
                    
                    # UI更新
                    index = self.timing_settings.profile_combo.findText(settings["current_machine"])
                    if index >= 0:
                        self.timing_settings.profile_combo.setCurrentIndex(index)
                
                # 表示設定
                if "show_recognition" in settings:
                    self.show_recognition_check.setChecked(settings["show_recognition"])
                    self.video_frame.set_show_recognition(settings["show_recognition"])
                
                if "show_timing" in settings:
                    self.show_timing_check.setChecked(settings["show_timing"])
                    self.video_frame.set_show_timing(settings["show_timing"])
                
                self.statusBar().showMessage("設定を読み込みました")
            else:
                self.statusBar().showMessage("設定ファイルが見つかりません。デフォルト設定を使用します。")
        
        except Exception as e:
            self.statusBar().showMessage(f"設定の読み込みに失敗しました: {str(e)}")
