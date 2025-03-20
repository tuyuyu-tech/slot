"""
自動キャリブレーション機能をUIに統合するための実装。
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from src.capture.auto_calibration import AutoCalibrationDialog


class CalibrationDialog(QDialog):
    """
    自動キャリブレーションを実行するダイアログウィンドウ。
    
    Attributes:
        screen_capture: スクリーンキャプチャオブジェクト
        video_processor: ビデオプロセッサオブジェクト
        auto_calibration: 自動キャリブレーションオブジェクト
        calibration_timer: キャリブレーションプロセスのタイマー
        preview_frame: プレビューフレーム表示用ラベル
        progress_bar: 進捗バー
        status_label: ステータス表示ラベル
        result: キャリブレーション結果
    """
    
    # キャリブレーション完了時のシグナル
    calibration_completed = pyqtSignal(dict)
    
    def __init__(self, screen_capture, video_processor, parent=None):
        """
        CalibrationDialogクラスの初期化。
        
        Args:
            screen_capture: スクリーンキャプチャオブジェクト
            video_processor: ビデオプロセッサオブジェクト
            parent: 親ウィジェット
        """
        super().__init__(parent)
        
        self.screen_capture = screen_capture
        self.video_processor = video_processor
        self.auto_calibration = AutoCalibrationDialog(screen_capture, max_frames=15)
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.on_calibration_timer)
        self.result = None
        
        self.setWindowTitle("自動キャリブレーション")
        self.resize(640, 500)
        
        self.init_ui()
    
    def init_ui(self):
        """UIの初期化。"""
        layout = QVBoxLayout(self)
        
        # プレビューフレーム
        self.preview_frame = QLabel()
        self.preview_frame.setAlignment(Qt.AlignCenter)
        self.preview_frame.setMinimumSize(600, 350)
        self.preview_frame.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.preview_frame)
        
        # ステータス表示
        self.status_label = QLabel("リール位置の自動検出を開始するには「開始」ボタンを押してください。")
        layout.addWidget(self.status_label)
        
        # 進捗バー
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # オプション設定
        options_layout = QHBoxLayout()
        
        self.apply_auto_checkbox = QCheckBox("検出結果を自動的に適用する")
        self.apply_auto_checkbox.setChecked(True)
        options_layout.addWidget(self.apply_auto_checkbox)
        
        layout.addLayout(options_layout)
        
        # ボタン
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("開始")
        self.start_btn.clicked.connect(self.start_calibration)
        btn_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("キャンセル")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.apply_btn = QPushButton("適用")
        self.apply_btn.clicked.connect(self.apply_calibration)
        self.apply_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_btn)
        
        layout.addLayout(btn_layout)
    
    def start_calibration(self):
        """キャリブレーションプロセスを開始する。"""
        self.auto_calibration.start_calibration()
        self.start_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.cancel_btn.setText("中止")
        self.status_label.setText("キャリブレーション中...")
        self.progress_bar.setValue(0)
        
        # タイマー開始
        self.calibration_timer.start(200)  # 200ms間隔でフレームを処理
    
    def on_calibration_timer(self):
        """タイマーイベントの処理。キャリブレーションのステップを実行する。"""
        # フレームを追加
        result = self.auto_calibration.add_calibration_frame()
        
        if not result:
            self.status_label.setText("フレームのキャプチャに失敗しました。")
            self.stop_calibration()
            return
        
        # 最後のフレームを表示
        if self.auto_calibration.frames:
            last_frame = self.auto_calibration.frames[-1]
            self._update_preview(last_frame)
        
        # 進捗を更新
        progress = int(len(self.auto_calibration.frames) / self.auto_calibration.max_frames * 100)
        self.progress_bar.setValue(progress)
        
        # キャリブレーションが完了したかチェック
        if self.auto_calibration.is_calibration_complete():
            self.finish_calibration()
    
    def finish_calibration(self):
        """キャリブレーションを完了し、結果を処理する。"""
        # タイマー停止
        self.calibration_timer.stop()
        
        # キャリブレーション完了
        self.result = self.auto_calibration.finish_calibration()
        
        # 結果をオーバーレイした画像を表示
        overlay_frame = self.auto_calibration.get_last_frame_with_overlay()
        if overlay_frame is not None:
            self._update_preview(overlay_frame)
        
        # UI更新
        self.status_label.setText(
            f"キャリブレーション完了。信頼度: {self.result['confidence']:.2f}"
        )
        self.cancel_btn.setText("閉じる")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("再試行")
        self.apply_btn.setEnabled(True)
        
        # 信頼度が低い場合は警告
        if self.result['confidence'] < 0.5:
            QMessageBox.warning(
                self, 
                "低信頼度の検出結果", 
                "リール位置の検出信頼度が低いです。再試行するか、手動でキャプチャ領域を設定することをおすすめします。"
            )
        elif self.apply_auto_checkbox.isChecked():
            # 自動適用が選択されていれば結果を適用
            self.apply_calibration()
    
    def apply_calibration(self):
        """キャリブレーション結果を適用する。"""
        if self.result is None or self.result['reel_area'] is None:
            QMessageBox.warning(
                self, 
                "適用エラー", 
                "有効なキャリブレーション結果がありません。"
            )
            return
        
        # キャリブレーション結果を適用
        self.calibration_completed.emit(self.result)
        
        # ダイアログを閉じる
        self.accept()
    
    def stop_calibration(self):
        """キャリブレーションプロセスを停止する。"""
        self.calibration_timer.stop()
        self.start_btn.setEnabled(True)
        self.start_btn.setText("再試行")
        self.cancel_btn.setText("閉じる")
    
    def _update_preview(self, frame):
        """
        プレビュー画像を更新する。
        
        Args:
            frame (np.ndarray): 表示するフレーム画像
        """
        h, w = frame.shape[:2]
        
        # OpenCV画像をQPixmapに変換
        bytes_per_line = 3 * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        # 表示サイズに合わせて拡大・縮小
        pixmap = pixmap.scaled(self.preview_frame.width(), self.preview_frame.height(), 
                              Qt.KeepAspectRatio)
        
        # 画像を表示
        self.preview_frame.setPixmap(pixmap)
