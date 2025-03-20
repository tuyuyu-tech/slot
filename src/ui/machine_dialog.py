"""
パチスロ機種データ管理用のダイアログ。
"""

import os
import json
import sys
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QTabWidget, QGroupBox, QCheckBox, QFileDialog, QHeaderView,
    QWidget
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QObject

from src.machine.machine_database import MachineDatabase


class MachineDialog(QDialog):
    """
    パチスロ機種データを管理するためのダイアログ。
    
    Attributes:
        machine_database (MachineDatabase): 機種データベース
        current_frame (np.ndarray): 現在のフレーム画像
        machine_updated (pyqtSignal): 機種データ更新時のシグナル
    """
    
    machine_updated = pyqtSignal(str)
    
    def __init__(self, machine_database: MachineDatabase, parent=None):
        """
        MachineDialog クラスの初期化。
        
        Args:
            machine_database (MachineDatabase): 機種データベース
            parent: 親ウィジェット
        """
        super().__init__(parent)
        self.machine_database = machine_database
        self.current_frame = None
        
        self.setWindowTitle("機種データ管理")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
        self.update_machine_list()
    
    def init_ui(self):
        """UIの初期化。"""
        layout = QVBoxLayout(self)
        
        # タブウィジェット
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 機種一覧・編集タブ
        machine_tab = QWidget()
        tab_widget.addTab(machine_tab, "機種一覧")
        self.setup_machine_tab(machine_tab)
        
        # 新規機種追加タブ
        add_tab = QWidget()
        tab_widget.addTab(add_tab, "新規機種追加")
        self.setup_add_tab(add_tab)
        
        # リール配列設定タブ
        reel_tab = QWidget()
        tab_widget.addTab(reel_tab, "リール配列設定")
        self.setup_reel_tab(reel_tab)
        
        # 機種判別学習タブ
        detect_tab = QWidget()
        tab_widget.addTab(detect_tab, "機種判別学習")
        self.setup_detect_tab(detect_tab)
        
        # ボタン
        button_layout = QHBoxLayout()
        
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def setup_machine_tab(self, tab_widget):
        """
        機種一覧・編集タブのセットアップ。
        
        Args:
            tab_widget: タブのウィジェット
        """
        layout = QVBoxLayout(tab_widget)
        
        # 機種一覧テーブル
        self.machine_table = QTableWidget()
        self.machine_table.setColumnCount(5)
        self.machine_table.setHorizontalHeaderLabels([
            "機種名", "すべりフレーム", "引き込み範囲", "ボタン-停止フレーム", "登録日時"
        ])
        self.machine_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.machine_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.machine_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.machine_table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.machine_table)
        
        # ボタングループ
        buttons_layout = QHBoxLayout()
        
        # 機種選択ボタン
        select_button = QPushButton("この機種を選択")
        select_button.clicked.connect(self.select_machine)
        buttons_layout.addWidget(select_button)
        
        # 編集ボタン
        edit_button = QPushButton("編集")
        edit_button.clicked.connect(self.edit_machine)
        buttons_layout.addWidget(edit_button)
        
        # 削除ボタン
        delete_button = QPushButton("削除")
        delete_button.clicked.connect(self.delete_machine)
        buttons_layout.addWidget(delete_button)
        
        layout.addLayout(buttons_layout)
    
    def setup_add_tab(self, tab_widget):
        """
        新規機種追加タブのセットアップ。
        
        Args:
            tab_widget: タブのウィジェット
        """
        layout = QGridLayout(tab_widget)
        
        # 機種名
        layout.addWidget(QLabel("機種名:"), 0, 0)
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit, 0, 1, 1, 3)
        
        # すべりフレーム数
        layout.addWidget(QLabel("すべりフレーム数:"), 1, 0)
        self.slip_spin = QSpinBox()
        self.slip_spin.setRange(0, 10)
        self.slip_spin.setValue(1)
        layout.addWidget(self.slip_spin, 1, 1)
        
        # 引き込み範囲
        layout.addWidget(QLabel("引き込み範囲:"), 1, 2)
        self.pull_in_spin = QSpinBox()
        self.pull_in_spin.setRange(0, 10)
        self.pull_in_spin.setValue(3)
        layout.addWidget(self.pull_in_spin, 1, 3)
        
        # ボタン～停止フレーム数
        layout.addWidget(QLabel("ボタン-停止フレーム数:"), 2, 0)
        self.button_to_stop_spin = QSpinBox()
        self.button_to_stop_spin.setRange(0, 10)
        self.button_to_stop_spin.setValue(2)
        layout.addWidget(self.button_to_stop_spin, 2, 1)
        
        # 登録ボタン
        add_button = QPushButton("機種を登録")
        add_button.clicked.connect(self.add_machine)
        layout.addWidget(add_button, 3, 0, 1, 4)
        
        # 余白を追加
        layout.setRowStretch(4, 1)
    
    def setup_reel_tab(self, tab_widget):
        """
        リール配列設定タブのセットアップ。
        
        Args:
            tab_widget: タブのウィジェット
        """
        layout = QVBoxLayout(tab_widget)
        
        # 機種選択
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("機種:"))
        self.reel_machine_combo = QComboBox()
        select_layout.addWidget(self.reel_machine_combo)
        layout.addLayout(select_layout)
        
        # リール配列編集グループ
        reel_group = QGroupBox("リール配列設定")
        reel_layout = QGridLayout(reel_group)
        
        # リール1
        reel_layout.addWidget(QLabel("リール1:"), 0, 0)
        self.reel1_edit = QLineEdit()
        self.reel1_edit.setPlaceholderText("図柄をカンマ区切りで入力（例: 7,BAR,ベル,...）")
        reel_layout.addWidget(self.reel1_edit, 0, 1)
        
        # リール2
        reel_layout.addWidget(QLabel("リール2:"), 1, 0)
        self.reel2_edit = QLineEdit()
        self.reel2_edit.setPlaceholderText("図柄をカンマ区切りで入力（例: 7,BAR,ベル,...）")
        reel_layout.addWidget(self.reel2_edit, 1, 1)
        
        # リール3
        reel_layout.addWidget(QLabel("リール3:"), 2, 0)
        self.reel3_edit = QLineEdit()
        self.reel3_edit.setPlaceholderText("図柄をカンマ区切りで入力（例: 7,BAR,ベル,...）")
        reel_layout.addWidget(self.reel3_edit, 2, 1)
        
        layout.addWidget(reel_group)
        
        # 保存ボタン
        save_reel_button = QPushButton("リール配列を保存")
        save_reel_button.clicked.connect(self.save_reel_array)
        layout.addWidget(save_reel_button)
        
        # リール配列表示
        self.reel_array_label = QLabel("リール配列: 未設定")
        layout.addWidget(self.reel_array_label)
        
        # 余白を追加
        layout.addStretch()
        
        # 機種選択時の処理
        self.reel_machine_combo.currentTextChanged.connect(self.load_reel_array)
    
    def setup_detect_tab(self, tab_widget):
        """
        機種判別学習タブのセットアップ。
        
        Args:
            tab_widget: タブのウィジェット
        """
        layout = QVBoxLayout(tab_widget)
        
        # 機種選択
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("機種:"))
        self.detect_machine_combo = QComboBox()
        select_layout.addWidget(self.detect_machine_combo)
        layout.addLayout(select_layout)
        
        # 現在のフレーム表示
        frame_group = QGroupBox("現在のフレーム")
        frame_layout = QVBoxLayout(frame_group)
        
        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(320, 240)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("border: 1px solid gray;")
        frame_layout.addWidget(self.frame_label)
        
        layout.addWidget(frame_group)
        
        # 特徴サンプル登録
        sample_group = QGroupBox("特徴サンプル登録")
        sample_layout = QGridLayout(sample_group)
        
        sample_layout.addWidget(QLabel("特徴ラベル:"), 0, 0)
        self.feature_label_combo = QComboBox()
        self.feature_label_combo.setEditable(True)
        self.feature_label_combo.addItems(["general", "title", "gameplay"])
        sample_layout.addWidget(self.feature_label_combo, 0, 1)
        
        sample_layout.addWidget(QLabel("現在のサンプル数:"), 1, 0)
        self.sample_count_label = QLabel("0")
        sample_layout.addWidget(self.sample_count_label, 1, 1)
        
        self.add_sample_button = QPushButton("このフレームを特徴サンプルとして追加")
        self.add_sample_button.clicked.connect(self.add_feature_sample)
        sample_layout.addWidget(self.add_sample_button, 2, 0, 1, 2)
        
        layout.addWidget(sample_group)
        
        # モデル学習
        train_button = QPushButton("機種判別モデルを学習")
        train_button.clicked.connect(self.train_machine_detector)
        layout.addWidget(train_button)
        
        # 自動判別テスト
        test_button = QPushButton("現在のフレームで機種判別テスト")
        test_button.clicked.connect(self.test_machine_detection)
        layout.addWidget(test_button)
        
        # 判別結果ラベル
        self.detection_result_label = QLabel("判別結果: なし")
        layout.addWidget(self.detection_result_label)
        
        # 余白を追加
        layout.addStretch()
        
        # 機種選択時の処理
        self.detect_machine_combo.currentTextChanged.connect(self.update_sample_count)
    
    def update_machine_list(self):
        """機種一覧を更新する。"""
        # テーブルをクリア
        self.machine_table.setRowCount(0)
        
        # 機種データを取得
        machines = self.machine_database.machines
        
        # テーブルに機種データを設定
        for i, (name, data) in enumerate(machines.items()):
            self.machine_table.insertRow(i)
            
            # 機種名
            self.machine_table.setItem(i, 0, QTableWidgetItem(name))
            
            # すべりフレーム
            slip_item = QTableWidgetItem(str(data.get("slip_frames", "-")))
            slip_item.setTextAlignment(Qt.AlignCenter)
            self.machine_table.setItem(i, 1, slip_item)
            
            # 引き込み範囲
            pull_in_item = QTableWidgetItem(str(data.get("pull_in_range", "-")))
            pull_in_item.setTextAlignment(Qt.AlignCenter)
            self.machine_table.setItem(i, 2, pull_in_item)
            
            # ボタン-停止フレーム
            button_to_stop_item = QTableWidgetItem(str(data.get("button_to_stop_frames", "-")))
            button_to_stop_item.setTextAlignment(Qt.AlignCenter)
            self.machine_table.setItem(i, 3, button_to_stop_item)
            
            # 登録日時
            added_date = data.get("added_date", "-")
            self.machine_table.setItem(i, 4, QTableWidgetItem(added_date))
        
        # コンボボックスの更新
        machine_names = self.machine_database.get_all_machine_names()
        
        # リール設定用コンボボックス
        current_reel_machine = self.reel_machine_combo.currentText()
        self.reel_machine_combo.clear()
        self.reel_machine_combo.addItems(machine_names)
        
        # 前の選択を復元（可能な場合）
        index = self.reel_machine_combo.findText(current_reel_machine)
        if index >= 0:
            self.reel_machine_combo.setCurrentIndex(index)
        
        # 機種判別用コンボボックス
        current_detect_machine = self.detect_machine_combo.currentText()
        self.detect_machine_combo.clear()
        self.detect_machine_combo.addItems(machine_names)
        
        # 前の選択を復元（可能な場合）
        index = self.detect_machine_combo.findText(current_detect_machine)
        if index >= 0:
            self.detect_machine_combo.setCurrentIndex(index)
    
    def select_machine(self):
        """選択した機種を現在の機種として設定する。"""
        selected_rows = self.machine_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "選択エラー", "機種を選択してください。")
            return
        
        # 選択行から機種名を取得
        row = selected_rows[0].row()
        machine_name = self.machine_table.item(row, 0).text()
        
        # 機種を設定
        self.machine_database.set_current_machine(machine_name)
        
        # 成功メッセージ
        QMessageBox.information(self, "機種選択", f"機種「{machine_name}」を選択しました。")
        
        # 更新シグナルを発行
        self.machine_updated.emit(machine_name)
    
    def edit_machine(self):
        """選択した機種を編集する。"""
        selected_rows = self.machine_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "選択エラー", "機種を選択してください。")
            return
        
        # 選択行から機種名を取得
        row = selected_rows[0].row()
        machine_name = self.machine_table.item(row, 0).text()
        
        # 機種データを取得
        machine_data = self.machine_database.get_machine(machine_name)
        if not machine_data:
            QMessageBox.warning(self, "データエラー", f"機種「{machine_name}」のデータが見つかりません。")
            return
        
        # 編集タブを選択
        self.name_edit.setText(machine_name)
        self.slip_spin.setValue(machine_data.get("slip_frames", 1))
        self.pull_in_spin.setValue(machine_data.get("pull_in_range", 3))
        self.button_to_stop_spin.setValue(machine_data.get("button_to_stop_frames", 2))
        
        # タブを切り替え
        parent = self.parent()
        if parent and hasattr(parent, "tabWidget"):
            parent.tabWidget.setCurrentIndex(1)
    
    def delete_machine(self):
        """選択した機種を削除する。"""
        selected_rows = self.machine_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "選択エラー", "機種を選択してください。")
            return
        
        # 選択行から機種名を取得
        row = selected_rows[0].row()
        machine_name = self.machine_table.item(row, 0).text()
        
        # デフォルト機種は削除できない
        if machine_name == "default":
            QMessageBox.warning(self, "削除エラー", "デフォルト機種は削除できません。")
            return
        
        # 確認ダイアログ
        reply = QMessageBox.question(
            self, "削除確認", f"機種「{machine_name}」を削除しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 機種を削除
            success = self.machine_database.remove_machine(machine_name)
            
            if success:
                QMessageBox.information(self, "削除完了", f"機種「{machine_name}」を削除しました。")
                self.update_machine_list()
                
                # 更新シグナルを発行
                self.machine_updated.emit("default")
            else:
                QMessageBox.warning(self, "削除エラー", f"機種「{machine_name}」の削除に失敗しました。")
    
    def add_machine(self):
        """新規機種を追加する。"""
        # 入力値を取得
        name = self.name_edit.text().strip()
        slip_frames = self.slip_spin.value()
        pull_in_range = self.pull_in_spin.value()
        button_to_stop_frames = self.button_to_stop_spin.value()
        
        # 入力チェック
        if not name:
            QMessageBox.warning(self, "入力エラー", "機種名を入力してください。")
            return
        
        # 機種データを作成
        machine_data = {
            "slip_frames": slip_frames,
            "pull_in_range": pull_in_range,
            "button_to_stop_frames": button_to_stop_frames,
            "reel_array": [],
            "visual_features": {}
        }
        
        # 機種を追加
        success = self.machine_database.add_machine(name, machine_data)
        
        if success:
            QMessageBox.information(self, "登録完了", f"機種「{name}」を登録しました。")
            self.update_machine_list()
            
            # 入力欄をクリア
            self.name_edit.clear()
            
            # 更新シグナルを発行
            self.machine_updated.emit(name)
        else:
            QMessageBox.warning(self, "登録エラー", f"機種「{name}」の登録に失敗しました。")
    
    def load_reel_array(self, machine_name: str):
        """
        選択した機種のリール配列を読み込む。
        
        Args:
            machine_name (str): 機種名
        """
        if not machine_name:
            return
        
        # 機種データを取得
        machine_data = self.machine_database.get_machine(machine_name)
        if not machine_data:
            self.reel_array_label.setText("リール配列: データが見つかりません")
            return
        
        # リール配列を取得
        reel_array = machine_data.get("reel_array", [])
        
        # リール配列が定義されている場合
        if reel_array and len(reel_array) > 0:
            # リール1
            if len(reel_array) > 0:
                self.reel1_edit.setText(",".join(reel_array[0]))
            else:
                self.reel1_edit.clear()
            
            # リール2
            if len(reel_array) > 1:
                self.reel2_edit.setText(",".join(reel_array[1]))
            else:
                self.reel2_edit.clear()
            
            # リール3
            if len(reel_array) > 2:
                self.reel3_edit.setText(",".join(reel_array[2]))
            else:
                self.reel3_edit.clear()
            
            # リール配列表示
            self.reel_array_label.setText(f"リール配列: 設定済み ({len(reel_array)}リール)")
        else:
            # 未設定の場合
            self.reel1_edit.clear()
            self.reel2_edit.clear()
            self.reel3_edit.clear()
            self.reel_array_label.setText("リール配列: 未設定")
    
    def save_reel_array(self):
        """リール配列を保存する。"""
        # 選択中の機種名を取得
        machine_name = self.reel_machine_combo.currentText()
        if not machine_name:
            QMessageBox.warning(self, "選択エラー", "機種を選択してください。")
            return
        
        # 入力されたリール配列を取得
        reel1_text = self.reel1_edit.text().strip()
        reel2_text = self.reel2_edit.text().strip()
        reel3_text = self.reel3_edit.text().strip()
        
        # 少なくとも1つのリールが設定されているか確認
        if not reel1_text and not reel2_text and not reel3_text:
            QMessageBox.warning(self, "入力エラー", "少なくとも1つのリールに図柄を入力してください。")
            return
        
        # リール配列を作成
        reel_array = []
        
        # リール1
        if reel1_text:
            reel1 = [s.strip() for s in reel1_text.split(",") if s.strip()]
            reel_array.append(reel1)
        
        # リール2
        if reel2_text:
            reel2 = [s.strip() for s in reel2_text.split(",") if s.strip()]
            reel_array.append(reel2)
        
        # リール3
        if reel3_text:
            reel3 = [s.strip() for s in reel3_text.split(",") if s.strip()]
            reel_array.append(reel3)
        
        # リール配列を保存
        success = self.machine_database.store_reel_array(machine_name, reel_array)
        
        if success:
            QMessageBox.information(self, "保存完了", f"機種「{machine_name}」のリール配列を保存しました。")
            self.reel_array_label.setText(f"リール配列: 設定済み ({len(reel_array)}リール)")
            
            # 更新シグナルを発行
            self.machine_updated.emit(machine_name)
        else:
            QMessageBox.warning(self, "保存エラー", f"機種「{machine_name}」のリール配列の保存に失敗しました。")
    
    def update_frame(self, frame: np.ndarray):
        """
        現在のフレームを更新する。
        
        Args:
            frame (np.ndarray): 新しいフレーム
        """
        self.current_frame = frame
        
        if frame is not None:
            # OpenCV画像をQPixmapに変換
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            # 表示サイズに合わせて拡大・縮小
            pixmap = pixmap.scaled(self.frame_label.width(), self.frame_label.height(), Qt.KeepAspectRatio)
            
            # 画像を表示
            self.frame_label.setPixmap(pixmap)
        else:
            self.frame_label.clear()
            self.frame_label.setText("フレーム未取得")
    
    def update_sample_count(self, machine_name: str):
        """
        特徴サンプル数を更新する。
        
        Args:
            machine_name (str): 機種名
        """
        if not machine_name:
            self.sample_count_label.setText("0")
            return
        
        # 機種データを取得
        machine_data = self.machine_database.get_machine(machine_name)
        if not machine_data:
            self.sample_count_label.setText("0")
            return
        
        # 視覚的特徴を取得
        visual_features = machine_data.get("visual_features", {})
        
        # サンプル数を計算
        total_samples = 0
        for label, features in visual_features.items():
            total_samples += len(features)
        
        self.sample_count_label.setText(str(total_samples))
    
    def add_feature_sample(self):
        """現在のフレームを特徴サンプルとして追加する。"""
        if self.current_frame is None:
            QMessageBox.warning(self, "フレームエラー", "フレームが取得されていません。")
            return
        
        # 選択中の機種名を取得
        machine_name = self.detect_machine_combo.currentText()
        if not machine_name:
            QMessageBox.warning(self, "選択エラー", "機種を選択してください。")
            return
        
        # 特徴ラベルを取得
        label = self.feature_label_combo.currentText().strip()
        if not label:
            label = "general"
        
        # 特徴サンプルを追加
        success = self.machine_database.add_visual_feature(machine_name, self.current_frame, label)
        
        if success:
            QMessageBox.information(self, "追加完了", f"機種「{machine_name}」に特徴サンプル（{label}）を追加しました。")
            self.update_sample_count(machine_name)
        else:
            QMessageBox.warning(self, "追加エラー", f"特徴サンプルの追加に失敗しました。")
    
    def train_machine_detector(self):
        """機種判別モデルを学習する。"""
        if not self.machine_database.machine_detector:
            QMessageBox.warning(self, "エラー", "機種判別モデルが初期化されていません。")
            return
        
        # 学習開始
        success = self.machine_database.machine_detector.train()
        
        if success:
            QMessageBox.information(self, "学習完了", "機種判別モデルの学習が完了しました。")
        else:
            QMessageBox.warning(self, "学習エラー", "機種判別モデルの学習に失敗しました。有効な学習データが不足している可能性があります。")
    
    def test_machine_detection(self):
        """現在のフレームで機種判別をテストする。"""
        if self.current_frame is None:
            QMessageBox.warning(self, "フレームエラー", "フレームが取得されていません。")
            return
        
        if not self.machine_database.machine_detector or not self.machine_database.machine_detector.model_trained:
            QMessageBox.warning(self, "モデルエラー", "機種判別モデルが学習されていません。")
            return
        
        # 機種判別を実行
        detected_machine = self.machine_database.detect_machine(self.current_frame)
        
        # 判別結果を表示
        self.detection_result_label.setText(f"判別結果: {detected_machine}")
        
        QMessageBox.information(self, "判別結果", f"このフレームは「{detected_machine}」と判別されました。")
