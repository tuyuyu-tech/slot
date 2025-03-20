"""
pytestの設定と共通フィクスチャを提供するモジュール。
"""
import os
import sys
import pytest
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication

# テスト対象モジュールのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PyQtアプリケーションのフィクスチャ
@pytest.fixture(scope="session")
def qapp():
    """PyQtアプリケーションのインスタンス"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

# テスト用の画像データ
@pytest.fixture
def test_image():
    """テスト用の画像を生成する"""
    # 単純な画像を生成（黒背景に白い四角）
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (150, 100), (250, 200), (255, 255, 255), -1)
    return img

# テスト用の図柄テンプレート
@pytest.fixture
def symbol_template():
    """テスト用の図柄テンプレートを生成する"""
    # 単純な図柄を生成（黒背景に白い円）
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.circle(template, (25, 25), 20, (255, 255, 255), -1)
    return template

# テスト用のリール画像
@pytest.fixture
def reel_image():
    """テスト用のリール画像を生成する"""
    # 縦長の画像を生成し、複数の図柄を配置
    img = np.zeros((500, 100, 3), dtype=np.uint8)
    
    # 複数の図柄を等間隔で配置
    for i in range(5):
        y = 50 + i * 100
        cv2.circle(img, (50, y), 30, (255, 255, 255), 2)
        cv2.rectangle(img, (35, y-15), (65, y+15), (128, 128, 128), -1)
    
    return img
