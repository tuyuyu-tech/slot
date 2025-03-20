"""
パチスロ動画ビタ押しタイミング認識システムのメインプログラム。
"""
import os
import sys
import logging
from PyQt5.QtWidgets import QApplication

# ロギングの設定
def setup_logging():
    """アプリケーションのロギングを設定する。"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("アプリケーション起動")

# メイン関数
def main():
    """アプリケーションのメインエントリーポイント。"""
    # ロギングの設定
    setup_logging()
    
    # 相対インポートのための処理
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    # UIのインポートはここで行う（循環インポートを避けるため）
    from src.ui.main_window import MainWindow
    
    # アプリケーションの作成と実行
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    # アプリケーションの実行
    try:
        sys.exit(app.exec_())
    finally:
        logging.info("アプリケーション終了")

if __name__ == "__main__":
    main()
