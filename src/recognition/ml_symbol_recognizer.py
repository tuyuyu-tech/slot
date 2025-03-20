"""
機械学習を用いたパチスロリール上の図柄認識を行うモジュール。
CNNを使ってより高精度な図柄認識を実現する。
"""
import os
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
import joblib

# ローカルモジュールのインポート
from src.recognition.symbol_recognizer import SymbolRecognizer, Symbol

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLSymbolRecognizer(SymbolRecognizer):
    """
    機械学習を用いた図柄認識クラス。
    HOG特徴量とSVMを利用した図柄認識を行う。
    
    Attributes:
        model_path (str): 学習済みモデルの保存パス
        model (object): 学習済みの分類器モデル
        hog (cv2.HOGDescriptor): HOG特徴量抽出器
        symbol_size (Tuple[int, int]): 特徴量抽出時の図柄サイズ
        min_detection_score (float): 検出スコアの最小閾値
    """
    
    def __init__(self, 
                 templates_dir: str = '../data/templates',
                 models_dir: str = '../data/models',
                 symbol_size: Tuple[int, int] = (64, 64),
                 min_detection_score: float = 0.6):
        """
        MLSymbolRecognizerクラスの初期化。
        
        Args:
            templates_dir (str, optional): テンプレート画像のディレクトリパス。
            models_dir (str, optional): 学習済みモデルのディレクトリパス。
            symbol_size (Tuple[int, int], optional): 特徴量抽出時の図柄サイズ。
            min_detection_score (float, optional): 検出スコアの最小閾値。
        """
        super().__init__(templates_dir)
        
        self.model_path = os.path.join(models_dir, 'symbol_classifier.joblib')
        self.model = None
        self.symbol_size = symbol_size
        self.min_detection_score = min_detection_score
        
        # HOG特徴量抽出器の初期化
        self.hog = cv2.HOGDescriptor(
            _winSize=(symbol_size[0], symbol_size[1]),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        # モデルディレクトリの作成
        os.makedirs(models_dir, exist_ok=True)
        
        # 学習済みモデルが存在する場合は読み込む
        self._load_model()
    
    def _load_model(self) -> None:
        """
        学習済みモデルを読み込む。
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"学習済みモデルを読み込みました: {self.model_path}")
            except Exception as e:
                logger.error(f"モデル読み込み中にエラーが発生しました: {str(e)}")
        else:
            logger.warning(f"学習済みモデルが見つかりません: {self.model_path}")
    
    def _save_model(self) -> None:
        """
        学習済みモデルを保存する。
        """
        if self.model is not None:
            try:
                joblib.dump(self.model, self.model_path)
                logger.info(f"学習済みモデルを保存しました: {self.model_path}")
            except Exception as e:
                logger.error(f"モデル保存中にエラーが発生しました: {str(e)}")
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        画像からHOG特徴量を抽出する。
        
        Args:
            image (np.ndarray): 特徴量を抽出する画像
            
        Returns:
            np.ndarray: 抽出されたHOG特徴量
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # サイズ変更
        resized = cv2.resize(gray, self.symbol_size)
        
        # HOG特徴量を抽出
        features = self.hog.compute(resized)
        
        return features.flatten()
    
    def train_model(self, augment_data: bool = True) -> bool:
        """
        登録されたテンプレートを用いてモデルを学習する。
        
        Args:
            augment_data (bool, optional): データ拡張を行うかどうか。デフォルトはTrue。
            
        Returns:
            bool: 学習が成功したかどうか
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        
        try:
            # 学習データの準備
            X = []  # 特徴量
            y = []  # ラベル（図柄名）
            
            logger.info("学習データの準備を開始")
            
            # 登録されている図柄ごとに特徴量を抽出
            for name, symbol in self.symbols.items():
                template = symbol.template
                
                # オリジナルテンプレートの特徴量を抽出
                features = self._extract_features(template)
                X.append(features)
                y.append(name)
                
                # データ拡張（回転、スケーリング、ノイズ追加）
                if augment_data:
                    # 回転
                    for angle in [-10, -5, 5, 10]:
                        rows, cols = template.shape
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                        rotated = cv2.warpAffine(template, M, (cols, rows))
                        features = self._extract_features(rotated)
                        X.append(features)
                        y.append(name)
                    
                    # スケーリング
                    for scale in [0.9, 1.1]:
                        scaled_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
                        scaled = cv2.resize(template, scaled_size)
                        # 元のサイズにパディング/クロップ
                        if scale < 1.0:  # パディング
                            padded = np.zeros_like(template)
                            x_offset = (template.shape[1] - scaled.shape[1]) // 2
                            y_offset = (template.shape[0] - scaled.shape[0]) // 2
                            padded[y_offset:y_offset+scaled.shape[0], x_offset:x_offset+scaled.shape[1]] = scaled
                            scaled = padded
                        else:  # クロップ
                            x_offset = (scaled.shape[1] - template.shape[1]) // 2
                            y_offset = (scaled.shape[0] - template.shape[0]) // 2
                            scaled = scaled[y_offset:y_offset+template.shape[0], x_offset:x_offset+template.shape[1]]
                        
                        features = self._extract_features(scaled)
                        X.append(features)
                        y.append(name)
                    
                    # ノイズ追加
                    for _ in range(2):
                        noisy = template.copy()
                        noise = np.random.normal(0, 10, template.shape).astype(np.uint8)
                        noisy = cv2.add(noisy, noise)
                        features = self._extract_features(noisy)
                        X.append(features)
                        y.append(name)
            
            # データが十分にあるか確認
            if len(X) < 2:
                logger.warning("学習データが不足しています")
                return False
            
            # 特徴量をnumpy配列に変換
            X = np.array(X)
            y = np.array(y)
            
            # 学習・テストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 特徴量のスケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデルの構築と学習
            logger.info("モデルの学習を開始")
            model = SVC(C=10, gamma='scale', probability=True)
            model.fit(X_train_scaled, y_train)
            
            # 評価
            accuracy = model.score(X_test_scaled, y_test)
            logger.info(f"モデルの精度: {accuracy:.4f}")
            
            # モデルとスケーラーを保存
            self.model = {
                'classifier': model,
                'scaler': scaler
            }
            self._save_model()
            
            logger.info("モデルの学習が完了しました")
            return True
        
        except Exception as e:
            logger.error(f"モデル学習中にエラーが発生しました: {str(e)}")
            return False
    
    def sliding_window(self, frame: np.ndarray, step_size: int = 16, min_size: Tuple[int, int] = (32, 32)) -> List[Dict[str, Any]]:
        """
        スライディングウィンドウ方式で画像内の図柄を検出する。
        
        Args:
            frame (np.ndarray): 検出対象のフレーム画像
            step_size (int, optional): スライディングウィンドウのステップサイズ。
            min_size (Tuple[int, int], optional): 検出する図柄の最小サイズ。
            
        Returns:
            List[Dict[str, Any]]: 検出された図柄情報のリスト
        """
        if self.model is None:
            logger.warning("モデルが読み込まれていません")
            return []
        
        # グレースケールに変換
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        detections = []
        
        # 複数のウィンドウサイズでスキャン
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
            # スケールに応じたウィンドウサイズを計算
            win_w = int(self.symbol_size[0] * scale)
            win_h = int(self.symbol_size[1] * scale)
            
            # 最小サイズより小さい場合はスキップ
            if win_w < min_size[0] or win_h < min_size[1]:
                continue
            
            # スケールに応じた画像を作成
            if scale != 1.0:
                scaled_w = int(frame.shape[1] * scale)
                scaled_h = int(frame.shape[0] * scale)
                scaled_frame = cv2.resize(gray, (scaled_w, scaled_h))
            else:
                scaled_frame = gray
            
            # スライディングウィンドウでスキャン
            for y in range(0, scaled_frame.shape[0] - win_h, step_size):
                for x in range(0, scaled_frame.shape[1] - win_w, step_size):
                    # ウィンドウを切り出し
                    window = scaled_frame[y:y+win_h, x:x+win_w]
                    
                    # 特徴量を抽出
                    features = self._extract_features(window)
                    
                    # 予測
                    scaler = self.model['scaler']
                    classifier = self.model['classifier']
                    
                    features_scaled = scaler.transform([features])
                    proba = classifier.predict_proba(features_scaled)[0]
                    max_proba_idx = np.argmax(proba)
                    confidence = proba[max_proba_idx]
                    
                    # 閾値以上の場合のみ検出結果に追加
                    if confidence >= self.min_detection_score:
                        # 元のフレームの座標に変換
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(win_w / scale)
                        orig_h = int(win_h / scale)
                        
                        # 予測されたクラス（図柄名）
                        predicted_class = classifier.classes_[max_proba_idx]
                        
                        # 検出結果を追加
                        detections.append({
                            'x': orig_x,
                            'y': orig_y,
                            'width': orig_w,
                            'height': orig_h,
                            'name': predicted_class,
                            'score': float(confidence)
                        })
        
        # 重複した検出結果を統合（Non-Maximum Suppression）
        return self._non_max_suppression(detections)
    
    def _non_max_suppression(self, detections: List[Dict[str, Any]], overlap_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        重複した検出結果を統合する（Non-Maximum Suppression）。
        
        Args:
            detections (List[Dict[str, Any]]): 検出結果のリスト
            overlap_threshold (float, optional): 重複とみなす閾値。
            
        Returns:
            List[Dict[str, Any]]: 統合後の検出結果リスト
        """
        # 検出結果がない場合は空リストを返す
        if not detections:
            return []
        
        # スコア順にソート（高い順）
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 結果リスト
        keep = []
        
        while detections:
            # スコアが最も高い検出を保持
            best = detections.pop(0)
            keep.append(best)
            
            # 残りの検出と比較
            remaining = []
            for det in detections:
                # 同じ図柄名でないものは保持
                if det['name'] != best['name']:
                    remaining.append(det)
                    continue
                
                # 重複度を計算
                # 矩形の座標
                x1 = max(best['x'], det['x'])
                y1 = max(best['y'], det['y'])
                x2 = min(best['x'] + best['width'], det['x'] + det['width'])
                y2 = min(best['y'] + best['height'], det['y'] + det['height'])
                
                # 重複領域の幅と高さ
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                
                # 重複領域の面積
                overlap_area = w * h
                
                # 検出領域の面積
                det_area = det['width'] * det['height']
                best_area = best['width'] * best['height']
                
                # 重複率
                overlap_ratio = overlap_area / min(det_area, best_area)
                
                # 重複が閾値未満なら保持
                if overlap_ratio < overlap_threshold:
                    remaining.append(det)
            
            # 残りの検出を更新
            detections = remaining
        
        return keep
    
    def recognize_symbols(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        フレーム内の図柄を認識する。
        
        Args:
            frame (np.ndarray): 認識対象のフレーム画像
            
        Returns:
            List[Dict[str, Any]]: 認識された図柄情報のリスト
        """
        # モデルが存在する場合は機械学習による認識
        if self.model is not None:
            return self.sliding_window(frame)
        # モデルがない場合は空のリストを返す
        else:
            logger.warning("モデルが読み込まれていないため、図柄認識できません")
            return []


class HybridSymbolRecognizer(SymbolRecognizer):
    """
    テンプレートマッチングと機械学習を組み合わせたハイブリッド図柄認識クラス。
    両方の手法の利点を活かし、より高精度な認識を実現する。
    
    Attributes:
        template_matcher (TemplateMatching): テンプレートマッチング認識器
        ml_recognizer (MLSymbolRecognizer): 機械学習認識器
        use_ml (bool): 機械学習認識器を使用するかどうか
    """
    
    def __init__(self, templates_dir: str = '../data/templates',
                 models_dir: str = '../data/models'):
        """
        HybridSymbolRecognizerクラスの初期化。
        
        Args:
            templates_dir (str, optional): テンプレート画像のディレクトリパス。
            models_dir (str, optional): 学習済みモデルのディレクトリパス。
        """
        super().__init__(templates_dir)
        
        # テンプレートマッチング認識器
        from src.recognition.symbol_recognizer import TemplateMatching
        self.template_matcher = TemplateMatching(templates_dir)
        
        # 機械学習認識器
        self.ml_recognizer = MLSymbolRecognizer(templates_dir, models_dir)
        
        # 機械学習認識器が使用可能かどうか
        self.use_ml = self.ml_recognizer.model is not None
    
    def register_template(self, 
                          name: str, 
                          template: np.ndarray, 
                          threshold: float = 0.7,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        新しい図柄テンプレートを登録する。
        
        Args:
            name (str): 図柄の名前
            template (np.ndarray): 図柄のテンプレート画像
            threshold (float, optional): マッチング閾値。デフォルトは0.7。
            metadata (Optional[Dict[str, Any]], optional): 図柄に関するメタデータ。
            
        Returns:
            bool: 登録が成功したかどうか
        """
        # テンプレートマッチング認識器に登録
        template_success = self.template_matcher.register_template(name, template, threshold, metadata)
        
        # 機械学習認識器に登録
        ml_success = self.ml_recognizer.register_template(name, template, threshold, metadata)
        
        # 登録後にモデルを再学習
        if ml_success:
            self.ml_recognizer.train_model()
            # 学習後にモデルが利用可能になったかチェック
            self.use_ml = self.ml_recognizer.model is not None
        
        # 親クラスのsymbolsに登録
        super().register_template(name, template, threshold, metadata)
        
        return template_success and ml_success
    
    def recognize_symbols(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        フレーム内の図柄を認識する。
        
        Args:
            frame (np.ndarray): 認識対象のフレーム画像
            
        Returns:
            List[Dict[str, Any]]: 認識された図柄情報のリスト
        """
        # テンプレートマッチングによる認識
        template_results = self.template_matcher.recognize_symbols(frame)
        
        # 機械学習による認識（モデルが利用可能な場合）
        ml_results = []
        if self.use_ml:
            ml_results = self.ml_recognizer.recognize_symbols(frame)
        
        # 結果を統合
        combined_results = self._merge_results(template_results, ml_results)
        
        return combined_results
    
    def _merge_results(self, 
                       template_results: List[Dict[str, Any]], 
                       ml_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        テンプレートマッチングと機械学習の認識結果を統合する。
        
        Args:
            template_results (List[Dict[str, Any]]): テンプレートマッチングの結果
            ml_results (List[Dict[str, Any]]): 機械学習の結果
            
        Returns:
            List[Dict[str, Any]]: 統合された認識結果
        """
        all_results = template_results + ml_results
        
        # 重複を除外するためのNon-Maximum Suppression
        merged_results = []
        
        # 結果がない場合は空リストを返す
        if not all_results:
            return []
        
        # スコア順にソート（高い順）
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        while all_results:
            # スコアが最も高い検出を保持
            best = all_results.pop(0)
            merged_results.append(best)
            
            # 残りの検出と比較
            remaining = []
            for det in all_results:
                # 同じ図柄名でないものは保持
                if det['name'] != best['name']:
                    remaining.append(det)
                    continue
                
                # 重複度を計算
                # 矩形の座標
                x1 = max(best['x'], det['x'])
                y1 = max(best['y'], det['y'])
                x2 = min(best['x'] + best['width'], det['x'] + det['width'])
                y2 = min(best['y'] + best['height'], det['y'] + det['height'])
                
                # 重複領域の幅と高さ
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                
                # 重複領域の面積
                overlap_area = w * h
                
                # 検出領域の面積
                det_area = det['width'] * det['height']
                best_area = best['width'] * best['height']
                
                # 重複率
                overlap_ratio = overlap_area / min(det_area, best_area)
                
                # 重複が30%未満なら保持
                if overlap_ratio < 0.3:
                    remaining.append(det)
            
            # 残りの検出を更新
            all_results = remaining
        
        return merged_results
