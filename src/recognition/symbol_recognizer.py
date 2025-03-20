"""
パチスロリール上の図柄を認識するためのモジュール。
"""
import os
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Symbol:
    """
    パチスロの図柄を表すクラス。
    
    Attributes:
        name (str): 図柄の名前
        template (np.ndarray): 図柄のテンプレート画像
        threshold (float): マッチング閾値
        metadata (Dict): 図柄に関するメタデータ
    """
    
    def __init__(self, 
                 name: str, 
                 template: np.ndarray, 
                 threshold: float = 0.7, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Symbolクラスの初期化。
        
        Args:
            name (str): 図柄の名前
            template (np.ndarray): 図柄のテンプレート画像
            threshold (float, optional): マッチング閾値。デフォルトは0.7。
            metadata (Optional[Dict[str, Any]], optional): 図柄に関するメタデータ。
        """
        self.name = name
        self.template = template
        self.threshold = threshold
        self.metadata = metadata if metadata else {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        図柄情報を辞書形式に変換する。
        
        Returns:
            Dict[str, Any]: 図柄情報の辞書
        """
        return {
            'name': self.name,
            'threshold': self.threshold,
            'metadata': self.metadata
        }


class SymbolRecognizer:
    """
    図柄認識の基底クラス。
    図柄の登録・認識を行うインターフェースを提供する。
    
    Attributes:
        symbols (Dict[str, Symbol]): 登録済み図柄の辞書
        templates_dir (str): 図柄テンプレートの保存ディレクトリ
    """
    
    def __init__(self, templates_dir: str = '../data/templates'):
        """
        SymbolRecognizerクラスの初期化。
        
        Args:
            templates_dir (str, optional): テンプレート画像のディレクトリパス。
        """
        self.symbols = {}
        self.templates_dir = templates_dir
        
        # テンプレートディレクトリが存在しない場合は作成
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # 保存されているテンプレートを読み込む
        self._load_templates()
    
    def _load_templates(self) -> None:
        """
        テンプレートディレクトリから図柄テンプレートを読み込む。
        """
        logger.info(f"テンプレートディレクトリから図柄を読み込み中: {self.templates_dir}")
        
        # テンプレートディレクトリに保存されているJSONファイルを探す
        index_file = os.path.join(self.templates_dir, 'index.json')
        if not os.path.exists(index_file):
            logger.warning(f"テンプレートインデックスファイルが見つかりません: {index_file}")
            return
        
        try:
            # インデックスファイルから図柄情報を読み込む
            with open(index_file, 'r', encoding='utf-8') as f:
                symbols_data = json.load(f)
            
            # 各図柄テンプレートを読み込む
            for symbol_data in symbols_data:
                name = symbol_data['name']
                image_path = os.path.join(self.templates_dir, f"{name}.png")
                
                if os.path.exists(image_path):
                    template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.symbols[name] = Symbol(
                            name=name,
                            template=template,
                            threshold=symbol_data.get('threshold', 0.7),
                            metadata=symbol_data.get('metadata', {})
                        )
                        logger.info(f"図柄テンプレートを読み込みました: {name}")
                    else:
                        logger.warning(f"図柄テンプレートの読み込みに失敗しました: {image_path}")
                else:
                    logger.warning(f"図柄テンプレートが見つかりません: {image_path}")
        
        except Exception as e:
            logger.error(f"テンプレート読み込み中にエラーが発生しました: {str(e)}")
    
    def _save_templates(self) -> None:
        """
        登録されている図柄テンプレートをファイルに保存する。
        """
        try:
            # インデックスファイルに図柄情報を保存
            symbols_data = [symbol.to_dict() for symbol in self.symbols.values()]
            index_file = os.path.join(self.templates_dir, 'index.json')
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(symbols_data, f, ensure_ascii=False, indent=2)
            
            # 各図柄テンプレート画像を保存
            for name, symbol in self.symbols.items():
                image_path = os.path.join(self.templates_dir, f"{name}.png")
                cv2.imwrite(image_path, symbol.template)
            
            logger.info("図柄テンプレートを保存しました")
        
        except Exception as e:
            logger.error(f"テンプレート保存中にエラーが発生しました: {str(e)}")
    
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
        try:
            # グレースケールに変換
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # 図柄を登録
            self.symbols[name] = Symbol(
                name=name,
                template=template,
                threshold=threshold,
                metadata=metadata if metadata else {}
            )
            
            # テンプレートを保存
            self._save_templates()
            
            logger.info(f"図柄テンプレートを登録しました: {name}")
            return True
        
        except Exception as e:
            logger.error(f"図柄テンプレート登録中にエラーが発生しました: {str(e)}")
            return False
    
    def recognize_symbols(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        フレーム内の図柄を認識する。
        サブクラスでオーバーライドする必要があるメソッド。
        
        Args:
            frame (np.ndarray): 認識対象のフレーム画像
            
        Returns:
            List[Dict[str, Any]]: 認識された図柄情報のリスト
        """
        raise NotImplementedError("このメソッドはサブクラスでオーバーライドする必要があります")


class TemplateMatching(SymbolRecognizer):
    """
    テンプレートマッチングによる図柄認識クラス。
    
    Attributes:
        method (int): テンプレートマッチングのメソッド
    """
    
    def __init__(self, templates_dir: str = '../data/templates',
                 method: int = cv2.TM_CCOEFF_NORMED):
        """
        TemplateMatchingクラスの初期化。
        
        Args:
            templates_dir (str, optional): テンプレート画像のディレクトリパス。
            method (int, optional): テンプレートマッチングのメソッド。
                                    デフォルトはcv2.TM_CCOEFF_NORMED。
        """
        super().__init__(templates_dir)
        self.method = method
    
    def match_template(self, 
                       frame: np.ndarray, 
                       template: np.ndarray,
                       threshold: float) -> List[Dict[str, Any]]:
        """
        テンプレートマッチングを実行し、フレーム内で指定されたテンプレートに一致する領域を検出する。
        
        Args:
            frame (np.ndarray): 検索対象のフレーム画像
            template (np.ndarray): 検索するテンプレート画像
            threshold (float): マッチング閾値
            
        Returns:
            List[Dict[str, Any]]: マッチング結果のリスト
        """
        # グレースケールに変換
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # テンプレートマッチングを実行
        result = cv2.matchTemplate(gray_frame, template, self.method)
        
        # 閾値以上のマッチ位置を検出
        locations = np.where(result >= threshold)
        matches = []
        
        # 検出結果を処理
        for pt in zip(*locations[::-1]):  # （x, y）の順に変換
            # 重複したマッチングを除外（既に検出した位置の近くは除外）
            is_duplicate = False
            for match in matches:
                # 既存のマッチング位置との距離を計算
                dx = pt[0] - match['x']
                dy = pt[1] - match['y']
                distance = np.sqrt(dx*dx + dy*dy)
                
                # 近い位置にあるマッチングは除外
                if distance < template.shape[0] / 2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # マッチング結果を追加
                score = result[pt[1], pt[0]]
                matches.append({
                    'x': pt[0],
                    'y': pt[1],
                    'width': template.shape[1],
                    'height': template.shape[0],
                    'score': float(score)
                })
        
        return matches
    
    def recognize_symbols(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        フレーム内の登録済み図柄を認識する。
        
        Args:
            frame (np.ndarray): 認識対象のフレーム画像
            
        Returns:
            List[Dict[str, Any]]: 認識された図柄情報のリスト
        """
        results = []
        
        # 登録されている各図柄についてマッチングを実行
        for name, symbol in self.symbols.items():
            matches = self.match_template(frame, symbol.template, symbol.threshold)
            
            # 検出結果に図柄名とメタデータを追加
            for match in matches:
                match['name'] = name
                match['metadata'] = symbol.metadata.copy()
                results.append(match)
        
        # スコア順にソート（高い順）
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results


class SymbolTracker:
    """
    認識された図柄の位置を追跡するクラス。
    
    Attributes:
        tracked_symbols (List[Dict]): 追跡中の図柄リスト
        max_tracking_frames (int): 追跡を継続する最大フレーム数
    """
    
    def __init__(self, max_tracking_frames: int = 10):
        """
        SymbolTrackerクラスの初期化。
        
        Args:
            max_tracking_frames (int, optional): 追跡を継続する最大フレーム数。
                                                デフォルトは10フレーム。
        """
        self.tracked_symbols = []
        self.max_tracking_frames = max_tracking_frames
    
    def update(self, 
               recognized_symbols: List[Dict[str, Any]], 
               frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        追跡中の図柄情報を更新する。
        
        Args:
            recognized_symbols (List[Dict[str, Any]]): 現在のフレームで認識された図柄リスト
            frame (np.ndarray): 現在のフレーム画像
            
        Returns:
            List[Dict[str, Any]]: 更新された追跡情報
        """
        # 前回追跡していた図柄がない場合、新規に追跡開始
        if not self.tracked_symbols:
            for symbol in recognized_symbols:
                # 追跡情報を追加
                symbol['tracking_frames'] = 0
                symbol['prev_positions'] = [(symbol['x'], symbol['y'])]
                symbol['velocity'] = (0, 0)  # (vx, vy) - 速度ベクトル
                self.tracked_symbols.append(symbol)
            return self.tracked_symbols
        
        # 前のフレームで追跡していた図柄と現在認識された図柄をマッチング
        updated_symbols = []
        
        for recognized in recognized_symbols:
            matched = False
            
            for tracked in self.tracked_symbols:
                # 同じ図柄名で距離が近いものをマッチングとみなす
                if recognized['name'] == tracked['name']:
                    dx = recognized['x'] - tracked['x']
                    dy = recognized['y'] - tracked['y']
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # 距離が閾値以下ならマッチング
                    max_distance = tracked['height'] * 0.5
                    if distance < max_distance:
                        # 位置情報を更新
                        recognized['tracking_frames'] = tracked['tracking_frames'] + 1
                        recognized['prev_positions'] = tracked['prev_positions'] + [(recognized['x'], recognized['y'])]
                        recognized['prev_positions'] = recognized['prev_positions'][-5:]  # 最新5フレーム分だけ保持
                        
                        # 速度計算（前のフレームとの差から）
                        if len(recognized['prev_positions']) >= 2:
                            prev_x, prev_y = recognized['prev_positions'][-2]
                            vx = recognized['x'] - prev_x
                            vy = recognized['y'] - prev_y
                            recognized['velocity'] = (vx, vy)
                        else:
                            recognized['velocity'] = tracked['velocity']
                        
                        updated_symbols.append(recognized)
                        matched = True
                        break
            
            if not matched and len(updated_symbols) < 10:  # 最大10個まで追跡
                # 新規の図柄として追加
                recognized['tracking_frames'] = 0
                recognized['prev_positions'] = [(recognized['x'], recognized['y'])]
                recognized['velocity'] = (0, 0)
                updated_symbols.append(recognized)
        
        # トラッキング情報を更新
        self.tracked_symbols = [s for s in updated_symbols if s['tracking_frames'] <= self.max_tracking_frames]
        
        return self.tracked_symbols
    
    def predict_movement(self, 
                         symbol: Dict[str, Any], 
                         frames_ahead: int = 1) -> Tuple[int, int]:
        """
        図柄の将来位置を予測する。
        
        Args:
            symbol (Dict[str, Any]): 追跡中の図柄情報
            frames_ahead (int, optional): 何フレーム先を予測するか。デフォルトは1。
            
        Returns:
            Tuple[int, int]: 予測されるx, y座標
        """
        if len(symbol['prev_positions']) < 2:
            return symbol['x'], symbol['y']
        
        # 現在の速度を使用して予測
        vx, vy = symbol['velocity']
        predicted_x = symbol['x'] + vx * frames_ahead
        predicted_y = symbol['y'] + vy * frames_ahead
        
        return int(predicted_x), int(predicted_y)
    
    def get_tracked_symbols(self) -> List[Dict[str, Any]]:
        """
        現在追跡中の図柄リストを取得する。
        
        Returns:
            List[Dict[str, Any]]: 追跡中の図柄リスト
        """
        return self.tracked_symbols
