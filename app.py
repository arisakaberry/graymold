import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
from datetime import datetime
from typing import List, Tuple
import numpy as np
import os
import re

# 灰色かび病リスクチェック関数
def check_gray_mold_risk(temp_humidity_data: List[Tuple[float, float]], timestamps) -> Tuple[str, int, str]:
    """
    最新の10日間(240時間)のデータからリスクレベルを算出する関数
    """
    # データをタイムスタンプでソート
    data_sorted = sorted(zip(temp_humidity_data, timestamps), key=lambda x: x[1])
    
    if not data_sorted:
        return "データなし", 0, "gray"
    
    # 最新の日時
    latest_timestamp = max(ts for _, ts in data_sorted)
    
    # 10日前の日時
    window_start = latest_timestamp - pd.Timedelta(hours=240)
    
    # 過去10日間のデータを抽出
    recent_window = [(temp, humidity) for (temp, humidity), ts in data_sorted 
                    if window_start <= ts <= latest_timestamp]
    
    # リスク時間を計算
    risk_hours = sum(
        1 for temp, humidity in recent_window
        if 15 <= float(temp) <= 25 and float(humidity) >= 94
    )
    
    # リスクレベルとカラーを決定
    if risk_hours == 0:
        return "極低", risk_hours, "gray"
    elif 0 < risk_hours <= 10:
        return "低", risk_hours, "blue"
    elif 10 < risk_hours <= 20:
        return "中", risk_hours, "green"
    elif 20 < risk_hours <= 30:
        return "高", risk_hours, "orange"
    else:
        return "極高", risk_hours, "red"

# CSVファイルから気温データと相対湿度データを読み込む関数
# 拡張版：様々なセンサーCSVファイルから気温・湿度データを読み込む関数
def read_temperature_and_humidity_data(file_obj, device_type=None):
    """
    様々な形式のセンサーデータファイルから温度と湿度のデータを読み込む関数
    
    Parameters:
    ----------
    file_obj : UploadedFile または str
        Streamlitでアップロードされたファイルオブジェクトまたはファイルパス
    device_type : str, optional
        デバイスタイプ ('HZ', 'PF', 'PF2', 'SB', 'OT', 'HN', None)
        None の場合はデフォルト形式またはファイル内容から推測を試みる
        
    Returns:
    -------
    tuple
        (temperature_list, humidity_list, timestamps)
    """
    # 内部ユーティリティ関数
    def try_multiple_encodings(file_path):
        """複数のエンコーディングを試してCSVファイルを読み込む関数"""
        encodings = ['shift-jis', 'utf-8', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                st.info(f"エンコーディング {encoding} で読み込み試行中...")
                df = pd.read_csv(file_path, encoding=encoding)
                st.success(f"エンコーディング {encoding} で読み込み成功")
                st.info(f"列名一覧: {df.columns.tolist()}")
                return df, encoding
            except Exception as e:
                st.info(f"エンコーディング {encoding} での読み込み失敗: {str(e)}")
                continue
        
        st.error("すべてのエンコーディングでの読み込みに失敗しました")
        return None, None

    def combine_date_time(df, date_col='日付', time_col='時刻'):
        """日付列と時刻列を結合してdatetime型の列を作成する関数"""
        try:
            st.info("日付と時刻の列を変換しています")
            
            # 時刻列のクリーニング（*を削除）
            if time_col in df.columns:
                st.info("時刻列を前処理しています")
                st.info(f"時刻の最初の5件: {df[time_col].head().tolist()}")
                df[time_col] = df[time_col].astype(str).str.replace('*', '', regex=False)
                df[time_col] = df[time_col].str.strip()
                st.info(f"処理後の時刻列サンプル: {df[time_col].head().tolist()}")
            
            # データ型を確認
            st.info(f"日付列のデータ型: {df[date_col].dtype}")
            st.info(f"時刻列のデータ型: {df[time_col].dtype}")
            
            # 行ごとに日時を変換
            dates = []
            for idx, row in df.iterrows():
                try:
                    date_str = str(row[date_col]).strip()
                    time_str = str(row[time_col]).strip()
                    
                    # 時刻が数字だけの場合は ":00" を追加
                    if time_str.isdigit() and len(time_str) <= 2:
                        time_str = f"{time_str}:00"
                    
                    # 日付と時刻を結合
                    date_time_str = f"{date_str} {time_str}"
                    
                    # datetimeに変換
                    date_time = pd.to_datetime(date_time_str, errors='coerce')
                    dates.append(date_time)
                    
                    # デバッグ用（最初の10行のみ）
                    if idx < 10:
                        st.info(f"行 {idx}: '{date_str}' + '{time_str}' => {date_time}")
                        
                except Exception as e:
                    st.info(f"行 {idx} の日時変換エラー: {str(e)}")
                    dates.append(pd.NaT)  # 変換できない場合はNaN値を追加
            
            # 新しい日時列を作成
            return pd.Series(dates)
            
        except Exception as e:
            st.error(f"日付と時刻の結合に失敗: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return pd.Series([pd.NaT] * len(df))

    def convert_to_numeric(df, columns):
        """指定された列を数値型に変換する関数"""
        for col in columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                st.info(f"{col}を数値型に変換しました。NaN値の数: {df[col].isna().sum()}")
            except Exception as e:
                st.error(f"{col}の数値変換エラー: {str(e)}")
        
        return df

    def resample_to_hourly(df):
        """データを1時間間隔にリサンプリングする関数"""
        try:
            # インデックスがdatetimeかチェック
            if not isinstance(df.index, pd.DatetimeIndex):
                st.error("データフレームのインデックスがdatetime型ではありません")
                raise ValueError("日時データの変換に失敗しました")
            
            # 1時間間隔にリサンプリング
            df.index = df.index.floor('H')
            st.info(f"時間切り捨て後のインデックスサンプル: {df.index[:5].tolist()}")
            
            # ここが重要: 数値型の列のみを抽出してグループ化処理を行う
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            st.info(f"数値型の列: {numeric_cols}")
            
            # 非数値列があれば警告を表示
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                st.warning(f"非数値型の列は除外されます: {non_numeric_cols}")
            
            if numeric_cols:
                # 数値列のみを使用してグループバイを実行
                df_hourly = df[numeric_cols].groupby(df.index).mean()
                st.success("グループバイ処理が成功しました")
            else:
                st.warning("数値型の列が見つからないため、処理を続行できません")
                return df
            
            # 欠損値の補間
            df_resampled = df_hourly.resample('1H').interpolate()
            st.info(f"補間後のデータ形状: {df_resampled.shape}")
            
            return df_resampled
        
        except Exception as e:
            st.error(f"リサンプリングエラー: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            
            # エラー時の代替処理
            try:
                st.warning("代替処理を試みます: 数値型の列のみを使用")
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    numeric_df.index = df.index
                    numeric_hourly = numeric_df.groupby(numeric_df.index).mean()
                    return numeric_hourly.resample('1H').interpolate()
                else:
                    st.error("数値型の列が見つかりません")
                    return df
            except Exception as e2:
                st.error(f"代替処理でもエラーが発生: {str(e2)}")
                return df

    # メイン処理開始
    temp_path = None
    try:
        # アップロードされたファイルを一時ファイルに保存
        import tempfile
        if hasattr(file_obj, 'read'):  # UploadedFileオブジェクトの場合
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(file_obj.getbuffer())
                temp_path = tmp_file.name
            st.info(f"一時ファイルを作成しました: {temp_path}")
        else:  # 文字列（ファイルパス）の場合
            temp_path = file_obj
            st.info(f"ファイルパスを使用します: {temp_path}")
        
        # デバイス設定の定義（辞書を活用）
        # 各デバイスタイプに対応する列名とパラメータを集約
        device_configs = {
            'HZ': {
                'temp_cols': ['温度'],
                'humid_cols': ['湿度'],
                'timestamp_cols': ['日付'],
                'encoding': 'shift-jis'
            },
            'PF': {
                'temp_cols': ['気温'],
                'humid_cols': ['相対湿度'],
                'timestamp_cols': ['年月日'],
                'encoding': 'shift-jis'
            },
            'PF2': {
                'temp_cols': ['PF 測定 気温'],
                'humid_cols': ['湿度'],
                'timestamp_cols': ['datetime'],
                'encoding': 'shift-jis',
                'date_time_cols': {'date_col': '日付', 'time_col': '時刻'}
            },
            'SB': {
                'temp_cols': ['Temperature_Celsius(°C)', 'Temperature_Celsius(℃)', 'Temperature_Celsius(ﾂｰC)'],  # 文字化けパターンを追加
                'humid_cols': ['Relative_Humidity(%)'],
                'timestamp_cols': ['Timestamp', 'Date'],
                'encoding': 'utf-8'  # エンコーディングを指定
            },
            'OT': {
                'temp_cols': ['室温'],
                'humid_cols': ['湿度'],
                'timestamp_cols': ['日付'],
                'encoding': 'shift-jis'
            },
            'HN': {
                'temp_cols': ['温度(℃)'],
                'humid_cols': ['相対湿度(％)'],
                'timestamp_cols': ['日時'],
                'encoding': 'utf-8-sig'
            }
        }
        
        # デバイスタイプの自動検出を試みる（デバイスタイプが未指定の場合）
        if device_type is None:
            st.info("デバイスタイプの自動検出を試みます...")
            
            # ファイルを読み込んでヘッダーを確認
            df, encoding = try_multiple_encodings(temp_path)
            if df is not None:
                cols = df.columns.tolist()
                st.info(f"検出された列名: {cols}")
                
                # 各デバイスタイプの特徴と照合
                for dev, config in device_configs.items():
                    # 温度列が存在するか確認
                    for col_name in config['temp_cols']:
                        if col_name in cols:
                            device_type = dev
                            st.success(f"{dev}形式のファイルを検出しました (列名: {col_name})")
                            break
                    if device_type:
                        break
            
            # 自動検出できない場合はデフォルトとしてSBを使用
            if device_type is None:
                device_type = 'SB'
                st.warning("ファイル形式を自動検出できませんでした。SwitchBot形式として処理します。")
        
        st.info(f"{device_type}形式として処理します")
        
        # 設定を取得
        config = device_configs.get(device_type, device_configs['SB'])
        
        # ファイル読み込み
        df, _ = try_multiple_encodings(temp_path)
        if df is None:
            st.error(f"ファイル読み込みに失敗しました")
            return None, None, None
        
        # タイムスタンプ列の特定と処理（デバイスタイプに応じた処理）
        timestamp_found = False
        
        # PF2形式の特別処理（日付と時刻が別々の列）
        if device_type == 'PF2' and '日付' in df.columns and '時刻' in df.columns:
            st.info("PF2形式として日付と時刻の列を処理します")
            
            # 時刻列からアスタリスクなどを削除して結合
            date_col = config.get('date_time_cols', {}).get('date_col', '日付')
            time_col = config.get('date_time_cols', {}).get('time_col', '時刻')
            
            df['datetime'] = combine_date_time(df, date_col, time_col)
            
            # NaT値をチェック
            nat_count = df['datetime'].isna().sum()
            if nat_count > 0:
                st.warning(f"{nat_count} 行の日時データが無効なため除外されます")
                df = df.dropna(subset=['datetime'])
            
            df = df.set_index('datetime')
            timestamp_found = True
            
        if device_type == 'SB':
            try:
                # SwitchBotファイルはUTF-8で強制的に読み直し
                st.info("SwitchBotファイルをUTF-8で読み込み直します")
                df = pd.read_csv(temp_path, encoding='utf-8')
                st.success("UTF-8での読み込みに成功しました")
                st.info(f"正しく読み込まれた列名: {df.columns.tolist()}")
            except Exception as e:
                st.warning(f"UTF-8での読み込みに失敗しました: {str(e)}")
        
        # 一般的なタイムスタンプ列の処理
        if not timestamp_found:
            for ts_col in config['timestamp_cols']:
                if ts_col in df.columns:
                    st.info(f"タイムスタンプ列を特定しました: {ts_col}")
                    
                    # タイムスタンプをdatetime型に変換
                    df['datetime'] = pd.to_datetime(df[ts_col], errors='coerce')
                    
                    # NaT値をチェック
                    nat_count = df['datetime'].isna().sum()
                    if nat_count > 0:
                        st.warning(f"{nat_count} 行の日時データが無効なため除外されます")
                        df = df.dropna(subset=['datetime'])
                    
                    df = df.set_index('datetime')
                    timestamp_found = True
                    break
        
        if not timestamp_found:
            # 日付と時刻が別々の列の場合（一般的なケース）
            if '日付' in df.columns and '時刻' in df.columns:
                st.info("日付と時刻の列を検出しました")
                df['datetime'] = combine_date_time(df, '日付', '時刻')
                
                # NaT値をチェック
                nat_count = df['datetime'].isna().sum()
                if nat_count > 0:
                    st.warning(f"{nat_count} 行の日時データが無効なため除外されます")
                    df = df.dropna(subset=['datetime'])
                
                df = df.set_index('datetime')
                timestamp_found = True
        
        if not timestamp_found:
            st.error("タイムスタンプ列が見つかりません")
            return None, None, None
        
        # 温度と湿度の列を特定
        temp_col = None
        for col in config['temp_cols']:
            if col in df.columns:
                temp_col = col
                st.info(f"温度列を特定しました: {temp_col}")
                break
        
        humid_col = None
        for col in config['humid_cols']:
            if col in df.columns:
                humid_col = col
                st.info(f"湿度列を特定しました: {humid_col}")
                break
        
        if temp_col is None or humid_col is None:
            st.error(f"必要な列が見つかりません。利用可能な列: {df.columns.tolist()}")
            return None, None, None
        
        # 数値型に変換
        df = convert_to_numeric(df, [temp_col, humid_col])
        
        # NaN値を含む行を除外
        before_count = len(df)
        df = df.dropna(subset=[temp_col, humid_col])
        after_count = len(df)
        if before_count > after_count:
            st.warning(f"{before_count - after_count} 行のデータが無効な数値のため除外されました")
        
        # 1時間間隔にリサンプリング
        df = resample_to_hourly(df)
        
        # 結果を返す
        return df[temp_col].tolist(), df[humid_col].tolist(), df.index
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None
    
    finally:
        # 一時ファイルを確実に削除
        if temp_path and hasattr(file_obj, 'read'):
            try:
                import os
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    st.info("一時ファイルを削除しました")
            except Exception as e:
                st.warning(f"一時ファイル削除エラー: {str(e)}")


def try_multiple_encodings(file_path):
    """複数のエンコーディングを試してCSVファイルを読み込む関数"""
    encodings = ['shift-jis', 'utf-8', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            st.info(f"エンコーディング {encoding} で読み込み試行中...")
            df = pd.read_csv(file_path, encoding=encoding)
            st.success(f"エンコーディング {encoding} で読み込み成功")
            st.info(f"列名一覧: {df.columns.tolist()}")
            return df, encoding
        except Exception as e:
            st.info(f"エンコーディング {encoding} での読み込み失敗: {str(e)}")
            continue
    
    st.error("すべてのエンコーディングでの読み込みに失敗しました")
    return None, None

def combine_date_time(df, date_col='日付', time_col='時刻'):
    """日付列と時刻列を結合してdatetime型の列を作成する関数"""
    try:
        st.info("日付と時刻の列を変換しています")
        
        # 時刻列のクリーニング（*を削除）
        if time_col in df.columns:
            st.info("時刻列を前処理しています")
            st.info(f"時刻の最初の5件: {df[time_col].head().tolist()}")
            df[time_col] = df[time_col].astype(str).str.replace('*', '', regex=False)
            df[time_col] = df[time_col].str.strip()
            st.info(f"処理後の時刻列サンプル: {df[time_col].head().tolist()}")
        
        # データ型を確認
        st.info(f"日付列のデータ型: {df[date_col].dtype}")
        st.info(f"時刻列のデータ型: {df[time_col].dtype}")
        
        # 行ごとに日時を変換
        dates = []
        for idx, row in df.iterrows():
            try:
                date_str = str(row[date_col]).strip()
                time_str = str(row[time_col]).strip()
                
                # 時刻が数字だけの場合は ":00" を追加
                if time_str.isdigit() and len(time_str) <= 2:
                    time_str = f"{time_str}:00"
                
                # 日付と時刻を結合
                date_time_str = f"{date_str} {time_str}"
                
                # datetimeに変換
                date_time = pd.to_datetime(date_time_str, errors='coerce')
                dates.append(date_time)
                
                # デバッグ用（最初の10行のみ）
                if idx < 10:
                    st.info(f"行 {idx}: '{date_str}' + '{time_str}' => {date_time}")
                    
            except Exception as e:
                st.info(f"行 {idx} の日時変換エラー: {str(e)}")
                dates.append(pd.NaT)  # 変換できない場合はNaN値を追加
        
        # 新しい日時列を作成
        return pd.Series(dates)
        
    except Exception as e:
        st.error(f"日付と時刻の結合に失敗: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return pd.Series([pd.NaT] * len(df))

def convert_to_numeric(df, columns):
    """指定された列を数値型に変換する関数"""
    for col in columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            st.info(f"{col}を数値型に変換しました。NaN値の数: {df[col].isna().sum()}")
        except Exception as e:
            st.error(f"{col}の数値変換エラー: {str(e)}")
    
    return df

def resample_to_hourly(df):
    """データを1時間間隔にリサンプリングする関数"""
    try:
        # インデックスがdatetimeかチェック
        if not isinstance(df.index, pd.DatetimeIndex):
            st.error("データフレームのインデックスがdatetime型ではありません")
            raise ValueError("日時データの変換に失敗しました")
        
        # 1時間間隔にリサンプリング
        df.index = df.index.floor('H')
        df_hourly = df.groupby(df.index).mean()
        st.info(f"時間集計後のデータ形状: {df_hourly.shape}")
        
        # 欠損値の補間
        df_resampled = df_hourly.resample('1H').interpolate()
        st.info(f"補間後のデータ形状: {df_resampled.shape}")
        
        return df_resampled
    
    except Exception as e:
        st.error(f"リサンプリングエラー: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return df


# 時系列リスク計算関数
def calculate_time_series_risk(temp_humidity_data, timestamps, days_to_show=14):
    """
    過去X日間の各日におけるリスクを計算する関数
    各日付から遡って10日間(240時間)のウィンドウでリスク判定
    """
    # データをタイムスタンプでソート
    data_sorted = sorted(zip(temp_humidity_data, timestamps), key=lambda x: x[1])
    
    # 各日のリスクを計算
    daily_risks = []
    
    if not data_sorted:
        return pd.DataFrame()  # データがない場合は空のDataFrameを返す
    
    # 最新と最古の日付
    latest_date = max(ts.date() for _, ts in data_sorted)
    earliest_date = min(ts.date() for _, ts in data_sorted)
    
    # 表示する日付範囲
    date_range = [latest_date - pd.Timedelta(days=i) for i in range(days_to_show)]
    date_range = [d for d in date_range if d >= earliest_date]
    
    for target_date in sorted(date_range):
        # その日の終わり（次の日の0時）
        end_datetime = datetime.combine(target_date + pd.Timedelta(days=1), datetime.min.time())
        
        # 過去10日間（240時間）のウィンドウ
        window_start = end_datetime - pd.Timedelta(hours=240)
        
        # このウィンドウ内のデータを抽出
        risk_window = [(temp, humidity) for (temp, humidity), ts in data_sorted 
                      if window_start <= ts < end_datetime]
        
        # リスク時間を計算
        risk_hours = sum(
            1 for temp, humidity in risk_window
            if 15 <= float(temp) <= 25 and float(humidity) >= 94
        )
        
        # リスクレベルとカラーを決定
        if risk_hours == 0:
            risk_level, color = "極低", "gray"
        elif 0 < risk_hours <= 10:
            risk_level, color = "低", "blue"
        elif 10 < risk_hours <= 20:
            risk_level, color = "中", "green"
        elif 20 < risk_hours <= 30:
            risk_level, color = "高", "orange"
        else:
            risk_level, color = "極高", "red"
        
        daily_risks.append({
            'date': target_date,
            'risk_hours': risk_hours,
            'risk_level': risk_level,
            'color': color,
            'window_data_count': len(risk_window)
        })
    
    return pd.DataFrame(daily_risks)

# 棒グラフ描画関数
def plot_risk_bar_chart(risk_df):
    """過去の日別リスクを棒グラフで表示する関数"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 日付を古い順にソート
    risk_df = risk_df.sort_values('date')
    
    # 日付を文字列フォーマットに変換
    date_labels = [d.strftime('%m/%d') for d in risk_df['date']]
    
    # 棒グラフをプロット
    bars = ax.bar(
        date_labels, 
        risk_df['risk_hours'],
        color=risk_df['color'],
        width=0.7
    )
    
    # X軸ラベルの回転
    plt.xticks(rotation=45, ha='right')
    
    # Y軸のラベルと上限
    ax.set_ylabel('条件を満たす時間数')
    ax.set_ylim(0, max(40, risk_df['risk_hours'].max() * 1.1))  # 最大値よりも少し大きめに
    
    ax.set_title('過去14日間の灰色かび病リスク推移')
    
    # リスクレベルの凡例を追加
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='極低 (0時間)'),
        Patch(facecolor='blue', label='低 (1-10時間)'),
        Patch(facecolor='green', label='中 (11-20時間)'),
        Patch(facecolor='orange', label='高 (21-30時間)'),
        Patch(facecolor='red', label='極高 (31時間以上)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 各棒の上に時間数を表示
    for bar, hours in zip(bars, risk_df['risk_hours']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{hours}時間',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    return fig

# スピードメーターのみを残して他の可視化関数を削除
def plot_speedometer(percentage, color, risk_level, risk_hours):
    """
    車のスピードメーターのような半円型のゲージでリスクを表示
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # 背景の半円弧（薄いグレー）
    theta = np.linspace(180, 0, 100) * np.pi / 180.0
    r = 0.8
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # 背景の弧（グレー）
    ax.plot(x, y, color='lightgray', linewidth=20, solid_capstyle='round')
    
    # リスクに応じた弧の描画
    end_angle = 180 - (percentage * 180)
    theta_risk = np.linspace(180, end_angle, 100) * np.pi / 180.0
    x_risk = r * np.cos(theta_risk)
    y_risk = r * np.sin(theta_risk)
    ax.plot(x_risk, y_risk, color=color, linewidth=20, solid_capstyle='round')
    
    # 針の描画
    needle_angle = (180 - (percentage * 180)) * np.pi / 180.0
    needle_length = 0.9
    ax.plot([0, needle_length * np.cos(needle_angle)], 
            [0, needle_length * np.sin(needle_angle)], 
            color='black', linewidth=2)
    
    # 針の中心点
    circle = plt.Circle((0, 0), 0.05, color='darkgray')
    ax.add_patch(circle)
    
    # リスクレベルとリスク時間のテキスト表示
    ax.text(0, -0.4, f"リスクレベル: {risk_level}", ha='center', va='center', fontsize=15, fontweight='bold', color=color)
    ax.text(0, -0.55, f"{risk_hours}時間", ha='center', va='center', fontsize=14)
    
    # 目盛り表示
    labels = ["極低", "低", "中", "高", "極高"]
    angles = np.linspace(180, 0, 5) * np.pi / 180.0
    for label, angle in zip(labels, angles):
        x = 1.0 * np.cos(angle)
        y = 1.0 * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    
    # スケール表示（時間数）
    scales = ["0", "10", "20", "30", "40+"]
    for scale, angle in zip(scales, angles):
        x = 0.85 * np.cos(angle)
        y = 0.85 * np.sin(angle)
        ax.text(x, y, scale, ha='center', va='center', fontsize=8, color='gray')
    
    # グラフの設定
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.7, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

# 簡略化されたメイン可視化関数（スピードメーターのみ）
def display_risk_visualization(risk_level, risk_hours):
    """
    リスク表示関数（スピードメーターのみ対応）
    """
    # リスクレベルに対応する色を取得
    colors = {
        "極低": "gray",
        "低": "blue",
        "中": "green", 
        "高": "orange",
        "極高": "red"
    }
    color = colors[risk_level]
    
    # パーセンテージ計算（40時間を最大値として）
    percentage = min(risk_hours / 40, 1)
    
    # スピードメーターを表示
    return plot_speedometer(percentage, color, risk_level, risk_hours)

def detect_device_type(file_path):
    """
    ファイルの内容からデバイスタイプを推測する関数
    """
    # ファイル名からの推測
    file_name = os.path.basename(file_path)
    if 'HZ' in file_name:
        return 'HZ'
    elif 'OT' in file_name:
        return 'OT'
    elif 'PF2' in file_name:
        return 'PF2'
    elif 'PF' in file_name:
        return 'PF'
    elif 'SwitchBot' in file_name:
        return 'SB'

    elif 'HN' in file_name or 'ハウスナビ' in file_name:
        return 'HN'
    
    # ファイルの先頭部分を読み込んで内容から推測
    try:
        # まずShift-JISで試す
        with open(file_path, 'r', encoding='shift-jis') as f:
            header = ''.join([f.readline() for _ in range(5)])
        
        if '日付' in header and '温度' in header:
            return 'HZ'
        elif '年月日' in header and '気温' in header:
            return 'PF'
        elif '日付' in header and '室温' in header or '湿度' in header:
            return 'OT'
        elif '日時' in header and '潅水' in header and 'PF' in header:
            return 'PF2'

    except UnicodeDecodeError:
        # UTF-8で試す
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                header = ''.join([f.readline() for _ in range(5)])
            
            if 'Temperature_Celsius' in header and 'Relative_Humidity' in header:
                return 'SB'
            elif '日時' in header and '温度' in header and '相対湿度' in header:
                return 'HN'
        except UnicodeDecodeError:
            # UTF-8-SIGで試す
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    header = ''.join([f.readline() for _ in range(5)])
                
                if '日時' in header and '温度' in header:
                    return 'HN'
            except:
                pass
    
    # 推測できない場合
    print("警告: デバイスタイプを自動検出できません。PFとして処理を試みます。")
    return 'PF'

def resample_to_hourly(df):
    """データを1時間間隔にリサンプリング"""
    try:
        # インデックスがdatetimeかチェック
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("データフレームのインデックスがdatetime型ではありません")
        
        # 1時間間隔にリサンプリング（平均値を使用）
        df_resampled = df.resample('1H').mean()
        
        # 欠損値の補間
        df_resampled = df_resampled.interpolate(method='time')
        
        return df_resampled
    
    except Exception as e:
        print(f"リサンプリングエラー: {str(e)}")
        return df

def main():
    st.set_page_config(page_title="イチゴ灰色かび病リスク計算", layout="wide")
    st.title("イチゴ灰色かび病リスク計算")

    # メインコンテンツエリアを2カラムに分割
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("CSVファイルアップロード")
        
        # デバイスタイプを選択するオプションを追加
        device_type = st.selectbox(
            "センサーのタイプを選択してください",
            ["自動検出", "HZ (はかる蔵)", "PF (プロファインダー旧型)", "PF2 (プロファインダー新型)", 
             "SB (スイッチボット)", "OT (おんどとり)", "HN (ハウスナビ)"]
        )
        
        # マッピング辞書
        device_map = {
            "自動検出": None,
            "HZ (はかる蔵)": "HZ",
            "PF (プロファインダー旧型)": "PF",
            "PF2 (プロファインダー新型)": "PF2",
            "SB (スイッチボット)": "SB",
            "OT (おんどとり)": "OT",
            "HN (ハウスナビ)": "HN"
        }
        
        # ファイルアップローダー
        uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            # 選択されたデバイスタイプを取得
            selected_device = device_map[device_type]
            
            # プログレスバーで処理状況を表示
            with st.spinner('ファイルを読み込んでいます...'):
                # 拡張版関数を使用してデータを読み込む
                temperature, relative_humidity, timestamps = read_temperature_and_humidity_data(
                    uploaded_file, device_type=selected_device
                )

            if temperature is not None and relative_humidity is not None:
                # データのプレビューを表示
                preview_df = pd.DataFrame({
                    '日時': timestamps[:5],
                    '温度': temperature[:5],
                    '湿度': relative_humidity[:5]
                })
                st.write("データプレビュー（最初の5行）:")
                st.dataframe(preview_df)
                
                st.write(f"読み込まれたデータポイント数: {len(temperature)}")
                st.write(f"期間: {min(timestamps)} から {max(timestamps)}")
                
                # 温度・湿度データをリスト化
                temp_humidity_data = list(zip(temperature, relative_humidity))

                # 現在のリスク計算（最新10日間のデータで計算）
                current_risk_level, current_risk_hours, current_color = check_gray_mold_risk(temp_humidity_data, timestamps)
                
                # 時系列リスク計算
                time_series_risk_df = calculate_time_series_risk(temp_humidity_data, timestamps)
                
                # タブを使用して結果を表示
                tab1, tab2 = st.tabs(["現在リスク", "時系列リスク推移"])
                
                with tab1:
                    st.subheader("現在の灰色かび病リスク")
                    st.write("（最新日から10日間さかのぼったデータに基づく計算）")
                    st.markdown(f"<h3 style='color: {current_color};'>灰色かび病の発病リスク: {current_risk_level}</h3>", unsafe_allow_html=True)
                    st.write(f"条件を満たす時間数: {current_risk_hours}時間")
                    # 以下は同じ

                    recommendations = {
                        "極低": "本病の発生リスクは極めて低いので、他作業（収穫等）に集中してください。",
                        "低": "本病の発生リスクは低いですが、耕種的防除を実施するとなお良いでしょう。",
                        "中": "予防的な耕種的防除及び薬剤防除の実施がお勧めです。",
                        "高": "耕種的防除と薬剤防除の実施が必要です。",
                        "極高": "今すぐに耕種的防除と薬剤防除の実施が必要です。"
                    }

                    st.write(f"推奨対策: {recommendations[current_risk_level]}")
                    st.write(f"データ総数: {len(temp_humidity_data)}時間分")

                    # スピードメーターのみを表示
                    risk_viz_fig = display_risk_visualization(current_risk_level, current_risk_hours)
                    st.pyplot(risk_viz_fig)
                
                with tab2:
                    st.subheader("過去14日間の灰色かび病リスク推移")
                    
                    if not time_series_risk_df.empty:
                        # 表示オプション
                        viz_type = st.radio(
                            "表示タイプ:",
                            ["棒グラフ", "ヒートマップ"],
                            horizontal=True
                        )
                        
                        if viz_type == "棒グラフ":
                            # 棒グラフを表示
                            bar_fig = plot_risk_bar_chart(time_series_risk_df)
                            st.pyplot(bar_fig)
                        else:
                            # ヒートマップを表示
                            heat_fig = plot_risk_heatmap(time_series_risk_df)
                            st.pyplot(heat_fig)
                        
                        # 詳細データテーブルを表示（オプション）
                        with st.expander("詳細データを表示"):
                            display_df = time_series_risk_df[['date', 'risk_hours', 'risk_level']].copy()
                            
                            # .dt アクセサーを使う前に型チェックを追加
                            if pd.api.types.is_datetime64_any_dtype(display_df['date']):
                                # datetime型の場合はdt.strftimeを使用
                                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                            else:
                                # すでに文字列または別の型の場合は単純に文字列変換
                                display_df['date'] = display_df['date'].astype(str)
                            
                            display_df.columns = ['日付', 'リスク時間数', 'リスクレベル']
                            st.dataframe(display_df)
                    else:
                        st.warning("表示可能な時系列データがありません。より長期間のデータを含むCSVファイルをアップロードしてください。")
    
    # アプリの使い方や説明を右側のカラムに表示
    with col2:
        st.header("灰色かび病リスクについて")
        st.info("""
        ## リスク判定基準
        - 温度15-25℃かつ相対湿度94%以上の条件が一定時間以上継続すると発病リスクが高まります
        - 過去10日間（240時間）の中で条件を満たす時間数をカウントしています
        
        ## リスクレベル
        - **極低**（0時間）：発病リスクなし
        - **低**（1-10時間）：発病リスク低
        - **中**（11-20時間）：発病リスク中
        - **高**（21-30時間）：発病リスク高
        - **極高**（31時間以上）：発病リスク極高
        """)
        
        # 時系列データに関する補足説明
        st.markdown("### 時系列データについて")
        st.info("""
        「時系列リスク推移」タブでは、過去14日間の日ごとのリスク状況を確認できます。
        各日付で、その日を含む過去10日間のリスク時間数を表示しています。
        """)

# ヒートマップ表示関数（新しい時系列表示オプション）
def plot_risk_heatmap(risk_df):
    """リスクをカレンダー形式のヒートマップで表示する関数"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 日付を古い順にソートし、月-日形式に変換
    risk_df = risk_df.sort_values('date')
    date_labels = [d.strftime('%m/%d') for d in risk_df['date']]
    
    # カラーマップの設定（白→灰→青→緑→オレンジ→赤）
    cmap = plt.cm.get_cmap('RdYlGn_r')
    norm = plt.Normalize(0, 40)  # 0-40時間の範囲
    
    # リスク時間数を2D配列に変換（1行×日数列のマトリックス）
    risk_matrix = risk_df['risk_hours'].values.reshape(1, -1)
    
    # ヒートマップの描画
    heatmap = ax.pcolormesh(risk_matrix, cmap=cmap, norm=norm, edgecolors='white', linewidth=1)
    
    # カラーバーの追加
    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('条件を満たす時間数')
    
    # Y軸ラベルの設定（空にする）
    ax.set_yticks([])
    
    # X軸に日付ラベルを設定
    ax.set_xticks(np.arange(len(date_labels)) + 0.5)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')
    
    # リスクレベルマーカーの追加（各セルの上）
    for i, row in enumerate(risk_df.itertuples()):
        level_marker = {
            '極低': 'o',   # 円形
            '低': '^',     # 三角形
            '中': 's',     # 四角形
            '高': 'D',     # ダイヤモンド
            '極高': '*'    # 星形
        }
        marker = level_marker.get(row.risk_level, '')
        ax.text(i + 0.5, 0.5, marker, ha='center', va='center', fontsize=15)
    
    # タイトルと凡例
    ax.set_title('過去14日間の灰色かび病リスク（ヒートマップ）')
    
    # 凡例の追加
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='極低'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=10, label='低'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=10, label='中'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='k', markersize=10, label='高'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='極高')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()