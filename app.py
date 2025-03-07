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
from matplotlib.patches import Patch

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
        return "極低", risk_hours, "blue"
    elif 0 < risk_hours <= 10:
        return "低", risk_hours, "blue"
    elif 10 < risk_hours <= 20:
        return "中", risk_hours, "green"
    elif 20 < risk_hours <= 30:
        return "高", risk_hours, "orange"
    else:
        return "極高", risk_hours, "red"

# CSVファイルから気温データと相対湿度データを読み込む関数
def read_temperature_and_humidity_data(file_obj, device_type=None, days_to_keep=30):
    """
    様々な形式のセンサーデータファイルから温度と湿度のデータを読み込む関数
    
    Parameters:
    ----------
    file_obj : UploadedFile または str
        Streamlitでアップロードされたファイルオブジェクトまたはファイルパス
    device_type : str, optional
        デバイスタイプ ('HZ', 'PF', 'PF2', 'SB', 'OT', 'HN', None)
    days_to_keep : int, optional
        保持する日数（最新のN日分）。デフォルトは30日。
    """
    
    # メイン処理開始
    temp_path = None
    try:
        # アップロードされたファイルを一時ファイルに保存
        import tempfile
        if hasattr(file_obj, 'read'):  # UploadedFileオブジェクトの場合
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(file_obj.getbuffer())
                temp_path = tmp_file.name
        else:  # 文字列（ファイルパス）の場合
            temp_path = file_obj
        
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
            
            # ファイルを読み込んでヘッダーを確認
            df, encoding = try_multiple_encodings(temp_path)
            if df is not None:
                cols = df.columns.tolist()
                
                # 各デバイスタイプの特徴と照合
                for dev, config in device_configs.items():
                    # 温度列が存在するか確認
                    for col_name in config['temp_cols']:
                        if col_name in cols:
                            device_type = dev
                            break
                    if device_type:
                        break
            
            # 自動検出できない場合はデフォルトとしてSBを使用
            if device_type is None:
                device_type = 'SB'
        
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
     
            # 時刻列からアスタリスクなどを削除して結合
            date_col = config.get('date_time_cols', {}).get('date_col', '日付')
            time_col = config.get('date_time_cols', {}).get('time_col', '時刻')
            
            df['datetime'] = combine_date_time(df, date_col, time_col)
            
            # NaT値をチェック
            nat_count = df['datetime'].isna().sum()
            if nat_count > 0:
                df = df.dropna(subset=['datetime'])
            
            df = df.set_index('datetime')
            timestamp_found = True
            
        if device_type == 'SB':
            try:
                # SwitchBotファイルはUTF-8で強制的に読み直し
                df = pd.read_csv(temp_path, encoding='utf-8')
            except Exception as e:
                st.warning(f"UTF-8での読み込みに失敗しました: {str(e)}")
        
        # 一般的なタイムスタンプ列の処理
        if not timestamp_found:
            for ts_col in config['timestamp_cols']:
                if ts_col in df.columns:
                    
                    # タイムスタンプをdatetime型に変換
                    df['datetime'] = pd.to_datetime(df[ts_col], errors='coerce')
                    
                    # NaT値をチェック
                    nat_count = df['datetime'].isna().sum()
                    if nat_count > 0:
                        df = df.dropna(subset=['datetime'])
                    
                    df = df.set_index('datetime')
                    timestamp_found = True
                    break
        
        if not timestamp_found:
            # 日付と時刻が別々の列の場合（一般的なケース）
            if '日付' in df.columns and '時刻' in df.columns:
                df['datetime'] = combine_date_time(df, '日付', '時刻')
                
                # NaT値をチェック
                nat_count = df['datetime'].isna().sum()
                if nat_count > 0:
                    df = df.dropna(subset=['datetime'])
                
                df = df.set_index('datetime')
                timestamp_found = True
        
        if not timestamp_found:
            st.error("タイムスタンプ列が見つかりません")
            return None, None, None
        
        if timestamp_found:
            if len(df) > 0:  # 空のDataFrameでないことを確認
                # 最新の日付を特定
                latest_date = df.index.max()
                # 期間を計算
                cutoff_date = latest_date - pd.Timedelta(days=days_to_keep)
                # 期間内のデータのみ保持
                df_filtered = df[df.index >= cutoff_date]

                # フィルタ後にデータが残っていることを確認
                if not df_filtered.empty:
                    # 元のデータフレームを更新
                    df = df_filtered
                else:
                    st.warning(f"最新の{days_to_keep}日間にデータがありません。すべてのデータを保持します。")
            
                
        # 温度と湿度の列を特定
        temp_col = None
        for col in config['temp_cols']:
            if col in df.columns:
                temp_col = col
                break
        
        humid_col = None
        for col in config['humid_cols']:
            if col in df.columns:
                humid_col = col
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
            except Exception as e:
                st.warning(f"一時ファイル削除エラー: {str(e)}")

def try_multiple_encodings(file_path):
    """複数のエンコーディングを試してCSVファイルを読み込む関数"""
    encodings = ['shift-jis', 'utf-8', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df, encoding
        except Exception as e:
            continue

    return None, None

def combine_date_time(df, date_col='日付', time_col='時刻'):
    """日付列と時刻列を結合してdatetime型の列を作成する関数"""
    try:
        st.info("日付と時刻の列を変換しています")
        
        # 時刻列のクリーニング（*を削除）
        if time_col in df.columns:
            df[time_col] = df[time_col].astype(str).str.replace('*', '', regex=False)
            df[time_col] = df[time_col].str.strip()

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

        # 非数値列を除外
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("数値型の列が見つかりません")
            return df
        
        # 数値列のみを使用
        numeric_df = df[numeric_cols].copy()
        
        # 1時間間隔にリサンプリング
        numeric_df.index = numeric_df.index.floor('H')
        
        # デバッグ情報
        sample_idx = numeric_df.index[:5].tolist() if len(numeric_df) > 0 else []
        
        # この時点でインデックスは時間単位に切り捨てられています
        # グループ化とリサンプリング
        df_hourly = numeric_df.groupby(numeric_df.index).mean()
        
        # 欠損値の補間 (時系列データに適した method='time' を使用)
        df_resampled = df_hourly.resample('1H').interpolate(method='time')
        
        return df_resampled
    
    except Exception as e:
        st.error(f"リサンプリングエラー: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return df

# 時系列リスク計算関数
def calculate_time_series_risk(temp_humidity_data, timestamps, days_to_show=30):
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

# スピードメーターのみを残して他の可視化関数を削除
def plot_speedometer(percentage, color, risk_level, risk_hours):
    """
    車のスピードメーターのような半円型のゲージでリスクを表示
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
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
    
    # リスクレベルを装飾ボックスで表示
    text_box_props = dict(
        boxstyle='round,pad=0.5',
        facecolor=color,
        alpha=0.8,
        edgecolor='none'
    )
    
    # リスクレベルテキスト - 白い文字で目立たせる
    ax.text(0, -0.4, f"リスクレベル: {risk_level}", ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white',
            bbox=text_box_props)
    
    # リスク時間のテキスト
    ax.text(0, -0.6, f"{risk_hours}時間", ha='center', va='center', 
            fontsize=18, color='#303030')
    
    # 目盛り表示
    labels = ["極低", "低", "中", "高", "極高"]
    angles = np.linspace(180, 0, 5) * np.pi / 180.0
    for label, angle in zip(labels, angles):
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', fontsize=18)
    
    # スケール表示（時間数）
    scales = ["0", "10", "20", "30", "40+"]
    for scale, angle in zip(scales, angles):
        x = 0.81 * np.cos(angle)
        y = 0.81 * np.sin(angle)
        ax.text(x, y, scale, ha='center', va='center', fontsize=16, color='black')
    
    # グラフの設定
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

# 簡略化されたメイン可視化関数（スピードメーターのみ）
def display_risk_visualization(risk_level, risk_hours):
    """
    リスク表示関数（スピードメーターのみ対応）
    """
    # リスクレベルに対応する色を更新
    colors = {
        "極低": "#0000cc",  # 青
        "低": "#0000cc",    # 青
        "中": "#00cc00",    # 緑
        "高": "#ff8000",    # オレンジ
        "極高": "#cc0000"   # 赤
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
    return 'PF'

# 棒グラフ描画関数
def plot_risk_bar_chart(risk_df):
    """過去の日別リスクを棒グラフで表示する関数（スマホ対応版、日付を上部に配置）"""
    # スマホに適したサイズ比率
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1]})
    ax = axes[0]  # メインの棒グラフ用
    ax_legend = axes[1]  # 凡例用
    
    # 日付を古い順にソート
    risk_df = risk_df.sort_values('date')
    
    # 日付を文字列フォーマットに変換
    date_labels = [d.strftime('%m/%d') for d in risk_df['date']]
    
    # リスクレベルに対応する色を定義
    risk_colors = {
        "極低": "#0000cc",  # 青
        "低": "#0000cc",    # 青
        "中": "#00cc00",    # 緑
        "高": "#ff8000",    # オレンジ
        "極高": "#cc0000"   # 赤
    }
    
    # データに基づく色のリスト作成
    bar_colors = [risk_colors[level] for level in risk_df['risk_level']]
    
    # インデックスベースのX軸位置を作成
    x_positions = np.arange(len(risk_df))
    
    # 棒グラフをプロット
    bars = ax.bar(
        x_positions,
        risk_df['risk_hours'],
        color=bar_colors,
        width=0.7
    )
    
    # X軸のティックとラベルを非表示にする
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Y軸の設定
    ax.set_ylabel('条件を満たす時間数', fontsize=16)
    
    # Y軸の上限を固定して、日付表示用のスペースを確保
    # 基本的に40時間を上限とし、それより大きい値がある場合はそれに少し余裕を持たせる
    max_data = risk_df['risk_hours'].max()
    y_max = max(40, max_data * 1.1)
    
    # 日付表示のために十分なスペースを確保
    ax.set_ylim(0, y_max * 1.2)  # 上限を40％増やして日付用のスペースを確保
    
    # 最初と最後の日付を取得して期間をタイトルに表示
    first_date = date_labels[0]
    last_date = date_labels[-1]
    ax.set_title(f'過去10日間の灰色かび病リスク推移({first_date}～{last_date})', fontsize=18)
    
    # 各棒の上に時間数を表示
    for i, (bar, hours) in enumerate(zip(bars, risk_df['risk_hours'])):
        height = bar.get_height()
        # 時間数を棒の上に表示
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{hours}時間',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 日付を40時間ラインの上に表示（棒グラフから離して）
    date_y_position = 43  # 40時間ラインより少し上
    for i, (x, date) in enumerate(zip(x_positions, date_labels)):
        # すべての日付を表示するか、偶数番目と最後の日付のみ表示するかを選択可能
        if i % 2 == 0 or i == len(date_labels) - 1:  # 偶数または最後のみ表示
            ax.text(x, date_y_position, date, 
                    ha='center', va='bottom', fontsize=14)
    
    # グリッド線を追加して読みやすくする
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # リスクレベルの凡例をスマホに最適化
    ax_legend.axis('off')  # 軸を非表示
    
    # リスクレベル区分を定義
    risk_levels = ["極低 (0時間)", "低 (1-10時間)", "中 (11-20時間)", "高 (21-30時間)", "極高 (31時間以上)"]
    risk_colors_list = [risk_colors["極低"], risk_colors["低"], risk_colors["中"], 
                         risk_colors["高"], risk_colors["極高"]]
    
    # 凡例を横に並べて表示（スマホに最適化）
    
    box_width = 0.15
    gap = 0.05
    total_width = (box_width + gap) * len(risk_levels) - gap
    start_x = (1 - total_width) / 2
    
    for i, (level, color) in enumerate(zip(risk_levels, risk_colors_list)):
        x = start_x + i * (box_width + gap)
        # 色付きのボックスを描画
        rect = patches.Rectangle((x, 0.4), box_width, 0.4, facecolor=color)
        ax_legend.add_patch(rect)
        # テキストラベルを追加
        ax_legend.text(x + box_width/2, 0.15, level, ha='center', va='center', fontsize=16)
    
    # 凡例のタイトル
    ax_legend.text(0.5, 0.9, "リスクレベル区分", ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # サブプロット間の隙間を調整
    return fig

def plot_risk_heatmap(risk_df):
    """リスクをカレンダー形式のヒートマップで表示する関数（スマホ対応版）"""
    # スマホ向けにサイズとレイアウトを調整
    fig, axes = plt.subplots(2, 1, figsize=(8, 1.8), gridspec_kw={'height_ratios': [1, 0.5]})
    ax = axes[0]  # メインのヒートマップ用
    ax_legend = axes[1]  # 凡例用
    
    # 日付を古い順にソートし、月-日形式に変換
    risk_df = risk_df.sort_values('date')
    date_labels = [d.strftime('%m/%d') for d in risk_df['date']]
    
    # カラーマップの設定（青→緑→オレンジ→赤）
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.0, 0.0, 0.8),  # 青（極低・低リスク）
              (0.0, 0.8, 0.0),  # 緑（中リスク）
              (1.0, 0.5, 0.0),  # オレンジ（高リスク）
              (0.8, 0.0, 0.0)]  # 赤（極高リスク）
    
    risk_cmap = LinearSegmentedColormap.from_list('risk_cmap', colors)
    norm = plt.Normalize(0, 40)  # 0-40時間の範囲
    
    # リスク時間数を2D配列に変換
    risk_matrix = risk_df['risk_hours'].values.reshape(1, -1)
    
    # ヒートマップの描画
    heatmap = ax.pcolormesh(risk_matrix, cmap=risk_cmap, norm=norm, edgecolors='white', linewidth=1)
    
    # Y軸ラベルの設定（空にする）
    ax.set_yticks([])
    
    # X軸の設定
    total_days = len(date_labels)
    xtick_positions = np.arange(total_days) + 0.5
    
    # 最初と最後の日付のみラベル表示
    ax.set_xticks([xtick_positions[0], xtick_positions[-1]])
    ax.set_xticklabels([date_labels[0], date_labels[-1]], fontsize=12, fontweight='bold')
    
    # 日付の区切り線（薄く）
    for x in xtick_positions:
        ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.3, alpha=0.2)
    
    
    # タイトル
    ax.set_title(f"過去30日間の灰色かび病リスク推移 ({date_labels[0]}～{date_labels[-1]})", fontsize=18)
    
    # 凡例を描画するサブプロットの設定
    ax_legend.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    return fig

def main():
    st.set_page_config(page_title="イチゴ灰色かび病リスク計算", layout="wide")
    st.title("イチゴ灰色かび病リスク計算")


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
    uploaded_file = st.file_uploader("CSVファイルを選択してください。", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # 選択されたデバイスタイプを取得
        selected_device = device_map[device_type]
        
        # プログレスバーで処理状況を表示
        with st.spinner('ファイルを読み込んでいます...'):
            # 拡張版関数を使用してデータを読み込む（30日分のみ）
            temperature, relative_humidity, timestamps = read_temperature_and_humidity_data(
                uploaded_file, device_type=selected_device, days_to_keep=30
            )

        if temperature is not None and relative_humidity is not None:
            # データのプレビューを表示
            
            # 温度・湿度データをリスト化
            temp_humidity_data = list(zip(temperature, relative_humidity))

            # 現在のリスク計算（最新の10日間）
            current_risk_level, current_risk_hours, current_color = check_gray_mold_risk(temp_humidity_data, timestamps)
            
            # 時系列リスク計算
            time_series_risk_df = calculate_time_series_risk(temp_humidity_data, timestamps, days_to_show=28)

            # 1. メインの現在リスク表示（スピードメーター）
            st.header("現在の灰色かび病リスク")
            st.markdown("<p style='color:gray; font-size:14px;'>（最新日から10日間さかのぼったデータに基づく計算）</p>", unsafe_allow_html=True)   
            
            # リスクレベル表示
            st.markdown(f"<h3 style='color: {current_color};'>灰色かび病の発病リスク: {current_risk_level}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px; font-weight:bold;'>条件を満たす時間数: {current_risk_hours}時間</p>", unsafe_allow_html=True)

            # 推奨対策表示
            recommendations = {
                "極低": "本病の発生リスクは極めて低いので、他作業（収穫等）に集中してください。",
                "低": "本病の発生リスクは低いですが、耕種的防除を実施するとなお良いでしょう。",
                "中": "予防的な耕種的防除及び薬剤防除の実施がお勧めです。",
                "高": "耕種的防除と薬剤防除の実施が必要です。",
                "極高": "今すぐに耕種的防除と薬剤防除の実施が必要です。"
            }
            st.markdown(f"<p style='font-size:16px;'><span style='font-weight:bold;'>推奨対策:</span> {recommendations[current_risk_level]}</p>", unsafe_allow_html=True)
            # スピードメーターを表示
            risk_viz_fig = display_risk_visualization(current_risk_level, current_risk_hours)
            st.pyplot(risk_viz_fig)
            
            # 区切り線を表示
            st.markdown("---")
            
            # 2. 時系列リスクの棒グラフ表示
            if not time_series_risk_df.empty:
                
                # 棒グラフ用に直近10日間のデータを抽出
                recent_10_days = time_series_risk_df.sort_values('date', ascending=False).head(10).sort_values('date')
                bar_fig = plot_risk_bar_chart(recent_10_days)
                st.pyplot(bar_fig)
                
                # 区切り線を表示
                st.markdown("---")
                
                # 3. 時系列リスクのヒートマップ表示（30日間）
                heat_fig = plot_risk_heatmap(time_series_risk_df)
                st.pyplot(heat_fig)
                
                # 詳細データテーブルを表示
                with st.expander("詳細データを表示"):
                    display_df = time_series_risk_df[['date', 'risk_hours', 'risk_level']].copy()
                    
                    # .dt アクセサーを使う前に型チェックを追加
                    if pd.api.types.is_datetime64_any_dtype(display_df['date']):
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    else:
                        display_df['date'] = display_df['date'].astype(str)
                    
                    display_df.columns = ['日付', 'リスク時間数', 'リスクレベル']
                    st.dataframe(display_df)
            else:
                st.warning("表示可能な時系列データがありません。より長期間のデータを含むCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()