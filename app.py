import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
from datetime import datetime
from typing import List, Tuple

def check_gray_mold_risk(temp_humidity_data: List[Tuple[float, float]]) -> Tuple[str, int, str]:
    window_size = 240  # 10日間 = 240時間
    max_risk_hours = 0
    
    for i in range(len(temp_humidity_data) - window_size + 1):
        window = temp_humidity_data[i:i+window_size]
        window_risk_hours = sum(
            1 for temp, humidity in window
            if 15 <= float(temp) <= 25 and float(humidity) >= 94
        )
        max_risk_hours = max(max_risk_hours, window_risk_hours)
    
    if max_risk_hours == 0:
        return "極低", max_risk_hours, "gray"
    elif 0 < max_risk_hours <= 10:
        return "低", max_risk_hours, "blue"
    elif 10 < max_risk_hours <= 20:
        return "中", max_risk_hours, "green"
    elif 20 < max_risk_hours <= 30:
        return "高", max_risk_hours, "yellow"
    else:
        return "極高", max_risk_hours, "red"

# CSVファイルから気温データと相対湿度データを読み込む関数
def read_temperature_and_humidity_data(csv_file):
    try:
        # CSVファイルを読み込む（ファイルオブジェクトとして渡される）
        data = pd.read_csv(csv_file, parse_dates=['Timestamp'])
        
        # 読み込んだカラム名を表示（デバッグ用）
        st.write("アップロードされたCSVのカラム:", data.columns.tolist())
        
        # 必須のカラムが存在するかを確認
        required_columns = ['Timestamp', 'Temperature', 'Relative Humidity']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("CSVファイルに 'Timestamp', 'Temperature' または 'Relative Humidity' 列がありません。")
        
        # 'Timestamp'をインデックスとして設定
        data.set_index('Timestamp', inplace=True)
        
        # 'Temperature' と 'Relative Humidity' を数値型に変換
        data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')
        data['Relative Humidity'] = pd.to_numeric(data['Relative Humidity'], errors='coerce')
        
        # タイムスタンプを時間単位で丸める
        data.index = data.index.floor('H')
        
        # 重複するインデックスのデータを平均化
        data = data.groupby(data.index).mean()
        
        # データを1時間ごとにリサンプリングし、欠損値を補間
        data = data.resample('1H').interpolate()

        # デバッグ用（本番環境ではコメントアウト）
        # st.write("処理後のデータ（最初の5行）:")
        # st.write(data.head())

        # Temperature と Relative Humidity 列とタイムスタンプの日付部分を返す
        return data['Temperature'].tolist(), data['Relative Humidity'].tolist(), data.index
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None

def main():
    st.set_page_config(page_title="イチゴの灰色かび病リスク計算 & 天気予報", layout="wide")

    st.title("イチゴの灰色かび病リスク計算 & 天気予報")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("CSVファイルアップロード")
        uploaded_file = st.file_uploader("CSVファイルを選択してください（Timestamp, Temperature, Relative Humidity の形式）", type="csv")

        if uploaded_file is not None:
            # CSVデータを読み込む
            temperature, relative_humidity, timestamps = read_temperature_and_humidity_data(uploaded_file)

            if temperature is not None and relative_humidity is not None:
                temp_humidity_data = list(zip(temperature, relative_humidity))

                # データのプレビュー
                st.subheader("アップロードされたデータのプレビュー")
                preview_data = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Temperature': temperature,
                    'Relative Humidity': relative_humidity
                })
                st.dataframe(preview_data.head())  # 最初の5行を表示

                # リスク計算
                risk_level, risk_hours, color = check_gray_mold_risk(temp_humidity_data)

                # 結果の表示
                st.subheader("解析結果")
                st.markdown(f"<h3 style='color: {color};'>灰色かび病の発病リスク: {risk_level}</h3>", unsafe_allow_html=True)
                st.write(f"条件を満たす時間数: {risk_hours}時間")

                recommendations = {
                    "極低": "本病の発生リスクは極めて低いので、他作業（収穫等）に集中してください。",
                    "低": "本病の発生リスクは低いですが、耕種的防除を実施するとなお良いでしょう。",
                    "中": "予防的な耕種的防除及び薬剤防除の実施がお勧めです。",
                    "高": "耕種的防除と薬剤防除の実施が必要です。",
                    "極高": "今すぐに耕種的防除と薬剤防除の実施が必要です。"
                }

                st.write(f"推奨対策: {recommendations[risk_level]}")
                st.write(f"データ総数: {len(temp_humidity_data)}時間分")

                st.progress(min(risk_hours / 40, 1))  # 40 hoursが最大としてプログレスバー

    with col2:
        st.header("天気予報")
        st.write("天気予報機能は現在無効化されています。")

if __name__ == "__main__":
    main()
