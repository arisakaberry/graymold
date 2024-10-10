import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
from datetime import datetime
from typing import List, Tuple

# 灰色かび病リスクチェック関数
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
        return "高", max_risk_hours, "orange"
    else:
        return "極高", max_risk_hours, "red"

# CSVファイルから気温データと相対湿度データを読み込む関数
def read_temperature_and_humidity_data(csv_file):
    try:
        data = pd.read_csv(csv_file, parse_dates=['Timestamp'])
        #st.write("アップロードされたCSVのカラム:", data.columns.tolist())
        required_columns = ['Timestamp', 'Temperature', 'Relative Humidity']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("CSVファイルに 'Timestamp', 'Temperature' または 'Relative Humidity' 列がありません。")
        
        data.set_index('Timestamp', inplace=True)
        data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')
        data['Relative Humidity'] = pd.to_numeric(data['Relative Humidity'], errors='coerce')
        data.index = data.index.floor('H')
        data = data.groupby(data.index).mean()
        data = data.resample('1H').interpolate()
        return data['Temperature'].tolist(), data['Relative Humidity'].tolist(), data.index
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None, None

# バッテリーのような円形表示を行う関数
def plot_gauge(percentage, color):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # 外側の円
    circle = patches.Circle((0, 0), 1.2, fill=True, color='lightgray', alpha=0.5)
    ax.add_patch(circle)
    
    # 内側の進捗部分
    ax.pie([percentage, 1 - percentage], colors=[color, 'white'], radius=1.1, startangle=90, wedgeprops=dict(width=0.3, edgecolor='w'))
    
    # テキストで進捗を表示
    ax.text(0, 0, f'{int(percentage * 100)}%', ha='center', va='center', fontsize=20)
    plt.axis('off')
    return fig

def main():
    st.set_page_config(page_title="イチゴ灰色かび病リスク計算", layout="wide")
    st.title("イチゴ灰色かび病リスク計算")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("CSVファイルアップロード")
        uploaded_file = st.file_uploader("CSVファイルを選択してください（Timestamp, Temperature, Relative Humidity の形式）", type="csv")

        if uploaded_file is not None:
            temperature, relative_humidity, timestamps = read_temperature_and_humidity_data(uploaded_file)

            if temperature is not None and relative_humidity is not None:
                temp_humidity_data = list(zip(temperature, relative_humidity))

                #st.subheader("アップロードされたデータのプレビュー")
                #preview_data = pd.DataFrame({
                #    'Timestamp': timestamps,
                #    'Temperature': temperature,
                #    'Relative Humidity': relative_humidity
                #})
                #st.dataframe(preview_data.head())

                risk_level, risk_hours, color = check_gray_mold_risk(temp_humidity_data)

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

                # バッテリーのような表示
                percentage = min(risk_hours / 40, 1)  # 最大40時間を100%とする
                fig = plot_gauge(percentage, color)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
