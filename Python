import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="QUANT AI DASHBOARD", layout="wide")

# --- CSS ตกแต่งให้สวยงาม (Dark Mode) ---
st.markdown("""
<style>
    .main { background-color: #0E1117; color: white; }
    .stMetric { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; border: 1px solid rgba(255,255,255,0.1); }
    h1 { color: #00D4FF !important; font-family: 'Arial'; }
</style>
""", unsafe_allow_html=True)

# --- ส่วนหัว ---
st.title("🤖 QUANT AI: Stock Trend Predictor")
st.write("วิเคราะห์แนวโน้มหุ้นล่วงหน้า 1 สัปดาห์ด้วย Machine Learning (XGBoost)")

# --- แถบเมนูด้านข้าง ---
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("ชื่อหุ้น (เช่น PTT.BK, TSLA, BTC-USD)", "TSLA").upper()
    btn = st.button("RUN ANALYSIS", use_container_width=True)

# --- ฟังก์ชันคำนวณ ---
def get_data(symbol):
    try:
        df = yf.download(symbol, period="2y", interval="1d")
        if len(df) < 50: return None
        
        # สร้าง Indicators พื้นฐาน
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Target: 1 = อีก 5 วันราคาขึ้น, 0 = ลง
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        return df.dropna()
    except:
        return None

# --- ส่วนประมวลผลเมื่อกดปุ่ม ---
if btn:
    with st.spinner('AI กำลังวิเคราะห์ข้อมูล...'):
        df = get_data(symbol)
        
        if df is not None:
            # เตรียม AI Model
            features = ['Close', 'SMA20', 'SMA50', 'Daily_Return']
            X = df[features]
            y = df['Target']
            
            model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
            model.fit(X[:-5], y[:-5]) # ไม่ใช้ 5 วันล่าสุดมาสอน เพื่อกันข้อมูลรั่ว
            
            # ทำนายผล
            last_row = X.tail(1)
            prob = model.predict_proba(last_row)[0] # [โอกาสลง, โอกาสขึ้น]
            
            # --- แสดงผลลัพธ์ ---
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("AI Prediction")
                if prob[1] > 0.5:
                    st.success(f"🔼 แนวโน้ม: ขาขึ้น (UP)")
                    st.write(f"ความน่าจะเป็นที่จะขึ้น: **{prob[1]*100:.2f}%**")
                else:
                    st.error(f"🔽 แนวโน้ม: ขาลง (DOWN)")
                    st.write(f"ความน่าจะเป็นที่จะลง: **{prob[0]*100:.2f}%**")
                
                st.metric("ราคาล่าสุด", f"{df['Close'].iloc[-1]:.2f}")

            with col2:
                # กราฟ Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00D4FF"}}
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)

            # --- กราฟแท่งเทียน ---
            st.subheader("Price Structure")
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='yellow', width=1), name='SMA 20'))
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("ไม่พบข้อมูลหุ้นตัวนี้ โปรดตรวจสอบชื่อหุ้นอีกครั้ง")
