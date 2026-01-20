import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import koreanize_matplotlib

# Koreanize-matplotlib 적용
try:
    import koreanize_matplotlib
    koreanize_matplotlib.set_rc()
except:
    # Fallback: 기본 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="감귤 당도 예측기", layout="wide")
st.title("🍊 감귤 당도 예측기")

# 모델 로드
model = joblib.load("brix_model.joblib")

# 세션 상태 초기화 (히스토리 저장)
if 'history' not in st.session_state:
    st.session_state.history = []

# 메인 레이아웃
st.subheader("🎯 파라미터별 영향도 분석")

# 좌우 2열 레이아웃
left_col, right_col = st.columns([1, 2])

with left_col:
    st.write("### 파라미터 슬라이더")
    
    # Minimum Temperature Slider | 최저기온 (먼저 설정)
    st.write("**최저기온**")
    min_temp_slider = st.slider(
        "Min Temp",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
        label_visibility="collapsed",
        key="min_temp_slider"
    )
    st.caption(f"값: {min_temp_slider:.2f}°C")
    
    # Average Temperature Slider | 평균기온 (최저기온 >= 평균기온 <= 최고기온)
    st.write("**평균기온**")
    avg_temp_slider = st.slider(
        "Avg Temp",
        min_value=min_temp_slider,
        max_value=35.0,
        value=min(20.0, 35.0) if min_temp_slider <= 20.0 else min_temp_slider,
        step=0.5,
        label_visibility="collapsed",
        key="avg_temp_slider"
    )
    st.caption(f"값: {avg_temp_slider:.2f}°C")
    
    # Maximum Temperature Slider | 최고기온 (평균기온 <= 최고기온)
    st.write("**최고기온**")
    max_temp_slider = st.slider(
        "Max Temp",
        min_value=avg_temp_slider,
        max_value=40.0,
        value=max(25.0, avg_temp_slider),
        step=0.5,
        label_visibility="collapsed",
        key="max_temp_slider"
    )
    st.caption(f"값: {max_temp_slider:.2f}°C")
    
    # Sunlight Hours Slider | 가조시간
    st.write("**가조시간**")
    sunlight_slider = st.slider(
        "Sunlight",
        min_value=9.9,
        max_value=14.4,
        value=12.0,
        step=0.1,
        label_visibility="collapsed",
        key="sunlight_slider"
    )
    st.caption(f"값: {sunlight_slider:.2f}시간")

with right_col:
    # 현재값으로 예측 | Predict with Current Values
    current_input = np.array([[avg_temp_slider, max_temp_slider, min_temp_slider, sunlight_slider]])
    current_prediction = model.predict(current_input)[0]
    
    # 예측 결과 저장 (히스토리)
    st.session_state.history.append({
        "평균기온": avg_temp_slider,
        "최고기온": max_temp_slider,
        "최저기온": min_temp_slider,
        "가조시간": sunlight_slider,
        "예측당도": current_prediction
    })
    
    # 예측 결과 표시 | Display Prediction Result
    st.write("### 예측 결과")
    
    # 예측값을 크게 표시
    col_main = st.columns([1])
    with col_main[0]:
        st.metric("감귤 당도 (Brix)", f"{current_prediction:.2f}°Bx", delta=None)
    
    st.divider()
    
    # 현재 입력값 요약 | Current Input Summary
    st.write("### 입력 파라미터")
    col_input = st.columns(4)
    with col_input[0]:
        st.info(f"평균기온\n{avg_temp_slider:.1f}°C", icon="🌡️")
    with col_input[1]:
        st.info(f"최고기온\n{max_temp_slider:.1f}°C", icon="🔥")
    with col_input[2]:
        st.info(f"최저기온\n{min_temp_slider:.1f}°C", icon="❄️")
    with col_input[3]:
        st.info(f"가조시간\n{sunlight_slider:.1f}시간", icon="☀️")

# 히스토리 표시 (메인 영역 아래)
st.divider()
st.subheader("📊 예측 히스토리")

if st.session_state.history:
    # 히스토리 데이터프레임
    history_df = pd.DataFrame(st.session_state.history)
    
    # 중복 제거 (최신값만 유지) - 슬라이더가 변할 때마다 추가되는 것을 방지
    history_df = history_df.drop_duplicates(subset=['평균기온', '최고기온', '최저기온', '가조시간'], keep='last')
    
    # 히스토리 테이블
    display_df = history_df.copy()
    display_df['예측당도'] = display_df['예측당도'].apply(lambda x: f"{x:.2f}°Bx")
    display_df['평균기온'] = display_df['평균기온'].apply(lambda x: f"{x:.2f}°C")
    display_df['최고기온'] = display_df['최고기온'].apply(lambda x: f"{x:.2f}°C")
    display_df['최저기온'] = display_df['최저기온'].apply(lambda x: f"{x:.2f}°C")
    display_df['가조시간'] = display_df['가조시간'].apply(lambda x: f"{x:.2f}시간")
    
    st.dataframe(display_df, use_container_width=True)
    
    # 히스토리 초기화 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ 히스토리 초기화"):
            st.session_state.history = []
            st.rerun()
else:
    st.info("아직 예측 기록이 없습니다.")
