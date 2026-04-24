import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .pass-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .fail-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('student-por.csv', sep=',')
df.columns = df.columns.str.strip()
df['pass'] = (df['G3'] >= 10).astype(int)
df['result'] = df['pass'].map({1: 'Pass', 0: 'Fail'})

st.markdown('<div class="main-header">🎓 Student Performance Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict student results using Machine Learning</div>',
            unsafe_allow_html=True)

page = st.sidebar.selectbox("Navigate",
    ["Home", "Predict", "Data Analysis", "About"])

if page == "Home":
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        pass_rate = f"{df['pass'].mean():.0%}"
        st.metric("Pass Rate", pass_rate)
    with col3:
        avg_grade = f"{df['G3'].mean():.1f}"
        st.metric("Average Grade", avg_grade)
    with col4:
        st.metric("Model Accuracy", "86.15%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grade Distribution")
        fig = px.histogram(df, x='G3', nbins=20,
                          color_discrete_sequence=['#4A90E2'],
                          title="Final Grade (G3) Distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pass vs Fail")
        counts = df['result'].value_counts()
        fig = px.pie(values=counts.values,
                    names=counts.index,
                    color_discrete_sequence=['#28a745','#dc3545'],
                    title="Pass/Fail Ratio")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predict":
    st.markdown("---")
    st.subheader("Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Academic Factors")
        studytime = st.slider("Study hours per week", 1, 4, 2,
                             help="1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs")
        failures = st.slider("Number of past failures", 0, 3, 0)
        absences = st.slider("Number of absences", 0, 30, 5)

    with col2:
        st.markdown("#### Personal Factors")
        goout = st.slider("Goes out with friends (1=low, 5=high)", 1, 5, 3)
        health = st.slider("Health status (1=bad, 5=good)", 1, 5, 3)
        Medu = st.selectbox("Mother's education",
                           [0,1,2,3,4],
                           format_func=lambda x:
                           ["None","Primary","5th-9th grade",
                            "Secondary","Higher"][x])
        Fedu = st.selectbox("Father's education",
                           [0,1,2,3,4],
                           format_func=lambda x:
                           ["None","Primary","5th-9th grade",
                            "Secondary","Higher"][x])

    st.markdown("---")
    if st.button("Predict Performance", use_container_width=True):
        inp = [[studytime, failures, absences, Medu, Fedu, goout, health]]
        result = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1]

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if result == 1:
                st.markdown(f"""
                <div class="pass-box">
                    <h1 style="color:#28a745">PASS</h1>
                    <h3>Confidence: {prob:.0%}</h3>
                    <p>This student is likely to pass!</p>
                </div>""", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="fail-box">
                    <h1 style="color:#dc3545">AT RISK</h1>
                    <h3>Pass Chance: {prob:.0%}</h3>
                    <p>This student needs extra support!</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("### Tips to Improve")
        if failures > 0:
            st.warning("Reduce past failures by seeking tutoring help")
        if absences > 10:
            st.warning("High absences detected — regular attendance is key!")
        if studytime < 2:
            st.warning("Increase study time to at least 2-5 hours per week")
        if goout > 3:
            st.info("Balance social activities with study time")
        if health < 3:
            st.info("Focus on health — it directly impacts performance")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Pass Probability %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4A90E2"},
                'steps': [
                    {'range': [0, 40], 'color': '#ffcccc'},
                    {'range': [40, 70], 'color': '#fff3cc'},
                    {'range': [70, 100], 'color': '#ccffcc'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Analysis":
    st.markdown("---")
    st.subheader("Student Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(df, x='result', y='G3',
                    color='result',
                    color_discrete_map={'Pass':'#28a745','Fail':'#dc3545'},
                    title="Grade Distribution by Result")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='absences', y='G3',
                        color='result',
                        color_discrete_map={'Pass':'#28a745','Fail':'#dc3545'},
                        title="Absences vs Final Grade")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(df, x='studytime', y='G3',
                    color_discrete_sequence=['#4A90E2'],
                    title="Study Time vs Final Grade")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df.groupby('failures')['G3'].mean().reset_index(),
                    x='failures', y='G3',
                    color_discrete_sequence=['#764ba2'],
                    title="Past Failures vs Average Grade")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Dataset")
    st.dataframe(df.drop('pass', axis=1), use_container_width=True)

elif page == "About":
    st.markdown("---")
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("""
        <div style='background:#4A90E2;padding:40px;
        border-radius:15px;text-align:center;color:white'>
            <h1>🎓</h1>
            <h2>Student Performance Analyzer</h2>
            <p>Version 1.0</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### About This Project")
        st.write("""
        This project uses Machine Learning to predict whether
        a student will pass or fail based on various factors
        like study time, absences, and family background.
        """)
        st.markdown("### Tech Stack")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("Python")
            st.info("Pandas")
        with col_b:
            st.info("Streamlit")
            st.info("Plotly")
        with col_c:
            st.info("Scikit-learn")
            st.info("Random Forest")

        st.markdown(" Dataset")
        st.write("UCI Student Performance Dataset — 649 students, 33 features")

        st.markdown(" Model Performance")
        st.success("Accuracy: 86.15%")