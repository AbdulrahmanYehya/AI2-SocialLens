import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
MODEL_PATH = "final_model.pkl"  # <<< PUT YOUR .PKL MODEL PATH HERE

st.set_page_config(page_title="SocialLens", page_icon="🧠", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource

def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("🌐 SocialLens")
section = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧠 Introvert vs Extrovert", "📊 What Affects Personality?", "🤖 Personality Predictor"]
)

# =========================
# HOME PAGE
# =========================
if section == "🏠 Home":
    st.title("SocialLens – Discover Your Personality")
    st.markdown("""
    **SocialLens** is a smart personality analysis system that helps you understand whether you lean more toward being an **introvert** or an **extrovert** using artificial intelligence.

    🔹 Learn the difference between personality types  
    🔹 Understand what affects them  
    🔹 Test yourself using a real trained ML model  
    
    ---
    ### 🎯 Project Goal
    To combine **data science, machine learning, and psychology** into an interactive platform that helps users understand themselves better.
    """)

# =========================
# INTROVERT VS EXTROVERT
# =========================
elif section == "🧠 Introvert vs Extrovert":
    st.title("Introvert vs Extrovert")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Introvert")
        st.markdown("""
        - Gains energy from **being alone**
        - Thinks deeply before speaking
        - Prefers quiet environments
        - Good listener
        - Small close friend groups
        """)

    with col2:
        st.subheader("🟠 Extrovert")
        st.markdown("""
        - Gains energy from **social interaction**
        - Enjoys being around people
        - Thinks while speaking
        - Expressive and talkative
        - Large social circles
        """)

    st.markdown("---")

    st.subheader("✅ Key Differences")
    st.table(pd.DataFrame({
        "Introvert": ["Quiet", "Enjoys solitude", "Deep focus", "Few friends", "Observer"],
        "Extrovert": ["Talkative", "Enjoys crowds", "Fast action", "Many friends", "Leader"]
    }))

# =========================
# WHAT AFFECTS PERSONALITY
# =========================
elif section == "📊 What Affects Personality?":
    st.title("What Affects Introversion & Extroversion?")

    st.markdown("""
    ### 🧬 1. Genetics
    Personality is partly inherited from parents.

    ### 🧠 2. Brain Chemistry
    Dopamine sensitivity plays a key role in extroversion.

    ### 👶 3. Childhood Environment
    Upbringing, parenting, and education shape behavior.

    ### 🌍 4. Culture & Society
    Different cultures encourage different traits.

    ### 💼 5. Life Experiences
    Trauma, success, and failures can shift personality.
    """)

# =========================
# ML PERSONALITY PREDICTOR
# =========================
elif section == "🤖 Personality Predictor":
    st.title("AI Personality Prediction")
    st.markdown("Enter your data and let **SocialLens AI** predict your personality type.")

    st.subheader("📥 User Input")

    # --- Inputs ONLY exist inside this page ---
    time_alone = st.slider("Time Spent Alone (hours/day)", 0, 24, 5)
    social_attend = st.slider("Social Event Attendance (per week)", 0, 14, 3)
    going_out = st.slider("Going Outside Frequency (per week)", 0, 14, 4)
    friends = st.slider("Friends Circle Size", 0, 50, 8)
    post_freq = st.slider("Social Media Post Frequency (per week)", 0, 50, 5)
    stage_fear = st.selectbox("Stage Fear", ["Yes", "No"])
    drained = st.selectbox("Drained After Socializing", ["Yes", "No"])

    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained == "Yes" else 0

    input_dict = {
        "Time_spent_Alone": time_alone,
        "Social_event_attendance": social_attend,
        "Going_outside": going_out,
        "Friends_circle_size": friends,
        "Post_frequency": post_freq,
        "Stage_fear": stage_fear_val,
        "Drained_after_socializing": drained_val
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("🔍 Predict Personality"):
        prediction = model.predict(input_df)[0]

        st.markdown("---")

        if str(prediction).lower().startswith("intro"):
            st.success("🧠 You are predicted to be an **INTROVERT**")
        else:
            st.success("🔥 You are predicted to be an **EXTROVERT**")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("**SocialLens – AI Powered Personality Detection** 🚀")