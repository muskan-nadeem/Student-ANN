import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import joblib
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Performance Evaluator", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .pass-box {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        border-left: 6px solid #10B981;
        padding: 1.2rem; border-radius: 12px; margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(16,185,129,0.2);
    }
    .fail-box {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-left: 6px solid #EF4444;
        padding: 1.2rem; border-radius: 12px; margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(239,68,68,0.2);
    }
    div[data-testid="metric-container"] {
        background: #F8FAFC; border-radius: 12px;
        padding: 0.8rem; border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🧠 Training ANN model...")
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset.xlsx")
    if not os.path.exists(dataset_path):
        st.error("dataset.xlsx not found!")
        st.stop()

    df = pd.read_excel(dataset_path)
    X = df[["attendance", "assignment", "quiz", "mid", "study_hours"]]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train)

    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])
    avg      = df.groupby("result")[["attendance","assignment","quiz","mid","study_hours"]].mean()

    try:
        joblib.dump(model,  os.path.join(base_dir, "model.joblib"))
        joblib.dump(scaler, os.path.join(base_dir, "scaler.joblib"))
    except Exception:
        pass

    return model, scaler, accuracy, report, cm, model.loss_curve_, avg, df


model, scaler, accuracy, report, cm, loss_curve, avg, df = train_model()
FEATURES  = ["Attendance", "Assignment", "Quiz", "Mid-term", "Study Hours"]
FEAT_KEYS = ["attendance", "assignment", "quiz", "mid", "study_hours"]


def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features   = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled     = scaler.transform(features)
    pred       = int(model.predict(scaled)[0])
    probs      = model.predict_proba(scaled)[0]
    pass_prob  = float(probs[1])
    confidence = float(max(probs))
    band = "High 🏆" if pass_prob >= 0.80 else ("Medium 📈" if pass_prob >= 0.50 else "Low ⚠️")
    return pred, pass_prob, confidence, band


def make_visuals(attendance, assignment, quiz, mid, study_hours, pass_prob, pred):
    student_vals = [attendance, assignment, quiz, mid, study_hours]
    pass_avg = [avg.loc[1, k] if 1 in avg.index else 0 for k in FEAT_KEYS]
    fail_avg = [avg.loc[0, k] if 0 in avg.index else 0 for k in FEAT_KEYS]

    PURPLE = "#8B5CF6"
    GREEN  = "#10B981"
    RED    = "#EF4444"
    BLUE   = "#6366F1"
    PINK   = "#EC4899"

    fig = plt.figure(figsize=(16, 12), facecolor="#FAFAFA")
    fig.suptitle("📊 Student Performance Analysis", fontsize=18,
                 fontweight="bold", color="#1E293B", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Graph 1: Bar Chart ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(FEATURES))
    w = 0.25
    b1 = ax1.bar(x - w, student_vals, w, label="Your Student", color=PURPLE, alpha=0.9, zorder=3)
    b2 = ax1.bar(x,     pass_avg,     w, label="Pass Avg",     color=GREEN,  alpha=0.8, zorder=3)
    b3 = ax1.bar(x + w, fail_avg,     w, label="Fail Avg",     color=RED,    alpha=0.6, zorder=3)

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{bar.get_height():.0f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=PURPLE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(FEATURES, fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Student vs Class Average", fontsize=12, fontweight="bold", color="#1E293B", pad=10)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 115)
    ax1.set_facecolor("#F1F5F9")
    ax1.grid(axis="y", alpha=0.4, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Graph 2: Gauge ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2], aspect="equal")
    gauge_color = GREEN if pass_prob >= 0.5 else RED
    theta2 = 180 - (pass_prob * 180)
    ax2.add_patch(mpatches.Wedge((0.5, 0.3), 0.40, 0, 180, width=0.15,
                                  facecolor="#E2E8F0", transform=ax2.transAxes))
    ax2.add_patch(mpatches.Wedge((0.5, 0.3), 0.40, theta2, 180, width=0.15,
                                  facecolor=gauge_color, transform=ax2.transAxes))
    ax2.text(0.5, 0.32, f"{pass_prob*100:.0f}%", transform=ax2.transAxes,
             ha="center", va="center", fontsize=26, fontweight="bold", color="#1E293B")
    ax2.text(0.5, 0.12, "Pass Probability", transform=ax2.transAxes,
             ha="center", fontsize=10, color="#64748B")
    result_text  = "✅ PASS" if pred == 1 else "❌ FAIL"
    result_color = GREEN if pred == 1 else RED
    ax2.text(0.5, 0.70, result_text, transform=ax2.transAxes,
             ha="center", fontsize=14, fontweight="bold", color=result_color)
    ax2.set_facecolor("#F8FAFC")
    ax2.axis("off")
    ax2.set_title("Result", fontsize=12, fontweight="bold", color="#1E293B", pad=10)

    # ── Graph 3: Radar Chart ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    maxvals = [100, 100, 100, 100, 20]
    s_norm  = [v/m for v, m in zip(student_vals, maxvals)] + [student_vals[0]/maxvals[0]]
    p_norm  = [v/m for v, m in zip(pass_avg,     maxvals)] + [pass_avg[0]/maxvals[0]]
    f_norm  = [v/m for v, m in zip(fail_avg,     maxvals)] + [fail_avg[0]/maxvals[0]]

    ax3.plot(angles, s_norm, "o-",  linewidth=2,   color=PURPLE, label="Student")
    ax3.fill(angles, s_norm, alpha=0.25, color=PURPLE)
    ax3.plot(angles, p_norm, "s--", linewidth=1.5, color=GREEN,  label="Pass Avg")
    ax3.fill(angles, p_norm, alpha=0.10, color=GREEN)
    ax3.plot(angles, f_norm, "^--", linewidth=1.5, color=RED,    label="Fail Avg")
    ax3.set_thetagrids(np.degrees(angles[:-1]), FEATURES, fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_title("Radar: Feature Comparison", fontsize=11,
                  fontweight="bold", color="#1E293B", pad=15)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax3.set_facecolor("#F8FAFC")
    ax3.grid(color="#CBD5E1", alpha=0.5)

    # ── Graph 4: Horizontal Score Bars ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    colors_bar = [BLUE, PURPLE, PINK, GREEN, "#F59E0B"]
    maxvals2   = [100, 100, 100, 100, 20]
    pcts       = [v/m*100 for v, m in zip(student_vals, maxvals2)]

    bars = ax4.barh(FEATURES, pcts, color=colors_bar, alpha=0.85, height=0.55)
    for bar, pct in zip(bars, pcts):
        ax4.text(min(pct + 1, 96), bar.get_y() + bar.get_height()/2,
                 f"{pct:.0f}%", va="center", fontsize=9, fontweight="bold")

    ax4.set_xlim(0, 110)
    ax4.set_xlabel("Score %", fontsize=9)
    ax4.set_title("Score Breakdown", fontsize=12, fontweight="bold", color="#1E293B", pad=10)
    ax4.set_facecolor("#F1F5F9")
    ax4.grid(axis="x", alpha=0.4)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.axvline(x=80, color=GREEN, linestyle="--", alpha=0.6, linewidth=1)
    ax4.text(81, -0.5, "Good", fontsize=7, color=GREEN)

    # ── Graph 5: Donut Chart ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    pass_count = int(df["result"].sum())
    fail_count = len(df) - pass_count
    wedges, texts, autotexts = ax5.pie(
        [pass_count, fail_count],
        explode=(0.05, 0),
        labels=["Pass", "Fail"],
        colors=[GREEN, RED],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
        pctdistance=0.75,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)

    student_angle = 90 + (pass_prob * 360)
    rad = np.radians(student_angle)
    ax5.plot(0.63 * np.cos(rad), 0.63 * np.sin(rad),
             "o", markersize=12, color=PURPLE, zorder=5,
             label=f"Your student ({pass_prob*100:.0f}%)")
    ax5.legend(fontsize=8, loc="lower center")
    ax5.set_title("Class Distribution", fontsize=12,
                  fontweight="bold", color="#1E293B", pad=10)

    return fig


# ═══════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════

st.markdown('<p class="main-title">🎓 Student Performance Evaluator</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748B;margin-top:-10px">ANN-powered · Beautiful Analytics · Pass / Fail Prediction</p>', unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### 📌 Model Info")
    st.markdown(f"**Accuracy:** `{accuracy*100:.1f}%`")
    st.markdown("**Architecture:** `5 → 64 → 32 → 1`")
    st.markdown("**Solver:** Adam | **Activation:** ReLU")
    st.divider()
    tn, fp, fn, tp = cm.ravel()
    st.markdown("### 📊 Test Results")
    st.metric("✅ True Pass",   int(tp))
    st.metric("✅ True Fail",   int(tn))
    st.metric("⚠️ False Alarm", int(fp))
    st.metric("⚠️ Missed Fail", int(fn))
    st.divider()
    st.markdown("### 📋 Ranges")
    st.markdown("""
| Feature | Max |
|---|---|
| Attendance | 100% |
| Assignment | 100 pts |
| Quiz | 100 pts |
| Mid-term | 100 pts |
| Study Hrs | 20/week |
    """)

tab1, tab2 = st.tabs(["🔮 Predict & Analyze", "📈 Model Report"])

# ── TAB 1 ───────────────────────────────────
with tab1:
    st.markdown("#### ✏️ Enter Student Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        attendance  = st.slider("🏫 Attendance (%)",   0,   100, 75)
        assignment  = st.slider("📄 Assignment Marks", 0,   100, 70)
    with col2:
        quiz        = st.slider("📝 Quiz Marks",        0,   100, 60)
        mid         = st.slider("📚 Mid-term Marks",    0,   100, 55)
    with col3:
        study_hours = st.slider("⏰ Study Hours/Week",  0.0, 20.0, 5.0, step=0.5)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict & Show Analysis",
                                use_container_width=True, type="primary")

    if predict_btn:
        pred, pass_prob, confidence, band = evaluate_student(
            attendance, assignment, quiz, mid, study_hours
        )

        if pred == 1:
            st.markdown(f"""
            <div class="pass-box">
                <h2 style="color:#065F46;margin:0">✅ PASS</h2>
                <p style="color:#065F46;margin:0.4rem 0 0 0">
                Pass probability: <strong>{pass_prob*100:.1f}%</strong> &nbsp;·&nbsp;
                Confidence: <strong>{confidence*100:.1f}%</strong> &nbsp;·&nbsp;
                Band: <strong>{band}</strong>
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fail-box">
                <h2 style="color:#991B1B;margin:0">❌ FAIL</h2>
                <p style="color:#991B1B;margin:0.4rem 0 0 0">
                Pass probability: <strong>{pass_prob*100:.1f}%</strong> &nbsp;·&nbsp;
                Confidence: <strong>{confidence*100:.1f}%</strong> &nbsp;·&nbsp;
                Band: <strong>{band}</strong>
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Pass Probability", f"{pass_prob*100:.1f}%")
        m2.metric("💡 Confidence",       f"{confidence*100:.1f}%")
        m3.metric("📊 Band",             band.split()[0])
        m4.metric("🏆 Model Accuracy",   f"{accuracy*100:.1f}%")

        st.markdown("---")
        st.markdown("### 📊 Full Visual Analysis")
        fig = make_visuals(attendance, assignment, quiz, mid, study_hours, pass_prob, pred)
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 💡 Improvement Tips")
        tips = []
        if attendance  < 75: tips.append("📅 Attendance 75% se upar lao")
        if assignment  < 60: tips.append("📄 Assignments properly complete karo")
        if quiz        < 50: tips.append("📝 Quiz preparation improve karo")
        if mid         < 50: tips.append("📚 Mid-term ke liye zyada study karo")
        if study_hours < 4:  tips.append("⏰ Rozana kam se kam 1 ghanta extra parho")

        if tips:
            t1, t2 = st.columns(2)
            half = len(tips)//2 + len(tips)%2
            with t1:
                for t in tips[:half]:  st.warning(t)
            with t2:
                for t in tips[half:]:  st.warning(t)
        else:
            st.success("🎉 Excellent! Student ki performance bahut achi hai!")

# ── TAB 2 ───────────────────────────────────
with tab2:
    st.markdown("#### 🎯 Classification Report")
    st.code(report)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📉 Training Loss Curve")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.plot(loss_curve, color="#6366F1", linewidth=2.5)
        ax2.fill_between(range(len(loss_curve)), loss_curve, alpha=0.15, color="#6366F1")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.set_title("ANN Training Loss", fontweight="bold")
        ax2.set_facecolor("#F8FAFC"); ax2.grid(alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with c2:
        st.markdown("#### 🔢 Confusion Matrix")
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=["Fail", "Pass"],
                    yticklabels=["Fail", "Pass"], ax=ax3,
                    linewidths=2, linecolor="white",
                    annot_kws={"size": 14, "weight": "bold"})
        ax3.set_xlabel("Predicted", fontsize=11)
        ax3.set_ylabel("Actual",    fontsize=11)
        ax3.set_title("Confusion Matrix", fontweight="bold", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    st.markdown("#### 🏗️ ANN Architecture")
    st.markdown("""
| Layer | Neurons | Activation |
|---|---|---|
| Input | 5 | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Output | 1 | Sigmoid |
    """)

st.divider()
st.caption("🎓 ANN Student Evaluator · scikit-learn · Streamlit · Made with ❤️")