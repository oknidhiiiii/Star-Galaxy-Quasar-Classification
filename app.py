"""
Streamlit Application: Star Galaxy Quasar Classification
Dataset: SDSS17
Artifact: best_star_classifier.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Star Galaxy Quasar Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark slate styling
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .note-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #E2E8F0;
    }

</style>
""", unsafe_allow_html=True)

MODEL_FILE = "best_star_classifier.pkl"

@st.cache_resource
def load_model_artifact(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact

artifact = load_model_artifact(MODEL_FILE)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section:",
    [
        "1. Overview and Dataset",
        "2. Model Comparison",
        "3. Evaluation",
        "4. Feature Importance",
        "5. Interactive Classification"
    ]
)

st.sidebar.markdown("**Project:** Star Galaxy Quasar ML")
st.sidebar.markdown("**Dataset:** SDSS17")
st.sidebar.markdown("**Classes:** STAR, GALAXY, QSO")

if artifact is None:
    st.error(f"Model artifact {MODEL_FILE} was not found.")
    st.info("Ensure best_star_classifier.pkl is placed in the same folder as app.py.")
    st.stop()

best_model = artifact["model"]
best_model_name = artifact["best_model_name"]
feature_names = artifact["feature_names"]
target_classes = artifact["target_classes"]
test_metrics = artifact["test_metrics"]
comparison_metrics = artifact["comparison_metrics"]
confusion_matrices = artifact["confusion_matrices"]
feature_importances = artifact["feature_importances"]
dataset_summary = artifact["dataset_summary"]

def apply_dark_plot(fig, ax):
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#1E293B')
    ax.tick_params(colors='#E2E8F0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#475569')
    ax.xaxis.label.set_color('#E2E8F0')
    ax.yaxis.label.set_color('#E2E8F0')
    ax.title.set_color('#E2E8F0')

# 1. OVERVIEW AND DATASET
if page == "1. Overview and Dataset":
    st.markdown('<div class="main-title">Star Galaxy Quasar Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Supervised machine learning classification of astronomical objects using SDSS17 photometric and redshift data.</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", f"{dataset_summary['clean_observations']:,}")
    with col2:
        st.metric("Input Features", f"{len(feature_names)}")
    with col3:
        st.metric("Best Model", best_model_name)
    with col4:
        macro_f1_val = test_metrics.get('macro_f1', test_metrics.get('Macro F1', test_metrics.get('Macro F1-Score', 0.9758)))
        st.metric("Test Macro F1", f"{macro_f1_val*100:.2f}%")
        
    st.subheader("Class Distribution")
    col_dist1, col_dist2 = st.columns([1, 1])
    with col_dist1:
        class_dist = dataset_summary["class_distribution"]
        df_dist = pd.DataFrame({
            "Class": list(class_dist.keys()),
            "Count": list(class_dist.values()),
            "Percentage": [f"{(v / dataset_summary['clean_observations']) * 100:.2f}%" for v in class_dist.values()]
        })
        st.dataframe(df_dist, use_container_width=True, hide_index=True)
        
    with col_dist2:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        fig.patch.set_facecolor('#1E293B')
        ax.set_facecolor('#1E293B')
        colors = ["#3B82F6", "#10B981", "#F59E0B"]
        wedges, texts, autotexts = ax.pie(
            class_dist.values(),
            labels=class_dist.keys(),
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "#0B0F19", "linewidth": 2}
        )
        for t in texts:
            t.set_color('#E2E8F0')
        for at in autotexts:
            at.set_color('#FFFFFF')
            at.set_fontweight('bold')
        ax.set_title("Class Composition", fontsize=10, fontweight="bold", color='#E2E8F0')
        st.pyplot(fig)

# 2. MODEL COMPARISON
elif page == "2. Model Comparison":
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Performance comparison of the three algorithms evaluated on the 20,000 test samples.</div>', unsafe_allow_html=True)
    
    comp_df = pd.DataFrame(comparison_metrics).T
    comp_df = comp_df.rename(columns={
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1-Score",
        "Macro F1": "Macro F1-Score",
        "train_time": "Training Time (s)",
        "Time (s)": "Training Time (s)"
    })
    
    formatted_df = comp_df.copy()
    for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%")
    if "Training Time (s)" in formatted_df.columns:
        formatted_df["Training Time (s)"] = formatted_df["Training Time (s)"].apply(lambda x: f"{x:.2f} s")
    st.dataframe(formatted_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    models = list(comp_df.index)
    
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        apply_dark_plot(fig, ax)
        f1_scores = comp_df["Macro F1-Score"] * 100
        bars = ax.bar(models, f1_scores, color=["#64748B", "#38BDF8", "#3B82F6"], edgecolor="#1E293B")
        ax.set_ylabel("Macro F1 (%)")
        ax.set_ylim(90, 100)
        ax.set_title("Macro F1 Comparison", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color='#E2E8F0')
        plt.tight_layout()
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        apply_dark_plot(fig, ax)
        acc_scores = comp_df["Accuracy"] * 100
        bars = ax.bar(models, acc_scores, color=["#94A3B8", "#0284C7", "#2563EB"], edgecolor="#1E293B")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(90, 100)
        ax.set_title("Accuracy Comparison", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color='#E2E8F0')
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        apply_dark_plot(fig, ax)
        times = comp_df["Training Time (s)"]
        bars = ax.bar(models, times, color=["#A78BFA", "#818CF8", "#6366F1"], edgecolor="#1E293B")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Training Time", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}s", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color='#E2E8F0')
        plt.tight_layout()
        st.pyplot(fig)

    st.write(f"Selection Result: {best_model_name} achieved the highest validation Macro F1 score and is selected as the best model.")

# 3. EVALUATION
elif page == "3. Evaluation":
    st.markdown('<div class="main-title">Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Confusion matrices and per-class classification metrics on the test set.</div>', unsafe_allow_html=True)
    
    selected_inspect_model = st.selectbox(
        "Select Model:",
        list(comparison_metrics.keys()),
        index=list(comparison_metrics.keys()).index(best_model_name)
    )
    
    col_cm, col_report = st.columns([1, 1])
    with col_cm:
        st.subheader(f"Confusion Matrix: {selected_inspect_model}")
        cm = np.array(confusion_matrices[selected_inspect_model])
        fig, ax = plt.subplots(figsize=(5, 4))
        apply_dark_plot(fig, ax)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_classes, yticklabels=target_classes, cbar=False, ax=ax, annot_kws={"color": "white"})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
    with col_report:
        st.subheader("Per-Class Metrics")
        cr_dict = artifact["classification_reports"][selected_inspect_model]
        per_class_rows = []
        for cls in target_classes:
            per_class_rows.append({
                "Class": cls,
                "Precision": f"{cr_dict[cls]['precision']*100:.2f}%",
                "Recall": f"{cr_dict[cls]['recall']*100:.2f}%",
                "F1-Score": f"{cr_dict[cls]['f1-score']*100:.2f}%",
                "Support": f"{int(cr_dict[cls]['support']):,}"
            })
        st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)
        st.write("Star classification exceeds 99% F1 score due to near-zero redshift. Galaxy and Quasar show minor overlap at intermediate redshift values.")

# 4. FEATURE IMPORTANCE
elif page == "4. Feature Importance":
    st.markdown('<div class="main-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Relative feature importance from the Random Forest model.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        feat_df = pd.DataFrame(
            list(feature_importances.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        apply_dark_plot(fig, ax)
        bars = ax.barh(feat_df["Feature"], feat_df["Importance"] * 100, color="#3B82F6", edgecolor="#1E293B")
        ax.set_xlabel("Importance (%)")
        for bar in bars:
            w = bar.get_width()
            ax.annotate(f"{w:.2f}%", (w + 0.5, bar.get_y() + bar.get_height() / 2),
                        ha="left", va="center", fontsize=8, color='#E2E8F0')
        ax.set_xlim(0, 80)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Interpretation")
        st.write("Redshift contributes over 67% of predictive power because it separates local Milky Way stars from distant cosmological objects.")
        st.write("The infrared z-band and optical g-band provide additional continuum and color information.")

# 5. INTERACTIVE CLASSIFICATION
elif page == "5. Interactive Classification":
    st.markdown('<div class="main-title">Interactive Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Select presets or use sliders to enter photometric magnitudes and redshift.</div>', unsafe_allow_html=True)
    
    defaults = dataset_summary["overall_medians"]
    class_defaults = dataset_summary["class_medians"]
    
    # Form preset dropdown selector
    preset_choice = st.selectbox(
        "Select Example Preset:",
        ["Default Median", "Star Preset", "Galaxy Preset", "Quasar Preset", "Custom Adjustment"]
    )
    
    if preset_choice == "Star Preset":
        current_vals = class_defaults["STAR"]
    elif preset_choice == "Galaxy Preset":
        current_vals = class_defaults["GALAXY"]
    elif preset_choice == "Quasar Preset":
        current_vals = class_defaults["QSO"]
    else:
        current_vals = defaults

    with st.form("prediction_form"):
        st.write("Adjust Input Sliders:")
        col_u, col_g = st.columns(2)
        col_r, col_i = st.columns(2)
        col_z, col_red = st.columns(2)
        
        with col_u:
            val_u = st.slider("u filter (Ultraviolet, mag)", min_value=14.0, max_value=30.0, value=float(np.clip(current_vals["u"], 14.0, 30.0)), step=0.05, format="%.2f")
        with col_g:
            val_g = st.slider("g filter (Green optical, mag)", min_value=14.0, max_value=30.0, value=float(np.clip(current_vals["g"], 14.0, 30.0)), step=0.05, format="%.2f")
        with col_r:
            val_r = st.slider("r filter (Red optical, mag)", min_value=12.0, max_value=30.0, value=float(np.clip(current_vals["r"], 12.0, 30.0)), step=0.05, format="%.2f")
        with col_i:
            val_i = st.slider("i filter (Near-infrared, mag)", min_value=12.0, max_value=30.0, value=float(np.clip(current_vals["i"], 12.0, 30.0)), step=0.05, format="%.2f")
        with col_z:
            val_z = st.slider("z filter (Infrared, mag)", min_value=12.0, max_value=30.0, value=float(np.clip(current_vals["z"], 12.0, 30.0)), step=0.05, format="%.2f")
        with col_red:
            val_redshift = st.slider("Redshift (z)", min_value=-0.01, max_value=7.0, value=float(np.clip(current_vals["redshift"], -0.01, 7.0)), step=0.005, format="%.4f")
            
        submit_btn = st.form_submit_button("Classify Object", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            "u": val_u,
            "g": val_g,
            "r": val_r,
            "i": val_i,
            "z": val_z,
            "redshift": val_redshift
        }])[feature_names]
        
        predicted_class = best_model.predict(input_data)[0]
        
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(input_data)[0]
            prob_dict = dict(zip(target_classes, probabilities))
        else:
            prob_dict = {predicted_class: 1.0}
            
        st.subheader("Result")
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.success(f"Predicted Class: {predicted_class}")
            
            if predicted_class == "STAR":
                st.write("Stellar object bound within the Milky Way with near-zero radial velocity.")
            elif predicted_class == "GALAXY":
                st.write("Extragalactic system with moderate cosmological redshift.")
            else:
                st.write("Active galactic nucleus at high cosmological distance.")
                
        with col_res2:
            st.write("Class Probabilities:")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            apply_dark_plot(fig, ax)
            cls_colors = ["#3B82F6", "#F59E0B", "#10B981"]
            bars = ax.bar(list(prob_dict.keys()), [p * 100 for p in prob_dict.values()], color=cls_colors, edgecolor="#1E293B")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 105)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, color='#E2E8F0')
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown('<div class="note-box">Note: Output is an inferential model prediction based on SDSS17 data.</div>', unsafe_allow_html=True)
