import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import io
import joblib
import os

st.set_page_config(page_title="Iris Classifier • ML Demo", page_icon="🌸", layout="wide")

# --- Sidebar: Model & Settings ---
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Choose a model and tweak parameters. The app trains quickly on the Iris dataset.")

model_name = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest", "SVM (RBF)"])

if model_name == "Logistic Regression":
    C = st.sidebar.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 500, step=50)
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, step=1)
else:
    C = st.sidebar.slider("C (Penalty)", 0.01, 10.0, 1.0)
    gamma = st.sidebar.select_slider("Gamma", options=["scale", "auto"])

test_size = st.sidebar.slider("Test Size (hold-out)", 0.1, 0.5, 0.2, step=0.05)
do_cv = st.sidebar.checkbox("Run 5-fold Cross-Validation", value=True)

st.sidebar.markdown("---")
st.sidebar.write("📦 You can **download the trained model** after training and reuse it later.")

# --- Load Data ---
@st.cache_data
def load_data():
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names
    df = iris.frame
    return X, y, target_names, feature_names, df

X, y, target_names, feature_names, df = load_data()

# --- Header ---
st.title("🌸 Iris Classifier — Interactive ML App")
st.caption("A simple, production-ready Streamlit app showcasing an end-to-end ML workflow with a clean UI.")

# --- Data Preview & Plot ---
with st.expander("👀 Preview Dataset & Basic EDA", expanded=False):
    st.write("The classic Iris dataset has 150 samples, 4 features, and 3 classes.")
    st.dataframe(df.head())
    st.write("Feature Summary:")
    st.dataframe(df.describe())

    # Pairwise scatter using matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 2])
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[2])
    ax.set_title("Quick Look: Sepal Length vs Petal Length")
    st.pyplot(fig)

# --- Build Pipeline ---
def build_model():
    if model_name == "Logistic Regression":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=max_iter, multi_class="auto"))
        ])
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=C, gamma=gamma, probability=True))
        ])
    return clf

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# --- Train Button ---
if st.button("🚀 Train Model"):
    model = build_model()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📈 Evaluation on Hold-out Test Set")
    st.write(pd.DataFrame(report).T)

    fig_cm = plt.figure()
    ax = fig_cm.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig_cm)

    if do_cv:
        st.subheader("🔁 5-Fold Cross-Validation (Accuracy)")
        model_cv = build_model()
        scores = cross_val_score(model_cv, X, y, cv=5, scoring="accuracy")
        st.write(f"Mean Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        st.write(pd.DataFrame({"Fold": np.arange(1, 6), "Accuracy": scores}))

    # Save model to buffer for download
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    st.download_button("💾 Download Trained Model (.joblib)", buffer, file_name="iris_model.joblib")

    st.success("Training complete! Scroll down to try interactive predictions.")

    st.session_state["trained_model"] = model

# --- Interactive Prediction UI ---
st.subheader("🔮 Try a Prediction")
col1, col2, col3, col4 = st.columns(4)
with col1:
    sepal_length = st.number_input("Sepal length (cm)", 4.3, 7.9, 5.1, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal width (cm)", 2.0, 4.4, 3.5, step=0.1)
with col3:
    petal_length = st.number_input("Petal length (cm)", 1.0, 6.9, 1.4, step=0.1)
with col4:
    petal_width = st.number_input("Petal width (cm)", 0.1, 2.5, 0.2, step=0.1)

user_sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

if "trained_model" in st.session_state:
    mdl = st.session_state["trained_model"]
    pred = mdl.predict(user_sample)[0]
    proba = getattr(mdl, "predict_proba", lambda X: np.zeros((len(X), len(target_names))))(user_sample)[0]

    st.info(f"**Predicted class:** {target_names[pred]}")
    st.write(pd.DataFrame({"class": target_names, "probability": proba}))
else:
    st.warning("Train a model first to enable predictions.")

st.markdown("---")
st.caption("Built with Streamlit • scikit-learn • pandas • numpy.")

# --- Allow direct Run in VS Code ---
if __name__ == "__main__":
    os.system("streamlit run app.py")
