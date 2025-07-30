import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import streamlit as st

# --- Simulation and ML ---
def simulate_data():
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq='h')
    data = []
    for ts in timestamps:
        temp = np.random.normal(55, 5)
        vib = np.random.normal(10, 2)
        pres = np.random.normal(250, 10)
        rpm = np.random.normal(1200, 50)
        failure = 0
        if ts > pd.to_datetime("2025-01-20"):
            temp += (ts - pd.to_datetime("2025-01-20")).days * 1.5
            vib += (ts - pd.to_datetime("2025-01-20")).days * 0.8
        if ts > pd.to_datetime("2025-02-05"):
            pres -= np.random.uniform(20, 40)
            rpm -= np.random.uniform(100, 200)
        if temp > 80 or vib > 30 or pres < 200:
            failure = 1
        data.append([ts, temp, vib, pres, rpm, failure])
    return pd.DataFrame(data, columns=["Timestamp", "Temperature", "Vibration", "Pressure", "RPM", "Failure"])

def train_model(df):
    X = df[['Temperature', 'Vibration', 'Pressure', 'RPM']]
    y = df['Failure']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def generate_dashboard(user_input, prediction, recommendation):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    sns.barplot(x=list(user_input.keys()), y=list(user_input.values()), ax=axs[0])
    axs[0].set_title('Sensor Input')
    axs[0].axhline(80, color='r', linestyle='--', label='Temp Threshold')
    axs[0].axhline(30, color='g', linestyle='--', label='Vibration Threshold')
    axs[0].axhline(200, color='b', linestyle='--', label='Pressure Threshold')
    axs[0].legend()

    axs[1].text(0.5, 0.5, f"{'Anomaly Detected ⚠️' if prediction else 'No Anomaly ✅'}", fontsize=20, ha='center')
    axs[1].axis('off')

    axs[2].text(0.5, 0.5, recommendation, fontsize=14, ha='center', wrap=True)
    axs[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# --- Streamlit App ---
st.title("Predictive Maintenance Dashboard with CrewAI")

temp = st.number_input("Temperature (°C)", value=85.0)
vib = st.number_input("Vibration", value=35.0)
pres = st.number_input("Pressure (psi)", value=180.0)
rpm = st.number_input("RPM", value=1100.0)

if st.button("Run Maintenance Prediction"):
    user_input = {"Temperature": temp, "Vibration": vib, "Pressure": pres, "RPM": rpm}
    df = simulate_data()
    df.loc[len(df)] = [pd.Timestamp.now(), temp, vib, pres, rpm, 0]
    model = train_model(df)
    pred = model.predict(df.iloc[[-1]][['Temperature', 'Vibration', 'Pressure', 'RPM']])[0]

    if pred == 1:
        if temp > 80 and vib > 30:
            recommendation = "Possible bearing fault detected. Immediate inspection recommended."
        elif pres < 200:
            recommendation = "Possible fluid leak detected. Inspect pressure system immediately."
        else:
            recommendation = "Anomaly detected. Further diagnostics recommended."
    else:
        recommendation = "No anomaly detected. System operating normally."

    st.success("CrewAI Maintenance Task Completed")
    generate_dashboard(user_input, pred, recommendation)
