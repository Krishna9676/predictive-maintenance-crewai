import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# --- 1. Simulate Equipment Data ---
def simulate_data():
    timestamps = pd.date_range(start="2025-01-01", periods=1000, freq='h')
    data = []

    for ts in timestamps:
        temp = np.random.normal(55, 5)
        vibration = np.random.normal(10, 2)
        pressure = np.random.normal(250, 10)
        rpm = np.random.normal(1200, 50)
        failure = 0

        if ts > pd.to_datetime("2025-01-20"):
            temp += (ts - pd.to_datetime("2025-01-20")).days * 1.5
            vibration += (ts - pd.to_datetime("2025-01-20")).days * 0.8

        if ts > pd.to_datetime("2025-02-05"):
            pressure -= np.random.uniform(20, 40)
            rpm -= np.random.uniform(100, 200)

        if temp > 80 or vibration > 30 or pressure < 200:
            failure = 1

        data.append([ts, temp, vibration, pressure, rpm, failure])

    df = pd.DataFrame(data, columns=["Timestamp", "Temperature", "Vibration", "Pressure", "RPM", "Failure"])
    return df

# --- 2. Add User Input ---
def add_user_input(df, **kwargs):
    new_row = pd.DataFrame([[pd.Timestamp.now(), kwargs["Temperature"], kwargs["Vibration"],
                             kwargs["Pressure"], kwargs["RPM"], 0]], columns=df.columns)
    return pd.concat([df, new_row], ignore_index=True)

# --- 3. Train Model ---
def train_model(df):
    X = df[['Temperature', 'Vibration', 'Pressure', 'RPM']]
    y = df['Failure']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- 4. Create CrewAI Tool ---
def create_tool(user_input, model, df):
    @tool
    def anomaly_detector_tool(data_file: str) -> str:
        latest = df.iloc[-1:]
        pred = model.predict(latest[['Temperature', 'Vibration', 'Pressure', 'RPM']])
        if pred[0] == 1:
            return f"Anomaly Detected:\n\n{latest.to_string(index=False)}"
        return "No anomaly detected. System is stable."
    return anomaly_detector_tool

# --- 5. Generate Dashboard ---
def generate_dashboard(df, model, user_input):
    latest = df.iloc[-1:]
    prediction = model.predict(latest[['Temperature', 'Vibration', 'Pressure', 'RPM']])[0]

    if prediction == 1:
        if user_input["Temperature"] > 80 and user_input["Vibration"] > 30:
            recommendation = "Possible bearing fault detected. Immediate inspection recommended."
        elif user_input["Pressure"] < 200:
            recommendation = "Possible fluid leak detected. Inspect pressure system immediately."
        else:
            recommendation = "Anomaly detected. Further diagnostics recommended."
    else:
        recommendation = "No anomaly detected. System operating normally."

    st.subheader("📊 Sensor Input Readings")
    st.bar_chart(pd.DataFrame(user_input, index=[0]))

    st.subheader("🔍 Failure Prediction")
    st.success("✅ No Anomaly Detected") if prediction == 0 else st.error("⚠️ Anomaly Detected")

    st.subheader("🛠 Maintenance Recommendation")
    st.info(recommendation)

# --- 6. Streamlit App ---
st.title("🔧 Predictive Maintenance with CrewAI")
df = simulate_data()

with st.form("sensor_form"):
    st.write("## Enter Sensor Readings")
    temp = st.slider("Temperature (°C)", 30.0, 120.0, 55.0)
    vib = st.slider("Vibration", 0.0, 50.0, 10.0)
    pres = st.slider("Pressure (psi)", 100.0, 300.0, 250.0)
    rpm = st.slider("RPM", 500.0, 1500.0, 1200.0)
    submit = st.form_submit_button("Run Analysis")

if submit:
    user_input = {"Temperature": temp, "Vibration": vib, "Pressure": pres, "RPM": rpm}
    df = add_user_input(df, **user_input)
    model = train_model(df)
    tool_instance = create_tool(user_input, model, df)

    anomaly_agent = Agent(
        role="Anomaly Detector",
        goal="Detect failure in user sensor readings",
        backstory="Expert in using ML models for real-time failure detection",
        tools=[tool_instance],
        verbose=False
    )

    diagnostic_agent = Agent(
        role="Maintenance Advisor",
        goal="Provide recommendations based on anomaly output",
        backstory="Expert maintenance engineer with pattern recognition skills",
        tools=[],
        verbose=False
    )

    task1 = Task(
        description="Run anomaly detection on the latest user input row.",
        expected_output="Anomaly or normal report.",
        agent=anomaly_agent
    )

    task2 = Task(
        description="Read anomaly report and provide actionable maintenance recommendation.",
        expected_output="Clear maintenance advice.",
        agent=diagnostic_agent,
        context=[task1]
    )

    crew = Crew(
        agents=[anomaly_agent, diagnostic_agent],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    st.subheader("🧠 CrewAI Result")
    st.code(result)
    generate_dashboard(df, model, user_input)
