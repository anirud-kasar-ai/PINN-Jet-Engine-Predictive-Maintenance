

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Jet Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .alert-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .critical-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_scalers():
    """Load trained PINN model and scalers from local path"""
    try:
        # Model path
        model_path = r"jet_engine_pinn_model.h5"
        scaler_x_path = r"scaler_X.pkl"
        scaler_y_path = r"scaler_y.pkl"
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None, None, False
            
        # Load model
        model = keras.models.load_model(model_path, compile=False)
        
        # Load scalers if they exist, otherwise create new ones
        if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
            with open(scaler_x_path, 'rb') as f:
                scaler_X = pickle.load(f)
            with open(scaler_y_path, 'rb') as f:
                scaler_y = pickle.load(f)
        else:
            # Create default scalers if files don't exist
            st.warning("Scaler files not found. Using default scalers.")
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            # Fit with dummy data
            scaler_X.fit(np.array([[500, 10, 0.3, 4500, 80, 0], 
                                   [600, 18, 0.8, 5500, 120, 200]]))
            scaler_y.fit(np.array([[0, 0, 0, 0, 0, 0], 
                                   [100, 100, 100, 100, 100, 100]]))
        
        return model, scaler_X, scaler_y, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, False

def generate_synthetic_engine_data(n_cycles=200, seed=None):
    """Generate synthetic data for a single engine"""
    if seed:
        np.random.seed(seed)
    
    failure_cycle = np.random.randint(150, n_cycles)
    
    temp_degrade = np.random.uniform(0.001, 0.003)
    pressure_degrade = np.random.uniform(0.001, 0.0025)
    vibration_degrade = np.random.uniform(0.002, 0.004)
    rpm_degrade = np.random.uniform(0.0005, 0.002)
    fuel_degrade = np.random.uniform(0.001, 0.003)
    
    data = []
    for cycle in range(n_cycles):
        temperature = 550 + np.random.normal(0, 5)
        pressure = 14.7 + np.random.normal(0, 0.3)
        vibration = 0.5 + np.random.normal(0, 0.05)
        rpm = 5000 + np.random.normal(0, 50)
        fuel_flow = 100 + np.random.normal(0, 2)
        
        if cycle > failure_cycle * 0.7:
            degradation_factor = (cycle - failure_cycle * 0.7) / (failure_cycle * 0.3)
            temperature += degradation_factor * 50 * temp_degrade * 100
            pressure -= degradation_factor * 2 * pressure_degrade * 100
            vibration += degradation_factor * 2 * vibration_degrade * 100
            rpm -= degradation_factor * 200 * rpm_degrade * 100
            fuel_flow += degradation_factor * 20 * fuel_degrade * 100
        
        temp_health = max(0, 100 - cycle * temp_degrade * 100)
        pressure_health = max(0, 100 - cycle * pressure_degrade * 100)
        vibration_health = max(0, 100 - cycle * vibration_degrade * 100)
        rpm_health = max(0, 100 - cycle * rpm_degrade * 100)
        fuel_health = max(0, 100 - cycle * fuel_degrade * 100)
        
        overall_health = np.mean([temp_health, pressure_health, vibration_health, 
                                 rpm_health, fuel_health])
        
        data.append({
            'cycle': cycle,
            'temperature': temperature,
            'pressure': pressure,
            'vibration': vibration,
            'rpm': rpm,
            'fuel_flow': fuel_flow,
            'overall_health': overall_health
        })
    
    return pd.DataFrame(data)

def predict_sensor_health(model, scaler_X, scaler_y, sensor_data):
    """Predict sensor health using PINN model"""
    try:
        # Prepare input
        X = np.array([[
            sensor_data['temperature'],
            sensor_data['pressure'],
            sensor_data['vibration'],
            sensor_data['rpm'],
            sensor_data['fuel_flow'],
            sensor_data['cycle']
        ]])
        
        # Scale and predict
        X_scaled = scaler_X.transform(X)
        prediction_scaled = model.predict(X_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)[0]
        
        # Extract individual sensor healths
        health_dict = {
            'Temperature': max(0, min(100, prediction[0])),
            'Pressure': max(0, min(100, prediction[1])),
            'Vibration': max(0, min(100, prediction[2])),
            'RPM': max(0, min(100, prediction[3])),
            'Fuel Flow': max(0, min(100, prediction[4])),
            'Overall': max(0, min(100, prediction[5]))
        }
        
        return health_dict
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_status_and_color(health):
    """Get status text and color based on health percentage"""
    if health >= 70:
        return "HEALTHY", "#28a745", "✓"
    elif health >= 50:
        return "WARNING", "#ffc107", "⚠"
    else:
        return "CRITICAL", "#dc3545", "✗"

def create_gauge_chart(value, title):
    """Create a gauge chart for sensor health"""
    status, color, icon = get_status_and_color(value)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{icon} {title}", 'font': {'size': 20, 'color': color}},
        delta = {'reference': 100, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 70], 'color': '#fff8e1'},
                {'range': [70, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_health_timeline(df, model, scaler_X, scaler_y):
    """Create health timeline chart"""
    sensor_features = ['temperature', 'pressure', 'vibration', 'rpm', 'fuel_flow', 'cycle']
    
    # Predict health for all cycles
    X = df[sensor_features].values
    X_scaled = scaler_X.transform(X)
    predictions_scaled = model.predict(X_scaled, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # Create figure
    fig = go.Figure()
    
    sensor_names = ['Temperature', 'Pressure', 'Vibration', 'RPM', 'Fuel Flow', 'Overall']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
    
    for i, (name, color) in enumerate(zip(sensor_names, colors)):
        fig.add_trace(go.Scatter(
            x=df['cycle'],
            y=predictions[:, i],
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f'{name}: %{{y:.1f}}%<br>Cycle: %{{x}}<extra></extra>'
        ))
    
    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                  annotation_text="Warning Threshold (70%)")
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="Critical Threshold (50%)")
    
    fig.update_layout(
        title="Sensor Health Degradation Timeline",
        xaxis_title="Operating Cycle",
        yaxis_title="Health (%)",
        hovermode='x unified',
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_sensor_comparison_bar(health_dict):
    """Create bar chart comparing all sensor healths"""
    sensors = list(health_dict.keys())
    values = list(health_dict.values())
    colors = [get_status_and_color(v)[1] for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sensors,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='outside',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical")
    
    fig.update_layout(
        title="Current Sensor Health Comparison",
        xaxis_title="Sensor",
        yaxis_title="Health (%)",
        yaxis_range=[0, 105],
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_radar_chart(health_dict):
    """Create radar chart for sensor health"""
    # Exclude overall from radar
    sensors = [k for k in health_dict.keys() if k != 'Overall']
    values = [health_dict[k] for k in sensors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=sensors,
        fill='toself',
        name='Current Health',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add reference circles
    fig.add_trace(go.Scatterpolar(
        r=[70]*len(sensors),
        theta=sensors,
        mode='lines',
        name='Warning Threshold',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[50]*len(sensors),
        theta=sensors,
        mode='lines',
        name='Critical Threshold',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Sensor Health Radar",
        height=450
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ Jet Engine Predictive Maintenance System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">Physics-Informed Neural Network (PINN) Based Real-Time Monitoring</p>', 
                unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading PINN model...'):
        model, scaler_X, scaler_y, model_loaded = load_model_and_scalers()
    
    if not model_loaded:
        st.error("⚠️ Failed to load model. Please ensure the model file exists at: D:\\Desktop\\PINN\\jet_engine_pinn_model.h5")
        st.info("💡 **Tip**: Make sure you have trained the model using the Jupyter notebook first!")
        st.stop()
    
    #st.success("✅ PINN Model loaded successfully!")
    
    # Sidebar
    # Try to load Aziro logo, fallback to placeholder if not found
    logo_path = r"D:\Downloads\aziro-logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        st.sidebar.markdown("### ✈️ Jet Engine Monitor")
        st.sidebar.info("Place 'aziro-logo.png' in D:\\Downloads\\ to display logo")
    st.sidebar.markdown("---")
    
    # Mode selection
    st.sidebar.markdown("### 🎛️ Operation Mode")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Real-Time Monitoring", "Historical Analysis", "Manual Input"],
        help="Choose how you want to interact with the system"
    )
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # MODE 1: REAL-TIME MONITORING
    # ========================================================================
    if mode == "Real-Time Monitoring":
        st.markdown('<h2 class="sub-header">📊 Real-Time Engine Monitoring</h2>', 
                    unsafe_allow_html=True)
        
        # Engine selection
        col1, col2 = st.columns([3, 1])
        with col1:
            engine_id = st.selectbox(
                "Select Engine:",
                [f"Engine {i:03d}" for i in range(1, 11)],
                help="Choose an engine to monitor"
            )
        with col2:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        # Generate synthetic data for selected engine
        seed = int(engine_id.split()[1])
        df = generate_synthetic_engine_data(n_cycles=200, seed=seed)
        
        # Current cycle selector
        current_cycle = st.slider(
            "Operating Cycle:",
            min_value=0,
            max_value=len(df)-1,
            value=len(df)//2,
            help="Slide to see predictions at different operating cycles"
        )
        
        # Get current sensor data
        current_data = df.iloc[current_cycle].to_dict()
        
        # Predict health
        health_dict = predict_sensor_health(model, scaler_X, scaler_y, current_data)
        
        if health_dict:
            # Display current sensor readings
            st.markdown("### 📈 Current Sensor Readings")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🌡️ Temperature", f"{current_data['temperature']:.1f} °C")
            with col2:
                st.metric("💨 Pressure", f"{current_data['pressure']:.2f} PSI")
            with col3:
                st.metric("📳 Vibration", f"{current_data['vibration']:.3f} g")
            with col4:
                st.metric("⚙️ RPM", f"{current_data['rpm']:.0f}")
            with col5:
                st.metric("⛽ Fuel Flow", f"{current_data['fuel_flow']:.1f} kg/h")
            
            st.markdown("---")
            
            # Overall health status
            overall_health = health_dict['Overall']
            status, color, icon = get_status_and_color(overall_health)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                        padding: 2rem; border-radius: 15px; border-left: 5px solid {color};'>
                <h2 style='margin: 0; color: {color};'>{icon} Overall Engine Health: {overall_health:.1f}%</h2>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; color: {color};'>Status: <strong>{status}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Individual Sensor Health")
            
            # Gauge charts for each sensor
            cols = st.columns(3)
            sensor_list = ['Temperature', 'Pressure', 'Vibration', 'RPM', 'Fuel Flow']
            
            for idx, sensor in enumerate(sensor_list):
                with cols[idx % 3]:
                    fig = create_gauge_chart(health_dict[sensor], sensor)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Alerts section
            st.markdown("### 🚨 Active Alerts")
            
            critical_sensors = [k for k, v in health_dict.items() if v < 50 and k != 'Overall']
            warning_sensors = [k for k, v in health_dict.items() if 50 <= v < 70 and k != 'Overall']
            
            if critical_sensors:
                st.markdown(f"""
                <div class='critical-box'>
                    <h4 style='margin: 0; color: #dc3545;'>⛔ CRITICAL ALERTS</h4>
                    <p style='margin: 0.5rem 0 0 0;'>The following sensors require immediate attention:</p>
                    <ul style='margin: 0.5rem 0 0 1.5rem;'>
                        {''.join([f"<li><strong>{s}</strong>: {health_dict[s]:.1f}%</li>" for s in critical_sensors])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            if warning_sensors:
                st.markdown(f"""
                <div class='alert-box'>
                    <h4 style='margin: 0; color: #856404;'>⚠️ WARNING ALERTS</h4>
                    <p style='margin: 0.5rem 0 0 0;'>The following sensors show degradation:</p>
                    <ul style='margin: 0.5rem 0 0 1.5rem;'>
                        {''.join([f"<li><strong>{s}</strong>: {health_dict[s]:.1f}%</li>" for s in warning_sensors])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            if not critical_sensors and not warning_sensors:
                st.markdown("""
                <div class='info-box'>
                    <h4 style='margin: 0; color: #28a745;'>✅ All Systems Normal</h4>
                    <p style='margin: 0.5rem 0 0 0;'>All sensors are operating within healthy parameters.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization tabs
            st.markdown("### 📊 Detailed Analytics")
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Radar Chart", "Health Timeline"])
            
            with tab1:
                fig_bar = create_sensor_comparison_bar(health_dict)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab2:
                fig_radar = create_radar_chart(health_dict)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab3:
                fig_timeline = create_health_timeline(df, model, scaler_X, scaler_y)
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    # ========================================================================
    # MODE 2: HISTORICAL ANALYSIS
    # ========================================================================
    elif mode == "Historical Analysis":
        st.markdown('<h2 class="sub-header">📈 Historical Engine Analysis</h2>', 
                    unsafe_allow_html=True)
        
        # Engine selection
        engine_id = st.selectbox(
            "Select Engine for Analysis:",
            [f"Engine {i:03d}" for i in range(1, 11)]
        )
        
        # Generate data
        seed = int(engine_id.split()[1])
        df = generate_synthetic_engine_data(n_cycles=200, seed=seed)
        
        # Predict for all cycles
        sensor_features = ['temperature', 'pressure', 'vibration', 'rpm', 'fuel_flow', 'cycle']
        X = df[sensor_features].values
        X_scaled = scaler_X.transform(X)
        predictions_scaled = model.predict(X_scaled, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # Add predictions to dataframe
        df['predicted_temp_health'] = predictions[:, 0]
        df['predicted_pressure_health'] = predictions[:, 1]
        df['predicted_vibration_health'] = predictions[:, 2]
        df['predicted_rpm_health'] = predictions[:, 3]
        df['predicted_fuel_health'] = predictions[:, 4]
        df['predicted_overall_health'] = predictions[:, 5]
        
        # Summary statistics
        st.markdown("### 📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_health = df['predicted_overall_health'].mean()
            st.metric("Average Health", f"{avg_health:.1f}%", 
                     delta=f"{avg_health - 100:.1f}%")
        
        with col2:
            min_health = df['predicted_overall_health'].min()
            min_cycle = df.loc[df['predicted_overall_health'].idxmin(), 'cycle']
            st.metric("Minimum Health", f"{min_health:.1f}%",
                     delta=f"at cycle {int(min_cycle)}")
        
        with col3:
            # Find first warning
            warning_cycles = df[df['predicted_overall_health'] < 70]
            if len(warning_cycles) > 0:
                first_warning = int(warning_cycles.iloc[0]['cycle'])
                st.metric("First Warning", f"Cycle {first_warning}",
                         delta="Warning threshold")
            else:
                st.metric("First Warning", "None", delta="All healthy")
        
        with col4:
            # Find first critical
            critical_cycles = df[df['predicted_overall_health'] < 50]
            if len(critical_cycles) > 0:
                first_critical = int(critical_cycles.iloc[0]['cycle'])
                st.metric("First Critical", f"Cycle {first_critical}",
                         delta="Critical threshold")
            else:
                st.metric("First Critical", "None", delta="No critical")
        
        # Main timeline chart
        st.markdown("### 🕐 Health Degradation Timeline")
        fig_timeline = create_health_timeline(df, model, scaler_X, scaler_y)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Sensor correlation heatmap
        st.markdown("### 🔥 Sensor Health Correlation")
        health_cols = ['predicted_temp_health', 'predicted_pressure_health', 
                      'predicted_vibration_health', 'predicted_rpm_health', 
                      'predicted_fuel_health']
        corr_matrix = df[health_cols].corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Temp', 'Pressure', 'Vibration', 'RPM', 'Fuel'],
            y=['Temp', 'Pressure', 'Vibration', 'RPM', 'Fuel'],
            colorscale='RdYlGn',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig_heatmap.update_layout(
            title="Sensor Health Correlation Matrix",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Download data
        st.markdown("### 💾 Export Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Historical Data (CSV)",
            data=csv,
            file_name=f'{engine_id}_historical_data.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    # ========================================================================
    # MODE 3: MANUAL INPUT
    # ========================================================================
    else:  # Manual Input
        st.markdown('<h2 class="sub-header">⚙️ Manual Sensor Input</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <p style='margin: 0;'><strong>Instructions:</strong> Enter sensor readings manually to get real-time health predictions from the PINN model.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Sensor Readings")
            temperature = st.number_input("Temperature (°C)", 
                                         min_value=400.0, max_value=700.0, 
                                         value=550.0, step=1.0)
            pressure = st.number_input("Pressure (PSI)", 
                                       min_value=10.0, max_value=20.0, 
                                       value=14.7, step=0.1)
            vibration = st.number_input("Vibration (g)", 
                                        min_value=0.1, max_value=2.0, 
                                        value=0.5, step=0.01)
        
        with col2:
            st.markdown("#### ⚙️ Operational Parameters")
            rpm = st.number_input("RPM", 
                                 min_value=3000, max_value=7000, 
                                 value=5000, step=10)
            fuel_flow = st.number_input("Fuel Flow (kg/h)", 
                                        min_value=50.0, max_value=150.0, 
                                        value=100.0, step=1.0)
            cycle = st.number_input("Operating Cycle", 
                                   min_value=0, max_value=500, 
                                   value=100, step=1)
        
        st.markdown("---")
        
        if st.button("🔮 Predict Sensor Health", type="primary", use_container_width=True):
            # Create sensor data dictionary
            sensor_data = {
                'temperature': temperature,
                'pressure': pressure,
                'vibration': vibration,
                'rpm': rpm,
                'fuel_flow': fuel_flow,
                'cycle': cycle
            }
            
            # Predict
            with st.spinner('Analyzing sensor data with PINN model...'):
                health_dict = predict_sensor_health(model, scaler_X, scaler_y, sensor_data)
            
            if health_dict:
                st.markdown("---")
                
                # Overall health
                overall_health = health_dict['Overall']
                status, color, icon = get_status_and_color(overall_health)
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                            padding: 2rem; border-radius: 15px; border-left: 5px solid {color};'>
                    <h2 style='margin: 0; color: {color};'>{icon} Overall Engine Health: {overall_health:.1f}%</h2>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; color: {color};'>Status: <strong>{status}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🎯 Predicted Sensor Health")
                
                # Gauge charts
                cols = st.columns(3)
                sensor_list = ['Temperature', 'Pressure', 'Vibration', 'RPM', 'Fuel Flow']
                
                for idx, sensor in enumerate(sensor_list):
                    with cols[idx % 3]:
                        fig = create_gauge_chart(health_dict[sensor], sensor)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Maintenance Recommendations")
                
                critical_sensors = [k for k, v in health_dict.items() if v < 50 and k != 'Overall']
                warning_sensors = [k for k, v in health_dict.items() if 50 <= v < 70 and k != 'Overall']
                
                if critical_sensors:
                    st.error(f"⛔ **IMMEDIATE ACTION REQUIRED**: {', '.join(critical_sensors)}")
                    st.markdown("- Schedule immediate maintenance")
                    st.markdown("- Replace or repair affected sensors")
                    st.markdown("- Ground aircraft until repairs completed")
                
                if warning_sensors:
                    st.warning(f"⚠️ **SCHEDULE MAINTENANCE**: {', '.join(warning_sensors)}")
                    st.markdown("- Plan maintenance within next 10 cycles")
                    st.markdown("- Monitor closely during operation")
                    st.markdown("- Prepare replacement parts")
                
                if not critical_sensors and not warning_sensors:
                    st.success("✅ **All systems healthy** - Continue normal operations")
                    st.markdown("- Maintain regular inspection schedule")
                    st.markdown("- Continue monitoring sensor readings")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar = create_sensor_comparison_bar(health_dict)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    fig_radar = create_radar_chart(health_dict)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Export results
                st.markdown("### 💾 Export Results")
                results_df = pd.DataFrame([{
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'cycle': cycle,
                    'temperature': temperature,
                    'pressure': pressure,
                    'vibration': vibration,
                    'rpm': rpm,
                    'fuel_flow': fuel_flow,
                    **health_dict
                }])
                
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Report (CSV)",
                    data=csv_results,
                    file_name=f'prediction_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
        <p style='margin: 0;'><strong>Jet Engine Predictive Maintenance System</strong></p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Powered by Physics-Informed Neural Networks (PINN)</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>© 2025 - Advanced Aerospace Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 📖 About PINN")
        st.info("""
        **Physics-Informed Neural Networks** combine:
        - Deep learning capabilities
        - Physical laws and constraints
        - Real-time sensor data
        
        This ensures accurate and reliable predictions for critical aerospace systems.
        """)
        
        st.markdown("### 🎯 Health Thresholds")
        st.success("**Healthy**: ≥ 70%")
        st.warning("**Warning**: 50-70%")
        st.error("**Critical**: < 50%")
        
        st.markdown("### 📞 Support")
        st.markdown("""
        For technical support:
        - 📧 support@aziro.com
        - 📱 +1-800-JET-HELP
        - 🌐 www.aziro.com
        """)

if __name__ == "__main__":

    main()
