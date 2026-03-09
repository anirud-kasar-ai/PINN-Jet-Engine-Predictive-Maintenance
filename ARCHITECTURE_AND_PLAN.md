# Jet Engine Predictive Maintenance Using PINN
## Comprehensive Plan & Architecture Documentation

---

## 📋 Project Overview

This project implements a **Physics-Informed Neural Network (PINN)** based predictive maintenance system for jet engines. It combines deep learning with physics constraints to predict sensor health degradation and provide early maintenance alerts.

### Key Objectives:
- Predict engine sensor health degradation patterns
- Identify critical component failures before they occur
- Provide real-time monitoring and analytics dashboard
- Generate actionable maintenance recommendations

---

## 🏗️ HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│              (Streamlit Web Application)                     │
│  • Dashboard View  • Analytics  • Predictions  • Reports     │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼──────────┐          ┌──────────▼──────────┐
│  DATA INPUT LAYER │          │ MODEL INFERENCE    │
├──────────────────┤          ├────────────────────┤
│ • Sensor Reading  │          │ • PINN Model       │
│ • Real-time Data  │          │ • Predictions      │
│ • Historical Data │          │ • Health Scoring   │
└────────┬──────────┘          └────────┬───────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
        ┌───────────────▼──────────────┐
        │   DATA PROCESSING LAYER      │
        ├──────────────────────────────┤
        │ • Feature Engineering        │
        │ • Data Normalization (MinMax)│
        │ • Scaler Transformations     │
        │ • Anomaly Detection          │
        └───────────────┬──────────────┘
                        │
        ┌───────────────▼──────────────┐
        │   STORAGE & PERSISTENCE      │
        ├──────────────────────────────┤
        │ • Model Files (.h5)          │
        │ • Scaler Objects (.pkl)      │
        │ • Historical Data (CSV/DB)   │
        │ • Configuration Files        │
        └──────────────────────────────┘
```

---

## 🔧 LOW-LEVEL ARCHITECTURE

### Component-Level Diagram

```
APPLICATION FLOW:
┌─────────────────────────────────────────────────────────────────┐
│  main() - Streamlit App Entry Point                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼────────┐
│ load_model_and │          │ generate_synth  │
│ _scalers()     │          │ _engine_data()  │
├────────────────┤          ├─────────────────┤
│ • Load .h5 model          │ • Random cycles │
│ • Load scalers from .pkl  │ • Degradation   │
│ • Error handling          │ • Health scores │
│ • Caching with @cache    │ • Return pandas │
└────────┬────────┘          └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
                     │
          ┌──────────▼──────────┐
          │ predict_sensor_     │
          │ health()            │
          ├─────────────────────┤
          │ • Input: sensor data│
          │ • Scale features    │
          │ • Model prediction  │
          │ • Inverse transform │
          │ • Return health %   │
          └──────────┬──────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
┌───────▼──────────┐      ┌───────▼──────────┐
│ create_gauge_    │      │ create_health_   │
│ chart()          │      │ timeline()       │
├──────────────────┤      ├──────────────────┤
│ • Gauge indicator        │ • Line charts    │
│ • Health-based colors    │ • Multi-sensor   │
│ • Status symbols         │ • Threshold lines│
└──────────────────┘      └──────────────────┘
```

### Data Flow Model

```
SENSOR DATA PROCESSING PIPELINE:

1. INPUT STAGE
   ├─ Temperature (°F)
   ├─ Pressure (PSI)
   ├─ Vibration (mm/s)
   ├─ RPM (Revolutions/min)
   ├─ Fuel Flow (lb/hr)
   └─ Operating Cycle (count)

2. PREPROCESSING
   ├─ Validation & Cleaning
   ├─ MinMaxScaler Transformation (0-1 range)
   └─ Feature Array Creation [6 features]

3. MODEL INFERENCE (PINN)
   ├─ Physics Constraints Applied
   ├─ Neural Network Processing
   └─ 6-Output Prediction Layer

4. POST-PROCESSING
   ├─ Inverse Scaling
   ├─ Value Clipping (0-100%)
   ├─ Health Score Calculation
   └─ Status Classification

5. VISUALIZATION & OUTPUT
   ├─ Gauge Charts (per sensor)
   ├─ Timeline Charts
   ├─ Status Indicators
   └─ Alerts & Recommendations
```

---

## 📊 System Components

### 1. **Frontend (User Interface)**
- **Framework**: Streamlit
- **Features**:
  - Real-time dashboard
  - Interactive gauge charts
  - Health timeline visualization
  - Sensor health metrics
  - Status indicators (Healthy/Warning/Critical)
  - Responsive design with custom CSS

### 2. **Core Model (PINN)**
- **Architecture**: Physics-Informed Neural Network
- **Input Features**: 6 (Temperature, Pressure, Vibration, RPM, Fuel Flow, Cycle)
- **Output**: 6 health scores (sensor-specific + overall)
- **Constraints**: Physics laws embedded during training
- **File**: `jet_engine_pinn_model.h5`

### 3. **Data Processing**
- **Scaler Type**: MinMaxScaler (sklearn)
- **Purpose**: Normalize features to [0, 1] range
- **Files**: 
  - `scaler_X.pkl` (input scaler)
  - `scaler_y.pkl` (output scaler)
- **Operations**: 
  - Transform (raw → normalized)
  - Inverse Transform (normalized → raw)

### 4. **Data Generation**
- **Synthetic Data Generator**: `generate_synthetic_engine_data()`
- **Purpose**: Demo and testing
- **Parameters**:
  - Configurable cycles (default: 200)
  - Random degradation patterns
  - Health score simulation

### 5. **Prediction Engine**
- **Function**: `predict_sensor_health()`
- **Process**:
  1. Prepare input features
  2. Scale using scaler_X
  3. Model inference
  4. Inverse scale using scaler_y
  5. Clip to valid range [0, 100]
  6. Return health dictionary

---

## 🔄 Data Schema

### Input Features (X):
```
├─ Temperature: 500-650°F
├─ Pressure: 10-18 PSI
├─ Vibration: 0.3-0.8 mm/s
├─ RPM: 4500-5500
├─ Fuel Flow: 80-120 lb/hr
└─ Operating Cycle: 0-n (integer)
```

### Output Predictions (Y):
```
├─ Temperature Health: 0-100%
├─ Pressure Health: 0-100%
├─ Vibration Health: 0-100%
├─ RPM Health: 0-100%
├─ Fuel Flow Health: 0-100%
└─ Overall Health: 0-100%
```

### Health Status Thresholds:
```
├─ HEALTHY: >= 70%  (Green, ✓)
├─ WARNING: 50-70%  (Yellow, ⚠)
└─ CRITICAL: < 50%  (Red, ✗)
```

---

## 🚀 Execution Flow

```
START
  │
  ├─► Page Configuration
  │    └─► CSS Styling
  │
  ├─► Load Model & Scalers
  │    ├─► Check files exist
  │    ├─► Load .h5 model
  │    ├─► Load .pkl scalers
  │    └─► Cache results
  │
  ├─► User Input Selection
  │    └─► Real data vs Synthetic
  │
  ├─► Data Acquisition
  │    ├─► Generate synthetic OR
  │    └─► Upload/Input real data
  │
  ├─► Data Prediction
  │    ├─► Preprocess features
  │    ├─► Model inference
  │    └─► Generate predictions
  │
  ├─► Visualization
  │    ├─► Create gauge charts
  │    ├─► Create timeline charts
  │    ├─► Display metrics
  │    └─► Show alerts
  │
  └─► END (await user interaction)
```

---

## 📁 File Structure

```
PINN/
├── app.py                          # Main Streamlit application (842 lines)
├── jet_engine_pinn_model.h5        # Trained PINN model (binary)
├── scaler_X.pkl                    # Input feature scaler
├── scaler_y.pkl                    # Output prediction scaler
├── PINN (1).ipynb                  # Jupyter notebook (training/analysis)
├── requirements.txt                # Python dependencies
├── README.md                        # Project documentation
├── ARCHITECTURE_AND_PLAN.md         # This file
└── .git/                            # GitHub repository (after setup)
```

---

## 🔌 Dependencies & Technologies

### Core Libraries:
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities (MinMaxScaler)
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations

### Supporting Libraries:
- **Joblib**: Model serialization
- **Matplotlib**: Static plotting
- **Pickle**: Object serialization

---

## 🎯 Key Features

1. **Real-time Health Monitoring**
   - Continuous sensor tracking
   - Multi-sensor health scores
   - Overall engine health calculation

2. **Predictive Degradation Analysis**
   - Historical trend analysis
   - Future failure prediction
   - Degradation rate estimation

3. **Intelligent Alerting**
   - Threshold-based notifications
   - Status classification
   - Visual indicators

4. **Interactive Dashboard**
   - Gauge charts for each sensor
   - Timeline visualization
   - Metric cards with KPIs
   - Professional styling

5. **Data Flexibility**
   - Synthetic data generation for demos
   - Support for real sensor data
   - Configurable parameters

---

## 🔐 Model Performance Metrics

The PINN model provides:
- **6 Output Predictions**: Granular health for each sensor + overall
- **Physics Constraints**: Embedded physical laws during training
- **Accuracy Range**: Optimized through physics-informed training
- **Inference Speed**: Real-time predictions (<100ms)

---

## 🛠️ Development Roadmap

### Phase 1: Core Implementation ✓
- [x] PINN model training
- [x] Streamlit dashboard development
- [x] Visualization components
- [x] Synthetic data generation

### Phase 2: Enhancement
- [ ] Real database integration
- [ ] Historical data storage
- [ ] API endpoints for external systems
- [ ] Advanced anomaly detection

### Phase 3: Production
- [ ] Containerization (Docker)
- [ ] Cloud deployment
- [ ] Scalability improvements
- [ ] Advanced logging & monitoring

---

## 📝 Usage Instructions

### Setup:
```bash
pip install -r requirements.txt
```

### Run Application:
```bash
streamlit run app.py
```

### Access Dashboard:
```
http://localhost:8501
```

---

## 🔗 Integration Points

- **Model Input**: Raw sensor readings
- **Model Output**: Health percentages
- **UI Output**: Interactive visualizations & alerts
- **Storage**: Pickle-based model persistence

---

## 📞 Support & Maintenance

- **Model Retraining**: Quarterly with new data
- **Scaler Calibration**: When sensor ranges change
- **Dashboard Updates**: Continuous improvement
- **Performance Monitoring**: Real-time tracking

---

## 📄 License & Attribution

This project uses Physics-Informed Neural Networks for jet engine predictive maintenance.

---

*Last Updated: January 16, 2026*
*Architecture Version: 1.0*
