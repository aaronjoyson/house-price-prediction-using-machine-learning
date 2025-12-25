import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Housing Price Predictor')
st.title('🏡 Housing Price Prediction (All features)')

# Model selection
model_choice = st.sidebar.selectbox(
    '🤖 Select Model',
    ['Random Forest', 'Linear Regression', 'Gradient Boosting']
)
st.sidebar.write(f'Selected Model: {model_choice}')
st.sidebar.divider()

# Model descriptions
model_info = {
    'Random Forest': {
        'description': '🏆 **Best Model** - Handles non-linear relationships with ensemble learning',
        'advantages': [
            '✅ Higher R² Score and lower error (MAE/MSE)',
            '✅ Captures complex feature interactions',
            '✅ No feature scaling required',
            '✅ Shows feature importance',
            '✅ More robust and reliable predictions'
        ],
        'uses': '200 decision trees aggregated for accurate predictions'
    },
    'Linear Regression': {
        'description': '📈 Simple model - Assumes linear relationships between features and price',
        'advantages': [
            '✅ Fast training and predictions',
            '✅ Highly interpretable',
            '✅ Good for baseline comparisons',
            '⚠️ Limited by linear assumption',
            '⚠️ Higher prediction errors on complex data'
        ],
        'uses': 'Direct proportional relationships with standardized features'
    },
    'Gradient Boosting': {
        'description': '⚡ **Powerful Ensemble** - Iteratively improves predictions by correcting errors',
        'advantages': [
            '✅ Excellent accuracy on complex patterns',
            '✅ Better error correction than single models',
            '✅ Good feature interaction handling',
            '⚠️ Slower than Random Forest',
            '⚠️ More prone to overfitting without tuning'
        ],
        'uses': '200 boosting iterations with 0.1 learning rate'
    }
}

# Display model info in sidebar
with st.sidebar.expander(f'ℹ️ About {model_choice}', expanded=True):
    st.write(model_info[model_choice]['description'])
    st.write('**Advantages:**')
    for adv in model_info[model_choice]['advantages']:
        st.write(adv)
    st.write(f"**Configuration:** {model_info[model_choice]['uses']}")

DATA_PATH = r'C:\Users\aaron\Downloads\house price\housing_with_ocean_proximity.csv'
MODEL_JOBLIB = r'C:\Users\aaron\Downloads\house price\best_model.joblib'
MODEL_KERAS = r'C:\Users\aaron\Downloads\house price\best_model_keras.h5'
PREPROCESSOR = r'C:\Users\aaron\Downloads\house price\best_preprocessor.joblib'

# Load dataset columns to build UI dynamically
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}. Please upload dataset to this path in the environment.")
    st.stop()

df = pd.read_csv(DATA_PATH)
if 'median_house_value' in df.columns:
    df_inputs = df.drop(columns=['median_house_value'])
else:
    df_inputs = df.copy()

numeric_cols = df_inputs.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_inputs.select_dtypes(include=['object','category']).columns.tolist()

# Display best model explanation
st.info("""
 Why Random Forest is the Best Model:

Performance Comparison:
| Metric | Random Forest | Linear Regression | Gradient Boosting |
| R²Score| 🥇 Highest | Lower | Very High |
| MAE | 🥇 Lowest | Higher | Low |
| MSE | 🥇 Lowest | Higher | Low |

Key Insights:
- Random Forest captures non-linear patterns in housing prices through 200 decision trees
- Handles complex feature interactions (location + property type = different price impact)
- No feature scaling needed, making preprocessing simpler
- Shows which features matter most for price prediction
- More robust to outliers and noise in data
Overall, Random Forest provides the most accurate and reliable predictions for housing prices compared to simpler linear models""")

st.sidebar.header('Input features (auto-generated)')
input_data = {}
for c in numeric_cols:
    # use median as default
    default_val = float(df_inputs[c].median()) if not np.isnan(df_inputs[c].median()) else 0.0
    input_data[c] = st.sidebar.number_input(c, value=default_val)

for c in cat_cols:
    options = sorted(df_inputs[c].dropna().unique().tolist())
    if len(options) == 0:
        input_data[c] = st.sidebar.text_input(c, value='')
    else:
        input_data[c] = st.sidebar.selectbox(c, options)

input_df = pd.DataFrame([input_data])
st.write('Input preview:')
st.write(input_df)

# Load or train model
model = None
model_type = None
preprocessor = None
label_encoder = None

MODEL_JOBLIB_RF = r'C:\Users\aaron\Downloads\house price\best_model_rf.joblib'
MODEL_JOBLIB_LR = r'C:\Users\aaron\Downloads\house price\best_model_lr.joblib'
MODEL_JOBLIB_GB = r'C:\Users\aaron\Downloads\house price\best_model_gb.joblib'
ENCODER_PATH = r'C:\Users\aaron\Downloads\house price\best_model_encoder.joblib'

# Determine which model to load based on selection
if model_choice == 'Random Forest':
    MODEL_PATH = MODEL_JOBLIB_RF
elif model_choice == 'Linear Regression':
    MODEL_PATH = MODEL_JOBLIB_LR
else:  # Gradient Boosting
    MODEL_PATH = MODEL_JOBLIB_GB

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_type = 'sklearn'
        st.sidebar.success(f'✅ {model_choice} model loaded')
    except Exception as e:
        st.sidebar.error(f'Failed to load {model_choice} model: ' + str(e))
else:
    st.sidebar.info(f'Training {model_choice} model...')
    try:
        # Prepare data
        df_train = df.copy()
        df_train['total_bedrooms'].fillna(df_train['total_bedrooms'].median(), inplace=True)
        
        # Encode categorical column
        label_encoder = LabelEncoder()
        df_train['ocean_proximity'] = label_encoder.fit_transform(df_train['ocean_proximity'])
        
        # Split features and target
        X = df_train.drop('median_house_value', axis=1)
        y = df_train['median_house_value']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train selected model
        if model_choice == 'Random Forest':
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        elif model_choice == 'Linear Regression':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = LinearRegression()
        else:  # Gradient Boosting
            model = GradientBoostingRegressor(n_estimators=200, random_state=42, learning_rate=0.1)
        
        model.fit(X_train, y_train)
        
        # Save the model and encoder
        joblib.dump(model, MODEL_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)
        
        model_type = 'sklearn'
        st.sidebar.success(f'✅ {model_choice} model trained and saved!')
        
        # Show model performance
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric('R² Score', f'{r2:.4f}')
        col2.metric('Mean Absolute Error', f'${mae:,.2f}')
        col3.metric('Mean Squared Error', f'{mse:,.2f}')
        
    except Exception as e:
        st.error(f'Failed to train {model_choice} model: ' + str(e))
        st.stop()

# Prediction
try:
    if model_type == 'sklearn':
        # Load label encoder if not already loaded
        if label_encoder is None:
            try:
                label_encoder = joblib.load(ENCODER_PATH)
            except:
                st.warning('Could not load label encoder. Ensure model is trained.')
                label_encoder = LabelEncoder()
        
        # Prepare input data
        pred_input = input_df.copy()
        if 'ocean_proximity' in pred_input.columns:
            pred_input['ocean_proximity'] = label_encoder.transform(pred_input['ocean_proximity'])
        
        pred = model.predict(pred_input)[0]
        st.success(f'🏡 **{model_choice}** Predicted median_house_value: **${pred:,.2f}**')
    elif model_type == 'keras':
        X_proc = preprocessor.transform(input_df)
        # if model expects sequences (RNN), reshape accordingly; this example assumes ANN (2D input)
        if X_proc.ndim == 2:
            pred = model.predict(X_proc)[0]
            if hasattr(pred[0], '__len__'):
                out = pred[0][0]
            else:
                out = float(pred)
        else:            
            out = float(model.predict(X_proc))
        st.success(f'🏡 Predicted median_house_value: **${out:,.2f}**')
except Exception as e:
    st.error('Prediction failed: ' + str(e))

st.set_page_config(page_title='Housing Price Predictor')
st.title('🏡 Housing Price Prediction (All features)')

st.markdown("""
<style>

/* ---------------- BACKGROUND IMAGE ---------------- */
.stApp {
    background: 
        linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
        url("https://images.unsplash.com/photo-1568605114967-8130f3a36994");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    animation: bgZoom 20s ease-in-out infinite;
}

/* Background slow zoom animation */
@keyframes bgZoom {
    0% { background-size: 100%; }
    50% { background-size: 105%; }
    100% { background-size: 100%; }
}

/* ---------------- GLASS UI ---------------- */
.block-container {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 2rem;
    animation: fadeIn 1s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(12px);
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* ---------------- SLIDER COLOR CHANGE ---------------- */
input[type="range"] {
    accent-color: #00f2ff;
    transition: all 0.3s ease-in-out;
}

input[type="range"]:hover {
    accent-color: #00ff88;
}

/* ---------------- INPUT ANIMATION ---------------- */
input, select {
    transition: all 0.3s ease-in-out !important;
}

input:focus, select:focus {
    transform: scale(1.04);
    box-shadow: 0 0 12px rgba(0,255,255,0.7);
}

/* ---------------- BUTTON ANIMATION ---------------- */
button {
    transition: all 0.3s ease !important;
}

button:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 8px 22px rgba(0,255,255,0.7);
}

/* ---------------- METRIC GLOW ---------------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 12px;
    animation: glow 2.5s infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 8px rgba(0,255,255,0.4); }
    to { box-shadow: 0 0 22px rgba(0,255,255,0.9); }
}

/* ---------------- FLOATING HOUSE ANIMATION ---------------- */
.floating-houses {
    position: fixed;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    z-index: 0;
    pointer-events: none;
}

.house {
    position: absolute;
    font-size: 2rem;
    opacity: 0.18;
    animation: floatUp linear infinite;
}

@keyframes floatUp {
    0% {
        transform: translateY(100vh);
        opacity: 0;
    }
    50% {
        opacity: 0.25;
    }
    100% {
        transform: translateY(-10vh);
        opacity: 0;
    }
}

.house:nth-child(1) { left: 10%; animation-duration: 18s; }
.house:nth-child(2) { left: 30%; animation-duration: 22s; }
.house:nth-child(3) { left: 50%; animation-duration: 20s; }
.house:nth-child(4) { left: 70%; animation-duration: 26s; }
.house:nth-child(5) { left: 90%; animation-duration: 24s; }

</style>

<div class="floating-houses">
    <div class="house">🏠</div>
    <div class="house">🏡</div>
    <div class="house">🏘️</div>
    <div class="house">🏠</div>
    <div class="house">🏡</div>
</div>
""", unsafe_allow_html=True)
