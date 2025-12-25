🏡 Housing Price Prediction using Machine Learning

This project is an interactive Housing Price Prediction Web Application built using Machine Learning and Streamlit.
It predicts the median house value based on multiple housing and location-related features and allows users to compare predictions across different ML models.

🚀 Features

🔮 Predict house prices using real-world housing data

🤖 Model selection:

Random Forest Regressor (Best Performing)

Linear Regression

Gradient Boosting Regressor

🎛️ Auto-generated input sliders and dropdowns

🎨 Modern UI with:

Background image

Animated effects

Color-changing sliders

📊 Model performance metrics (R², MAE, MSE)

💾 Automatic model saving and loading

🧠 Machine Learning Models Used
Model	Description
Random Forest	Best accuracy, handles non-linear relationships
Linear Regression	Simple baseline model
Gradient Boosting	Strong ensemble with error correction

Why Random Forest?

Highest R² score

Lowest prediction error

Robust to outliers

Captures complex feature interactions

📂 Project Structure
house_price_prediction/
│
├── app.py
├── data/
   └── housing_with_ocean_proximity.csv

├── models/
    ├── best_model_rf.joblib
    ├── best_model_lr.joblib
    ├── best_model_gb.joblib
    └── best_model_encoder.joblib

├── requirements.txt
└── README.md

🛠️ Technologies Used

Python

Streamlit

Pandas

NumPy

Scikit-learn

Joblib

📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

🧪 Dataset

Source: California Housing Dataset

Features Include:

Median income

Housing age

Total rooms & bedrooms

Population

Households

Ocean proximity

🎯 How It Works

User selects a machine learning model

Inputs housing features using sliders/dropdowns

App loads or trains the selected model

Model predicts the median house value

Prediction is displayed with smooth UI animations
