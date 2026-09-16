# 🔭 ChurnVision

ChurnVision is an end-to-end Machine Learning project that predicts the probability of a telecom customer churning (leaving the service). It includes data cleaning and modeling Jupyter notebooks, and a fully functional Streamlit web application that lets you input customer details and instantly get a churn risk prediction.

> **Note:** This project was built as the **NTI Final Project** by our team:
> - Mohamed Elmhlawy
> - Zeyad Elsheikh
> - Ali Rizk
> - Abanob Salama

## 🚀 Features
- **Machine Learning Model:** Uses a robust Random Forest Classifier trained on telecom customer data.
- **Interactive UI:** Built with Streamlit for a seamless, fast, and responsive user experience.
- **Bilingual Support:** Full English and Arabic language support with RTL/LTR dynamic styling.
- **Actionable Insights:** Not just a percentage! The app interprets the prediction and suggests actionable advice (e.g., "High churn risk — recommend immediate action").
- **Reproducible Research:** Includes Jupyter notebooks documenting the entire Data Cleaning, Exploratory Data Analysis (EDA), and Modeling process.

## 📂 Project Structure
```
ChurnVision/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
│
├── utils/                  # Helper modules for the app
│   ├── translations.py     # English & Arabic text dictionaries
│   └── styles.py           # Custom CSS for the UI
│
├── data/
│   ├── raw/                # Original raw dataset
│   ├── processed/          # Cleaned dataset ready for modeling
│   └── models/             # Pickled ML models, scaler, and encoder
│
└── notebooks/
    ├── DataCleaning.ipynb  # Data wrangling and preprocessing steps
    └── Models.ipynb        # Model training, tuning, and evaluation
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/ChurnVision.git
   cd ChurnVision
   ```

2. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the App

To launch the Streamlit app locally, run the following command in your terminal:
```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

## 🧠 Modeling Details
- **Algorithm:** Random Forest Classifier (along with a baseline Logistic Regression model).
- **Preprocessing:** 
  - Continuous variables (Tenure, MonthlyCharges) are scaled using `StandardScaler`.
  - Categorical variables are encoded using `OneHotEncoder`.
- **Target:** `Churn` (Yes / No).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).

