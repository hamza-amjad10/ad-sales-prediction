Ad Sales Prediction Web App

This project predicts Sales based on the amount spent on TV, Radio, and Newspaper advertising.
The model is built using Linear Regression and deployed as an interactive Streamlit Web App.

🚀 Features

End-to-end Machine Learning pipeline (data preprocessing → model training → evaluation → saving model)

Predict sales using TV, Radio, Newspaper ad budgets

Interactive Streamlit web app for user input and predictions

Model and scaler stored with Joblib for easy reusability

🛠️ Tech Stack

Python

Pandas, Numpy → Data processing

Scikit-learn → Linear Regression, Scaling, Model Evaluation

Joblib → Save/Load model

Streamlit → Web app

📂 Project Structure
├── app.py                  # Streamlit web app
├── train_model.py          # Model training & evaluation script
├── ad_sales_model.pkl      # Trained Linear Regression model
├── ad_sales_scaler.pkl     # StandardScaler object
├── 6 advertising.csv       # Dataset
└── README.md               # Project documentation

📊 Dataset

The dataset (6 advertising.csv) contains advertising spending and sales:

TV	Radio	Newspaper	Sales
230.1	37.8	69.2	22.1
44.5	39.3	45.1	10.4
17.2	45.9	69.3	9.3
...	...	...	...
📈 Model Training

Load and split dataset (80% training, 20% testing)

Scale features using StandardScaler

Train Linear Regression model

Evaluate with:

R² Score → Goodness of fit

Mean Squared Error (MSE) → Prediction error

Model & scaler are saved as .pkl files for future predictions.

▶️ How to Run
Clone the Repository
git clone https://github.com/your-username/ad-sales-prediction.git
cd ad-sales-prediction


Run the Streamlit App
streamlit run app.py

🖥️ Example Prediction

Input:

TV = 230.1
Radio = 37.8
Newspaper = 69.2


Output:

Predicted Sales: 21.37


⚡ Built with ❤️ using Python & Streamlit
