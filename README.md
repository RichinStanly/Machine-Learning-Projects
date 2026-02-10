# 💳 Credit Card Fraud Detection System

## 📌 Overview
This project utilizes **Unsupervised Deep Learning** (Self-Organizing Maps - SOM) to identify potential fraud in credit card applications. Unlike traditional supervised models that require labeled "fraud" data, this system identifies **outliers** in the application pattern to flag suspicious customers automatically.

## 📂 Project Structure
* **`credit.ipynb`**: The core research notebook. It performs data preprocessing, trains the SOM to identify outliers, and visualizes the "Winning Nodes" (fraud patterns).
* **`app.py`**: A deployment script (likely using Streamlit) to visualize the results or predict new applications.
* **`Credit_Card_Applications.csv`**: The dataset containing anonymized application attributes.

## 🧠 The Logic (How it works)
1.  **Data Preprocessing:** Scales all attributes (Age, Income, Credit Score, etc.) to a 0-1 range using MinMax Scaling.
2.  **Unsupervised Learning (SOM):** The model builds a 2D map of the data. Customers who don't fit the standard "approved" patterns are pushed to the edges or specific "outlier nodes."
3.  **Fraud Detection:** The algorithm identifies the Mean Inter-Neuron Distance (MID). Higher MID values indicate significant deviation from the norm—flagging these specific application IDs as potential fraud.

## 🛠️ Tech Stack
* **Python 3.x**
* **MiniSom:** For implementing Self-Organizing Maps.
* **Scikit-Learn:** For data scaling and preprocessing.
* **Pandas & NumPy:** For data manipulation.
* **Matplotlib:** For plotting the SOM grid and identifying outliers.

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Analysis Notebook:**
    Open `credit.ipynb` in Jupyter Notebook to see the training process and visualization.
3.  **Run the App (Optional):**
    ```bash
    streamlit run app.py
    ```

## 📈 Results
The model successfully segments customers into "Approved" and "High Risk" clusters, providing a list of Application IDs that require manual review for potential fraud.