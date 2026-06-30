import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Result Analyzer", layout="wide")
st.title("🎓 Student Performance Predictor & Analyzer")

# --- 1. DATA UPLOAD & OVERVIEW ---
st.header("1. Upload Student Dataset")
st.write("Upload your dataset to analyze results and train the prediction model.")

uploaded_file = st.file_uploader("Choose a CSV file (e.g., students.csv)", type="csv")

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df)

    # --- 2. PRACTICAL TASK: GRADES & TOP PERFORMER ---
    st.header("2. Student Result Analyzer")

    if 'Previous_Scores' in df.columns and 'Name' in df.columns:
        top_student = df.loc[df['Previous_Scores'].idxmax()]
        st.info(f"🏆 **Top Performer:** {top_student['Name']} with a score of {top_student['Previous_Scores']}!")
    else:
        st.warning("Ensure your CSV has 'Name' and 'Previous_Scores' columns for the Top Performer feature.")

    # --- 3. DATA VISUALIZATION ---
    st.header("3. Data Visualizations")
    
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Attendance vs Scores")
        if 'Attendance_Percentage' in df.columns and 'Previous_Scores' in df.columns:
            fig, ax = plt.subplots()
            ax.scatter(df['Attendance_Percentage'], df['Previous_Scores'], color='teal')
            ax.set_xlabel("Attendance (%)")
            ax.set_ylabel("Previous Scores")
            st.pyplot(fig)
        else:
            st.warning("Missing 'Attendance_Percentage' or 'Previous_Scores'.")

    with col_chart2:
        st.subheader("Student Performance Chart")
        if 'Name' in df.columns and 'Previous_Scores' in df.columns:
            chart_data = df.set_index('Name')['Previous_Scores']
            st.bar_chart(chart_data)
        else:
            st.warning("Missing 'Name' or 'Previous_Scores' for the bar chart.")

    # --- 4. MACHINE LEARNING: PERFORMANCE PREDICTOR ---
    st.header("4. Predict Future Performance")
    st.write("Train the Scikit-Learn model using the uploaded data to predict future performance.")

    # Added 'Internal_Marks' to the list of required columns
    required_ml_columns = ['Attendance_Percentage', 'Study_Hours_Per_Week', 'Internal_Marks', 'Performance_Category']
    if all(col in df.columns for col in required_ml_columns):
        
        le = LabelEncoder()
        df['Category_Encoded'] = le.fit_transform(df['Performance_Category'])

        # Included 'Internal_Marks' in the feature set (X)
        X = df[['Attendance_Percentage', 'Study_Hours_Per_Week', 'Internal_Marks']]
        y = df['Category_Encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Created 3 columns for the UI inputs instead of 2
        col1, col2, col3 = st.columns(3)
        with col1:
            input_attendance = st.number_input("Enter Attendance (%)", min_value=0, max_value=100, value=80)
        with col2:
            input_study = st.number_input("Enter Study Hours/Week", min_value=0, max_value=50, value=10)
        with col3:
            input_internal = st.number_input("Enter Internal Marks", min_value=0, max_value=100, value=75) # New Input

        if st.button("Predict Performance"):
            # Passed the internal marks variable to the prediction function
            prediction_encoded = clf.predict([[input_attendance, input_study, input_internal]])
            prediction_label = le.inverse_transform(prediction_encoded)[0]
            
            st.success(f"🔮 Predicted Performance Category: **{prediction_label}**")
    else:
        st.error(f"Cannot train model. The uploaded CSV must contain these columns: {', '.join(required_ml_columns)}")

else:
    st.info("Waiting for a CSV file to be uploaded...")
    st.stop()