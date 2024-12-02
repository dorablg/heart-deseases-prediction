import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# Load model and dataset
model = joblib.load('model.pkl')
df = pd.read_csv('heart.csv')  # replace with your dataset path

# Helper function for model prediction
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Sidebar menu for navigation
page = st.sidebar.selectbox("Select Page", ["Visualization", "Model Results", "Prediction"])

# Page 1: Data Visualization
if page == "Visualization":
    st.title("Data Visualization")

    # Visualization Options
    chart = st.sidebar.selectbox("Select Chart Type", ["Distribution", "Correlation Matrix", "Feature Importance"])

    if chart == "Distribution":
        feature = st.sidebar.selectbox("Select Feature for Distribution", df.columns)
        fig = px.histogram(df, x=feature)  # More colorful palette
        st.plotly_chart(fig)

    elif chart == "Correlation Matrix":
        st.write("Correlation Matrix:")
        corr = df.corr()
        fig = px.imshow(corr, 
                        title="Correlation Matrix",
                        labels=dict(x="Features", y="Features"))
        st.plotly_chart(fig)

    elif chart == "Feature Importance":
        # Assuming you have a decision tree model and can extract feature importance
        feature_names = df.columns[:len(model.feature_importances_)]  # Ensure the number of features matches

        # Get feature importance from the model
        feature_importance = model.feature_importances_

        # Create a bar plot for feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_names, feature_importance, color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance))))  # Colorful bar chart
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

elif page == "Model Results":
    st.title("Model Results")

    # Decision Tree Image
    fig, ax = plt.subplots(figsize=(15,10))
    plot_tree(model, feature_names=df.columns[:len(model.feature_importances_)], class_names=['No Risk', 'High Risk'], filled=True, proportion=True, rounded=True, precision=2)
    st.pyplot(fig)

    # Accuracy and Loss
    X = df.drop('target', axis=1)  # Assuming 'target' is your target column name
    y = df['target']  # replace 'target' with your target column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy**: {accuracy:.4f}")
    st.write("The accuracy measures the proportion of correct predictions made by the model. "
             "It is the percentage of predictions where the predicted label matches the actual label.")

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Mean Squared Error (Loss)**: {mse:.4f}")
    st.write("The Mean Squared Error (MSE) calculates the average of the squared differences between predicted and actual values. "
             "Lower values indicate better model performance. For classification, it is less common than accuracy, but can still give insight.")

    # Confusion Matrix Plot
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['No Risk', 'High Risk'], yticklabels=['No Risk', 'High Risk'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Precision, Recall, F1-Score
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"**Precision**: {precision:.4f}")
    st.write("Precision calculates the proportion of positive predictions that were actually correct. "
             "In binary classification, this is important when we want to minimize false positives.")

    st.write(f"**Recall**: {recall:.4f}")
    st.write("Recall (also known as Sensitivity or True Positive Rate) measures the proportion of actual positives that were correctly identified by the model. "
             "It is crucial when we want to minimize false negatives.")

    st.write(f"**F1-Score**: {f1:.4f}")
    st.write("The F1-Score is the harmonic mean of precision and recall. It is a balanced measure when there is an uneven class distribution "
             "or when both precision and recall are important.")

    # Bar Plot for Predicted vs Actual Counts
    st.write("Predicted vs Actual Count Comparison:")

    actual_counts = y_test.value_counts()
    predicted_counts = pd.Series(y_pred).value_counts()

    comparison_df = pd.DataFrame({
        'Actual': actual_counts,
        'Predicted': predicted_counts
    }).fillna(0)

    # Plot comparison using bar plot
    comparison_df.plot(kind='bar', figsize=(10, 6), color=['orange', 'green'])
    plt.title('Comparison of Actual vs Predicted Class Counts')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(plt)
    
    # Loss (Mean Squared Error)
    st.write(f"**Mean Squared Error (Loss)**: {mse:.4f}")
    st.write("This value indicates the overall performance of the model. The smaller the error, the better the model.")

elif page == "Prediction":
    st.title("Make Prediction")

    # Input fields for new data
    st.write("Please enter the following details to predict the heart disease risk:")

    # Helper function to get user input and make a prediction
    def get_user_input():
        # Age (Numerical Input)
        age = st.number_input("Enter Age", min_value=1, max_value=120, value=25, help="Your age in years")

        # Sex (Dropdown Select Box)
        sex = st.selectbox("Select Gender", ["Male", "Female"], help="Select your gender")

        # Chest Pain Type (Dropdown Select Box)
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], help="Select type of chest pain")
        cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp = cp_dict[cp]

        # Resting Blood Pressure (Numerical Input)
        trestbps = st.number_input("Enter Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, help="Enter your resting blood pressure")

        # Serum Cholesterol (Numerical Input)
        chol = st.number_input("Enter Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200, help="Enter your serum cholesterol level")

        # Fasting Blood Sugar > 120 mg/dl (Yes/No)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], help="Indicate if your fasting blood sugar is greater than 120 mg/dl")
        fbs = 1 if fbs == "Yes" else 0

        # Resting Electrocardiographic Results (Dropdown Select Box)
        restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], help="Select the result of your resting electrocardiogram")
        restecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
        restecg = restecg_dict[restecg]

        # Maximum Heart Rate Achieved (Slider)
        thalach = st.slider("Maximum Heart Rate Achieved (bpm)", min_value=70, max_value=200, value=150, help="Enter the maximum heart rate you achieved during exercise")

        # Exercise Induced Angina (Yes/No)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Indicate if you experience chest pain during exercise")
        exang = 1 if exang == "Yes" else 0

        # Depression Induced by Exercise (Numerical Input)
        oldpeak = st.number_input("Depression Induced by Exercise (ST depression)", min_value=0.0, max_value=10.0, value=1.0, help="Enter the depression level induced by exercise")

        # Slope of Peak Exercise ST Segment (Dropdown Select Box)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"], help="Indicate the slope of the peak exercise ST segment")
        slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope = slope_dict[slope]

        # Number of Major Vessels Colored by Fluoroscopy (Dropdown Select Box)
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", ["0 vessels", "1 vessel", "2 vessels", "3 vessels"], help="Enter the number of major vessels colored by fluoroscopy")
        ca_dict = {"0 vessels": 0, "1 vessel": 1, "2 vessels": 2, "3 vessels": 3}
        ca = ca_dict[ca]

        # Thalassemia (Dropdown Select Box)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"], help="Select the type of thalassemia")
        thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
        thal = thal_dict[thal]

        # Prepare the input data as a 2D array
        input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        return input_data

    # Display title
    st.title("Heart Disease Prediction")

    # Create the form for inputs
    with st.form(key='input_form'):
        # Get user input
        input_data = get_user_input()

        # Button to generate prediction at the bottom of the form
        submit_button = st.form_submit_button(label='Generate Prediction')

    # Display the prediction when the button is clicked
    if submit_button:
        # Make the prediction (no scaling required)
        prediction = model.predict(input_data)
        prediction = True if prediction[0] == 1 else False  # Convert 0/1 to True/False

        # Display the prediction result in big font
        if prediction:
            st.markdown("<h2 style='text-align: center; color: red;'>High Risk of Heart Disease</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>No Risk of Heart Disease</h2>", unsafe_allow_html=True)
else: # If the model is not loaded, display a message
    st.title("there's an error predicting")