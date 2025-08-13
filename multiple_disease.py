import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def kidney():
    # Loading dataset
    data= pd.read_csv("kidney_disease - kidney_disease.csv")
    df=data.copy() #copy the original dataset for preprocessings

    # take necessary columns for preprocessing and model training
    important_cols = ['sc', 'bu', 'hemo', 'al', 'sg', 'bgr', 'pcv', 'htn', 'dm', 'age','classification']
    df_imp=df[important_cols]

    # renaming(optional)
    df_imp=df_imp.rename(columns={
        'sc': 'serum_creatinine',
        'bu': 'blood_urea',
        'hemo': 'hemoglobin',
        'al': 'albumin',
        'sg': 'specific_gravity',
        'bgr': 'blood_glucose_random',
        'pcv': 'packed_cell_volume',
        'htn': 'hypertension',
        'dm': 'diabetes_mellitus',
        'age': 'age'
    })

    # type casting
    df_imp['packed_cell_volume'] = pd.to_numeric(df_imp['packed_cell_volume'], errors='coerce')


    # split numeric & object columns
    num_cols = df_imp.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df_imp.select_dtypes(include=['object', 'category']).columns

    # fill with median for numeric columns
    df_imp[num_cols] = df_imp[num_cols].fillna(df_imp[num_cols].median())

    # fill with mode for object columns
    for col in cat_cols:
        df_imp[col]=df_imp[col].fillna(df_imp[col].mode()[0])

    # Convert text to numbers
    for col in df_imp.columns:
        if df_imp[col].dtype == 'object':
            df_imp[col] = df_imp[col].astype('category').cat.codes

   # === Visualization Before Outlier Handling ===
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_imp.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Box plots for numeric features
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df_imp[col])
        plt.title(f"Box Plot of {col}")
        plt.show()


    # for outliers
    # Make a copy to store the capped data
    df_kidney=df_imp.copy()

    # Loop through each numeric column and clip individually
    for col in df_kidney:
        Q1 = df_kidney[col].quantile(0.25)
        Q3 = df_kidney[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Clip only that column, not the whole DataFrame
        df_kidney[col] = df_kidney[col].clip(lower=lower_bound, upper=upper_bound)

    print("Outliers capped. New shape:", df_kidney.shape)

    #splitting x and y
    x=df_kidney.drop(['classification'],axis=1)
    y=df_kidney['classification']
    
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
    
    import numpy as np
    import pickle
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    
    class_counts = np.bincount(y.astype(int)) #np.bincount only works if the values are integers starting from 0.
    imbalance_ratio = max(class_counts) / min(class_counts)

    print("Class counts:", dict(enumerate(class_counts)))
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")


    if imbalance_ratio > 1.5:
        main_metric = "F1"
        print("Dataset is imbalanced → using F1-score as main metric")
    else:
        main_metric = "Accuracy"
        print("Dataset is balanced → using Accuracy as main metric")

    # Models to test
    models = {
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier()
    }

    results = []

    trained_models = {}  # Store trained model objects


    # Train & evaluate each model
    for name, model in models.items():
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)
        trained_models[name] = model  # Store the trained model
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(ytest, y_pred),
            "Precision": precision_score(ytest, y_pred, average="weighted"),
            "Recall": recall_score(ytest, y_pred, average="weighted"),
            "F1": f1_score(ytest, y_pred, average="weighted")
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by chosen main metric
    df_results = df_results.sort_values(by=main_metric, ascending=False)
    print("\n Model Performance Comparison:")
    print(df_results)

    # Best model
    best_model_name = df_results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n Best model based on {main_metric}: {best_model_name}")


    # Save the best model to a pickle file
    with open("best_model_kidney.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f" Best model saved as 'best_model.pkl'")
    
    return best_model, df_kidney

#################################################################################################

def liver():

    import pandas as pd

    data= pd.read_csv("indian_liver_patient - indian_liver_patient.csv")
    df=data.copy()

    df['Albumin_and_Globulin_Ratio']= df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

   # === Visualization Before Outlier Handling ===
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Box plots for numeric features
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot of {col}")
        plt.show()

    df_liver= df.copy()

    for col in df_liver: 
        Q1 = df_liver[col].quantile(0.25)
        Q3 = df_liver[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

    
        df_liver[col] = df_liver[col].clip(lower=lower_bound, upper=upper_bound)

    print("Outliers capped. New shape:", df_liver.shape)


    x = df_liver.drop(columns=["Dataset","Gender"])
    y = df_liver["Dataset"].apply(lambda x: 1 if x == 2 else 0)

    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


    import numpy as np
    import pickle
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    class_counts = np.bincount(y.astype(int))
    imbalance_ratio = max(class_counts) / min(class_counts)

    print("Class counts:", dict(enumerate(class_counts)))
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 1.5:
        main_metric = "F1"
        print("Dataset is imbalanced → using F1-score as main metric")
    else:
        main_metric = "Accuracy"
        print("Dataset is balanced → using Accuracy as main metric")

    # Models to test
    models = {
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier()
    }

    results = []

    trained_models = {}  # Store trained model objects


    # Train & evaluate each model
    for name, model in models.items():
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)

        trained_models[name] = model  # Store the trained model

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(ytest, y_pred),
            "Precision": precision_score(ytest, y_pred, average="weighted"),
            "Recall": recall_score(ytest, y_pred, average="weighted"),
            "F1": f1_score(ytest, y_pred, average="weighted")
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by chosen main metric
    df_results = df_results.sort_values(by=main_metric, ascending=False)
    print("\n Model Performance Comparison:")
    print(df_results)

    # Best model
    # Best model
    best_model_name = df_results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n Best model based on {main_metric}: {best_model_name}")


    # Save the best model to a pickle file
    with open("best_model_liver.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f" Best model saved as 'best_model.pkl'")
    return best_model, df_liver



##################################################################################################################

def parkinsons():

    import pandas as pd

    data= pd.read_csv("parkinsons - parkinsons.csv")
    df=data.copy()

    df=df.drop(['name'],axis=1)

   # heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Box plots for numeric features
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot of {col}")
        plt.show()


    x=df.drop(['status'],axis=1)
    y=df['status']


    from sklearn.model_selection import train_test_split

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


    import numpy as np
    import pickle
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    class_counts = np.bincount(y.astype(int))
    imbalance_ratio = max(class_counts) / min(class_counts)

    print("Class counts:", dict(enumerate(class_counts)))
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 1.5:
        main_metric = "F1"
        print("Dataset is imbalanced → using F1-score as main metric")
    else:
        main_metric = "Accuracy"
        print("Dataset is balanced → using Accuracy as main metric")

    # Models to test
    models = {
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier()
    }

    results = []

    trained_models = {}  # Store trained model objects


    # Train & evaluate each model
    for name, model in models.items():
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)

        trained_models[name] = model  # Store the trained model

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(ytest, y_pred),
            "Precision": precision_score(ytest, y_pred, average="weighted"),
            "Recall": recall_score(ytest, y_pred, average="weighted"),
            "F1": f1_score(ytest, y_pred, average="weighted")
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by chosen main metric
    df_results = df_results.sort_values(by=main_metric, ascending=False)
    print("\n Model Performance Comparison:")
    print(df_results)

    # Best model
    # Best model
    best_model_name = df_results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n Best model based on {main_metric}: {best_model_name}")


    # Save the best model to a pickle file
    with open("best_model_parkinsons.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f" Best model saved as 'best_model.pkl'")
    return best_model, df_results

######################################################################################################################


import streamlit as st
from streamlit_option_menu import option_menu
import pickle

# Load models
kidney_model = pickle.load(open('best_model_kidney.pkl', 'rb'))
liver_model = pickle.load(open('best_model_liver.pkl', 'rb'))
parkinsons_model = pickle.load(open('best_model_parkinsons.pkl', 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Kidney Prediction", "Liver Prediction", "Parkinsons Prediction"],
        icons=['droplet', 'heart', 'person'],
        menu_icon="hospital",
        default_index=0
    )
# ---------------------- Kidney Prediction ----------------------
if selected == "Kidney Prediction":
    st.title("Kidney Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', value=None, placeholder="Enter value")
        diabetes_mellitus = st.number_input('Diabetes Mellitus', value=None, placeholder="Enter value")
        hypertension = st.number_input('Hypertension', value=None, placeholder="Enter value")
        blood_urea = st.number_input('Blood Urea', value=None, placeholder="Enter value")
    with col2:
        specific_gravity = st.number_input('Specific Gravity', value=None, placeholder="Enter value")
        albumin = st.number_input('Albumin', value=None, placeholder="Enter value")
        packed_cell_volume = st.number_input('Packed Cell Volume', value=None, placeholder="Enter value")
    with col3:
        hemoglobin = st.number_input('Hemoglobin', value=None, placeholder="Enter value")
        blood_glucose_random = st.number_input('Blood Glucose Random', value=None, placeholder="Enter value")
        serum_creatinine = st.number_input('Serum Creatinine', value=None, placeholder="Enter value")

    if st.button('Predict Kidney Disease'):
        inputs = [
            serum_creatinine, blood_urea, hemoglobin, albumin,
            specific_gravity, blood_glucose_random, packed_cell_volume,
            hypertension, diabetes_mellitus, age
        ]

        if any(v is None for v in inputs):
            st.warning("Enter all fields before prediction")
        else:
            prediction = kidney_model.predict([inputs])
            if prediction[0] == 1:
                st.error("The person has Kidney Disease")
            else:
                st.success("The person does not have Kidney Disease")


# ---------------------- Liver Prediction ----------------------
if selected == "Liver Prediction":
    st.title("Liver Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input('Age', value=None, placeholder="Enter value")
        Total_Bilirubin = st.number_input('Total Bilirubin', value=None, placeholder="Enter value")
        Direct_Bilirubin = st.number_input('Direct Bilirubin', value=None, placeholder="Enter value")
    with col2:
        Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase', value=None, placeholder="Enter value")
        Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase', value=None, placeholder="Enter value")
        Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase', value=None, placeholder="Enter value")
    with col3:
        Total_Protiens = st.number_input('Total Proteins', value=None, placeholder="Enter value")
        Albumin = st.number_input('Albumin', value=None, placeholder="Enter value")
        Albumin_and_Globulin_Ratio = st.number_input('Albumin and Globulin Ratio', value=None, placeholder="Enter value")

    if st.button('Predict Liver Disease'):
        inputs = [
            Age, Total_Bilirubin, Direct_Bilirubin,
            Alkaline_Phosphotase, Alamine_Aminotransferase,
            Aspartate_Aminotransferase, Total_Protiens, Albumin,
            Albumin_and_Globulin_Ratio
        ]

        if any(v is None for v in inputs):
            st.warning("Enter all fields before prediction")
        else:
            prediction = liver_model.predict([inputs])
            if prediction[0] == 1:
                st.error("The person has Liver Disease")
            else:
                st.success("The person does not have Liver Disease")



# ---------------------- Parkinson's Prediction ----------------------

if selected == "Parkinsons Prediction":
    st.title("Parkinsons Disease Prediction")

    parkinsons_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
       'spread2', 'D2', 'PPE']

    # Create inputs in 3 columns
    values = []
    cols = st.columns(3)
    for idx, feature in enumerate(parkinsons_features):
        col = cols[idx % 3]
        val = col.number_input(feature,value=None,step=1e-6, format="%.6f", placeholder="Enter value")
        values.append(val)

    if st.button("Predict Parkinsons Disease"):
        if any(v is None for v in values):
            st.warning("Enter all fields before prediction")
        else:
            # prediction = parkinsons_model.predict([values])
            prediction = [1]  # Dummy for testing
            if prediction[0] == 1:
                st.error("The person has Parkinsons Disease")
            else:
                st.success("The person does not have Parkinsons Disease")
