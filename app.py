import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder


def main():
    # Set the title of the Streamlit app
    st.title('Feature Selection App')

    # Allow the user to upload a file
    uploaded_file = st.file_uploader("Choose a CSV, Excel or Parquet file", type=["csv", "xlsx", "parquet"])
    if uploaded_file is not None:
        # Load the uploaded file into a DataFrame
        if uploaded_file.type == "text/csv":
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "application/octet-stream":
            data = pd.read_parquet(uploaded_file)
        # Display the DataFrame
        st.dataframe(data)

        # Allow the user to select features
        features = st.multiselect('Select features', data.columns.tolist())
        if features:
            # Filter the DataFrame based on the selected features
            data = data[features]
            # Display the filtered DataFrame
            st.dataframe(data)


        # Allow the user to select a column
        selected_column = st.selectbox('Select column to filter', data.columns.tolist())
        if selected_column:
            # Get the minimum and maximum values of the selected column
            min_value = data[selected_column].min()
            max_value = data[selected_column].max()

            # Allow the user to select a range of values
            values = st.slider('Select a range of values', min_value, max_value, (min_value, max_value))

            # Filter the DataFrame based on the selected range
            data = data[data[selected_column].between(values[0], values[1])]

        # Add aggregation functionality
        if st.checkbox('Aggregate Data'):
            st.subheader('Data Aggregation')
            agg_methods = ['sum', 'count', 'mean', 'min', 'max']
            selected_method = st.selectbox('Select aggregation method', agg_methods)
            selected_feature = st.multiselect('Select feature to aggregate', data.columns.tolist())
            groupby_columns = st.multiselect('Select columns to group by', data.columns.tolist())

            # if st.button('Aggregate'):
            # Group the data by the selected columns and Aggregate the data based on the selected method

            if len(groupby_columns)>0 and len(selected_feature)>0:
                if selected_method == 'sum':
                    data = data.groupby(groupby_columns, as_index=False)[selected_feature].sum()
                elif selected_method == 'count':
                    data = data.groupby(groupby_columns, as_index=False)[selected_feature].count()
                elif selected_method == 'mean':
                    data = data.groupby(groupby_columns, as_index=False)[selected_feature].mean()
                elif selected_method == 'min':
                    data = data.groupby(groupby_columns, as_index=False)[selected_feature].min()
                elif selected_method == 'max':
                    data = data.groupby(groupby_columns, as_index=False)[selected_feature].max()

                # Display the aggregated data
                st.dataframe(data)

        # Add discretization functionality

        if st.checkbox('Discretize Data'):
            st.subheader('Data Discretization')
            discretize_columns = st.multiselect('Select columns to discretize', data.columns.tolist())
            for column in discretize_columns:
                # Calculate the optimal number of bins
                bins = calculate_optimal_bins(data[column])
                # Create labels for the bins
                labels = range(1,bins+1)
                # Discretize the data and add it as a new column
                data[column+'_discretized'] = pd.cut(data[column], bins=bins, labels=labels)
            # Display the DataFrame with the discretized data
            st.dataframe(data)




        # Correlation Matrix
        le = LabelEncoder()
        for column in data.select_dtypes(include=['object']).columns:
            data[column] = le.fit_transform(data[column])

        selected_columns = st.multiselect('Select columns for correlation matrix', data.columns.tolist())
        if selected_columns:
            # Filter the DataFrame based on the selected columns
            filtered_data = data[selected_columns]

            correlation_matrix = filtered_data.corr()
            # Create a matplotlib figure
            fig, ax = plt.subplots()
            # Visualize the correlation matrix
            sns.heatmap(correlation_matrix, annot=True, ax=ax)
            st.pyplot(fig)


        # Add model building functionality
        if st.checkbox('Build Model'):
            st.subheader('Model Building')
            model_types = ['Classification', 'Regression']
            selected_model_type = st.selectbox('Select model type', model_types)
            st.dataframe(data)
            target_feature = st.selectbox('Select target feature', data.columns.tolist())
            feature_columns = st.multiselect('Select feature columns',[col for col in data.columns.tolist() if col != target_feature])

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                data[feature_columns], data[target_feature], test_size=0.2, random_state=42)

            if selected_model_type == 'Classification':
                # Train a Logistic Regression model
                model = LogisticRegression()
                model.fit(X_train, y_train)
                # Make predictions on the testing set
                predictions = model.predict(X_test)
                # Calculate the accuracy of the model
                accuracy = accuracy_score(y_test, predictions)
                st.write(f'Accuracy: {accuracy}')
            elif selected_model_type == 'Regression':
                # Train a Linear Regression model
                model = LinearRegression()
                model.fit(X_train, y_train)
                # Make predictions on the testing set
                predictions = model.predict(X_test)
                # Calculate the mean squared error of the model
                mse = mean_squared_error(y_test, predictions)
                st.write(f'Mean Squared Error: {mse}')



def calculate_optimal_bins(data):
    # Calculate the IQR of the data
    iqr = np.subtract(*np.percentile(data, [75, 25]))

    # Calculate the bin width based on the Freedman-Diaconis rule
    bin_width = 2 * iqr * (len(data) ** (-1/3))

    # Calculate the number of bins
    bins = round((data.max() - data.min()) / bin_width)

    return bins

if __name__ == "__main__":
    main()