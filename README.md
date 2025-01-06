1. Setup Phase

    1.1: Create a project folder structure as outlined earlier.
    1.2: Set up version control:
        Initialize a Git repository with git init.
        Add a .gitignore file to exclude sensitive files (e.g., configs/api_keys.json and large datasets).
    1.3: Create a virtual environment:

    1.4: Install dependencies and save them to requirements.txt:
        pip install pandas numpy matplotlib seaborn scikit-learn tensorflow streamlit requests
        pip freeze > requirements.txt

2. Data Collection

    2.1: Obtain an API key from CoinGecko (if required). <br />
    2.2: Write a script (src/data/data_loader.py) to fetch historical cryptocurrency data from CoinGecko. <br />
    2.3: Save the raw data in the data/raw/ folder as CSV files. <br />
    2.4: Run the script and verify that the data is being downloaded correctly. <br />

3. Data Preprocessing

    3.1: Write a script (src/data/data_cleaning.py) to clean and preprocess the raw data:<br />
        - Convert timestamps to datetime. <br />
        - Handle missing values. <br />
        - Remove duplicates.<br /> 

    3.2: Save the cleaned data in the data/processed/ folder. <br />
    3.3: Explore the data using a Jupyter notebook (notebooks/data_exploration.ipynb):<br />
        - Visualize price trends using line charts.<br />
        - Perform basic descriptive analysis. <br />

4. Feature Engineering

    4.1: Develop additional features to improve predictions:
        Moving averages (e.g., 7-day, 30-day).
        Price change percentages.
        Sentiment scores (if available).
        On-chain metrics like transaction volume or active wallets.
    4.2: Save feature-enhanced datasets in the data/features/ folder. <br />

5. Model Development

    5.1: Write scripts in the src/models/ folder to:
        Split the data into training and testing sets.
        Train machine learning models (e.g., Linear Regression, Random Forest).
        Evaluate models using metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE). <br />
     5.2: Experiment with models in a Jupyter notebook (notebooks/model_training.ipynb).<br />
    5.3: Save the best-performing model using joblib or pickle.

6. Build the Dashboard

    6.1: Use Streamlit to create an interactive dashboard:
        Allow users to select the cryptocurrency and date range.
        Display data visualizations like line charts, heatmaps, and sentiment word clouds.
        Show prediction results.
    6.2: Save the Streamlit app as src/app/dashboard.py.
    6.3: Run the app locally to verify functionality:

    streamlit run src/app/dashboard.py

7. Security and Monitoring

    7.1: Secure API keys using environment variables or configs/api_keys.json (ensure this file is excluded from Git).
    7.2: Implement logging for monitoring application activity:
        Save logs in the logs/ folder.
    7.3: Ensure HTTPS is used for API calls (e.g., with requests).

8. Evaluation

    8.1: Evaluate the accuracy of your predictions:
        Use testing data and record performance metrics in a report (reports/evaluation_metrics.txt).
    8.2: Gather user feedback (if applicable) to refine the dashboard or models.

9. Deployment

    9.1: Choose a deployment platform:
        Local: Deploy locally for demonstration.
        Cloud: Use Heroku, AWS, or PythonAnywhere for web hosting.
    9.2: Configure the hosting environment to run the Streamlit app.
    9.3: Test the deployed application to ensure functionality.

10. Documentation and Presentation

    10.1: Create a project documentation file (README.md):
        Explain the project’s purpose, setup instructions, and usage guide.
    10.2: Prepare a presentation summarizing:
        The business problem.
        Data collection and preprocessing steps.
        Model results and application functionality.
    10.3: Submit all deliverables as per the capstone requirements.

11. Maintenance

    11.1: Monitor the application regularly for errors (using logs).
    11.2: Update models periodically with new data for improved predictions.
    11.3: Fix bugs or enhance features based on user feedback.