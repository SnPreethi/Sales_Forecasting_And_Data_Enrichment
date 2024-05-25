# SALES FORECASTING & DATA ENRICHMENT

<p align="justify">This is a basic machine learning project that aims to predict the next three months of demand of an item in a store using the historical data of past 5 years. The problem statement used in this project is from a Kaggle competition named "Store Item Demand Forecasting Challenge". </p>

### WHAT DOES THE DATASET CONTAIN?
<p align="justify">The dataset contains five years of store item sales data. This is time-series data as the sales are dependent on time. It is structured in a tabular format with four columns:
<ul>
<li> 1st column: date </li>
<li> 2nd column: store id </li>
<li> 3rd column: item id (referring to the id of each item within the store) </li>
<li> 4th column: the number of times the item has been sold (a particular item has been sold X number of times in Y particular store on Zth date)</li>
</ul></p>

### WHAT ARE WE GOING TO DO WITH WITH THIS DATASET?
<p align="justify">Based on five years of sales data from 10 different stores, this project aims to predict the next three months of sales for each individual item in these stores. By analyzing historical trends, the model will forecast future demand to aid in inventory management and sales planning.</p>

### WHAT ARE WE USING TO SOLVE THIS PROBLEM?
<ol type="A" align="justify">
<li><b>CAT BOOST</b>: It is an algorithm for gradient boosting on decision trees. It is a very popular machine learning algorithm which is used in recommendation systems and forecasting.</li>
<li><b>UPGINI</b>: Upgini is a Python library that helps achieve highly accurate forecasting models. The data we have is sparse, with only two main features: the date of sales and the number of sales. It is not a lot of information for our machine learning model to understand how to perform the prediction process for the sales of various items. Upgini solves this problem by automatically searching through thousands of public data sources to find the most relevant features.It then integrates these features with our existing dataset, improving the model's performance.</li>
<li><b>PANDAS</b>: Handle dataframes by downloading a CSV file, converting it into a pandas dataframe, and then feeding it into our model.</li>
</ol>

### IDE TO USE?
<ul>
<li>Google Colab</li>
<li>Jupyter notebook (or)</li>
<li>Any other IDE you like</li>
</ul>

### HOW TO INSTALL THE LIBRARIES?
<ul>
<li> catboost - pip install catboost </li>
<li> upgini - pip install upgini (only works with python version >=3.7 and < 3.10) </li>
<li> pandas - pip install pandas </li>
</ul>

### TASK TYPE - REGRESSION

### STEPS
<ol type="1">
<li>Install the libraries.</li>
<li>Download the dataset and prepare the input data.</li>
<li>Split the dataset into test and train sets.</li>
<li>Split the datasets into features (input values) and labels (what we want to predict)</li>
<li>Enrich the features using upgini library to get relevant features and their corresponding SHAP value (It is a mathematical value that indicates how relevant or how influential this feature is towards the prediction.)</li>
<li>Defining the catboost model</li>
<li>Adding the new features to the original dataset</li>
<li>
Model's performance and evaluation under:
    <ul><li>original dataset without any enrichment</li>
    <li>newly formed enriched dataset</li></ul>
</li>
</ol>
