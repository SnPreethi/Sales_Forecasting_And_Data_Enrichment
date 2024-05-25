# INSTALLING THE LIBRARIES
pip install catboost
pip install pandas
pip install upgini

# PREPARING THE INPUT DATA
from os.path import exists
import pandas as pd
# check if the file is downloaded else download it to avoid re-downloading the file every single time
df_path = "train.csv.zip" if exists ("train.csv.zip") else "https://github.com/upgini/upgini/raw/main/notebooks/train.csv.zip"
#reading the csv file and storing it as pandas data frames
df = pd.read_csv(df_path)
#randomly selecting a sample of 10k points from the data set
df = df.sample(n=10_000, random_state=0)
#converting store and items columns into string formats
df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)
#converting the date column into pandas datetime format for sorting
df["date"] = pd.to_datetime(df["date"])
#chronological sorting on the grounds of date column
df.sort_values("date", inplace = True)
df.reset_index(inplace = True, drop = True)
#gives the first 5 rows
df.head()

#SPLITTING THE DATASET
train = df[df["date"] < "2017-01-01"] #dates before the specified are added into the training dataset
test = df[df["date"] >= "2017-01-01"] #dates from and after the specified are added into the testing dataset

# SPLITTING THE DATASET INTO FEATURES AND LABELS
train_features = train.drop(columns=["sales"]) # except sales, every other column is used for training
train_target = train["sales"] #sales column is what we have to predict
test_features = test.drop(columns = ["sales"])
test_target = test["sales"]

# ENRICH FEATURES USING UPGINI LIBRARY
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType
# Features enricher to add brand new features to the dataset with the help of date
#as the searchkey and telling the enricher that it is a time-series data
enricher = FeaturesEnricher(search_keys={"date":SearchKey.DATE}, cv=CVType.time_series)
enricher.fit(train_features, train_target, eval_set=[(test_features,test_target)])

# DEFINE THE CATBOOST MODEL
from catboost import CatBoostRegressor
from catboost.utils import eval_metric
model = CatBoostRegressor(verbose = False, allow_writing_files = False, random_state = 0)
enricher.calculate_metrics(train_features, train_target,eval_set = [(test_features,test_target)], estimator = model, scoring = "mean_absolute_percentage_error")

# ADDING THE NEW ADDITION FEATURES TO THE ORIGINAL DATASET
enriched_train_features = enricher.transform(train_features, keep_input = True) #adding the enricher features to the train features without skipping the original features in the train dataset
enriched_test_features = enricher.transform(test_features, keep_input = True) #adding the enricher features to the test features without skipping the original features in the test dataset
enriched_train_features.head() #displays the first five values

# MODEL PERFORMANCE AND EVALUATION

#model's performance on original dataset before enrichment
model.fit(train_features, train_target) #training the model
preds = model.predict(test_features)
eval_metric (test_target.values, preds, "SMAPE") #gives error rate of the model when the original dataset without any enrichment is used

#model's performance after enrichment
model.fit(enriched_train_features, train_target) #training the model
enriched_preds = model.predict(enriched_test_features)
eval_metric (test_target.values, enriched_preds, "SMAPE") #gives error rate of the model when the enriched dataset is used
