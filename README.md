# Predictive Analysis using OkCupid Dataset
This modular project aims to explore the OKCupid dataset, a vast collection of data on people's interests, lifestyles, and personal characteristics. The project involves data loading, exploration, and cleaning, followed by the use of machine learning models to forecast outcomes. The project also includes a unique sex prediction task based on wealth and educational attainment. The final deliverables include a comprehensive model comparison and evaluation, coding in a Jupyter Notebook, a detailed project report, and presentation slides for stakeholder engagement. The ultimate goal is to provide insights into variables affecting Zodiac signs, body type, wealth, and gender forecasts on OKCupid, and to improve understanding of user preferences and behaviors in online dating scenarios.
 # stack used
 Programming Language: Python, IDE: Jupyter Notebook, Machine Learning Models: Artificial Neural Network, Support Vector Classifier, Decision Tree, Random Forest
# Load in the DataFrame
The data is stored in **profiles.csv**. We can start to work with it in **OkCupid.ipynb** by using Pandas, which we have imported for you with the line:

```
import pandas as pd
```
and then loading the csv into a DataFrame:

```
df = pd.read_csv("profiles.csv")
```
# Explore the Data
Let's make sure we understand what these columns represent!

Pick some columns and call .head() on them to see the first five rows of data. For example, we were curious about job, so we called:
```
df.job.head()
```
You can also call value_counts() on a column to figure out what possible responses there are, and how many of each response there was.
or example, we were curious about the distribution of ages on the site, so we made a histogram of the `age` column:


```
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()
```


Try this code in your own file and take a look at the histogram it produces!


### Formulate a Question


As we started to look at this data, we started to get more and more curious about Zodiac signs. First, we looked at all of the possible values for Zodiac signs:


```
df.sign.value_counts()
```
We started to wonder if there was a way to predict a user's Zodiac sign from the information in their profile. Thinking about the columns we had already explored, we thought that maybe we could classify Zodiac signs using drinking, smoking, drugs, and essays as our features.




### Augment your Data


In order to answer the question you've formulated, you will probably need to create some new columns in the DataFrame. This is especially true because so much of our data here is categorical (i.e. `diet` consists of the options `vegan`, `vegetarian`, `anything`, etc. instead of numerical values).


Categorical data is great to use as labels, but we want to create some numerical data as well to use for features.


For our question about Zodiac signs, we wanted to transform the `drinks` column into numerical data. We used:

```
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}


all_data["drinks_code"] = all_data.drinks.map(drink_mapping)
```


These lines of code created a new column called 'drinks_code' that mapped the following `drinks` values to these numbers:


| drinks      | drinks_code |
|-------------|-------------|
| not at all  | 0           |
| rarely      | 1           |
| socially    | 2           |
| often       | 3           |
| very often  | 4           |
| desperately | 5           |
```

We did the same for `smokes` and `drugs`.

### Normalize your Data!


In order to get accurate results, we should make sure our numerical data all has the same weight.


For our Zodiac features, we used:


```
feature_data = all_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
```

# Analysis
This solution will use descriptive statistics and data visualization to find key figures in understanding the distribution, count, and relationship between variables.

# Evaluation
The project will conclude with the evaluation of the machine learning model selected with a validation data set. The output of the predictions can be checked through a confusion matrix, and metrics such as accuracy, precision, recall and F1 scores.
