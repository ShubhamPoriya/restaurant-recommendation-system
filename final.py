# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import numpy as np

token = "pk.eyJ1Ijoic2h1YmhhbXBvcml5YSIsImEiOiJjbDlyYzA3M20xMWFzM3ZvMGE5amN3aHQ0In0.g1pczB9pvkS4B1B3r0BFrg" # you will need your own token

import warnings
warnings.filterwarnings("ignore")


# Script handling
files = sys.argv

for filename in files:
    if filename == "business.csv":
        df_business = pd.read_csv(r'{}'.format(filename))
    elif filename == "review.csv":
        df_review = pd.read_csv(r'{}'.format(filename))
    elif filename == "checkin.csv":
        df_checkin = pd.read_csv(r'{}'.format(filename))
    elif filename == "restaurants.csv":
        df_restaurant_review = pd.read_csv(r'{}'.format(filename))
    else:
        pass

"""
Data Preprocessing
"""

# made a new column name "year" derived from "date" attribute available in "df_review"
df_review["year"] = pd.DatetimeIndex(df_review["date"]).year

df_review.head()

# make a custome dataframe "df_review_count" which shows the count of reviews per year
df_review_count = df_review.groupby(["year"]).agg({"review_id": "count"})

df_review["date"] = pd.to_datetime(df_review["date"])
df_review = df_review.set_index("date")

df_business_checkin = pd.merge(df_checkin, df_business, how='right')



"""
Data Analysis
"""
# initializer for Analysis table
# find mean for all attributes

star_rating_mean = df_review.mean()[0]
useful_review_mean = df_review.mean()[1]
funny_review_mean = df_review.mean()[2]
cool_review_mean = df_review.mean()[3]
review_year_mean = df_review.mean()[4]
# make a list of review mean
review_mean = [
    star_rating_mean,
    useful_review_mean,
    funny_review_mean,
    cool_review_mean,
    review_year_mean,
]

# find minimum and maximum value for all attributes
star_rating_range = [df_review.min()[3], df_review.max()[3]]
useful_review_range = [df_review.min()[4], df_review.max()[4]]
funny_review_range = [df_review.min()[5], df_review.max()[5]]
cool_review_range = [df_review.min()[6], df_review.max()[6]]
review_year_range = [df_review.min()[8], df_review.max()[8]]
# make a list of review range with all attributes's min and max value list
review_range = [
    star_rating_range,
    useful_review_range,
    funny_review_range,
    cool_review_range,
    review_year_range,
]

# temp_mode will find mode for review dataframe on axis 0 and we convert the result in np.array for ease of use
temp_mode = df_review.mode(axis=0)
temp_mode_list = np.array(temp_mode)

# find mode for all attributes
star_rating_mode = temp_mode_list[0][3]
useful_review_mode = temp_mode_list[0][4]
funny_review_mode = temp_mode_list[0][5]
cool_review_mode = temp_mode_list[0][6]
review_year_mode = temp_mode_list[0][8]
# make a list of review mode
review_mode = [
    star_rating_mode,
    useful_review_mode,
    funny_review_mode,
    cool_review_mode,
    review_year_mode,
]

# make analysis dataframe
# temp_index= ['Range','Mean','Mode']

# set column names and values for data frames with the list generated before for range, mean and mode
df_analysis = pd.DataFrame(
    {
        "Star rating": [star_rating_range, star_rating_mean, star_rating_mode],
        "Useful Review": [useful_review_range, useful_review_mean, useful_review_mode],
        "Funny Review": [funny_review_range, funny_review_mean, funny_review_mode],
        "Cool Review": [cool_review_range, cool_review_mean, cool_review_mode],
        "Review Year": [review_year_range, review_year_mean, review_year_mode],
        "index": ["Range", "Mean", "Mode"],
    }
)
# df = pd.DataFrame(review_range, review_mean, review_mode,index=index)
df_analysis = df_analysis.set_index("index")

print("========")
print("Analysis")
print("========")
print(df_analysis)


"""
Data visulization
"""

# number of reviews vs year plot
# plt.plot(df_review_count["review_id"])
# plt.xlabel("year", fontsize=12)
# plt.ylabel("No. of reviews", fontsize=12)
# plt.title("No. of reviews per month", fontsize=16)

# plt.figure()

# Star ratings plot
temp_business = df_business["stars"].value_counts()
temp_business = temp_business.sort_index()


# ax = sns.barplot(x = temp_business.index, y = temp_business.values, alpha=0.9)
# plt.title("Distribution of star ratings", fontsize=16)
# plt.ylabel("No. of businesses", fontsize=12)
# plt.xlabel("Star Ratings ", fontsize=12)

# plt.figure(figsize=(10, 6))

# rating distribution of reviews get by per city
city_review = df_business["city"].value_counts()
city_review = city_review.sort_values(ascending=False)
city_review = city_review.iloc[0:20]

# ax = sns.barplot(x=city_review.index, y=city_review.values, alpha=0.8)
# plt.title("Which city has the most reviews?", fontsize=16)
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=50)
# plt.ylabel("no. of businesses", fontsize=12)
# plt.xlabel("City", fontsize=12)

# plt.figure(figsize=(16, 4))

# Pie chart for text mining

df_piechart = pd.DataFrame(df_restaurant_review)
s_words = ['pizza','burger','chinese','fries','chicken','indian']

vec = CountVectorizer(vocabulary=s_words,lowercase=False)

word_count = vec.fit_transform(df_piechart['text'].values.astype('U'))
vec.get_feature_names()

word_array = word_count.toarray()
word_array.shape

word_array.sum(axis=0)

t_df = pd.DataFrame(index=vec.get_feature_names(),data=word_array.sum(axis=0)).rename(columns={0:'Word Count'})

# plt.pie(word_array.sum(axis=0),labels=vec.get_feature_names(),explode=(0.1,0.1,0.1,0.1,0.1,0.1))
# plt.title('Distribution of Food Related Words in User Reviews')

# plt.figure(figsize=(16, 4))

# plt.show()

# Geo map of all ratings and review counts
# px.set_mapbox_access_token(token)
# fig = px.scatter_mapbox(df_business_checkin, lat="latitude", lon="longitude", color="stars", size="review_count",
#                   color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

# fig.show()


# python main.py business.csv checkin.csv restaurants.csv reviews.csv

# ------ Machine Learning application --------

df_business.rename(columns={"name": "business_name"}, inplace=True)
restaurants = []
df_business_cleaned = df_business[df_business.categories.isnull() == False]

for index, row in df_business_cleaned.iterrows():
  list_cat = row.categories.split(",")
  categories = [item.strip() for item in list_cat]
  # df_business_cleaned.replace(row.categories, str(categories))
  if "Restaurants" in categories:
    restaurants.append(True)
  else:
    restaurants.append(False)

df_business_cleaned["is_restaurant"] =  restaurants
df_restaurants = df_business_cleaned[df_business_cleaned.is_restaurant == True]

# left join between restaurant and review datasets
df_restaurant_reviews = df_restaurants.merge(df_review, how='left', on='business_id')
# cleaning dataset to remove NUll values
df_reviews_cleaned = df_restaurant_reviews[df_restaurant_reviews.review_id.isnull() == False]

attributes_list = []
for index, row in df_reviews_cleaned.iterrows():
  attributes = df_reviews_cleaned.attributes[index].strip('{}').replace("\'", "").split(",")
  x = [attr.strip() for attr in attributes]
  attributes_list.append(x)

df_reviews_cleaned["list_attributes"] = attributes_list

# loop will split string into list of restaurant categories  
words_list = []
for index, row in df_restaurants.iterrows():
  words = df_restaurants.categories[index].strip().split(",")
  
  y = [''.join(('"',word.strip(),'"')) for word in words]
  y_str = ' '.join(y)
  
  words_list.append(y_str)

# add a new column in restaurants dataset
df_restaurants["Restaurant_Categories"] = words_list

# this function will recommend top 10 restaurants based on the city and retaurant selected
def recommend_restaurant(business_name, cityName):
  df_restaurants_city = df_restaurants[df_restaurants.city == cityName]
  df_restaurants_city.reset_index(drop=True, inplace=True)
  df_restaurant_cat = df_restaurants_city[["business_name", "Restaurant_Categories"]]
  count = CountVectorizer()
  count_matrix = count.fit_transform(df_restaurant_cat['Restaurant_Categories'])
  cosine_similarity_df = pd.DataFrame(cosine_similarity(count_matrix))

  restaurant_index = df_restaurants_city.index[df_restaurants_city["business_name"] == business_name].tolist()
  indexes = cosine_similarity_df[restaurant_index].sort_values(by=restaurant_index, ascending=False).index.tolist()[1:10]

  top_10_restro = []
  top_10_restro_stars = []
  restro_address = []
  for i in range(len(indexes)):
    top_10_restro.append(df_restaurants_city["business_name"].loc[indexes[i]])
    top_10_restro_stars.append(df_restaurants_city["stars"].loc[indexes[i]])
    restro_address.append(df_restaurants_city['address'].loc[indexes[i]])

  df_top_10 = pd.DataFrame({
      "Restaurants": top_10_restro,
      "Ratings": top_10_restro_stars,
      "Address": restro_address
  })
  return df_top_10


'''
- This chunk of code will take an input from user for prefered city,
by using the city name we filter the dataset for restaurants.
- Second input choice of restaurant in the selected city will give an output 
of top 10 restaurant which have high similarity score.

'''

print("Please select your current city \n")
print(df_restaurants['city'].tolist(), '\n')
input_city = input("Enter your current city: \n")
print("\n")

print(df_restaurants[df_restaurants['city'] == input_city]['business_name'].tolist(), '\n')
input_name = input("Enter restaurant name: ")
print("\n You might also like these top 10 restaurants \n")
df = recommend_restaurant(input_name, input_city)
print(df, '\n')

'''
we try to ue location based reccomendation system using K-means clustering
for creating local group of restaurants 
'''
X = df_restaurants[df_restaurants['city'] == "Philadelphia"][["longitude", "latitude"]]
kmean = KMeans(n_clusters=5)
X["clusters"] = kmean.fit_predict(X)

colors = ['#DF2020', '#81DF20', '#2095DF', '#32a852', '#bd14cc']
X['c'] = X.clusters.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4]})

# plot of restaurant dataset for Philadelphia city with k=5
plt.scatter(df_restaurants[df_restaurants['city'] == "Philadelphia"]["latitude"], 
            df_restaurants[df_restaurants['city'] == "Philadelphia"]['longitude'],
            c=X.c,
            alpha=0.6,
            s=10)
plt.title("Clustering for Restaurants")
plt.show()


inertia = []
k_values = [i for i in range(2, 50)]

for k in k_values:
  model = KMeans(n_clusters=k, random_state=0)
  model.fit(df_restaurants[df_restaurants['city'] == "Philadelphia"][["longitude", "latitude"]])
  inertia.append(model.inertia_)

plt.title("Elbow method for k-means")
plt.plot(k_values, inertia)
plt.show()

negative_k_values = {}
k_values_s = [i for i in range(5, 10)]

# creating temporary subset of data for city
df_temp = df_restaurants[df_restaurants['city'] == "Philadelphia"][["longitude", "latitude"]]

# looping over k values estimatwd from elbow method
for k in k_values_s:
  model = KMeans(n_clusters=k, random_state=0)
  model.fit(df_temp)
  # silhouette score for checking quality of clusters
  labels = model.predict(df_temp)
  silhouette_score = silhouette_samples(df_temp, labels)

# checking values of silhouette score for negative values
  for s in silhouette_score:    
    
    if s < 0:
      if k not in negative_k_values:
          negative_k_values[k] = 1
      
      else:
          negative_k_values[k] += 1

print("\n Silhouette Score \n")
# printing cluster and silhouette score
for key, val in negative_k_values.items():
    print(f'Cluster: {key} \t Values: {val}')
