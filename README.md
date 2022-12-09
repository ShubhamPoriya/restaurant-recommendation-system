# Restaurant Recommendation System

A content-based recommendation system implemented using YELP business public dataset.

Yelp dataset is a huge business data pool with subsets of reviews, user data, business data and tip data. For this project
only restaurants were filtered out in pre-processing step for project goal. 

## 1. Finding similar restaurants by restuarant names

Recommendation system works on principle of association rule. So I found similarity between multiple restaurant based on a list of categories.
This similarity score was calculated using cosine similarity. 


<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow is="true">
    <mtext is="true">Similarity</mtext>
    <mrow is="true">
      <mo is="true">(</mo>
      <mrow is="true">
        <mi is="true">x</mi>
        <mtext is="true">,</mtext>
        <mi is="true">y</mi>
      </mrow>
      <mo is="true">)</mo>
    </mrow>
    <mo linebreak="badbreak" is="true">=</mo>
    <mi is="true">cos</mi>
    <mrow is="true">
      <mo is="true">(</mo>
      <mi is="true">&#x3B8;</mi>
      <mo is="true">)</mo>
    </mrow>
    <mo linebreak="goodbreak" is="true">=</mo>
    <mfrac is="true">
      <mrow is="true">
        <mi is="true">x</mi>
        <mrow is="true">
          <mo linebreak="badbreak" is="true">&#xB7;</mo>
        </mrow>
        <mi is="true">y</mi>
      </mrow>
      <mrow is="true">
        <mrow is="true">
          <mo is="true">|</mo>
          <mi is="true">x</mi>
          <mo is="true">|</mo>
        </mrow>
        <mrow is="true">
          <mo is="true">|</mo>
          <mi is="true">y</mi>
          <mo is="true">|</mo>
        </mrow>
      </mrow>
    </mfrac>
  </mrow>
</math>

## 2. Similar restaurants based on current location

Recommendation based on location needs an extra step of clustering restaurants in particular city in finite groups. For this, I used
un-supervised learning: K-Means clustering technique to classify restaurants. For finding optimal k-value, there were 2 methods implemented 
that are: 
1. Elbow Method - Plotting inertia of dataset against number of clusters. Point where plot pivots or changes trens is a useful cluster range.
![Elbow plot](https://github.com/ShubhamPoriya/restaurant-recommendation-system/blob/main/elbow.png?raw=true)


2. Silhouette Method - We calculate Silhouette score to check quality of range of clusters found from elbow method. 
![Silhouette score values](https://github.com/ShubhamPoriya/restaurant-recommendation-system/blob/main/silhouette.png)

## Results

Restaurant recommender system will give top 10 similar restaurants sorted by similarity scores as output when any restaunrant is entered as input.
As data conists of mutiple cities, user can also choose their current city before getting the results. For location based recommendation system, 
user will only have to put their current location to indentify city and restaurants suggested.
