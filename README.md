#Visualizing a Decision Tree

Tutorial for Visualizing a Decision Tree with scikit-learn
Based on the [Visualizing a Decision Tree - Machine Learning Recipes #2](https://www.youtube.com/watch?v=tNa99PG8hR8)

## Summary

Table for Training Data

Weight  #Texture    #Label
150g    Bumpy       Orange
170g    Bumpy       Orange
140g    Smooth      Apple
130g    Smooth      Apple

Think features as the input to the classifier, label as the output.
0 = Bumpy, 1 = Smooth
0 = Apple, 1 = Orange

Decision tree
                Weight >= 150g ?
                Yes    /\   No
    Texture = Bumpy ?
        Yes   / \   No
     Orange        Apple