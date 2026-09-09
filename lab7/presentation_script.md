# Lab 7 Presentation Script - Decision Tree Classifier on Iris Dataset

Use this script to present each code cell to the evaluator. Read the narration, then execute/explain the code.

---

## Cell 1: Imports and Setup

**Narration:**
"We begin by importing all necessary libraries. NumPy and Pandas for data manipulation, Matplotlib for visualization, and from scikit-learn we import the Iris dataset, train-test split, cross-validation utilities, the Decision Tree classifier, and evaluation metrics. We also configure Pandas display options and suppress warnings for cleaner output."

---

## Cell 2: Dataset Exploration

**Narration:**
"First, we load the Iris dataset using scikit-learn's built-in loader. We create a DataFrame with the four features: sepal length, sepal width, petal length, and petal width. We add a target column with numeric class labels and a target_name column mapping them to setosa, versicolor, and virginica.

From the output, we can see the dataset contains 150 samples with 4 features and 3 balanced classes of 50 each."

---

## Cell 3: First 5 Records

**Narration:**
"Here we display the first 5 rows of the dataset to understand the feature ranges and data structure. We can see petal measurements vary significantly across species, while sepal measurements overlap more."

---

## Cell 4: Class Distribution

**Narration:**
"We visualize the class distribution with a bar chart. As we can see, Iris is perfectly balanced with 50 samples per class. This balance means we don't need to worry about class imbalance in our baseline experiments."

---

## Cell 5: Data Preparation

**Narration:**
"We split the data into training and testing sets using an 80-20 split with stratification. Stratification ensures the class distribution is preserved in both sets. We get 120 training samples and 30 testing samples, maintaining the 40-40-40 and 10-10-10 class split respectively."

---

## Cell 6: Stratification Note

**Narration:**
"This markdown cell confirms that Iris is balanced and that our stratified split preserves these ratios, which is important for reliable model evaluation."

---

## Cell 7: Default Decision Tree

**Narration:**
"We train a Decision Tree classifier with default parameters. The model achieves 100% accuracy on the training set and approximately 93.3% on the test set. The confusion matrix shows only one misclassification. This already strong baseline demonstrates that decision trees can capture the underlying patterns in Iris very effectively."

---

## Cell 8: Tree Rules

**Narration:**
"We export the tree structure as text rules using sklearn's export_text function. This reveals the decision logic: the first split is on petal length, which cleanly separates setosa from the other species. This interpretability is one of decision trees' key advantages."

---

## Cell 9: Tree Visualization

**Narration:**
"We visualize the full decision tree. The root node splits on petal length, and we can see how the tree branches to classify each species. The tree has a certain depth and number of leaf nodes as we'll examine next."

---

## Cell 10: Tree Statistics

**Narration:**
"We extract key statistics: the root node uses petal length (the most informative feature), and we print the tree's maximum depth and number of leaf nodes. The inference confirms that petal features dominate the splits while sepal features are rarely used."

---

## Cell 11: Gini vs Entropy Setup

**Narration:**
"Now we compare the two splitting criteria: Gini index and entropy. We'll train separate models with each criterion and evaluate them on multiple metrics including accuracy, cross-validation scores, tree complexity, and which feature becomes the root."

---

## Cell 12: Gini vs Entropy Results

**Narration:**
"Running both criteria, we see they produce nearly identical results on Iris. Both achieve the same test accuracy and CV accuracy. The tree depth, leaf count, and root feature are identical. This confirms that for small, clean datasets like Iris, the choice between Gini and entropy has minimal impact."

---

## Cell 13: Side-by-Side Trees

**Narration:**
"We visualize both trees side by side to confirm they're structurally identical. The visual comparison reinforces that Gini and entropy produce the same splits on this dataset."

---

## Cell 14: max_depth Experiment Setup

**Narration:**
"Now we investigate the effect of maximum tree depth. We test depths of 1, 2, 3, 4, and unlimited (None). For each, we record training accuracy, testing accuracy, tree depth, and leaf count. This will reveal the bias-variance tradeoff."

---

## Cell 15: max_depth Results

**Narration:**
"The results clearly show the bias-variance tradeoff: depth 1 underfits with only 66.7% accuracy, depth 3 achieves optimal generalization at 96.7%, while depth 4 and above show overfitting with 100% training accuracy but dropping test performance. The sweet spot is depth 3."

---

## Cell 16: max_depth Visualization

**Narration:**
"We plot training vs testing accuracy across depths. The gap between training and testing accuracy represents overfitting. We see this gap widens significantly after depth 3, confirming our earlier observation."

---

## Cell 17: max_depth Tree Comparison

**Narration:**
"We visualize trees at different depths. At depth 1, the tree is too simple. At depth 3, it captures the essential patterns. At unlimited depth, it becomes overly complex with many branches that capture noise rather than signal."

---

## Cell 18: min_samples_split Setup

**Narration:**
"Next, we examine min_samples_split, which controls the minimum number of samples required to split an internal node. We test values of 2, 5, 10, and 20. Higher values prevent the tree from growing too deep."

---

## Cell 19: min_samples_split Results

**Narration:**
"As min_samples_split increases, the tree becomes simpler with fewer leaves. This acts as a regularization mechanism. However, setting it too high (like 20 on 120 samples) can cause underfitting by preventing necessary splits."

---

## Cell 20: min_samples_split Trees

**Narration:**
"We visualize how the tree structure changes with different min_samples_split values. Higher values produce more conservative trees that stop splitting earlier."

---

## Cell 21: min_samples_leaf Setup

**Narration:**
"Now we look at min_samples_leaf, the minimum number of samples required at a leaf node. We test 1, 2, 5, and 10. This parameter smooths decision boundaries by ensuring leaves represent enough samples to be reliable."

---

## Cell 22: min_samples_leaf Results

**Narration:**
"Higher min_samples_leaf values produce smoother trees with fewer leaves. This reduces overfitting by preventing the tree from creating very specific leaves for just one or two samples. However, 10 is too high for our 120-sample training set, causing underfitting."

---

## Cell 23: min_samples_leaf Trees

**Narration:**
"The visual comparison shows how increasing min_samples_leaf progressively simplifies the tree structure, making it more generalizable."

---

## Cell 24: Cross-Validation Setup

**Narration:**
"Moving to Enhancement 1, we implement 10-fold stratified cross-validation with a fixed max_depth of 3. Stratification ensures each fold maintains class balance. We also prepare to generate a learning curve to assess how model performance scales with training data size."

---

## Cell 25: CV Results and Learning Curve

**Narration:**
"We execute cross-validation and generate two visualizations: a boxplot showing the distribution of accuracy across folds, and a learning curve plotting training and validation accuracy against training set size. The converging curves indicate the model is well-fitted without significant overfitting or underfitting. The mean CV accuracy is approximately 96% with low variance across folds."

---

## Cell 26: CV Interpretation

**Narration:**
"The key insight: converging training and validation curves mean the model generalizes well. If the gap widened, it would indicate overfitting. Here, the small gap confirms our depth-3 model is appropriately regularized."

---

## Cell 27: Class Imbalance Simulation

**Narration:**
"For Enhancement 2, we simulate a real-world scenario with class imbalance by reducing versicolor to 20 samples and virginica to just 10 samples, while keeping setosa at 30. This creates a 30-20-10 distribution that mimics imbalanced datasets."

---

## Cell 28: Class Balancing Comparison

**Narration:**
"We train two models on this imbalanced data: one without class weighting and one with class_weight='balanced'. The classification reports show that without balancing, the minority class virginica has lower recall. With balancing, the model pays more attention to minority classes, improving their recall at a slight cost to precision. This demonstrates the importance of class_weight for imbalanced datasets."

---

## Cell 29: Confusion Matrix Comparison

**Narration:**
"We visualize the confusion matrices side by side. The balanced model shows improved classification of the minority class virginica, with fewer false negatives. This confirms that class_weight='balanced' effectively redistributes the model's attention to underrepresented classes."

---

## Cell 30: Early Stopping Concept

**Narration:**
"Enhancement 3 introduces a custom early stopping mechanism called SlopeEarlyStopping. Instead of monitoring validation error, this callback monitors the slope of the accuracy curve across increasing tree depths. When the slope flattens below a threshold, indicating diminishing returns, training stops."

---

## Cell 31: Early Stopping Implementation

**Narration:**
"We implement the SlopeEarlyStopping class with three parameters: a window size for slope calculation, a slope threshold, and patience to avoid stopping on noise. The callback uses linear regression on a sliding window of accuracy scores to compute the slope. When the absolute slope falls below the threshold for a specified number of iterations, it signals to stop."

---

## Cell 32: Early Stopping Execution

**Narration:**
"We run the early stopping algorithm, testing depths from 1 to 19. For each depth, we compute 5-fold CV accuracy and feed it to the stopper. The output shows depth, CV accuracy, current slope, and whether stopping was triggered. The algorithm stops when the improvement slope flattens, identifying the optimal depth automatically."

---

## Cell 33: Early Stopping Visualization

**Narration:**
"We plot two graphs: CV accuracy vs depth showing where performance plateaus, and the slope vs depth showing when improvement slows. The vertical line marks the best depth found. This demonstrates how early stopping can automate hyperparameter selection."

---

## Cell 34: Feature Importance Setup

**Narration:**
"Enhancement 4 analyzes feature importance from our depth-3 tree. Feature importance measures how much each feature contributes to the splitting decisions, computed as the normalized total reduction in impurity brought by that feature."

---

## Cell 35: Feature Importance Results

**Narration:**
"We extract and visualize feature importances. The bar chart confirms that petal length and petal width are the most important features, while sepal features contribute minimally. This aligns with our earlier observation from the tree structure where petal features dominated the root splits."

---

## Cell 36: GridSearchCV Setup

**Narration:**
"Enhancement 5 performs comprehensive hyperparameter tuning using GridSearchCV. We define a parameter grid exploring 2 criteria, multiple max_depth values, min_samples_split values, and min_samples_leaf values, resulting in 90 combinations. We use 10-fold CV to evaluate each combination."

---

## Cell 37: GridSearchCV Results

**Narration:**
"GridSearchCV evaluates all 90 combinations and identifies the best parameters: gini criterion with max_depth=3, min_samples_leaf=5, and min_samples_split=2. We display the top 5 configurations with their mean test scores and standard deviations. The best model achieves approximately 96% CV accuracy."

---

## Cell 38: Optimized Model Evaluation

**Narration:**
"We retrain the best model on the full training set and evaluate on the test set. The optimized model achieves approximately 96% test accuracy with a tree depth of 3 and 5 leaf nodes. The classification report shows strong performance across all classes with high precision, recall, and F1 scores."

---

## Cell 39: Optimized Tree Visualization

**Narration:**
"We visualize the final optimized decision tree. This is our recommended model - simple, interpretable, and achieving high accuracy through careful hyperparameter tuning."

---

## Cell 40: Standardization Comparison

**Narration:**
"Enhancement 6 demonstrates an important property of decision trees: they are invariant to feature scaling. We compare a plain decision tree with one wrapped in a StandardScaler pipeline. Both achieve identical CV accuracy, confirming that standardization provides no benefit for tree-based models since they make decisions based on ordered splits, not distances."

---

## Cell 41: Final Summary

**Narration:**
"We conclude with our final recommended model. The GridSearchCV-optimized decision tree with parameters gini, max_depth=3, min_samples_leaf=5, and min_samples_split=2 achieves approximately 96% CV accuracy and 96% test accuracy. This model balances performance with interpretability, using only 3 levels and 5 leaf nodes, making it both accurate and explainable."

---

## Cell 42: Analysis Question 1

**Narration:**
"Question 1 asks about the role of the criterion parameter. Gini index measures the probability of misclassification, while entropy measures information gain based on information theory. Both optimize the same goal of creating pure splits, which is why they produce similar results on Iris."

---

## Cell 43: Analysis Question 2

**Narration:**
"Question 2 addresses max_depth effect. Low depth causes underfitting with high bias as the model is too simple. Optimal depth balances bias and variance. High depth causes overfitting with high variance as the model memorizes training noise."

---

## Cell 44: Analysis Question 3

**Narration:**
"Question 3 covers min_samples_split and min_samples_leaf. Both parameters impose minimum sample thresholds that limit tree growth. Higher values produce simpler, more generalizable models by preventing the tree from creating splits based on very few samples."

---

## Cell 45: Analysis Question 4

**Narration:**
"Question 4 asks which hyperparameter is most impactful. Max_depth is the most critical because it causes sharp transitions from underfitting to overfitting. The other parameters fine-tune gradually, but max_depth fundamentally controls model complexity."

---

## Cell 46: Analysis Question 5

**Narration:**
"Question 5 requests the recommended model. Based on our GridSearchCV over 90 combinations with 10-fold CV, we recommend: criterion=gini, max_depth=3, min_samples_leaf=5, min_samples_split=2. This achieves 96% CV accuracy while maintaining simplicity and interpretability."

---

## Cell 47: Final Visualization

**Narration:**
"We end with a final visualization of our recommended model and print the final metrics: approximately 96% test accuracy and 96% 10-fold CV accuracy. This completes our comprehensive analysis of decision trees on the Iris dataset."

---

## Presentation Tips

1. **Execute cells in order** - Don't skip ahead, the narrative builds sequentially
2. **Pause for questions** - After each major section (Tasks 1-9, Enhancements)
3. **Highlight key numbers** - 96% accuracy, depth 3, petal features dominate
4. **Emphasize the tradeoff** - Underfitting vs overfitting is the central theme
5. **Mention practical relevance** - Early stopping, class balancing, and feature importance are applicable to real-world problems
