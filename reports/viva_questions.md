# Viva / Teacher Questions — Heart Disease Risk Predictor

## Q1: Why is feature scaling (StandardScaler) needed for SVM and KNN but not for Decision Trees?

**Answer:** SVM and KNN are distance-based algorithms. SVM uses distances to find the optimal separating hyperplane, and KNN calculates distances between data points to find neighbors. If features have different scales (e.g., age ranges 29–77 while cholesterol ranges 126–564), the larger-scale feature dominates the distance calculation, biasing the model. StandardScaler normalizes all features to mean=0, std=1, giving them equal influence.

Decision Trees and Random Forests, on the other hand, split on individual features using thresholds — they compare values within a single feature, not across features. So scaling doesn't affect their performance at all.

---

## Q2: Why do you use stratified splitting instead of regular random splitting?

**Answer:** Stratified splitting ensures the class distribution in both train and test sets matches the original dataset. Our dataset has ~54% disease and ~46% no disease. With regular random splitting on a small dataset (303 rows), we might accidentally get a test set with 70% disease or 30% disease, which would make our evaluation misleading.

Stratified splitting guarantees both sets have approximately the same 54/46% ratio, giving us a more reliable estimate of real-world performance. This is especially important with small datasets where random variation can significantly skew the split.

---

## Q3: What does overfitting look like in a Decision Tree, and how do you prevent it?

**Answer:** An overfitted Decision Tree memorizes the training data by growing very deep — creating a leaf node for almost every sample. Signs of overfitting include:
- **Training accuracy = 100%** but much lower test accuracy (large gap).
- Very deep trees with many nodes and few samples per leaf.
- The tree has learned noise and outliers rather than general patterns.

**Prevention methods:**
1. **Limit max_depth** (we tuned this: 3, 5, 7, None) — shallower trees capture general patterns, not noise.
2. **Set min_samples_split** (we tuned: 2, 5, 10) — requires a minimum number of samples to split a node.
3. **Pruning:** Remove branches that don't improve validation performance.
4. **Use Random Forest** — an ensemble of many trees that averages out individual tree overfitting.

In our results, the Decision Tree with max_depth=3 performed best, confirming that restricting depth prevents overfitting.

---

## Q4: Why do you use cross-validation instead of just comparing models on the test set?

**Answer:** With only 303 samples (61 in the test set), a single train/test split can produce misleading results due to random variation — the particular 61 samples chosen might happen to be easy or hard for a specific model.

**5-fold cross-validation** addresses this by:
1. Splitting the training data into 5 equal folds.
2. Training on 4 folds and validating on the 5th, rotating through all 5.
3. Averaging the 5 scores to get a more stable estimate.

This gives us both a **mean score** (how well the model performs on average) and a **standard deviation** (how stable it is across different subsets). For example, KNN had the highest CV F1 (0.853) with the lowest std (0.037), suggesting stable but not necessarily generalizable performance.

We use CV scores to **select** the model and the held-out test set to **confirm** — never optimizing directly on the test set.

---

## Q5: What is data leakage, and how does your project prevent it?

**Answer:** Data leakage occurs when information from the test set accidentally influences the training process, leading to overly optimistic results that don't reflect real-world performance.

**Common example:** If you apply StandardScaler to the entire dataset *before* splitting, the scaler's mean and standard deviation are computed using test samples. The model then "knows" something about the test distribution during training, which inflates performance metrics.

**How we prevent it:**
1. **Pipeline architecture:** Our preprocessing (StandardScaler + OneHotEncoder) is inside a scikit-learn `Pipeline`. When `GridSearchCV` calls `pipe.fit()`, the scaler learns parameters only from the training fold. When it calls `pipe.predict()`, it applies those same parameters to the validation/test fold without refitting.
2. **Strict train/test separation:** The 20% test set is held out before any model training. We never tune hyperparameters using test-set performance.
3. **ColumnTransformer:** Handles numeric and categorical features separately, each fitted only on training data.

This ensures our reported metrics (accuracy, F1, etc.) are honest estimates of how the model would perform on truly unseen patient data.
