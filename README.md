# SocialLens — Machine Learning Model Comparison Pipeline

An end-to-end classification project focused on rigorous model *selection*, not just model training — seven classical ML algorithms are trained and compared under identical, leakage-safe conditions to pick a final model on evidence rather than a single lucky train/test split.

## Stack
`pandas` · `numpy` · `scikit-learn` (Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder) · `matplotlib` / `seaborn` · `joblib`

## Approach
1. **Preprocessing** — a `ColumnTransformer` handles numeric and categorical features separately: imputation (`SimpleImputer`) for missing values, `StandardScaler` for numeric columns, `OneHotEncoder` for categorical ones — all wrapped in a single `Pipeline` so preprocessing is fit only on training folds (no leakage).
2. **Stratified split** — `train_test_split(..., stratify=y)` keeps class balance consistent across train/test.
3. **Model bank** — Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, and AdaBoost are each wrapped in the same preprocessing pipeline.
4. **Cross-validation** — `StratifiedKFold(n_splits=5, shuffle=True)` scores every model on training data before it ever touches the test set.
5. **Selection** — the final model is chosen by comparing cross-validation mean/std accuracy *and* held-out test accuracy side by side, not by whichever model happened to score highest on one split.
6. **Final evaluation** — confusion matrix and classification report on the selected model.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)

for name, model in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"{name} | CV Mean: {cv_scores.mean():.4f} | Test Acc: {test_acc:.4f}")
```

## Why this project matters
Most beginner ML projects train one model and report accuracy. This one demonstrates the actual practice of model selection: consistent preprocessing across candidates, cross-validation to avoid overfitting to one split, and a final decision backed by two independent accuracy estimates.

How to run the website:
1- Download files
2- Open command prompt or any terminal
3- CD to the folder in which the files are location
4- Then type py -m streamlit run website.py
5- A window should open with the website
6- Enjoy :)
