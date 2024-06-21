from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def perform_grid_search(X_train, y_train, param_grid, cv=5, class_weight='balanced', random_state=42):
    """
    Performs grid search to find the best Random Forest parameters.

    Parameters:
    - X_train: Training feature matrix.
    - y_train: Training target variable.
    - param_grid: Grid of parameters to search over.
    - cv: Number of cross-validation folds.
    - class_weight: Weights associated with classes.
    - random_state: Seed used by the random number generator.

    Returns:
    - best_params: Best parameters found by grid search.
    """
    rf = RandomForestClassifier(random_state=random_state, class_weight=class_weight)
    grid_search = GridSearchCV(rf, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

from sklearn.model_selection import train_test_split

def train_random_forest(X, y, best_params, test_size=0.3, random_state=42, class_weight='balanced'):
    """
    Trains a Random Forest model with the given best parameters.

    Parameters:
    - X: Feature matrix.
    - y: Target variable.
    - best_params: Best parameters for RandomForestClassifier.
    - test_size: Fraction of the dataset to be used as test set.
    - random_state: Seed used by the random number generator.
    - class_weight: Weights associated with classes.

    Returns:
    - model: Trained Random Forest model.
    - X_train, X_test, y_train, y_test: Split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = RandomForestClassifier(random_state=random_state, class_weight=class_weight, **best_params)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test