import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression


def load_dataset(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 1:]  
    y = data[:, 0].astype(int)  
    return X, y


def split_dataset(X, y):
    train_idx = np.arange(0, len(y), 2) 
    test_idx = np.arange(1, len(y), 2) 
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def evaluate_classifiers_for_combinations(X_train, y_train, X_test, y_test):
    results = []
    n_features = X_train.shape[1]
    
    for num_features in range(1, n_features + 1):  
        for feature_indices in combinations(range(n_features), num_features):

            X_train_subset = X_train[:, feature_indices]
            X_test_subset = X_test[:, feature_indices]
            

            qda = QuadraticDiscriminantAnalysis()
            qda.fit(X_train_subset, y_train)
            y_pred_qda = qda.predict(X_test_subset)
            error_qda = 1 - accuracy_score(y_test, y_pred_qda)
            

            reg = LinearRegression()
            y_train_ls = np.where(y_train == 1, 1, -1)
            reg.fit(X_train_subset, y_train_ls)
            y_pred_ls = reg.predict(X_test_subset)
            y_pred_ls_class = np.where(y_pred_ls >= 0, 1, 2)
            error_ls = 1 - accuracy_score(y_test, y_pred_ls_class)
            
  
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_subset, y_train)
            y_pred_knn = knn.predict(X_test_subset)
            error_knn = 1 - accuracy_score(y_test, y_pred_knn)
            

            results.append({
                'num_features': num_features,
                'feature_indices': feature_indices,
                'error_qda': error_qda,
                'error_ls': error_ls,
                'error_knn': error_knn
            })
    
    return results


datasets = ['ds-1.txt', 'ds-2.txt', 'ds-3.txt']
for i, file_path in enumerate(datasets, start=1):
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    results = evaluate_classifiers_for_combinations(X_train, y_train, X_test, y_test)
    

    print(f"\nResults for Dataset {i}:")
    for result in results:
        feature_str = ", ".join(map(str, result['feature_indices']))
        print(f"Features: {feature_str} | "
              f"QDA Error: {result['error_qda']:.4f} | "
              f"LS Error: {result['error_ls']:.4f} | "
              f"1-NN Error: {result['error_knn']:.4f}")
