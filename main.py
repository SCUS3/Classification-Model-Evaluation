import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support

# Load data
data = pd.read_csv('cancer-data-train.csv')

X_train = data.iloc[:, :-1]
Y_train = data.iloc[:, -1]
X_test = data.iloc[:, :-1]
Y_test = data.iloc[:, -1]

# (a) Confifure SVM: Tune C parameter
Cs = [0.01, 0.1, 1, 10, 100]
svm_scores = []
for c in Cs:
    svm = SVC(kernel='linear', C=c)
    cv_scores = cross_validate(svm, X_train, Y_train, scoring='f1_macro', cv=10)
    svm_scores.append(cv_scores['test_score'].mean())

plt.figure(figsize=(8, 6))
plt.plot(Cs, svm_scores, marker='o')
plt.xscale('log')
plt.xlabel('C (Log Scale)')
plt.ylabel('Average F-measure')
plt.title('SVM: Tuning C Parameter')
plt.show()

# (b) LDA, Naive Bayes, Nearest Neighbor: kNN - Tune k parameter
ks = [1, 2, 5, 10, 50]
knn_scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_validate(knn, X_train, Y_train, scoring='f1_macro', cv=10)
    knn_scores.append(cv_scores['test_score'].mean())

plt.figure(figsize=(8, 6))
plt.plot(ks, knn_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Average F-measure')
plt.title('kNN: Tuning k Parameter')
plt.show()

# (c) Confifure a Multi layer perception: MLP - Evaluate different configurations
mlp_configs = [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)]
mlp_scores = []
for config in mlp_configs:
    mlp = MLPClassifier(hidden_layer_sizes=config)
    cv_scores = cross_validate(mlp, X_train, Y_train, scoring='f1_macro', cv=10)
    mlp_scores.append(cv_scores['test_score'].mean())

plt.figure(figsize=(10, 6))
plt.bar(range(len(mlp_scores)), mlp_scores)
plt.xticks(range(len(mlp_configs)), mlp_configs)
plt.xlabel('MLP Configuration')
plt.ylabel('Average F-measure')
plt.title('MLP: Evaluating Configurations')
plt.show()

# (d) Compare classifiers
best_svm = SVC(kernel='linear', C=1)
best_knn = KNeighborsClassifier(n_neighbors=5)
best_mlp = MLPClassifier(hidden_layer_sizes=(50, 50))

models = {'SVM': best_svm, 'kNN': best_knn, 'MLP': best_mlp, 'LDA': LDA(), 'NB': GaussianNB()}

for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(Y_test, y_pred, average='macro')
    print(name)
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
