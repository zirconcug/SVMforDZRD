import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# read data
df = pd.read_excel('trainingdata.xlsx')

# training model
X = df[['Dose', 'FWHM']].values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma=0.4, max_iter=10000, class_weight='balanced')) # set the optimal values for C and gamma.
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# predict unknown sample data
df_unknown = pd.read_excel('predicteddata.xlsx') 
X_unknown = df_unknown[['Dose', 'FWHM']].values
predictions = model.predict(X_unknown)
df_unknown['predicted_label'] = predictions
df_unknown.to_excel('predictions.xlsx', index=False)

plt.figure(figsize=(6.2, 4.5))
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.legend(['Unannealed zircon', 'Annealed zircon'], loc='upper right')
plt.xlabel('$Dose (Alpha-events x 10^{15}/mg)$')
plt.ylabel('$v_{3}SiO_{4} FWHM (cm^{-1})$')
plt.title('SVM Predection')
plt.scatter(X_unknown[df_unknown['predicted_label'] == 0][:, 0], X_unknown[df_unknown['predicted_label'] == 0][:, 1], marker='s', c='blue', s=50, edgecolor='black', label='Predicted Type 0')
plt.scatter(X_unknown[df_unknown['predicted_label'] == 1][:, 0], X_unknown[df_unknown['predicted_label'] == 1][:, 1], marker='s', c='orange', s=50, edgecolor='black', label='Predicted Type 1')
plt.xlim(0, 22)
plt.ylim(0, 34)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1.5)
plt.gca().xaxis.set_major_locator(MultipleLocator(4))
plt.gca().yaxis.set_major_locator(MultipleLocator(4))  
plt.savefig('prediction.pdf')
plt.show()