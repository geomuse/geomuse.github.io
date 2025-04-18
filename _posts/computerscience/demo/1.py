#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

url = 'https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv'
default = pd.read_csv(url)
print(default.head())
print(default.info())
print(default['Default'].value_counts())

y = default['Default']
X = default.drop(['Default'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

#%%
len(X_resampled) , len(X_train) , len(X)
#%%
model = LogisticRegression()
model.fit(X_resampled,y_resampled)
y_pred = model.predict(X_test)

confusion_matrix(y_test,y_pred)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
