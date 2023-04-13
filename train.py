import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,roc_auc_score,precision_score,f1_score,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





from sklearn.datasets import load_wine
X,y = load_wine(return_X_y=True)

#Train_test_split
x_train, x_test, y_train, y_test = train_test_split(
 X, y, test_size=0.1, random_state=42)





model = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = model.score(x_test, y_test)
f1 = f1_score(y_test,y_pred,average="macro")
precision = precision_score(y_test,y_pred,average="macro")
print(acc)

#save metrics to metrics.txt
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")
    outfile.write("f1 Score: " + str(f1) + "\n")
    outfile.write("Precision: " + str(precision) + "\n")


# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    model, x_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("confusion_matrix.png")


#create dataset

data = pd.DataFrame({'predicted': y_pred,'actual':y_test})
data.to_csv("classes.csv",index=False)
