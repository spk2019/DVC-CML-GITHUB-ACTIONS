import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





from sklearn.datasets import load_wine
X,y = load_wine(return_X_y=True)

#Train_test_split
x_train, x_test, y_train, y_test = train_test_split(
 X, y, test_size=0.1, random_state=42)





model = RandomForestClassifier().fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

#save metrics to metrics.txt
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    model, x_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("confusion_matrix.png")


#create dataset
y_pred = model.predict(x_test)
data = pd.DataFrame({'predicted': y_pred,'actual':y_test})
data.to_csv("classes.csv",index=False)
