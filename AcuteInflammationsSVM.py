
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# fetch dataset 
import numpy as np
from sklearn.model_selection import LearningCurveDisplay,ShuffleSplit
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#get Dataset
acute_inflammations = fetch_ucirepo(id=184) 

# # data (as pandas dataframes) 
X = acute_inflammations.data.features 
#map categorical features to numeric values
X['nausea']= np.where(X["nausea"] == 'no',0,1)
X['lumbar-pain']= np.where(X["lumbar-pain"] == 'no',0,1)
X['urine-pushing']= np.where(X["urine-pushing"] == 'no',0,1)
X['micturition-pains']= np.where(X["micturition-pains"] == 'no',0,1)
X['burning-urethra']= np.where(X["burning-urethra"] == 'no',0,1)
#Y contains two classes, which we will split into two targets
y = acute_inflammations.data.targets 
y['bladder-inflammation']= np.where(y["bladder-inflammation"] == 'no',0,1)
y['nephritis']= np.where(y["nephritis"] == 'no',0,1)
inflamamationY = y['bladder-inflammation']
nephritisY = y['nephritis']
infX_train, infX_test, infy_train, infy_test = train_test_split(X, inflamamationY, test_size=0.2, random_state=42)
nephX_train, nephX_test, nephy_train, nephy_test = train_test_split(X, nephritisY, test_size=0.2, random_state=42)

#create deature correlation heatmap
sns.heatmap(X.corr())

#Use grid to find optimal parameters
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}

#Use GridSearchCV to test multiple combinations of SVC parameters and pick the best ones
gridInf = GridSearchCV(svm.SVC(), param_grid, refit = True)
gridNeph = GridSearchCV(svm.SVC(), param_grid, refit = True)
#Create a model using the optimal grid and training data
inflammationModel = gridInf.fit(infX_train, infy_train) 
nephritisModel = gridNeph.fit(nephX_train, nephy_train)
#Output Best Parameters
print("Optimal Parameters for Inflammation Model:", gridInf.best_params_)
print("Optimal Parameters for Nephritis Model:", gridNeph.best_params_)

#predict using test data
inflammationModelPredictions = inflammationModel.predict(infX_test)
nephritisModelPredictions = nephritisModel.predict(nephX_test)

#Classification Reports
print(classification_report(infy_test, inflammationModelPredictions)) 
print(classification_report(nephy_test, nephritisModelPredictions)) 
#confusion matrices
cm = confusion_matrix(infy_test, inflammationModelPredictions, labels=inflammationModel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=inflammationModel.classes_)

disp.plot()
plt.title("confusion matrix for Inflammation")
plt.show()

cm = confusion_matrix(nephy_test, nephritisModelPredictions, labels=nephritisModel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=nephritisModel.classes_)
disp.plot()
plt.title("confusion matrix for Nephritis")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)
inf_params = {
    "X": X,
    "y": inflamamationY,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}
LearningCurveDisplay.from_estimator(gridInf.best_estimator_,**inf_params,ax=ax[0])
handles, label = ax[0].get_legend_handles_labels()
ax[0].legend(handles[:2], ["Training Score", "Test Score"])
ax[0].set_title("Learning Curve for UTI classification")

neph_params= {
    "X": X,
    "y": nephritisY,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}
LearningCurveDisplay.from_estimator(gridNeph.best_estimator_,**neph_params,ax=ax[1])
handles, label = ax[1].get_legend_handles_labels()
ax[1].legend(handles[:2], ["Training Score", "Test Score"])
ax[1].set_title("Learning Curve for Nephritis classification")
plt.show()

#scatterplots
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(infX_test))
plt.scatter(transformed[inflammationModelPredictions==0][0], transformed[inflammationModelPredictions==0][1], label='Not Inflammed', c='red')
plt.scatter(transformed[inflammationModelPredictions==1][0], transformed[inflammationModelPredictions==1][1], label='Inflammed', c='blue')

plt.legend()
plt.title("Scatterplot of Inflammation")
plt.show()

transformed = pd.DataFrame(pca.fit_transform(nephX_test))
plt.scatter(transformed[nephritisModelPredictions==0][0], transformed[nephritisModelPredictions==0][1], label='Not Nephritis', c='red')
plt.scatter(transformed[nephritisModelPredictions==1][0], transformed[nephritisModelPredictions==1][1], label='Nephritis', c='blue')

plt.legend()
plt.title("Scatterplot of Nephritis")
plt.show()
