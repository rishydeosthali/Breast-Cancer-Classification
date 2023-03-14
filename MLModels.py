from sklearn import linear_model

# Binary boundary classifier function
# Classify whether a cell is malignant of beningn based on single input parameter and target boundary value
# the input parameter is contained in the input_mean_series array. The target boundary can be specified in the target_boundary variable

def boundary_classifier(target_boundary, classifier_para_series):
  result = []
  for i in range(len(classifier_para_series)):
    if(classifier_para_series[i]>target_boundary):
      result.append(1)
      
    else:
      result.append(0)
     
  return result
  

def linreg(X,Y):
   model = linear_model.LinearRegression()
   model.fit(X,Y)
   preds = model.predict(X)

   return preds
   
def logreg_train(X,Y):
   logreg_model = linear_model.LogisticRegression()
   logreg_model.fit(X,Y)
   
   return logreg_model
   
def logreg_predict(logreg_model,X):
   Y = logreg_model.predict(X)
   
   return Y