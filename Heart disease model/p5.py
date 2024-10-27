# Heart disease checking model 


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd 


df = pd.read_csv('Heart_Disease_Prediction.csv')


x = df.drop(columns='HeartDisease')
y = df['HeartDisease'] 

model = DecisionTreeClassifier()
model.fit(x,y)

prediction = model.predict([[70,1,4,130,322,0,2,109,0,2.4,2,3,3]])

print(prediction)
