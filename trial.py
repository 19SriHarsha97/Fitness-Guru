# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("megaGymDataset.csv")
# # print(df.head())
# # print(df.tail())

# #print(df['BodyPart'].unique())
# #['Abdominals' 'Adductors' 'Abductors' 'Biceps' 'Calves' 'Chest' 'Forearms''Glutes' 'Hamstrings' 'Lats' 'Lower Back' 'Middle Back' 
# # 'Traps' 'Neck' 'Quadriceps' 'Shoulders' 'Triceps']

# # print(df['Equipment'].unique())
# # ['Bands' 'Barbell' 'Kettlebells' 'Dumbbell' 'Other' 'Cable' 'Machine' 'Body Only' 'Medicine Ball' 'Exercise Ball' 'Foam Roll']

# equipment = ['Dumbbell','Barbell','Cable','Machine','Body Only']
# muscles = list(map(str,input("Enter the muscles u want to work today: ").split()))
# print("Your workout is as follows: ")
# # print(df[df['BodyPart'].isin(muscles)].head())
# for i in muscles:
#     print("Workout for",i)
#     workout = []
#     for j in equipment:
#         exercises = df[(df['BodyPart']==i) & (df['Equipment']==j)]['Title'][:1]
#         workout.extend(exercises.tolist())
#     for k in range(0,5,2):
#         if(k==0 or k==2):
#             print(workout[k]+" / "+workout[k+1]+" - 2 heavy sets to failure")
#         else:
#             print(workout[k]+ " - 3 sets of 15 reps")
#     print('------------------------------------------------------------')




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("megaGymDataset.csv")

# Encode categorical variables
# Encode 'BodyPart'
bodypart_encoder = LabelEncoder()
df['BodyPart_encoded'] = bodypart_encoder.fit_transform(df['BodyPart'])

# Encode 'Equipment'
equipment_encoder = LabelEncoder()
df['Equipment_encoded'] = equipment_encoder.fit_transform(df['Equipment'])

# Define features and target
X = df[['BodyPart_encoded', 'Equipment_encoded']]
y = df['Title']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    random_state=42        # Ensures reproducibility of results
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Function to recommend exercise based on body part and equipment
def recommend_exercise(body_part, equipment):
    # Encode the input
    body_part_encoded = bodypart_encoder.transform([body_part])[0]
    equipment_encoded = equipment_encoder.transform([equipment])[0]
    
    # Make prediction
    prediction = model.predict([[body_part_encoded, equipment_encoded]])
    
    # Return the predicted exercise
    return prediction[0]

# Example usage
user_body_part = 'Lats'  # Example input
user_equipment = 'Barbell'  # Example input
recommended_exercise = recommend_exercise(user_body_part, user_equipment)
print(f'Recommended Exercise: {recommended_exercise}')
