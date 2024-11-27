
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import pandas as pd
import pickle

max_iter = 250

# read cli arguments
improper_input_reason = ''
if len(sys.argv) == 3:
    X_filename = sys.argv[1]
    Y_filename = sys.argv[2]
else: # TODO
    print('To feature engineer, include 2 filenames as sys args (or import and call the function):\n\tfilename 1: X train data\n\tfilename 2: Y train data')
    exit()
    
# Read in provided csvs
print('Reading in...')
sys.stdout.flush()
X = pd.read_csv(X_filename)
Y = pd.read_csv(Y_filename)

# Do logistic regression sklearn
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y['isFraud'], test_size=0.2, random_state=42)

# Train model
print('Training model...')
model = LogisticRegression(max_iter=max_iter)
model.fit(X_train, y_train)

# Predict
print('Predicting...')
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print(f'Model saved as {filename}')
