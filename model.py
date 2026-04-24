import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load data - changed sep to comma
df = pd.read_csv('student-por.csv', sep=',')

# Fix column names
df.columns = df.columns.str.strip()

print("Columns:", df.shape)
print("G3 found:", 'G3' in df.columns)

# Create pass/fail column
df['pass'] = (df['G3'] >= 10).astype(int)

# Select features
features = ['studytime','failures','absences',
            'Medu','Fedu','goout','health']
X = df[features]
y = df['pass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Check accuracy
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained!")
print(f"Accuracy: {acc:.2%}")

# Save model
pickle.dump(model, open('model.pkl','wb'))
print("Model saved!")