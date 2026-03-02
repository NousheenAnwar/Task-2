# ============================================================
#   CREDIT RISK PREDICTION
#   Dataset: credit_risk_dataset.csv
#   Goal: Predict if a loan applicant will DEFAULT or NOT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier        # Our ML model
from sklearn.model_selection import train_test_split   # Split data into train/test
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder         # Convert text to numbers

# STEP 1: LOAD THE DATASET
# pd.read_csv() reads a CSV file and stores it as a DataFrame

df = pd.read_csv("credit_risk_dataset.csv")

print(f"   Rows    : {df.shape[0]:,}")   # How many people in the dataset
print(f"   Columns : {df.shape[1]}")      # How many features/columns

print("\n   First 5 rows:")
print(df.head())

# person_age              → Age of the applicant
# person_income           → Annual income
# person_home_ownership   → RENT / OWN / MORTGAGE / OTHER
# person_emp_length       → How long they've been employed (years)
# loan_intent             → Why they need the loan (EDUCATION, MEDICAL, etc.)
# loan_grade              → Bank's rating of the loan (A=best, G=worst)
# loan_amnt               → How much money they want to borrow
# loan_int_rate           → Interest rate on the loan
# loan_percent_income     → Loan amount as % of income
# cb_person_default_on_file → Has this person defaulted before? (Y/N)
# cb_person_cred_hist_length → How many years of credit history
# loan_status             → TARGET: 1 = Defaulted, 0 = Did NOT default

# 2a. Check data types and missing values
print("\n   Column Info (data types + missing values):")
print(df.info())

# 2b. Check how many values are missing in each column
print("\n   Missing Values per Column:")
print(df.isnull().sum())
# Only loan_int_rate and person_emp_length have missing values.

# 2c. Look at basic statistics (min, max, average, etc.)
print("\n   Basic Statistics:")
print(df.describe())

# 2d. How many people defaulted vs didn't?
print("\n   Target Column — loan_status counts:")
print(df['loan_status'].value_counts())
print("   (0 = No Default,  1 = Defaulted)")

# A simple strategy: fill with the MEDIAN (middle value)
# We use median instead of average because it's not affected by extreme values

print(f"\n   Missing in 'loan_int_rate'    : {df['loan_int_rate'].isnull().sum()}")
print(f"   Missing in 'person_emp_length': {df['person_emp_length'].isnull().sum()}")

# Fill missing numbers with the median of that column
df['loan_int_rate']     = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())

# Remove rows where age is unrealistically high (e.g. age 144 is impossible)
df = df[df['person_age'] <= 100]

print("\n   ✔ Missing values filled with median!")
print(f"   ✔ Missing after fix: {df.isnull().sum().sum()} (should be 0)")
print(f"   ✔ Dataset size after cleaning: {df.shape[0]:,} rows")

# We'll draw 6 simple charts to explore the data and see what patterns we can find before we train the model.
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Credit Risk Dataset — Exploratory Data Analysis",
             fontsize=16, fontweight='bold')

# ── Chart 1: How many people defaulted vs didn't? ────────────
ax = axes[0, 0]
counts = df['loan_status'].value_counts()
ax.bar(['No Default (0)', 'Default (1)'], counts.values,
       color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
ax.set_title('How Many People Defaulted?', fontweight='bold')
ax.set_ylabel('Number of People')
# Add count labels on top of each bar
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, counts.max() * 1.25)

# ── Chart 2: What loan amounts are most common? ───────────────
ax = axes[0, 1]
ax.hist(df[df['loan_status'] == 0]['loan_amnt'], bins=30, alpha=0.6,
        color='#2ecc71', label='No Default', edgecolor='white')
ax.hist(df[df['loan_status'] == 1]['loan_amnt'], bins=30, alpha=0.6,
        color='#e74c3c', label='Default', edgecolor='white')
ax.set_title('Loan Amount: Who Defaults?', fontweight='bold')
ax.set_xlabel('Loan Amount ($)')
ax.set_ylabel('Number of People')
ax.legend()

# ── Chart 3: Does loan grade affect default rate? ─────────────
ax = axes[0, 2]
# Calculate the % of people who defaulted for each loan grade
grade_default_rate = df.groupby('loan_grade')['loan_status'].mean() * 100
grade_default_rate = grade_default_rate.reindex(['A','B','C','D','E','F','G'])
grade_default_rate.plot(kind='bar', ax=ax, color='#3498db', edgecolor='white', rot=0)
ax.set_title('Default Rate by Loan Grade\n(A=Best, G=Worst)', fontweight='bold')
ax.set_xlabel('Loan Grade')
ax.set_ylabel('Default Rate (%)')
# Add % labels on bars
for i, v in enumerate(grade_default_rate):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

# ── Chart 4: Does interest rate differ for defaulters? ────────
ax = axes[1, 0]
ax.hist(df[df['loan_status'] == 0]['loan_int_rate'], bins=30, alpha=0.6,
        color='#2ecc71', label='No Default', edgecolor='white')
ax.hist(df[df['loan_status'] == 1]['loan_int_rate'], bins=30, alpha=0.6,
        color='#e74c3c', label='Default', edgecolor='white')
ax.set_title('Interest Rate: Who Defaults?', fontweight='bold')
ax.set_xlabel('Interest Rate (%)')
ax.set_ylabel('Number of People')
ax.legend()

# ── Chart 5: What is each loan used for? ─────────────────────
ax = axes[1, 1]
intent_rate = df.groupby('loan_intent')['loan_status'].mean().sort_values() * 100
intent_rate.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='white')
ax.set_title('Default Rate by Loan Purpose', fontweight='bold')
ax.set_xlabel('Default Rate (%)')
for i, v in enumerate(intent_rate):
    ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=9)

# ── Chart 6: Has the person defaulted before? ────────────────
ax = axes[1, 2]
prev_default = df.groupby('cb_person_default_on_file')['loan_status'].mean() * 100
ax.bar(['No Previous\nDefault (N)', 'Previous\nDefault (Y)'],
       prev_default.reindex(['N','Y']).values,
       color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
ax.set_title('Default Rate: Previous Default History', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
for i, v in enumerate(prev_default.reindex(['N','Y']).values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_beginner.png', dpi=150, bbox_inches='tight')
plt.close()


# We need to convert text columns (like 'RENT', 'EDUCATION') into numbers

# LabelEncoder converts text categories to numbers automatically
# e.g.  RENT→0, MORTGAGE→1, OWN→2, OTHER→3

le = LabelEncoder()

# List of columns that contain text (not numbers)
text_columns = ['person_home_ownership', 'loan_intent',
                'loan_grade', 'cb_person_default_on_file']

for col in text_columns:
    df[col] = le.fit_transform(df[col])
    print(f"   ✔ '{col}' converted to numbers")

# Now pick which columns to use as INPUT features (X)
# and which column is the TARGET we want to predict (y)

feature_columns = [
    'person_age',           # Age
    'person_income',        # Income
    'person_home_ownership',# Home ownership (now a number)
    'person_emp_length',    # Employment length
    'loan_intent',          # Loan purpose (now a number)
    'loan_grade',           # Loan grade (now a number)
    'loan_amnt',            # Loan amount
    'loan_int_rate',        # Interest rate
    'loan_percent_income',  # Loan as % of income
    'cb_person_default_on_file',      # Previous default (now a number)
    'cb_person_cred_hist_length'      # Credit history length
]

X = df[feature_columns]   # Input features (what we know about the person)
y = df['loan_status']     # Target (what we want to predict: 0 or 1)

print(f"\n   Input  (X) shape: {X.shape}  ← {X.shape[1]} features, {X.shape[0]:,} people")
print(f"   Target (y) shape: {y.shape}  ← 1 value per person (0 or 1)")

# We split our data into:
#   Training set (80%) → model learns from this
#   Testing  set (20%) → we test how well the model does on NEW data
#
# Why split? If we test on data the model already saw, it's like giving
# someone the answers before the exam — not a fair test!

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to testing
    random_state=42,    # random_state=42 means we get the same split every time
    stratify=y          # make sure both sets have the same ratio of 0s and 1s
)


print(f"\n   Training set : {len(X_train):,} rows  (80%) ← model learns from this")
print(f"   Testing  set : {len(X_test):,}  rows  (20%) ← we test on this")

# We use a Decision Tree — a simple model that works like a flowchart:
# "Is interest rate > 12%?  → Yes → Is loan_grade D or worse? → ..."
#
# max_depth=6 means the tree can ask at most 6 questions deep
# (keeps it simple and prevents overfitting)

model = DecisionTreeClassifier(max_depth=6, random_state=42)

# .fit() = TRAIN the model (it learns patterns from training data)
model.fit(X_train, y_train)

print("\n   ✔ Model trained successfully!")


# Now we test the model on data it has NEVER seen before (X_test)

# .predict() → model looks at each person's features and guesses 0 or 1
y_pred = model.predict(X_test)

# ── Accuracy ──────────────────────────────────────────────────
# Accuracy = how many predictions were CORRECT out of all predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   📌 Accuracy: {accuracy * 100:.2f}%")
print(f"      → The model correctly predicted {accuracy * 100:.2f}% of loan outcomes")

# ── Classification Report ─────────────────────────────────────
# Shows precision, recall, and F1-score for each class
# Precision = when model says "Default", how often is it right?
# Recall    = of all actual defaults, how many did the model catch?
print("\n   📋 Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['No Default (0)', 'Default (1)']))

# ── Confusion Matrix ──────────────────────────────────────────
# A table showing:
#   True Negatives  (TN) → correctly predicted NO default
#   False Positives (FP) → predicted default, but they DIDN'T default
#   False Negatives (FN) → predicted no default, but they DID default ← costly!
#   True Positives  (TP) → correctly predicted DEFAULT

cm = confusion_matrix(y_test, y_pred)

print("\n   🔢 Confusion Matrix (raw numbers):")
print(f"   {cm}")

tn, fp, fn, tp = cm.ravel()
print(f"\n   Breaking it down:")
print(f"   ✔ True  Negatives  (correct 'No Default') : {tn:,}")
print(f"   ✖ False Positives  (wrong  'Default')      : {fp:,}")
print(f"   ✖ False Negatives  (missed real defaults)  : {fn:,}  ← most important to minimise!")
print(f"   ✔ True  Positives  (correct 'Default')    : {tp:,}")


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Evaluation — Decision Tree', fontsize=15, fontweight='bold')

# ── Plot 1: Confusion Matrix heatmap ─────────────────────────
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted\nNo Default', 'Predicted\nDefault'],
            yticklabels=['Actual\nNo Default', 'Actual\nDefault'],
            linewidths=2, linecolor='white', cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Confusion Matrix', fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# ── Plot 2: Feature Importance — which columns matter most? ──
ax = axes[1]
# Get importance score for each feature (higher = more useful)
importances = pd.Series(model.feature_importances_, index=feature_columns)
importances = importances.sort_values(ascending=True)
# Color the top feature red, rest blue
colors = ['#e74c3c' if v == importances.max() else '#3498db' for v in importances]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Feature Importance\n(Which features matter most?)', fontweight='bold')
ax.set_xlabel('Importance Score')
for i, v in enumerate(importances):
    ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

# ── Plot 3: Accuracy summary bar ─────────────────────────────
ax = axes[2]
scores = ['Accuracy']
values = [accuracy * 100]
bar = ax.bar(scores, values, color='#3498db', edgecolor='white', width=0.4)
ax.set_ylim(0, 100)
ax.set_title('Model Accuracy', fontweight='bold')
ax.set_ylabel('Score (%)')
ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% target line')
ax.legend()
ax.text(0, values[0] + 1.5, f'{values[0]:.2f}%',
        ha='center', fontsize=14, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.savefig('model_results_beginner.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   ✔ Result charts saved as 'model_results_beginner.png'")


print(f"""
   Dataset        : credit_risk_dataset.csv
   Total Records  : {len(df):,} loan applicants
   Features Used  : {len(feature_columns)} columns

   Model          : Decision Tree (max_depth = 6)
   Training Size  : {len(X_train):,} rows
   Testing  Size  : {len(X_test):,} rows

   ✅ Accuracy     : {accuracy * 100:.2f}%
   ✅ True Positives  (defaults caught)   : {tp:,}
   ⚠️  False Negatives (defaults missed)  : {fn:,}

   Top Feature    : {importances.idxmax()}
   (This was the most useful column for predicting defaults)
""")
