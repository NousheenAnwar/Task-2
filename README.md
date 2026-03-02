Output

 Rows    : 32,581
   Columns : 12

   First 5 rows:
   person_age  person_income person_home_ownership  ...  loan_percent_income cb_person_default_on_file cb_person_cred_hist_length
0          22          59000                  RENT  ...                 0.59                         Y                          3
1          21           9600                   OWN  ...                 0.10                         N                          2
2          25           9600              MORTGAGE  ...                 0.57                         N                          3
3          23          65500                  RENT  ...                 0.53                         N                          2
4          24          54400                  RENT  ...                 0.55                         Y                          4

[5 rows x 12 columns]

   Column Info (data types + missing values):
<class 'pandas.DataFrame'>
RangeIndex: 32581 entries, 0 to 32580
Data columns (total 12 columns):
 #   Column                      Non-Null Count  Dtype
---  ------                      --------------  -----
 0   person_age                  32581 non-null  int64
 1   person_income               32581 non-null  int64
 2   person_home_ownership       32581 non-null  str
 3   person_emp_length           31686 non-null  float64
 4   loan_intent                 32581 non-null  str
 5   loan_grade                  32581 non-null  str
 6   loan_amnt                   32581 non-null  int64
 7   loan_int_rate               29465 non-null  float64
 8   loan_status                 32581 non-null  int64
 9   loan_percent_income         32581 non-null  float64
 10  cb_person_default_on_file   32581 non-null  str
 11  cb_person_cred_hist_length  32581 non-null  int64
dtypes: float64(3), int64(5), str(4)
memory usage: 3.0 MB
None

   Missing Values per Column:
person_age                       0
person_income                    0
person_home_ownership            0
person_emp_length              895
loan_intent                      0
loan_grade                       0
loan_amnt                        0
loan_int_rate                 3116
loan_status                      0
loan_percent_income              0
cb_person_default_on_file        0
cb_person_cred_hist_length       0
dtype: int64

   Basic Statistics:
         person_age  person_income  person_emp_length     loan_amnt  loan_int_rate   loan_status  loan_percent_income  cb_person_cred_hist_length
count  32581.000000   3.258100e+04       31686.000000  32581.000000   29465.000000  32581.000000         32581.000000                32581.000000    
mean      27.734600   6.607485e+04           4.789686   9589.371106      11.011695      0.218164             0.170203                    5.804211    
std        6.348078   6.198312e+04           4.142630   6322.086646       3.240459      0.413006             0.106782                    4.055001    
min       20.000000   4.000000e+03           0.000000    500.000000       5.420000      0.000000             0.000000                    2.000000    
25%       23.000000   3.850000e+04           2.000000   5000.000000       7.900000      0.000000             0.090000                    3.000000    
50%       26.000000   5.500000e+04           4.000000   8000.000000      10.990000      0.000000             0.150000                    4.000000    
75%       30.000000   7.920000e+04           7.000000  12200.000000      13.470000      0.000000             0.230000                    8.000000    
max      144.000000   6.000000e+06         123.000000  35000.000000      23.220000      1.000000             0.830000                   30.000000    

   Target Column — loan_status counts:
loan_status
0    25473
1     7108
Name: count, dtype: int64
   (0 = No Default,  1 = Defaulted)

   Missing in 'loan_int_rate'    : 3116
   Missing in 'person_emp_length': 895

   ✔ Missing values filled with median!
   ✔ Missing after fix: 0 (should be 0)
   ✔ Dataset size after cleaning: 32,576 rows
   ✔ 'person_home_ownership' converted to numbers
   ✔ 'loan_intent' converted to numbers
   ✔ 'loan_grade' converted to numbers
   ✔ 'cb_person_default_on_file' converted to numbers

   Input  (X) shape: (32576, 11)  ← 11 features, 32,576 people
   Target (y) shape: (32576,)  ← 1 value per person (0 or 1)

   Training set : 26,060 rows  (80%) ← model learns from this
   Testing  set : 6,516  rows  (20%) ← we test on this

   ✔ Model trained successfully!

   📌 Accuracy: 92.23%
      → The model correctly predicted 92.23% of loan outcomes

   📋 Classification Report:
                precision    recall  f1-score   support

No Default (0)       0.91      0.99      0.95      5094
   Default (1)       0.97      0.66      0.79      1422

      accuracy                           0.92      6516
     macro avg       0.94      0.83      0.87      6516
  weighted avg       0.93      0.92      0.92      6516


   🔢 Confusion Matrix (raw numbers):
   [[5066   28]
 [ 478  944]]

   Breaking it down:
   ✔ True  Negatives  (correct 'No Default') : 5,066
   ✖ False Positives  (wrong  'Default')      : 28
   ✖ False Negatives  (missed real defaults)  : 478  ← most important to minimise!
   ✔ True  Positives  (correct 'Default')    : 944

   ✔ Result charts saved as 'model_results_beginner.png'

   Dataset        : credit_risk_dataset.csv
   Total Records  : 32,576 loan applicants
   Features Used  : 11 columns

   Model          : Decision Tree (max_depth = 6)
   Training Size  : 26,060 rows
   Testing  Size  : 6,516 rows

   ✅ Accuracy     : 92.23%
   ✅ True Positives  (defaults caught)   : 944
   ⚠️  False Negatives (defaults missed)  : 478

   Top Feature    : loan_percent_income
   (This was the most useful column for predicting defaults)
