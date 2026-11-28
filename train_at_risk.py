"""Retrain the existing GradeGenius AI binary early-warning bundle."""
from pathlib import Path
import sys
import subprocess
import importlib.util
import zipfile
import tempfile
for module, package in [('xgboost','xgboost'), ('sklearn','scikit-learn'), ('pandas','pandas'), ('joblib','joblib'), ('numpy','numpy')]:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path('/content') if Path('/content/student_performance_dataset.csv').is_file() else (Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd())
DATA = ROOT / 'student_performance_dataset.csv'
OUT = ROOT / 'at_risk'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA)
features = ['study_time_hours','attendance_percent','sleep_deviation','parental_education','internet_access','extracurricular_activities','part_time_job','previous_grade']
binary = ['internet_access','extracurricular_activities','part_time_job']
edu = {'High School':0,'Bachelors':1,'Masters':2,'PhD':3}
assert len(df) == 25000, f'Expected 25000 rows, got {len(df)}'
assert set(df.final_grade.unique()) == {'A','B','C','D','F'}, 'Unexpected grade labels'
df['parental_education'] = df['parental_education'].fillna('High School').map(edu)
h = df['sleep_hours']
df['sleep_deviation'] = np.maximum(np.maximum(6-h,h-9),0)
assert not df[features].isna().any().any(), 'Missing or unexpected input values'
# Same outer split as grade prediction, stratified on all five grades.
train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df.final_grade)
X_train = train[features].copy(); X_test = test[features].copy()
y_train = train.final_grade.isin(['D','F']).astype(int); y_test = test.final_grade.isin(['D','F']).astype(int)
encoders = {}
for col in binary:
    encoders[col] = LabelEncoder().fit(X_train[col])
    X_train[col] = encoders[col].transform(X_train[col])
    X_test[col] = encoders[col].transform(X_test[col])
scaler = StandardScaler().fit(X_train)
a = scaler.transform(X_train); b = scaler.transform(X_test)
model = XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,objective='binary:logistic',eval_metric='logloss',monotone_constraints=(-1,-1,1,-1,0,0,0,-1),random_state=42,n_jobs=4,tree_method='hist')
print('Training XGBoost at-risk model:',a.shape,flush=True)
model.fit(a,y_train)
p = model.predict_proba(b)[:,1]
# Preserve the application's existing threshold; do not tune it on test data.
threshold = .30
pred = (p >= threshold).astype(int)
print(f'\nTest records: {len(test)} | Risk threshold: {threshold:.2f}')
for name, score in [('Accuracy',accuracy_score(y_test,pred)),('Precision',precision_score(y_test,pred,zero_division=0)),('Recall',recall_score(y_test,pred,zero_division=0)),('F1 Score',f1_score(y_test,pred,zero_division=0)),('ROC AUC',roc_auc_score(y_test,p))]:
    print(f'{name}: {score*100:.2f}%')
print('Confusion matrix (rows actual; columns predicted; order On Track, At Risk):\n',confusion_matrix(y_test,pred,labels=[0,1]))
# SHAP uses scaled training examples, never test examples.
background = a[np.random.default_rng(42).choice(len(a),40,replace=False)]
with tempfile.TemporaryDirectory() as tmp:
    native_path = Path(tmp) / 'model.json'
    model.save_model(str(native_path))
    native_bytes = native_path.read_bytes()
bundle = {'at_risk_model.pkl':{'format':'xgboost-json-v1','model':native_bytes},'at_risk_scaler.pkl':scaler,'at_risk_bin_encoders.pkl':encoders,'at_risk_features.pkl':features,'shap_background.pkl':background}
for name,obj in bundle.items():
    joblib.dump(obj,OUT/name)
loaded = XGBClassifier()
loaded.load_model(bytearray(joblib.load(OUT/'at_risk_model.pkl')['model']))
assert np.allclose(loaded.predict_proba(b[:10]),model.predict_proba(b[:10]))
archive = ROOT/'at_risk.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in bundle: z.write(OUT/name, 'at_risk/'+name)
print('\nSaved all five original filenames in at_risk/.')
print('Archive:',archive)
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(str(archive))
