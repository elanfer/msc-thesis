import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn import tree
from sklearn.metrics import roc_auc_score
from joblib import dump, load


# Load merged data set and perform split into train/test/eval (60/20/20)

df = pd.read_csv('data_sets/OnePlusNordN10/ground_truth/ONE_PLUS_merged_set.csv', header=0)

X = df[['attenuation','rssi24','freq24','rssi5','freq5']]
y = df['y']
X_train, X_eval, y_train, y_eval = train_eval_split(X, y, eval_size = 0.4, shuffle=True, random_state=42,  stratify = y)
X_eval, X_eval, y_eval, y_eval = train_eval_split(X_eval, y_eval, eval_size = 0.50, shuffle=True, random_state=42,  stratify = y_eval)


# save sets
X_train.to_csv('TRAIN_OnePlus_X.csv')
y_train.to_csv('TRAIN_OnePLus_y.csv')

X_eval.to_csv('eval_OnePlus_X.csv')
y_eval.to_csv('eval_OnePlus_y.csv')

X_eval.to_csv('EVAL_OnePlus_X.csv')
y_eval.to_csv('EVAL_OnePlus_y.csv')


# to load special bus eval set uncomment this block: 
"""
eval_bus = pd.read_csv('data_sets/OnePlusNordN10/bus_eval/oneplus_bus_eval_mapped.csv')
X_eval = eval_bus[['attenuation','rssi24','freq24','rssi5','freq5']]
y_eval = eval_bus[['y']]
"""

# to load special meeting eval set uncomment this block: 
"""
eval_bus = pd.read_csv('data_sets/OnePlusNordN10/meeting_eval/EVAL__oneplus_merged_meeting_matched.csv')
X_eval = eval_bus[['attenuation','rssi24','freq24','rssi5','freq5']]
y_eval = eval_bus[['y']]
"""


# Train DT 2.4 GHz
dt24ghz = tree.DecisionTreeClassifier(max_depth=8)
# inspected, some last big splits, depth higher brings very small splits and tends to overfit
dt24ghz.fit(X_train[['rssi24', 'freq24']], y_train)

# Train DT 5 GHz
dt5ghz = tree.DecisionTreeClassifier(max_depth=8)
# inspected, some last big splits, depth higher brings very small splits and tends to overfit
dt5ghz.fit(X_train[['rssi5', 'freq5']], y_train)


# view tree
dot_data = tree.export_graphviz(dt24ghz, out_file=None, 
                     feature_names=["rssi_24ghz",  "freq_24ghz"],
                     class_names=["close", "safe", "very_close"],
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view()

print("#########################")
print("eval RESULT DT 2.4 GHZ ONLY")
# eval 2.4 ghz classifier
y_24ghz_pred = dt24ghz.predict(X_eval[['rssi24', 'freq24']])
print(roc_auc_score(y_eval, dt24ghz.predict_proba(X_eval[['rssi24', 'freq24']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))

print("#########################")
print("EVAL RESULT DT 2.4 GHZ ONLY")
# EVAL 2.4 ghz classifier
y_24ghz_pred = dt24ghz.predict(X_eval[['rssi24', 'freq24']])
print(roc_auc_score(y_eval, dt24ghz.predict_proba(X_eval[['rssi24', 'freq24']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))


# eval 5 ghz classifier
print("#########################")
print("eval RESULT DT 5 GHZ ONLY")
y_5ghz_pred = dt5ghz.predict(X_eval[['rssi5', 'freq5']])
print(roc_auc_score(y_eval, dt5ghz.predict_proba(X_eval[['rssi5', 'freq5']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))

# EVAL 5 ghz classifier
print("#########################")
print("EVAL RESULT DT 5 GHZ ONLY")
y_5ghz_pred = dt5ghz.predict(X_eval[['rssi5', 'freq5']])
print(roc_auc_score(y_eval, dt5ghz.predict_proba(X_eval[['rssi5', 'freq5']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))

# Train RF 2.4 GHz
rf24ghz = RandomForestClassifier(max_depth=8)
rf24ghz.fit(X_train[['rssi24', 'freq24']], y_train)

# Train RF 5 GHz
rf5ghz = RandomForestClassifier(max_depth=8)
rf5ghz.fit(X_train[['rssi5', 'freq5']], y_train)

print("#########################")
print("eval RESULT RF 2.4 GHZ ONLY")
# eval 2.4 ghz classifier
y_24ghz_pred = rf24ghz.predict(X_eval[['rssi24', 'freq24']])
print(roc_auc_score(y_eval, rf24ghz.predict_proba(X_eval[['rssi24', 'freq24']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))

print("#########################")
print("EVAL RESULT RF 2.4 GHZ ONLY")
# EVAL 2.4 ghz classifier
y_24ghz_pred = rf24ghz.predict(X_eval[['rssi24', 'freq24']])
print(roc_auc_score(y_eval, rf24ghz.predict_proba(X_eval[['rssi24', 'freq24']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_24ghz_pred, labels=['very_close','close','safe']))


# eval 5 ghz classifier
print("#########################")
print("eval RESULT RF 5 GHZ ONLY")
y_5ghz_pred = rf5ghz.predict(X_eval[['rssi5', 'freq5']])
print(roc_auc_score(y_eval, rf5ghz.predict_proba(X_eval[['rssi5', 'freq5']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))

# EVAL 5 ghz classifier
print("#########################")
print("EVAL RESULT RF 5 GHZ ONLY")
y_5ghz_pred = rf5ghz.predict(X_eval[['rssi5', 'freq5']])
print(roc_auc_score(y_eval, rf5ghz.predict_proba(X_eval[['rssi5', 'freq5']]), multi_class='ovr'))
print(confusion_matrix(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, y_5ghz_pred, labels=['very_close','close','safe']))


# Combined Classifier BLE thresholding
def number_to_class(num):
  if num == 0:
    return 'very_close'
  elif num == 1:
    return 'close'
  elif num == 2:
    return 'safe'
  else:
    print('ERROR')
    


def class_to_number(cl):
  if cl == 'very_close':
    return 0
  elif cl == 'close':
    return 1
  elif cl == 'safe':
    return 2
  else:
    print('Error wrong class')
    exit(0)
    

def clf_btle(attn):
  if attn < 55:
    return 'very_close'
  elif attn < 63:
    return 'close'
  else:
    return 'safe'


# Combined Classifier with DTs
y_24ghz_pred = dt24ghz.predict(X_eval[['rssi24', 'freq24']])
y_5ghz_pred = dt5ghz.predict(X_eval[['rssi5', 'freq5']])

# EVAL Combined Classifier with DTs
y_24ghz_pred = dt24ghz.predict(X_eval[['rssi24', 'freq24']])
y_5ghz_pred = dt5ghz.predict(X_eval[['rssi5', 'freq5']])

nums_5ghz = []
nums_24ghz = []
nums_ble = []
pred_ble = []
for i, item in enumerate(y_5ghz_pred):
  nums_5ghz.append(class_to_number(item))  
  nums_24ghz.append(class_to_number(y_24ghz_pred[i]))
  nums_ble.append(class_to_number(clf_btle(X_eval['attenuation'].values[i])))
  pred_ble.append(clf_btle(X_eval['attenuation'].values[i]))

dt_combi_df = pd.DataFrame()
dt_combi_df['5ghz_num_pred'] = nums_5ghz
dt_combi_df['24ghz_num_pred'] = nums_24ghz
dt_combi_df['ble_num_pred'] = nums_ble
dt_combi_df['ble_pred'] = pred_ble

dt_combi_df['combi_num_pred'] = round((dt_combi_df['5ghz_num_pred'] + dt_combi_df['24ghz_num_pred'] + dt_combi_df['ble_num_pred'])/3,0)
dt_combi_df['combi_num_pred_weighted'] = round((1.5*dt_combi_df['5ghz_num_pred'] + 0.75*dt_combi_df['24ghz_num_pred'] + 0.75*dt_combi_df['ble_num_pred'])/3,0)

combi_pred = []
for item in dt_combi_df['combi_num_pred'].values:
  combi_pred.append(number_to_class(item))
dt_combi_df['combi_pred'] = combi_pred

combi_pred = []
for item in dt_combi_df['combi_num_pred_weighted'].values:
  combi_pred.append(number_to_class(item))
dt_combi_df['combi_pred_weighted'] = combi_pred

# eval BLE only
print("#########################")
print("eval RESULT BLE ONLY")
print(confusion_matrix(y_eval, dt_combi_df['ble_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['ble_pred'], labels=['very_close','close','safe']))

# eval Combi DT
print("#########################")
print("eval RESULT COMBI DT")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))

# eval Combi DT weighted
print("#########################")
print("eval RESULT COMBI DT Weighted")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))


# EVAL BLE only
print("#########################")
print("EVAL RESULT BLE ONLY")
print(confusion_matrix(y_eval, dt_combi_df['ble_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['ble_pred'], labels=['very_close','close','safe']))

# EVAL Combi DT
print("#########################")
print("EVAL RESULT COMBI DT")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))

# EVAL Combi DT weighted
print("#########################")
print("EVAL RESULT COMBI DT Weighted")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))

# Combined Classifier with RFs
y_24ghz_pred = rf24ghz.predict(X_eval[['rssi24', 'freq24']])
y_5ghz_pred = rf5ghz.predict(X_eval[['rssi5', 'freq5']])

# EVAL
y_24ghz_pred = rf24ghz.predict(X_eval[['rssi24', 'freq24']])
y_5ghz_pred = rf5ghz.predict(X_eval[['rssi5', 'freq5']])

nums_5ghz = []
nums_24ghz = []
nums_ble = []
pred_ble = []
for i, item in enumerate(y_5ghz_pred):
  nums_5ghz.append(class_to_number(item))  
  nums_24ghz.append(class_to_number(y_24ghz_pred[i]))
  nums_ble.append(class_to_number(clf_btle(X_eval['attenuation'].values[i])))
  pred_ble.append(clf_btle(X_eval['attenuation'].values[i]))

dt_combi_df = pd.DataFrame()
dt_combi_df['5ghz_num_pred'] = nums_5ghz
dt_combi_df['24ghz_num_pred'] = nums_24ghz
dt_combi_df['ble_num_pred'] = nums_ble
dt_combi_df['ble_pred'] = pred_ble

dt_combi_df['combi_num_pred'] = round((dt_combi_df['5ghz_num_pred'] + dt_combi_df['24ghz_num_pred'] + dt_combi_df['ble_num_pred'])/3,0)
dt_combi_df['combi_num_pred_weighted'] = round((1.5*dt_combi_df['5ghz_num_pred'] + 0.75*dt_combi_df['24ghz_num_pred'] + 0.75*dt_combi_df['ble_num_pred'])/3,0)

combi_pred = []
for item in dt_combi_df['combi_num_pred'].values:
  combi_pred.append(number_to_class(item))
dt_combi_df['combi_pred'] = combi_pred

combi_pred = []
for item in dt_combi_df['combi_num_pred_weighted'].values:
  combi_pred.append(number_to_class(item))
dt_combi_df['combi_pred_weighted'] = combi_pred

# eval BLE only
print("#########################")
print("eval RESULT BLE ONLY")
print(confusion_matrix(y_eval, dt_combi_df['ble_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['ble_pred'], labels=['very_close','close','safe']))

# eval Combi RF
print("#########################")
print("eval RESULT COMBI RF")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))

# eval Combi RF weighted
print("#########################")
print("eval RESULT COMBI RF Weighted")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))

# EVAL Combi RF
print("#########################")
print("EVAL RESULT COMBI RF")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred'], labels=['very_close','close','safe']))

# EVAL Combi RF weighted
print("#########################")
print("EVAL RESULT COMBI RF Weighted")
print(confusion_matrix(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))
print(classification_report(y_eval, dt_combi_df['combi_pred_weighted'], labels=['very_close','close','safe']))



# General Classifier
# evaling with single tree
full_dt = tree.DecisionTreeClassifier(max_depth=8)
# inspected, some last big splits, depth higher brings very small splits and tends to overfit
full_dt.fit(X_train, y_train['y'])
full_dt_pred = full_dt.predict(X_eval)

print("#########################")
print("eval RESULT GEN DT")
print(roc_auc_score(y_eval, full_dt.predict_proba(X_eval), multi_class='ovr'))
print(confusion_matrix(y_eval, full_dt_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, full_dt_pred, labels=['very_close','close','safe']))

# view tree
dot_data = tree.export_graphviz(full_dt, out_file=None, 
                     feature_names=['attn_btle', 'rssi_24ghz', 'freq_24ghz', 'rssi_5ghz', 'freq_5ghz'],
                     class_names=["close", "safe", "very_close"],
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view()


# General Classifier
# evaling with RF
full_rf = RandomForestClassifier(max_depth=8)
# inspected, some last big splits, depth higher brings very small splits and tends to overfit
full_rf.fit(X_train, y_train['y'])
full_rf_pred = full_rf.predict(X_eval)

print("#########################")
print("eval RESULT GEN RF")
print(roc_auc_score(y_eval, full_rf.predict_proba(X_eval), multi_class='ovr'))
print(confusion_matrix(y_eval, full_rf_pred, labels=['very_close','close','safe']))
print(classification_report(y_eval, full_rf_pred, labels=['very_close','close','safe']))


