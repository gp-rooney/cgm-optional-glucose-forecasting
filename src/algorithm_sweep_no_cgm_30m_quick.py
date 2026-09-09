from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CADENCE_MIN=5
H_STEPS=6
W30,W60,W120=6,12,24
MIN_TRAIN_ROWS=250
MIN_TEST_ROWS=150

def rmse(y,p):
    return np.sqrt(mean_squared_error(y,p))

def add_time_since_event(g,col,out_col):
    event = g[col].fillna(0).to_numpy() > 0
    idx = np.where(event, np.arange(len(g)), np.nan)
    last_idx = pd.Series(idx).ffill().to_numpy()
    delta = np.arange(len(g)) - last_idx
    delta[np.isnan(last_idx)] = np.nan
    g[out_col] = delta * CADENCE_MIN

def build_features(g):
    g = g.sort_values('time').copy()
    mins = g['time'].dt.hour*60 + g['time'].dt.minute
    angle = 2*np.pi*mins/(24*60)
    g['tod_sin']=np.sin(angle); g['tod_cos']=np.cos(angle)
    g['dow']=g['time'].dt.dayofweek; g['is_weekend']=(g['dow']>=5).astype(int)
    g['hr_now']=g['heart_rate']; g['steps_now']=g['steps']; g['cal_now']=g['calories']
    g['carb_now']=g['carb_input']; g['bolus_now']=g['bolus_volume_delivered']; g['basal_now']=g['basal_rate']
    g['hr_mean_30']=g['heart_rate'].shift(1).rolling(W30).mean(); g['hr_std_30']=g['heart_rate'].shift(1).rolling(W30).std()
    g['steps_sum_30']=g['steps'].shift(1).rolling(W30).sum(); g['steps_sum_60']=g['steps'].shift(1).rolling(W60).sum()
    g['cal_sum_30']=g['calories'].shift(1).rolling(W30).sum(); g['cal_sum_60']=g['calories'].shift(1).rolling(W60).sum()
    g['carb_sum_60']=g['carb_input'].shift(1).rolling(W60).sum(); g['carb_sum_120']=g['carb_input'].shift(1).rolling(W120).sum()
    g['bolus_sum_30']=g['bolus_volume_delivered'].shift(1).rolling(W30).sum(); g['bolus_sum_60']=g['bolus_volume_delivered'].shift(1).rolling(W60).sum()
    g['basal_mean_30']=g['basal_rate'].shift(1).rolling(W30).mean()
    for lag in [1,3,6,12]:
        g[f'hr_lag_{lag}']=g['heart_rate'].shift(lag)
        g[f'steps_lag_{lag}']=g['steps'].shift(lag)
        g[f'cal_lag_{lag}']=g['calories'].shift(lag)
        g[f'carb_lag_{lag}']=g['carb_input'].shift(lag)
        g[f'bolus_lag_{lag}']=g['bolus_volume_delivered'].shift(lag)
    add_time_since_event(g,'carb_input','mins_since_carb')
    add_time_since_event(g,'bolus_volume_delivered','mins_since_bolus')
    g['y_30']=g['glucose'].shift(-H_STEPS)
    return g

def main():
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / 'data' / 'HUPA-UCM Diabetes Dataset' / 'Preprocessed'
    results_dir = project_dir / 'results' / 'tables'
    results_dir.mkdir(parents=True, exist_ok=True)

    parts=[]
    for f in sorted(data_dir.glob('*.csv')):
        d=pd.read_csv(f,sep=';')
        d['time']=pd.to_datetime(d['time'])
        d['user_id']=f.stem
        for c in ['glucose','calories','heart_rate','steps','basal_rate','bolus_volume_delivered','carb_input']:
            d[c]=pd.to_numeric(d[c],errors='coerce')
        parts.append(d)
    df=pd.concat(parts,ignore_index=True).sort_values(['user_id','time']).reset_index(drop=True)
    feat=df.groupby('user_id',group_keys=False).apply(build_features).reset_index(drop=True)

    features=[
        'tod_sin','tod_cos','dow','is_weekend','hr_now','steps_now','cal_now','carb_now','bolus_now','basal_now',
        'hr_mean_30','hr_std_30','steps_sum_30','steps_sum_60','cal_sum_30','cal_sum_60','carb_sum_60','carb_sum_120',
        'bolus_sum_30','bolus_sum_60','basal_mean_30','mins_since_carb','mins_since_bolus',
        'hr_lag_1','hr_lag_3','hr_lag_6','hr_lag_12','steps_lag_1','steps_lag_3','steps_lag_6','steps_lag_12',
        'cal_lag_1','cal_lag_3','cal_lag_6','cal_lag_12','carb_lag_1','carb_lag_3','carb_lag_6','carb_lag_12',
        'bolus_lag_1','bolus_lag_3','bolus_lag_6','bolus_lag_12'
    ]

    models={
        'ridge': Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',Ridge(alpha=8.0))]),
        'elasticnet': Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',ElasticNet(alpha=0.002,l1_ratio=0.15,max_iter=5000,random_state=42))]),
        'hist_gbr': Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingRegressor(learning_rate=0.04,max_depth=8,max_iter=200,min_samples_leaf=60,random_state=42))]),
        'extra_trees': Pipeline([('imputer',SimpleImputer(strategy='median')),('model',ExtraTreesRegressor(n_estimators=90,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42))]),
    }

    rows=[]
    users=sorted(feat['user_id'].unique())
    for user in users:
        u=feat[feat.user_id==user].dropna(subset=['y_30']).sort_values('time').reset_index(drop=True)
        if len(u) < (MIN_TRAIN_ROWS+MIN_TEST_ROWS):
            continue
        cut=int(len(u)*0.8)
        tr=u.iloc[:cut]
        te=u.iloc[cut:]
        if len(tr)<MIN_TRAIN_ROWS or len(te)<MIN_TEST_ROWS:
            continue
        y=te['y_30'].values
        for name,m in models.items():
            m.fit(tr[features], tr['y_30'])
            p=m.predict(te[features])
            rows.append({'user_id':user,'model':name,'rmse':rmse(y,p),'mae':mean_absolute_error(y,p),'n_test':len(te)})

    res=pd.DataFrame(rows)
    summary=(res.assign(rmse_w=lambda d:d.rmse*d.n_test, mae_w=lambda d:d.mae*d.n_test)
             .groupby('model',as_index=False)
             .agg(total_n=('n_test','sum'), rmse_w=('rmse_w','sum'), mae_w=('mae_w','sum')))
    summary['RMSE']=summary['rmse_w']/summary['total_n']
    summary['MAE']=summary['mae_w']/summary['total_n']
    summary=summary[['model','RMSE','MAE','total_n']].sort_values(['RMSE','MAE'])

    out_csv=results_dir/'personalized_no_cgm_algorithm_sweep.csv'
    out_json=results_dir/'personalized_no_cgm_top_models.json'
    res.to_csv(out_csv,index=False)
    out_json.write_text(json.dumps(summary.head(3).to_dict(orient='records'), indent=2))

    print('Saved:',out_csv)
    print('Saved:',out_json)
    print(summary.to_string(index=False))

if __name__=='__main__':
    main()
