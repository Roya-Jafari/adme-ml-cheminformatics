import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Setup & Feature Extractor
remover = SaltRemover()
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
descriptor_funcs = Descriptors._descList

def compute_expert_features(mol):
    if mol is None:
        return None
    fp = morgan_gen.GetCountFingerprintAsNumPy(mol)
    desc_values = []
    for name, func in descriptor_funcs:
        try:
            val = func(mol)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
        except Exception:
            val = 0.0
        desc_values.append(val)
    return np.concatenate([fp, np.array(desc_values, dtype=np.float32)])

def process_molecule(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return remover.StripMol(mol)

def scaffold_train_test_split(df, test_ratio=0.2):
    def get_scaffold(m):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            return ""

    scaffolds = {}
    for idx, m in enumerate(df["Mol"]):
        scaffold = get_scaffold(m)
        scaffolds.setdefault(scaffold, []).append(idx)
        
    sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    
    train_indices, test_indices = [], []
    test_cutoff = len(df) * test_ratio
    
    for scaffold, indices in sorted_scaffolds:
        if len(test_indices) < test_cutoff:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)
            
    return np.array(train_indices), np.array(test_indices)

# 2. Data Preprocessing
df = pd.read_csv("ADME_public_set_3521.csv")
df["Mol"] = df["SMILES"].apply(process_molecule)
valid_df = df[df["Mol"].notna()].reset_index(drop=True)

print("Pre-computing hybrid features for all molecules...")
raw_features = [compute_expert_features(m) for m in valid_df["Mol"]]
X_all = np.nan_to_num(np.vstack(raw_features), nan=0.0, posinf=0.0, neginf=0.0)

# List of all 6 ADME Target Endpoints
endpoints = [
    "LOG HLM_CLint (mL/min/kg)",
    "LOG MDR1-MDCK ER (B-A/A-B)",
    "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
    "LOG RLM_CLint (mL/min/kg)"
]

results = []

# 3. Multi-Endpoint Loop
print("\n================ BENCHMARKING ALL ENDPOINTS ================")
for target_col in endpoints:
    # Filter dataset for compounds containing measurement for this target
    sub_df = valid_df.dropna(subset=[target_col]).reset_index(drop=True)
    valid_indices = valid_df[valid_df[target_col].notna()].index
    
    X_sub = X_all[valid_indices]
    y_sub = sub_df[target_col].values
    
    # Scaffold Split
    train_idx, test_idx = scaffold_train_test_split(sub_df, test_ratio=0.2)
    
    X_train_raw, X_test_raw = X_sub[train_idx], X_sub[test_idx]
    y_train, y_test = y_sub[train_idx], y_sub[test_idx]
    
    # Scale & Feature Select (Fit on Train set ONLY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    selector = VarianceThreshold(threshold=0.01)
    X_train_final = selector.fit_transform(X_train_scaled)
    X_test_final = selector.transform(X_test_scaled)
    
    # Train Model
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(X_train_final, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Endpoint": target_col,
        "N_Samples": len(sub_df),
        "Scaffold Test RMSE": round(rmse, 3),
        "Scaffold Test R^2": round(r2, 3)
    })
    print(f"Done: {target_col[:30]}... -> R^2: {r2:.3f} | RMSE: {rmse:.3f}")

# 4. Print Final Summary Table
print("\n================ FINAL MULTI-ENDPOINT SUMMARY ================")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
