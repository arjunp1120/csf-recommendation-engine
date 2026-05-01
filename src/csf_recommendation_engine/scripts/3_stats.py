import pickle
from pathlib import Path
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

DIR_PATH = Path(__file__).parent

print("1. Loading Model and Pre-built Matrices...")
with open(DIR_PATH / 'champion_model_v1.pkl', 'rb') as f:
    payload = pickle.load(f)

model = payload['model']
item_features_matrix = payload['item_features_matrix']

# Load the explicitly split train and test matrices
with open(DIR_PATH / 'training_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

train_interactions = artifacts['train_interactions']
test_interactions = artifacts['test_interactions']
user_features_matrix = artifacts['user_features_matrix']

print("2. Running Statistical Validation (Excluding Training Data)...")

print("\nCalculating AUC...")
test_auc = auc_score(
    model, 
    test_interactions=test_interactions, 
    train_interactions=train_interactions, 
    user_features=user_features_matrix, 
    item_features=item_features_matrix,
    num_threads=4 
).mean()

print("Calculating Precision@5...")
test_precision = precision_at_k(
    model, 
    test_interactions=test_interactions, 
    train_interactions=train_interactions, 
    k=5, 
    user_features=user_features_matrix, 
    item_features=item_features_matrix,
    num_threads=4
).mean()

print("Calculating Recall@5...")
test_recall = recall_at_k(
    model, 
    test_interactions=test_interactions, 
    train_interactions=train_interactions, 
    k=5, 
    user_features=user_features_matrix, 
    item_features=item_features_matrix,
    num_threads=4
).mean()

print("\n========================================")
print("     MODEL DIAGNOSTIC REPORT")
print("========================================")
print(f"AUC Score:      {test_auc:.4f}  (Target: > 0.75)")
print(f"Precision@5:    {test_precision:.4f}  (Target: > 0.15 for sparse graphs)")
print(f"Recall@5:       {test_recall:.4f}  (Target: > 0.20 for sparse graphs)")
print("========================================")