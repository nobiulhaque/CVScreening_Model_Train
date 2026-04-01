"""
Training pipeline for Resume Screening Model
Uses skill vectors + TF-IDF text features
Models: Random Forest, CatBoost, LightGBM (auto-selects best)
"""

import numpy as np
import csv
import json
import joblib
import warnings
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone as sk_clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from config import PROCESSED_DATA_DIR, MODEL_SAVE_DIR, TECHNICAL_SKILLS, SOFT_SKILLS


def load_data():
    """Load skill vectors + raw text for TF-IDF.
    Categories are discovered dynamically from the data."""
    csv_path = PROCESSED_DATA_DIR / "resume_skill_vectors.csv"
    json_path = PROCESSED_DATA_DIR / "resume_features.json"
    cat_meta_path = PROCESSED_DATA_DIR / "categories.json"

    if not csv_path.exists():
        print("Dataset not found! Run create_dataset.py first.")
        return None, None, None, None

    # Load categories
    if cat_meta_path.exists():
        with open(cat_meta_path, 'r', encoding='utf-8') as f:
            cat_meta = json.load(f)
        categories = cat_meta["categories"]
    else:
        # Fallback: discover from CSV data
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            categories = sorted(set(row["category"] for row in reader))

    print(f"Discovered {len(categories)} categories: {categories}")

    # Load skill vectors
    features = []
    labels = []
    filenames = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            label = categories.index(category)
            labels.append(label)
            filenames.append(row.get("filename", ""))

            feature_vec = []
            for key, value in row.items():
                if key not in ["filename", "category"]:
                    try:
                        feature_vec.append(float(value))
                    except ValueError:
                        feature_vec.append(0.0)
            features.append(feature_vec)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Load raw text for TF-IDF
    corpus_path = PROCESSED_DATA_DIR / "resume_raw_texts.json"
    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        text_map = {
            item.get("filename", ""): item.get("raw_text", "")
            for item in corpus_data
        }
        texts = [text_map.get(fname, "") for fname in filenames]
        missing = sum(1 for t in texts if not t)
        print(f"Loaded raw text corpus ({len(corpus_data)} docs)")
        if missing:
            print(f"WARNING: Missing raw text for {missing} sample(s); using empty text.")
    else:
        # Fallback: use extracted skills from JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        fallback_map = {}
        for item in json_data:
            text_parts = []
            text_parts.extend(item.get("technical_skills", []))
            text_parts.extend(item.get("soft_skills", []))
            text_parts.extend(item.get("job_titles_list", "").split("|") if isinstance(item.get("job_titles_list"), str) else item.get("job_titles", []))
            text_parts.extend(item.get("certifications", []))
            edu = item.get("education_level", "")
            if edu:
                text_parts.append(edu)
            fallback_map[item.get("filename", "")] = " ".join(text_parts)
        texts = [fallback_map.get(fname, "") for fname in filenames]
        print(f"Using extracted skills as text (raw corpus not found)")

    if len(texts) != len(features):
        print("ERROR: Text/feature sample size mismatch. Re-run create_dataset.py.")
        return None, None, None, None

    print(f"Loaded {len(features)} samples with {features.shape[1]} skill features")
    return features, labels, texts, categories


def train_model():
    """Train ensemble classifier"""
    print("=" * 60)
    print("RESUME CATEGORY CLASSIFIER TRAINING")
    print("(Random Forest + CatBoost + LightGBM + SMOTE + TF-IDF)")
    print("=" * 60 + "\n")

    # Load data
    features, labels, texts, categories = load_data()
    if features is None or categories is None:
        return
    
    # Type narrowing: if features & categories loaded, so did labels & texts
    assert labels is not None
    assert texts is not None

    # Class distribution
    print("Class distribution:")
    for idx, cat in enumerate(categories): 
        count = np.sum(labels == idx)
        print(f"  {cat:20} {count:4}")

    # ==================== TF-IDF Features ====================
    print("\nBuilding TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_features = tfidf.fit_transform(texts)
    print(f"TF-IDF features: {tfidf_features.shape[1]}")

    # Combine skill vectors + TF-IDF
    skill_sparse = csr_matrix(features)
    X_combined_sparse = csr_matrix(hstack([skill_sparse, tfidf_features]))
    total_features = int(features.shape[1] + tfidf_features.shape[1])
    print(f"Total features: {total_features}")

    # ==================== Proper Evaluation (SMOTE inside each fold) ====================
    print("\n" + "=" * 60)
    print("ROBUST EVALUATION (5-fold CV, SMOTE per fold)")
    print("=" * 60)

    # Check GPU availability for boosting models
    try:
        import torch # type: ignore
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"
    except ImportError:
        has_gpu = False
        gpu_name = "N/A"
    print(f"  GPU: {'YES - ' + gpu_name if has_gpu else 'NO (CPU only)'}")

    # Dynamic fold count — can't have more folds than smallest class size
    min_class_size = min(Counter(labels).values())
    if min_class_size < 2:
        print("WARNING: At least one class has only 1 sample. Stratified CV with SMOTE is not feasible.")
        print("         CV will be skipped and model will be trained on all data instead.")
        cv = None
    else:
        n_folds = min(5, min_class_size)
        print(f"  CV folds: {n_folds} (smallest class has {min_class_size} samples)")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    catboost_params = {
        'iterations': 300, 'depth': 6, 'learning_rate': 0.1,
        'l2_leaf_reg': 3, 'random_seed': 42, 'verbose': 0,
        'auto_class_weights': 'Balanced',
    }
    if has_gpu:
        catboost_params['task_type'] = 'GPU'
        catboost_params['devices'] = '0'

    lgbm_params = {
        'n_estimators': 150, 'max_depth': 15, 'learning_rate': 0.1,
        'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 5,
        'class_weight': 'balanced', 'random_state': 42,
        'n_jobs': -1, 'verbose': -1,
    }
    if has_gpu:
        lgbm_params['device'] = 'gpu'

    model_templates = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt',
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(**catboost_params),
        'LightGBM': LGBMClassifier(**lgbm_params),
    }

    cv_results = {}
    if cv is not None:
        for name, template in model_templates.items():
            print(f"  Training {name}...", end=" ", flush=True)
            fold_accs = []
            for train_idx, val_idx in cv.split(np.zeros(len(labels)), labels):
                X_fold_train, X_fold_val = X_combined_sparse[train_idx], X_combined_sparse[val_idx]
                y_fold_train, y_fold_val = labels[train_idx], labels[val_idx]

                # Convert fold slices to dense right before SMOTE to avoid global dense copy.
                X_fold_train = X_fold_train.toarray()
                X_fold_val = X_fold_val.toarray()

                # SMOTE inside the fold if possible
                min_count = min(Counter(y_fold_train).values())
                if min_count > 1:
                    k = min(5, min_count - 1)
                    sm = SMOTE(random_state=42, k_neighbors=k)
                    X_res, y_res = sm.fit_resample(X_fold_train, y_fold_train) # type: ignore
                else:
                    X_res, y_res = X_fold_train, y_fold_train

                # Scale
                sc = StandardScaler()
                X_res = sc.fit_transform(X_res) # type: ignore
                X_fold_val = sc.transform(X_fold_val)

                # Train & predict (suppress noisy warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = sk_clone(template)
                    m.fit(X_res, y_res)
                    preds = m.predict(X_fold_val)
                fold_accs.append(accuracy_score(y_fold_val, preds))

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            cv_results[name] = (mean_acc, std_acc)
            print(f"{mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")

        best_cv_name = max(cv_results, key=lambda k: cv_results[k][0])
        print(f"\n  Best by CV: {best_cv_name} ({cv_results[best_cv_name][0]*100:.1f}%)")
    else:
        # No reliable CV possible (singleton class, etc.)
        print("\nSkipping cross-validation due to insufficient class sample sizes.")
        for name in model_templates:
            cv_results[name] = (0.0, 0.0)
        best_cv_name = 'Random Forest'
        print(f"Using default model: {best_cv_name}")

    # ==================== Train Final Model on ALL Data ====================
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*60}\n")

    # Use the best model from CV
    print(f"Training final {best_cv_name} on all data...")
    final_model_template = model_templates[best_cv_name]

    # Apply SMOTE on all data if possible
    min_class_count = min(Counter(labels).values())
    X_all_dense = X_combined_sparse.toarray()
    if min_class_count > 1:
        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_all_resampled, y_all_resampled = smote.fit_resample(X_all_dense, labels) # type: ignore
    else:
        print("WARNING: Not enough samples for SMOTE on full data; using original dataset without SMOTE.")
        X_all_resampled, y_all_resampled = X_all_dense, labels

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_resampled) # type: ignore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = sk_clone(final_model_template)
        final_model.fit(X_all_scaled, y_all_resampled)

    # Quick check: accuracy on original (non-resampled) data
    X_orig_scaled = scaler.transform(X_all_dense)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pred = final_model.predict(X_orig_scaled)
    train_acc = accuracy_score(labels, train_pred)
    print(f"Accuracy on original data: {train_acc*100:.1f}%")
    print(f"\nClassification Report (on original data):")
    target_names = [cat[:12] for cat in categories]
    print(classification_report(labels, train_pred, target_names=target_names, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(labels, train_pred)
    header = "            " + " ".join(f"{c[:6]:>7}" for c in categories)
    print(header)
    for i, cat in enumerate(categories):
        row = f"{cat[:12]:12}" + " ".join(f"{cm[i][j]:7d}" for j in range(len(categories)))
        print(row)

    # ==================== Save Everything ====================
    print(f"\nSaving models...")

    # Save best model (compressed)
    model_path = MODEL_SAVE_DIR / "best_classifier.pkl"
    joblib.dump(final_model, model_path, compress=3)
    print(f"  Best model: {model_path}")

    # Save scaler
    scaler_path = MODEL_SAVE_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path, compress=3)
    print(f"  Scaler: {scaler_path}")

    # Save TF-IDF vectorizer
    tfidf_path = MODEL_SAVE_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(tfidf, tfidf_path, compress=3)
    print(f"  TF-IDF: {tfidf_path}")

    # Save metadata
    best_cv_acc, best_cv_std = cv_results[best_cv_name]
    meta = {
        "best_model": best_cv_name,
        "cv_accuracy": best_cv_acc,
        "cv_std": best_cv_std,
        "all_cv_results": {name: {"mean": float(m), "std": float(s)} for name, (m, s) in cv_results.items()},
        "train_accuracy": float(train_acc),
        "categories": categories,
        "skill_list": TECHNICAL_SKILLS + SOFT_SKILLS,
        "num_skill_features": features.shape[1],
        "num_tfidf_features": tfidf_features.shape[1],
        "total_features": total_features,
        "total_samples": len(labels),
    }
    meta_path = MODEL_SAVE_DIR / "model_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best: {best_cv_name} — CV: {best_cv_acc*100:.1f}% (+/- {best_cv_std*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_model()
