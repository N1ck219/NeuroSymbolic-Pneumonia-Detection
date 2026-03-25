# --- IMPORTS ---
import os
import clingo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_asp_rules(results_meta, patient_targets, rules_path):
    """
    Translates deep learning metadata into ASP facts, runs the Clingo solver
    with the specified rules, and evaluates the neuro-symbolic performance.
    """
    if not os.path.exists(rules_path):
        print(f"[ERROR] ASP rules file '{rules_path}' not found.")
        return

    print(f"\n[INFO] Integrating ASP Logic using: {rules_path}")

    # 1. Fact Generation
    facts = []
    asp_id_to_patient = {} 
    
    for idx, data in enumerate(results_meta):
        asp_id_to_patient[idx] = data['patientId']
        score_int = int(data['score'] * 100)
        intensity_int = int(data['intensity'] * 100)
        texture_int = int(data.get('texture', 0.30) * 100)
        
        x, y, w, h = data['pred_box']
        facts.append(f"pred({idx}, {score_int}, {x}, {y}, {w}, {h}, {intensity_int}, {texture_int}).")

    # 2. Clingo Solving
    ctl = clingo.Control()
    valid_box_indices = set()

    try:
        ctl.load(rules_path)
        ctl.add("base", [], "\n".join(facts))
        ctl.ground([("base", [])])
        
        # Callback to collect 'valid' predictions
        def on_model(m):
            for s in m.symbols(shown=True):
                if s.name == "valid":
                    valid_box_indices.add(int(str(s.arguments[0])))
                    
        ctl.solve(on_model=on_model)
        print(f"[INFO] ASP Resolution Complete. Validated boxes: {len(valid_box_indices)}")
    except Exception as e:
        print(f"[ERROR] Clingo execution failed: {e}")
        return

    # 3. Results Aggregation
    final_preds_raw = {pid: 0 for pid in patient_targets}
    final_preds_asp = {pid: 0 for pid in patient_targets}

    # Standard DL Threshold Logic (50%)
    for data in results_meta:
        if data['score'] >= 0.50: 
            final_preds_raw[data['patientId']] = 1

    # Neuro-Symbolic Logic
    for idx in valid_box_indices:
        final_preds_asp[asp_id_to_patient[idx]] = 1

    y_true, y_raw, y_asp = [], [], []
    for pid, target in patient_targets.items():
        y_true.append(target)
        y_raw.append(final_preds_raw[pid])
        y_asp.append(final_preds_asp[pid])

    # 4. Evaluation & Visualization
    cm_raw = confusion_matrix(y_true, y_raw)
    cm_asp = confusion_matrix(y_true, y_asp)
    
    fp_raw, fp_asp = cm_raw[0][1], cm_asp[0][1]
    fn_raw, fn_asp = cm_raw[1][0], cm_asp[1][0]
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 14})
    axes[0].set_title(f"RAW DL Model (Thresh: 50%)\nAccuracy: {accuracy_score(y_true, y_raw):.2%}", fontsize=14)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_asp, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False, annot_kws={"size": 14})
    axes[1].set_title(f"Neuro-Symbolic (DL + ASP)\nAccuracy: {accuracy_score(y_true, y_asp):.2%}", fontsize=14)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True Label')

    plt.suptitle(f"ASP Ruleset: {rules_path}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Text Report
    print(f"\n--- NEURO-SYMBOLIC IMPROVEMENT REPORT ({rules_path}) ---")
    print(f"False Positives (Healthy misclassified): {fp_raw} -> {fp_asp} (Diff: {fp_asp - fp_raw})")
    print(f"False Negatives (Missed Pneumonia):      {fn_raw} -> {fn_asp} (Diff: {fn_asp - fn_raw})")
    
    diff_fn = fn_raw - fn_asp
    diff_fp = fp_raw - fp_asp

    if diff_fn > 0: print(f" [+] ASP recovered {diff_fn} missed pneumonia cases.")
    elif diff_fn < 0: print(f" [-] ASP incorrectly discarded {abs(diff_fn)} true positive cases.")
    
    if diff_fp > 0: print(f" [+] ASP filtered out {diff_fp} false alarms.")
    elif diff_fp < 0: print(f" [-] ASP introduced {abs(diff_fp)} additional false alarms.")
    print("-" * 60)