import os
import json

def get_label_counts(feedback_dir):
    counts = {}
    for label in os.listdir(feedback_dir):
        label_path = os.path.join(feedback_dir, label)
        if os.path.isdir(label_path):
            counts[label] = len([f for f in os.listdir(label_path) if f.endswith('.jpg')])
    return counts

def should_trigger_retrain(feedback_dir, record_path='retrain/last_counts.json', threshold=20):
    current_counts = get_label_counts(feedback_dir)
    if not os.path.exists(record_path):
        with open(record_path, 'w') as f:
            json.dump(current_counts, f)
        return False

    with open(record_path, 'r') as f:
        last_counts = json.load(f)

    all_labels = set(current_counts) & set(last_counts)
    enough_new_data = all(
        current_counts[label] - last_counts.get(label, 0) >= threshold
        for label in all_labels
    )

    if enough_new_data:
        with open(record_path, 'w') as f:
            json.dump(current_counts, f)
        return True
    return False
    