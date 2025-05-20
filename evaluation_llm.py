import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

true_labels = {}
with open('dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        key = (data['video_id'], data['instance_id'])
        true_labels[key] = 'HUMOR' if data['label'] == 'humor' else 'NO HUMOR'

pred_labels = {}
with open('predictions_llm.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        key = (data['video_id'], data['instance_id'])
        text = data['gpt_label'].upper()
        pred_labels[key] = 'HUMOR' if 'HUMOR' in text and 'NO HUMOR' not in text else 'NO HUMOR'

keys = sorted(true_labels.keys())
y_true = [ true_labels[k] for k in keys ]
y_pred = [ pred_labels.get(k, 'NO HUMOR') for k in keys ]

print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=['HUMOR','NO HUMOR'], zero_division=0))
#print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=['HUMOR','NO HUMOR']))
