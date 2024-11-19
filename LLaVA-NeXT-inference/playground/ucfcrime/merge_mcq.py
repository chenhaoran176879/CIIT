import json

# File paths
test_file_path = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_test_mcq.jsonl'
train_file_path = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_train_mcq.jsonl'
val_file_path = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_val_mcq.jsonl'

# Load JSONL data into dictionaries
def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            data[item['index']] = item
    return data

# Load all data
test_data = load_jsonl(test_file_path)
train_data = load_jsonl(train_file_path)
val_data = load_jsonl(val_file_path)

# Update test data with train and val data
for index, item in {**train_data, **val_data}.items():
    if index in test_data:
        test_data[index].update(item)

# Write the updated test data back to a new file
output_file_path = '/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_mcq.jsonl'
with open(output_file_path, 'w') as file:
    for item in test_data.values():
        file.write(json.dumps(item) + '\n')

print("Test data has been updated successfully!")
