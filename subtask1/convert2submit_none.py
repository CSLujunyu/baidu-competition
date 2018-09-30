# -*- coding: utf-8 -*-

import csv
import sys
import json
import numpy as np
from operator import itemgetter

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def convert_sample(sample):
    # sample.append([sample[0][0], 'None', 0])
    sort_sample = sorted(sample, key=itemgetter(2), reverse=True)
    probs = softmax([x[2] for x in sort_sample])
    converted_sample = {
        'example-id' : sample[0][0],
        'candidate-ranking' : []
    }
    for s, p in zip(sort_sample[:100], probs[:100]):
        converted_sample['candidate-ranking'].append(
            {
                'candidate-id' : s[1],
                'confidence' : p
            }
        )
    return converted_sample

def main():
    file_name = '/hdd/lujunyu/model/DSTC7/ubuntu/s1/model_1/test_score.txt'
    save_path = '/hdd/lujunyu/model/DSTC7/ubuntu/s1/model_1/Ubuntu_subtask_1.json'
    with open(file_name, 'r') as fin:
        raw_result = csv.reader(fin, delimiter='\t')
        converted_result = []
        sample = []
        sample_id = None
        for row in raw_result:
            if sample_id is None or sample_id != row[0]:
                if sample:
                    converted_result.append(convert_sample(sample))
                sample_id = row[0]
                sample = []
            row[2] = float(row[2])
            sample.append(row[:3])
        if sample:
            converted_result.append(convert_sample(sample))
    with open(save_path, "w") as f:
        f.write(json.dumps(converted_result, indent=4))
        print("Finished...")

if __name__ == "__main__":
    main()
