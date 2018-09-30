import numpy as np

def get_p_at_n_in_m(data, n, m, ind):
    curr = data[ind:ind+m]
    candidate_id, score, label = zip(*curr)

    assert len(set(label)) == 1

    candidate_id = candidate_id[:n]
    if label[0] in candidate_id:
        return 1
    else:
        return 0

def MRR(data,ind):
    curr = data[ind:ind + 100]

    for n, item in enumerate(curr):
        if item[0] == item[2]:
            return 1/(n+1)

    return 0.0

def evaluate(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split("\t")
            if len(tokens) != 4:
                continue

            data.append([str(tokens[1]), float(tokens[2]), str(tokens[3])])

    p_at_1_in_100 = 0.0
    p_at_2_in_100 = 0.0
    p_at_5_in_100 = 0.0
    p_at_10_in_100 = 0.0
    p_at_50_in_100 = 0.0
    mrr = 0.0

    length = int(len(data) / 100)
    for i in range(length):
        ind = i * 100

        p_at_1_in_100 += get_p_at_n_in_m(data, 1, 100, ind)
        p_at_2_in_100 += get_p_at_n_in_m(data, 2, 100, ind)
        p_at_5_in_100 += get_p_at_n_in_m(data, 5, 100, ind)
        p_at_10_in_100 += get_p_at_n_in_m(data, 10, 100, ind)
        p_at_50_in_100 += get_p_at_n_in_m(data, 50, 100, ind)
        mrr += MRR(data, ind)

    p_at_1_in_100 = p_at_1_in_100 / length
    p_at_2_in_100 = p_at_2_in_100 / length
    p_at_5_in_100 = p_at_5_in_100 / length
    p_at_10_in_100 = p_at_10_in_100 / length
    p_at_50_in_100 = p_at_50_in_100 / length
    mrr = mrr / length
    average = (p_at_10_in_100 + mrr) / 2

    return (p_at_1_in_100, p_at_2_in_100, p_at_5_in_100, p_at_10_in_100, p_at_50_in_100, mrr, average)
