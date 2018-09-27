import numpy as np

def get_p_at_n_in_m(data, n, m, ind, none_flag):
    curr = data[ind:ind+m]
    if none_flag!=1:
        curr.append([0,1])
    curr = sorted(curr, key = lambda x:x[0], reverse=True)[:m]
    flag = np.sum(np.array(curr)[:n,1])
    if flag == 1:
        return 1
    else:
        return 0

def MRR(data,ind, none_flag):
    curr = data[ind:ind + 100]
    if none_flag != 1:
        curr.append([0, 1])
    curr = sorted(curr, key=lambda x: x[0], reverse=True)[:100]
    for n, item in enumerate(curr):
        if item[1] == 1:
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

            data.append([float(tokens[2]), int(tokens[3])])

    p_at_1_in_100 = 0.0
    p_at_2_in_100 = 0.0
    p_at_5_in_100 = 0.0
    p_at_10_in_100 = 0.0
    p_at_50_in_100 = 0.0
    mrr = 0.0
    tmp_label = list(zip(*data))[1]

    length = int(len(data) / 100)
    for i in range(length):
        ind = i * 100
        none_flag = sum(tmp_label[ind:ind+100])

        p_at_1_in_100 += get_p_at_n_in_m(data, 1, 100, ind, none_flag)
        p_at_2_in_100 += get_p_at_n_in_m(data, 2, 100, ind, none_flag)
        p_at_5_in_100 += get_p_at_n_in_m(data, 5, 100, ind, none_flag)
        p_at_10_in_100 += get_p_at_n_in_m(data, 10, 100, ind, none_flag)
        p_at_50_in_100 += get_p_at_n_in_m(data, 50, 100, ind, none_flag)
        mrr += MRR(data, ind, none_flag)

    p_at_1_in_100 = p_at_1_in_100 / length
    p_at_2_in_100 = p_at_2_in_100 / length
    p_at_5_in_100 = p_at_5_in_100 / length
    p_at_10_in_100 = p_at_10_in_100 / length
    p_at_50_in_100 = p_at_50_in_100 / length
    mrr = mrr / length
    average = (p_at_10_in_100 + mrr) / 2

    return (p_at_1_in_100, p_at_2_in_100, p_at_5_in_100, p_at_10_in_100, p_at_50_in_100, mrr, average)
