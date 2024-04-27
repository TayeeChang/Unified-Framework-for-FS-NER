import json
import os.path


def load_data_finetune(data):
    texts, labels = data
    d = ['']
    preFlag = 'O'
    for i in range(len(labels)):
        char = texts[i]
        flag = labels[i]
        if flag != 'O':
            if flag != preFlag:
                d.append([len(d[0]), len(d[0]) + len(char) - 1, flag])
            else:
                d[-1][1] = len(d[0]) + len(char) - 1
        preFlag = flag
        d[0] += char + ' '
    d[0] = d[0].rstrip()
    return d


group = 'intra'
suffix = 'test_10_5.jsonl'
file = f'D://dataset//FewNerd//episode-data//{group}//{suffix}'
base = os.path.basename(file)[:-6]

with open(file, 'r', encoding='utf-8') as fr:
    for index, line in enumerate(fr):
        line = json.loads(line)
        support = line['support']
        query = line['query']
        types = line['types']

        support_examples = []
        for i in range(len(support['word'])):
            support_example = load_data_finetune((support['word'][i], support['label'][i]))
            support_examples.append(support_example)

        query_examples = []
        for i in range(len(query['word'])):
            query_example = load_data_finetune((query['word'][i], query['label'][i]))
            query_examples.append(query_example)

        label2dict = dict()
        for l in types:
            label2dict[l] = l

        rootPath = f'{group}/episode/{base}/{index}'
        if not os.path.exists(rootPath):
            os.makedirs(rootPath)

        with open(rootPath + '/support.example', 'w', encoding='utf-8') as fw:
            for line in support_examples:
                fw.write(json.dumps(line) + '\n')

        with open(rootPath + '/query.example', 'w', encoding='utf-8') as fw:
            for line in query_examples:
                fw.write(json.dumps(line) + '\n')

        with open(rootPath + '/fewshot.jsonl', 'w', encoding='utf-8') as fw:
            fw.write(json.dumps(label2dict, indent=4))

        if index == 49:  # stop
            break





