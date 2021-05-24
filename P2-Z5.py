import re
import random as rnd
from collections import defaultdict
import numpy as np
import glob
from pprint import pprint
from itertools import permutations
from functools import reduce
import pprint


def load_unigram(file='1grams'):
    dictionary = {}
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().lower()
            line = line.split(' ')
            dictionary[line[1]] = int(line[0])

    return dictionary


def load_digram(file='2grams', k=10):
    dictionary = defaultdict(list)
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            line = line.strip().lower()
            line = line.split(' ')
            if int(line[0]) >= k:
                dictionary[line[1]].append((line[2], int(line[0])))
            else:
                break

    return dictionary


def load_tags(file='supertags.txt'):
    tags = {}
    with open(file, 'r', encoding='utf8') as t:
        for line in t:
            key, value = line.split()
            tags[key.lower()] = value.lower()

    return tags


def tag_mapper_digram(tags, digrams):
    mapper = defaultdict(list)
    for start in digrams:
        if start not in tags:
            continue
        for end, weight in digrams[start]:
            if end not in tags:
                continue
            tag1 = tags[start]
            tag2 = tags[end]
            mapper[(tag1, tag2)].append(((start, end), weight))
    return mapper


uni = load_unigram()
di = load_digram()
tags = load_tags()
zad4_mapper = tag_mapper_digram(tags, di)


def PPMI(word, k):
    RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ\-]+')
    word = word.lower()
    if word not in di:
        print('Unique word: {1}'.format(k))
        return []
    result = []
    count = sum([v for _, v in di[word]])
    uni_count = sum([uni[c] for c in uni])
    for w, num in di[word]:
        if w not in uni:
            continue
        res = re.findall(RE, w)
        if len(res):
            w = res[0]
        if any([v == w for v, _ in result]):
            continue
        result.append((w, np.log((num / count) / ((uni[word] / uni_count) * (uni[w] / uni_count)))))
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:k]


def PPMI_ssc(word, k):
    RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ\-]+')
    word = word.lower()
    if word not in di:
        print('Unique word: {1}'.format(word))
        return []
    result = []
    count = sum([v for _, v in di[word]])
    uni_count = sum([uni[c] for c in uni])
    for w, num in di[word]:
        if w not in uni:
            continue
        res = re.findall(RE, w)
        if len(res):
            w = res[0]
        if any([v == w for v, _ in result]):
            continue
        x = (num / uni_count)
        result.append((w, (x - ((uni[word] / uni_count) * (uni[w] / uni_count))) / np.sqrt(np.power(x * (1.0 - x), 2) / uni_count)))
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:k]


t_sum = defaultdict(int)
for w in uni:
    if w in tags:
        t_sum[tags[w]] += 1


def PPMI_tags(word, w2):
    result = []

    t1 = tags[word]

    count = sum([sum([x[1] for x in zad4_mapper[w]]) for w in zad4_mapper])
    t_count = sum([1 for c in tags])
    if w2 not in tags:
        return 0
    t2 = tags[w2]
    if (t1, t2) not in zad4_mapper:
        return 0
    weight = sum([x[1] for x in zad4_mapper[(t1, t2)]])
    return (np.log((weight / count) / (t_sum[t1] / t_count) * (t_sum[t2] / t_count)))


def PPMI_T(word, k):
    RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ]+')
    word = word.lower()
    if word not in di:
        print('Unique word: {1}'.format(k))
        return []
    result = []
    count = sum([v for _, v in di[word]])
    uni_count = sum([uni[c] for c in uni])
    for w, num in di[word]:
        if w not in uni:
            continue
        res = re.findall(RE, w)
        if len(res):
            w = res[0]
        else:
            continue
        if any([v == w for v, _ in result]):
            continue
        r = PPMI_tags(word, w)
        result.append((w, np.log((num / count) / ((uni[word] / uni_count) * (uni[w] / uni_count))) + r))
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:k]


test_words = ["dom", "szpieg", "pies", "muzyka", "rower", "gra", "student", "wywar"]

P1 = []
P2 = []
P3 = []
for w in test_words:
    P1.append(PPMI(w, 10))
    P2.append(PPMI_ssc(w, 10))
    P3.append(PPMI_T(w, 10))
merged = [(w, list(zip(e1, e2, e3)))for w, e1, e2, e3 in zip(test_words, P1, P2, P3)]

pp = pprint.PrettyPrinter(width=200, depth=7)
pp.pprint(merged)