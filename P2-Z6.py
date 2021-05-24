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


super_dict = {}
def superbase_mapping(word_list):
    if len(super_dict) == 0:
        with open('superbazy.txt', 'r', encoding='utf8') as bazy:
            for line in bazy:
                w1, w2 = (line.split(' ')[0], line.split(' ')[1])
                super_dict[w1] = w2.strip()
    return [super_dict[w] if w in super_dict else w for w in word_list]


uni = load_unigram()
di = load_digram()


PT_lines = []
RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ\-]+')
with open('pan-tadeusz.txt', 'r', encoding='utf8') as pt:
    nf = False
    for line in pt:
        if not line.strip():
            continue
        if not nf:
            if line.strip().split(' ')[0] == 'Litwo!':
                nf = True
        if nf:
            if line.strip()[0] == '-':
                break
            else:
                PT_lines.append([word.lower() for word in re.findall(RE, line)])

vovels = ['a', 'ą', 'e', 'ę', 'i', 'o', 'u', 'y', 'ó']


def accent(sentence):
    res = []
    for w in sentence:
        c = 0
        for c1, c2 in zip('^' + w, w):
            if c2 in vovels and c1 != 'i':
                c += 1
        res.append(c)
    return res


word_to_tag = {}
tag_to_word = defaultdict(list)
with open('supertags.txt', 'r', encoding='utf8') as file:
    for line in file:
        word, tag = line.split()
        word_to_tag[word] = tag
        tag_to_word[tag].append(word)


def get_rhyme(word):
    for i in range(1, len(word)):
        if accent([word[-i:]])[0] == 2:
            return word[-i:]
    return None


def single_verset(sentences):
    idx = rnd.randint(1, len(sentences) - 2)
    rhyme = get_rhyme(sentences[idx][-1])

    if get_rhyme(sentences[idx - 1][-1]) == rhyme:
        return sentences[idx - 1], sentences[idx]
    elif get_rhyme(sentences[idx + 1][-1]) == rhyme:
        return sentences[idx], sentences[idx + 1]
    return None, None


def same_tags(w):
    if w in word_to_tag and rnd.random() > 0.05:
        return tag_to_word[word_to_tag[w]]
    elif ('#' + w)[-3:] in word_to_tag and rnd.random() > 0.05:
        return tag_to_word[word_to_tag[('#' + w)[-3:]]]
    else:
        return tag_to_word[rnd.choice(list((word_to_tag.values())))]


def random_werse():
    while True:
        sentence1 = []
        sentence2 = []
        start = rnd.randint(0, len(PT_lines) - 2)
        w1, w2 = single_verset(PT_lines)
        if not w1:
            continue

        if get_rhyme(w1[-1]) != get_rhyme(w2[-1]):
            continue

        a1, a2 = accent(w1), accent(w2)

        t1 = [same_tags(w) for w in w1]
        t2 = [same_tags(w) for w in w2]

        if None in t1 or None in t2:
            continue

        candidates1 = [[c for c in candidate if accent([c])[0] == a] for candidate, a in zip(t1, a1)]
        candidates2 = [[c for c in candidate if accent([c])[0] == a] for candidate, a in zip(t2, a2)]

        cn1 = []
        cn2 = []

        for w in candidates1[-1]:
            for c2 in candidates2[-1]:
                if get_rhyme(w) == get_rhyme(c2):
                    cn1.append(w)
                    break
        for w in candidates2[-1]:
            for c2 in candidates1[-1]:
                if get_rhyme(w) == get_rhyme(c2):
                    cn2.append(w)
                    break

        candidates1[-1] = cn1
        candidates2[-1] = cn2

        if [] in candidates1 or [] in candidates2:
            continue

        weights1 = [[uni[w] if w in uni else 2 for w in l] for l in candidates1]
        weights2 = [[uni[w] if w in uni else 2 for w in l] for l in candidates2]

        for _ in range(1000):
            res1 = []
            res2 = []
            for wl, weights in zip(candidates1, weights1):
                k = []
                ww = []
                for x, w2 in zip(wl, weights):
                    if x in res1 or x in res2:
                        break
                    else:
                        k.append(x)
                        ww.append(w2)
                if not k:
                    break
                res1.append(rnd.choices(k, weights=ww)[0])

            for wl, weights in zip(candidates2, weights2):
                k = []
                ww = []
                for x, w2 in zip(wl, weights):
                    if x in res2 or x in res1:
                        break
                    else:
                        k.append(x)
                        ww.append(w2)
                if not k:
                    break

                res2.append(rnd.choices(k, weights=ww)[0])

            if len(res1) < len(candidates1) or len(res2) < len(candidates2):
                continue
            if get_rhyme(res1[-1]) != get_rhyme(res2[-1]):
                continue
            return "{0}\n{1}\n".format(" ".join(res1), " ".join(res2))

    return ""


PPMI_ucount = sum([uni[c] for c in uni])


def PPMI_value(w1, w2, sb=0):
    if sb == 3:
        return 0.0
    if w1 not in uni or w2 not in uni:
        return PPMI_value(superbase_mapping([w1])[0], superbase_mapping(w2)[0], sb + 1)
    p = False
    for w, _ in di[w1]:
        if w == w2:
            p = True
            break
    if not p:
        return PPMI_value(w1, superbase_mapping([w2])[0], sb + 1)
    num = 0
    PPMI_count = sum([v for _, v in di[w1]])
    for w, n in di[w1]:
        if w == w2:
            num = n
            break
    return np.log((num / PPMI_count) / ((uni[w1] / PPMI_ucount) * (uni[w2] / PPMI_ucount)))


for i in range(5):
    print(random_werse())
    print()


def PPMI_value_sentence(sentence):
    score = 0.0
    for w1, w2 in zip(sentence, sentence[1:]):
        score += PPMI_value(w1, w2)
    return score


def same_tags(w):
    if w in word_to_tag and rnd.random() > 0.05:
        return tag_to_word[word_to_tag[w]]
    elif ('#' + w)[-3:] in word_to_tag and rnd.random() > 0.05:
        return tag_to_word[word_to_tag[('#' + w)[-3:]]]
    else:
        return tag_to_word[rnd.choice(list((word_to_tag.values())))]


def random_werse_PPMI():
    while True:
        sentence1 = []
        sentence2 = []
        start = rnd.randint(0, len(PT_lines) - 2)
        w1, w2 = single_verset(PT_lines)
        if not w1:
            continue

        if get_rhyme(w1[-1]) != get_rhyme(w2[-1]):
            continue

        a1, a2 = accent(w1), accent(w2)

        t1 = [same_tags(w) for w in w1]
        t2 = [same_tags(w) for w in w2]

        if None in t1 or None in t2:
            continue

        candidates1 = [[c for c in candidate if accent([c])[0] == a] for candidate, a in zip(t1, a1)]
        candidates2 = [[c for c in candidate if accent([c])[0] == a] for candidate, a in zip(t2, a2)]

        cn1 = []
        cn2 = []

        for w in candidates1[-1]:
            for c2 in candidates2[-1]:
                if get_rhyme(w) == get_rhyme(c2):
                    cn1.append(w)
                    break
        for w in candidates2[-1]:
            for c2 in candidates1[-1]:
                if get_rhyme(w) == get_rhyme(c2):
                    cn2.append(w)
                    break

        candidates1[-1] = cn1
        candidates2[-1] = cn2

        if [] in candidates1 or [] in candidates2:
            continue

        weights1 = [[uni[w] if w in uni else 2 for w in l] for l in candidates1]
        weights2 = [[uni[w] if w in uni else 2 for w in l] for l in candidates2]

        results_final1 = []
        results_final2 = []
        for _ in range(20000):
            res1 = []
            res2 = []
            for wl, weights in zip(candidates1, weights1):
                k = []
                ww = []
                for x, w2 in zip(wl, weights):
                    if x in res1 or x in res2:
                        break
                    else:
                        k.append(x)
                        ww.append(w2)
                if not k:
                    break
                res1.append(rnd.choices(k, weights=ww)[0])

            for wl, weights in zip(candidates2, weights2):
                k = []
                ww = []
                for x, w2 in zip(wl, weights):
                    if x in res2 or x in res1:
                        break
                    else:
                        k.append(x)
                        ww.append(w2)
                if not k:
                    break

                res2.append(rnd.choices(k, weights=ww)[0])

            if len(res1) < len(candidates1) or len(res2) < len(candidates2):
                continue
            if get_rhyme(res1[-1]) != get_rhyme(res2[-1]):
                continue
            results_final1.append((res1, PPMI_value_sentence(res1)))
            results_final2.append((res2, PPMI_value_sentence(res2)))

        if len(results_final1) < 5:
            continue

        res_final = []
        bs = 0.0
        for (r1, s1), (r2, s2) in zip(results_final1, results_final2):
            res_final.append((s1 + s2, (r1, r2)))
        res_final.sort(key=lambda x: x[0], reverse=True)
        return ["score {0}\n{1}\n{2}\n\n".format(x[0], " ".join(x[1][0]), " ".join(x[1][1])) for x in res_final[:5]]


for i in range(5):
    res = random_werse_PPMI()
    for w in res:
        print(w)
