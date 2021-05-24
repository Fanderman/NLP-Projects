import itertools
import re
import random
import pandas as pd
import numpy as np
from decision_trees import *
from timeit import default_timer as timer


base_forms = ["adj", "adja", "adjc", "adjp", "adv", "burk", "depr", "ger", "conj", "comp", "num", "pact",
               "pant", "pcon", "ppas", "ppron12", "ppron3", "pred", "prep", "siebie", "subst", "verb", "brev",
               "interj", "qub"]

verb_forms = ["nom", "gen", "acc", "dat", "inst", "loc", "voc"]

raw_form = {"subst:nom":[], "subst:gen":[], "subst:acc":[], "subst:dat":[], "subst:inst":[], "subst:loc":[], "subst:voc":[],
            "adj":[], "adja":[], "adjc":[], "adjp":[], "adv":[], "burk":[], "depr":[], "ger":[], "conj":[], "comp":[], "num":[], "pact":[],
            "pant":[], "pcon":[], "ppas":[], "ppron12":[], "ppron3":[], "pred":[], "prep":[], "siebie":[], "verb":[], "brev":[],
            "interj":[], "qub":[], "target":[]}

empty_form = {"subst:nom":0, "subst:gen":0, "subst:acc":0, "subst:dat":0, "subst:inst":0, "subst:loc":0, "subst:voc":0,
            "adj":0, "adja":0, "adjc":0, "adjp":0, "adv":0, "burk":0, "depr":0, "ger":0, "conj":0, "comp":0, "num":0, "pact":0,
            "pant":0, "pcon":0, "ppas":0, "ppron12":0, "ppron3":0, "pred":0, "prep":0, "siebie":0, "verb":0, "brev":0,
            "interj":0, "qub":0, "target":0}


signs = ['.', '(', ')', ';', '"', '[', ']', ',', '?', '!', ':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
polish = [('ź', 'z'), ('ż', 'z'), ('ą', 'a'), ('ę', 'e'), ('ó', 'o'), ('ł', 'l'), ('ć', 'c'), ('ń', 'n')]


def tokenize(line):
    line2 = []
    line = line.split(' ')
    for base in line:
        base = base.lower()
        for sign in signs:
            base = base.replace(sign, ' ')
        base = base.strip()
        base = base.split(' ')
        if base != '' and base != ['']:
            line2.extend(base)

    return line2


def remove_polish(line):
    line2 = []
    for word in line:
        for sign in polish:
            word = word.replace(sign[0], sign[1])
        line2.append(word)
    return line2


def load_polimorph(file='polimorfologik-2.1.txt'):
    dictionary = {}
    with open(file, 'r', encoding='utf8') as base_file:
        for line in base_file:
            line = line.strip().lower()
            line = line.split(";")
            line[2] = line[2].split("+")
            nl = []
            for comp in line[2]:
                spl = comp.split(":")
                if spl[0] != "subst":
                    nl.append(spl[0])
                else:
                    nl.append(spl[0] + ":" + spl[2])
            line[2] = nl
            dictionary[line[1]] = (line[0], line[2])

    return dictionary


def create_casts(base_poli):
    dictionary = {}
    for key in base_poli:
        weak_key = remove_polish([key])[0]
        if weak_key not in dictionary:
            dictionary[weak_key] = []
        dictionary[weak_key].append(key)

    return dictionary


def load_unigrams(file='1grams'):
    dictionary = {}
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            line = line.strip().lower()
            line = line.split(' ')
            dictionary[line[1]] = int(line[0])

    return dictionary


def load_2grams(file='2grams', k=10):
    dictionary = {}
    i = 0
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            if i % 100000 == 0:
                print(i)
            line = line.strip().lower()
            line = line.split(' ')
            if int(line[0]) >= k:
                if line[1] not in dictionary:
                    dictionary[line[1]] = []
                dictionary[line[1]].append((line[2], int(line[0])))
            else:
                break
            i += 1

    return dictionary


def load_3grams(file='3grams', k=10):
    dictionary = {}
    i = 0
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            if i % 100000 == 0:
                print(i)
            line = line.strip().lower()
            line = line.split(' ')
            if int(line[0]) >= k:
                if line[1] not in dictionary:
                    dictionary[line[1]] = []
                dictionary[line[1]].append((line[2:], int(line[0])))
            else:
                break
            i += 1

    return dictionary


def load_set(file='train_shuf.txt', k=10000):
    i = 0
    lines = []
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            if i == k:
                break
            lines.append(tokenize(line))
            i += 1

    return lines


def divide_set(total_set, k=0.7):
    s = round(len(total_set)*k)
    return total_set[:s], total_set[s:]


def permute(line, casts):
    line2 = []
    for word in line:
        if word in casts:
            line2.append(casts[word])
        else:
            line2.append([word])

    conc = line2[0]
    for part in line2[1:]:
        conc = list(map(list, itertools.product(conc, part)))

    def flatten(listt):
        a = []
        for itemm in listt:
            if isinstance(itemm, list):
                a += flatten(itemm)
            else:
                a.append(itemm)
        return a

    conc2 = []
    for item in conc:
        conc2.append(flatten(item))
    return conc2


def wrong(line, casts):
    if line not in permute(line, casts):
        return True
    return False


def windows(line, k=12):
    if len(line) <= k:
        return [line]
    else:
        lines = []
        line2 = line[0:k]
        lines.append(line2.copy())
        for word in line[k:]:
            del line2[0]
            line2.append(word)
            lines.append(line2.copy())
        return lines


def train(training_set, poli, casts):
    # df = pd.DataFrame(data=raw_form)
    df = []
    # duos = pd.DataFrame(data={"fst":[], "snd":[], "target":[]})
    # trios = pd.DataFrame(data={"fst":[], "snd":[], "trd":[], "target":[]})
    trios = []
    dgrams = {}
    j = -1
    for base_line in training_set:
        j += 1
        if j % 1000 == 0:
            print(j)

        for line in windows(base_line, k=3):
            last_fst = None
            last_snd = None
            last_trd = None
            wrong_ans = wrong(line, casts)
            if not wrong_ans:
                for perm in permute(line, casts):
                    fst = None
                    snd = None
                    trio = {"fst":0, "snd":0, "trd":0, "target":0}
                    i = 0
                    for word in perm:
                        if word in poli:
                            if snd is not None:
                                if (snd, word) not in dgrams:
                                    dgrams[(snd, word)] = 0
                                dgrams[(snd, word)] += 1
                            for form in poli[word][1]:
                                if fst is not None and snd is not None and (fst != last_fst or snd != last_snd or word != last_trd):
                                    last_fst = fst
                                    last_snd = snd
                                    last_trd = word
                                    if line[i-2] == perm[i-2] and line[i-1] == perm[i-1] and line[i] == perm[i]:
                                        trio["target"] = "y"
                                    else:
                                        trio["target"] = "n"
                                    for form1 in poli[fst][1]:
                                        for form2 in poli[snd][1]:
                                            trio["fst"] = form1
                                            trio["snd"] = form2
                                            trio["trd"] = form
                                            trios.append(trio)
                            fst = snd
                            snd = word
                        else:
                            fst = snd
                            snd = None
                        i += 1
                    #if perm == line:
                    #    composition["target"] = "y"
                    #else:
                    #    composition["target"] = "n"
                    #df.append(composition)

    #df = pd.DataFrame(df)
    trios = pd.DataFrame(trios)
    return trios, dgrams


def train2(training_set, poli, casts, trios_tree, digrams1, digrams2):
    compilation = []
    j = -1
    for base_line in training_set:
        j += 1
        if j % 1000 == 0:
            print(j)
        for line in windows(base_line, k=3):
            wrong_ans = wrong(line, casts)
            if not wrong_ans:
                for perm in permute(line, casts):
                    if len(perm) == 3:
                        entry = {"trio": 0, "digram1": 0, "digram2": 0, "target": 0}
                        if perm == line:
                            entry["target"] = "y"
                        else:
                            entry["target"] = "n"
                        if (perm[0], perm[1]) in digrams1:
                            entry["digram1"] += 1
                        if (perm[1], perm[2]) in digrams1:
                            entry["digram1"] += 1
                        if (perm[0], perm[1]) in digrams2:
                            entry["digram2"] += 1
                        if (perm[1], perm[2]) in digrams2:
                            entry["digram2"] += 1
                        for i in range(1):
                            strng = "trio"
                            lst = []
                            for word in perm[i:i+3]:
                                if word not in poli:
                                    lst.append([None])
                                else:
                                    lst.append(poli[word])

                            conc = lst[0]
                            for part in lst[1:]:
                                conc = list(map(list, itertools.product(conc, part)))

                            def flatten(listt):
                                a = []
                                for itemm in listt:
                                    if isinstance(itemm, list):
                                        a += flatten(itemm)
                                    else:
                                        a.append(itemm)
                                return a

                            lst = []
                            for item in conc:
                                lst.append(flatten(item))
                            #mn = 1
                           # mx = 0
                            ttl = 0
                            for lst2 in lst:
                                dct = {"fst": lst2[0], "snd": lst2[1], "trd": lst2[2]}
                                sc = trios_tree.classify(dct)
                                ttl += sc
                                #if sc > mx:
                                #    mx = sc
                                #if sc < mn:
                                #    mn = sc
                            #if 0.5 - mn >= mx - 0.5:
                            #    entry[strng] = mn
                            #else:
                            #    entry[strng] = mx
                            entry[strng] = ttl/len(lst)
                        compilation.append(entry)

    df = pd.DataFrame(compilation)
    return df


def fix_polish(phrase, poli, casts, digrams1, digrams2, trios_tree, final_tree):
    ans = {}
    for i in range(len(phrase)):
        ans[i] = []
    i = 0
    for line in windows(phrase, k=3):
        i += 1
        mx = 0
        mperm = line
        for perm in permute(line, casts):
            if len(perm) == 3:
                entry = {"trio": 0, "digram1": 0, "digram2": 0}
                if (perm[0], perm[1]) in digrams1:
                    entry["digram1"] += 1
                if (perm[1], perm[2]) in digrams1:
                    entry["digram1"] += 1
                if (perm[0], perm[1]) in digrams2:
                    entry["digram2"] += 1
                if (perm[1], perm[2]) in digrams2:
                    entry["digram2"] += 1

                lst = []
                for word in perm[0:3]:
                    if word not in poli:
                        lst.append([None])
                    else:
                        lst.append(poli[word])

                conc = lst[0]
                for part in lst[1:]:
                    conc = list(map(list, itertools.product(conc, part)))

                def flatten(listt):
                    a = []
                    for itemm in listt:
                        if isinstance(itemm, list):
                            a += flatten(itemm)
                        else:
                            a.append(itemm)
                    return a

                lst = []
                for item in conc:
                    lst.append(flatten(item))
                #mn1 = 1
                #mx1 = 0
                ttl = 0
                for lst2 in lst:
                    dct = {"fst": lst2[0], "snd": lst2[1], "trd": lst2[2]}
                    sc1 = trios_tree.classify(dct)
                    ttl += sc1
                    #if sc1 > mx1:
                    #    mx1 = sc1
                    #if sc1 < mn1:
                    #    mn1 = sc1
                #if 0.5 - mn1 >= mx1 - 0.5:
                #    entry["trio"] = mn1
                #else:
                #    entry["trio"] = mx1
                entry["trio"] = ttl/len(lst)

                sc = final_tree.classify(entry)
                if sc > mx:
                    mx = sc
                    mperm = perm

        ans[i-1].append((mperm[0], mx))
        ans[i].append((mperm[1], mx))
        ans[i+1].append((mperm[2], mx))

    output = []
    for i in ans:
        mx = 0
        oword = None
        for word, sc in ans[i]:
            if sc > mx:
                mx = sc
                oword = word
        output.append(oword)

    return output


def score(phrase1, phrase2):
    s = 0
    for i in range(len(phrase1)):
        if phrase1[i] == phrase2[i]:
            s += 1

    return s/len(phrase1)


def start():
    poli = load_polimorph()
    casts = create_casts(poli)
    #unigrams = load_unigrams()
    digrams = load_2grams()
    #trigrams = load_3grams()
    total_set = load_set(k=100000)
    train_set1, train_set2 = divide_set(total_set, k=0.6)
    train_set1, validation_set1 = divide_set(train_set1, k=0.7)
    train_set2, train_set3 = divide_set(train_set2, k=0.7)
    trios, digrams2 = train(train_set1, poli, casts)
    trios2, _ = train(validation_set1, poli, casts)
    trios_tree = Tree(trios)
    trios_tree.start_prune(trios2)
    df = train2(train_set2, poli, casts, trios_tree, digrams, digrams2)
    final_tree = Tree(df)

    k = 0
    for line in train_set3:
        k += score(line, fix_polish(remove_polish(line), poli, casts, digrams, digrams2, trios_tree, final_tree))

    print(k / len(train_set3))

    print("rysowanie...")
    trios_tree.draw().render('test-output/round-table3.gv', view=False)
    final_tree.draw().render('test-output/final2.gv', view=False)


start()

