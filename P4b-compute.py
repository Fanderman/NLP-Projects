#%%

import itertools
import re
import random
import pandas as pd
import numpy as np
import math
from timeit import default_timer as timer

from pysuffixarray.core import SuffixArray
from bayes_opt import BayesianOptimization

alphabet = [' ', 'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n',
            'ń', 'o', 'ó', 'p', 'q', 'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ż', 'ź']

#%%

import sys
import itertools

class RMQ:
    def __init__(self, n):
        self.sz = 1
        self.inf = (1 << 31) - 1
        while self.sz <= n: self.sz = self.sz << 1
        self.dat = [self.inf] * (2 * self.sz - 1)

    def update(self, idx, x):
        idx += self.sz - 1
        self.dat[idx] = x
        while idx > 0:
            idx = (idx - 1) >> 1
            self.dat[idx] = min(self.dat[idx * 2 + 1], self.dat[idx * 2 + 2])

    def query(self, a, b):
        return self.query_help(a, b, 0, 0, self.sz)

    def query_help(self, a, b, k, l, r):
        if r <= a or b <= l:
            return 9999999
        elif a <= l and r <= b:
            return self.dat[k]
        else:
            return min(self.query_help(a, b, 2 * k + 1, l, (l + r)>>1),
                       self.query_help(a, b, 2 * k + 2, (l + r) >> 1, r))

#%%

def load_text(file='.\P4b-data\sentences_for_task1.txt'):
    complete_text = ''
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            line = line.strip().lower()
            complete_text += ' ' + line

    complete_text = complete_text + ' '
    return complete_text

text = load_text()
sabase = SuffixArray(text)
SA = sabase.suffix_array()
LCP = sabase.longest_common_prefix()

#%%

precomp_length=18
precomp_adj=6

def precompute_counts(ln):
    word_counts = {}
    template = {}
    for a in alphabet:
        template[a] = 0
    adj_table = {}
    for i in range(len(SA)):
        for j in range(1, ln):
            wrd = text[SA[i]:SA[i]+j]
            if wrd not in word_counts:
                word_counts[wrd] = [0, i]
                if j <= precomp_adj:
                    adj_table[wrd] = [template.copy(), template.copy()]
            word_counts[wrd][0] += 1
            if j <= precomp_adj:
                if SA[i]-1 > 0:
                    adj_table[wrd][0][text[SA[i]-1]] += 1
                if SA[i]+j < len(text):
                    adj_table[wrd][1][text[SA[i]+j]] += 1

    return word_counts, adj_table

#%%

word_counts, adj_table = precompute_counts(precomp_length)

#%%

rmq = RMQ(len(SA))

for i in range(len(LCP)):
    rmq.update(i, LCP[i])

#%%

print(LCP[0:5])
print(text[SA[2]:SA[2]+10])
print(text[SA[3]:SA[3]+10])
rmq.query(2, 1)

#%%

# count the average amount of each letter
def count_letters():
    letters = {}
    for a in alphabet:
        letters[a] = 0
    for l in text:
        letters[l] += 1
    sm = 0
    for a in letters:
        sm += letters[a]
    return letters, sm

letters, letters_sum = count_letters()

#%%

def check_prefix(pre, w, init=0):
    ln = len(pre)
    lw = len(w)
    """
    w = w[:ln]
    if pre == w:
        return ln, True
    return False, w < pre
    """
    for i in range(init, ln, 1):
        if lw <= i:
            return False, True
        if w[i] != pre[i]:
            return i, w[i] < pre[i]
    return ln, True

def find_highest(w, li, ri):
    i = li
    while li < ri:
        preli = li
        preri = ri
        prei = i
        i = (li+ri)//2
        lw = len(w)
        ans = rmq.query(li+1, i+1)
        if ans < lw:
            ri = i
        else:
            li = i
        if preli == li and preri == ri:
            return li
    return li

def find_lowest(w, li, ri):
    i = ri
    while li < ri:
        preli = li
        preri = ri
        prei = i
        i = (li+ri)//2
        lw = len(w)
        ans = rmq.query(i+1, ri+1)
        if ans < lw:
            li = i
        else:
            ri = i
        if preli == li and preri == ri:
            return ri
    return ri

def count_words(w):
    if len(w) <= precomp_length:
        if w in word_counts:
            return word_counts[w]
        return 0, 10

    li = 0
    ri = len(SA)-1
    lw = len(w)
    i = -1
    ans = 0
    while li < ri:
        preli = li
        preri = ri
        prei = i
        i = (li+ri)//2
        if prei < i:
            init = rmq.query(prei+1, i+1)
        else:
            init = rmq.query(i+1, prei+1)
        init = min(init, ans)
        ans, direction = check_prefix(w, text[SA[i]:], init)
        if ans < lw:
            if direction:
                li = i
            else:
                ri = i
            if preli == li and preri == ri:
                return 0, 0
        else:
            h = find_highest(w, i, ri)
            l = find_lowest(w, li, i)
            return h - l + 1, l


#%%

def find_split(w, min_split=1, a=1, b=1):
    abc = check_adj(w)
    adj00 = check_alpha(abc[0])
    adj01 = check_alpha(abc[1])
    bc = max(adj00, adj01)
    hs = bc
    #bc, _ = count_words(w)
    #hs = bc
    si = 0

    for i in range(1, len(w)):
        w1 = w[:i]
        w2 = w[i:]
        if w2 not in potential:
            #bw1, _ = count_words(w1)
            #bw2, _ = count_words(w2)
            adj1 = check_alpha(check_adj(w1)[1])
            adj2 = check_alpha(check_adj(w2)[0])

            if max(adj1, adj2) < hs:
                hs = max(adj1, adj2)
                si = i
        #if min(bw1, bw2) > hs:
        #    hs = min(bw1, bw2)
        #    si = i

    if hs > bc * min_split:
        return True, si, hs/bc
    else:
        return False, si, hs/bc

#%%

c, p = count_words("ztabu")
print(c, p)
print(text[SA[p-1]:SA[p-1]+10])
print(text[SA[p]:SA[p]+10])
print(text[SA[p+c-1]:SA[p+c-1]+10])
print(text[SA[p+c]:SA[p+c]+10])


#%%

def check_adj(w):
    if len(w) <= precomp_adj:
        return adj_table[w]
    bs, sp = count_words(w)
    prefixes = {}
    suffixes = {}
    if bs > 1000000:
        for a in alphabet:
            pre, _ = count_words(a + w)
            suf, _ = count_words(w + a)
            prefixes[a] = pre
            suffixes[a] = suf
    else:
        for a in alphabet:
            prefixes[a] = 0
            suffixes[a] = 0
        for i in range(sp, sp+bs):
            if SA[i] - 1 > 0:
                prefixes[text[SA[i]-1]] += 1
            if SA[i] + len(w) < len(text):
                suffixes[text[SA[i]+len(w)]] += 1

    return prefixes, suffixes

#%%

def check_alpha(alpha):
    sm = 0
    mx = 0
    for l in alpha:
        sm += alpha[l]
        if alpha[l] > mx:
            mx = alpha[l]

    score = 0
    for a in alphabet:
        d1 = (letters[a])/letters_sum
        d2 = (alpha[a])/sm
        score += (d2-d1)*(d2-d1)
    if alpha[' '] == 0 and mx > 15:
        score += 0.5

    return score

#%%

def find_phrases(w):
    c, p = count_words(w)
    phrases = []
    for i in range(p, p+c, 1):
        lf = SA[i]
        while text[lf] != ' ':
            lf -= 1
        rt = SA[i]
        while text[rt] != ' ':
            rt += 1
        phrases.append((text[lf+1:rt], SA[i]-lf))
    return phrases

#%%

def split_phrase(ph, p, l):
    w1 = ph[:p-1]
    w2 = ph[p+l-1:]
    bs1 = 2
    bs2 = 2

    for i in range(5, min(10,len(w2))):
        wrd = w2[:i]
        ww1 = check_adj(wrd)
        ans11 = check_alpha(ww1[0])
        _, si, _ = find_split(wrd)
        if si > 0:
            ww21 = wrd[:si]
            ans11 = min(ans11, check_alpha(check_adj(ww21)[0]))
        #print(wrd, ans11)
        if ans11 < bs2:
            bs2 = ans11

    for i in range(5, min(12, len(w1))):
        wrd = w1[-i:]
        ww1 = check_adj(wrd)
        ans12 = check_alpha(ww1[1])
        _, si, _ = find_split(wrd)
        if si > 0:
            ww21 = wrd[:si]
            ans12 = min(ans12, check_alpha(check_adj(ww21)[0]))
        #print(wrd, ans12)
        if ans12 < bs1:
            bs1 = ans12

    return bs1, bs2


#%%

#czasach 0.031123297961619303 0.020721112185516068
#eliczne 0.6118221203851923 0.0252523649090908
#niem 0.023549158377852437 999
#demo 0.03331824185837327 0.14103016019263753
#kowych 0.6639164598939549 999
#wzięte 0.018950865005565084 0.12897087396779425
def check_phrases(w, mx=50):
    phs = find_phrases(w)
    phs = random.sample(phs, min(mx,len(phs)))
    mn1 = 0
    mn2 = 0
    for ph in phs:
        sc1, sc2 = split_phrase(ph[0], ph[1], len(w))
        if sc1 == 2:
            sc1 = 0.05
        if sc2 == 2:
            sc2 = 0.05
        mn1 += sc1
        mn2 += sc2

    return mn1/len(phs), mn2/len(phs)

def check_phrases0(w, mx=50):
    phs = find_phrases(w)
    phs = random.sample(phs, min(mx,len(phs)))
    mn1 = 2
    mn2 = 2
    for ph in phs:
        sc1, sc2 = split_phrase(ph[0], ph[1], len(w))
        if sc1 < mn1:
            mn1 = sc1
        if sc2 < mn2:
            mn2 = sc2
    if len(w) > 4:
        if mn1 == 2:
            mn1 = 0.05
        if mn2 == 2:
            mn2 = 0.05
    return mn1, mn2

#%%

def find_suffixes(breakpoint = 0.04, breakpoint2 = 100, breakpoint3 = 0.001):
    suffixes = {}
    for word in word_counts:
        if len(word) <= 4 and count_words(word)[0] > breakpoint2:
            adjs = check_adj(word)
            sm1 = 0
            for a in alphabet:
                sm1 += adjs[0][a]
            if adjs[0][' ']/sm1 <= breakpoint3:
                sm2 = 0
                for a in alphabet:
                    sm2 += adjs[1][a]
                if adjs[1][' ']/sm2 > breakpoint*len(word):
                    suffixes[word] = adjs[1][' ']/sm2
    return suffixes

potential = find_suffixes()
#%%

def resolve_word(w, debug=False):
    if len(w) <= 3:
        ans, s, _ = find_split(w, 50)
    else:
        ans, s, _ = find_split(w)
    if ans:
        if not debug:
            return False
        print(w[:s], w[s:])
        ans1 = check_alpha(check_adj(w[:s])[0])
        ans2 = check_alpha(check_adj(w[s:])[1])
        return False, ans, ans1, ans2
    else:
        ww = check_adj(w)
        ans1 = check_alpha(ww[0])
        ans2 = check_alpha(ww[1])

    if not debug:
        if ans1 and ans2:
            return True
        return False
    else:
        if ans1 and ans2:
            return True, ans, ww, ans1, ans2
        return False, ans, ww, ans1, ans2

#%%

def resolve_pair(w1, w2, a=0.15, b=3.5, c=3, d=0.2, e=0.2):
    _, _, h1 = find_split(w1)
    _, _, h2 = find_split(w2)

    ww1 = check_adj(w1)
    ans11 = check_alpha(ww1[0])
    ans12 = check_alpha(ww1[1])
    ww2 = check_adj(w2)
    ans21 = check_alpha(ww2[0])
    ans22 = check_alpha(ww2[1])
    ans1 = max(ans11, ans12)
    ans2 = max(ans21, ans22)

    if not (h1 == 1 and h2 == 1):
        if h2 == 1:
            return -1, h1, h2, ans1, ans2
        if h1 == 1:
            return 1, h1, h2, ans1, ans2

    sp11, sp12 = check_phrases(w1)
    sp21, sp22 = check_phrases(w2)

    if not ((sp11 > sp21 * b or sp12 > sp22 * b) and (sp11 * b < sp21 or sp12 * b < sp22)):
        if sp11 > sp21 * b or sp12 > sp22 * b:
            return -1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22
        if sp11 * b < sp21 or sp12 * b < sp22:
            return 1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22

    if h1/h2 < a:
        return -1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22
    if h2/h1 < a:
        return 1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22

    if ans1 > ans2 * c:
        return -1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22
    if ans1 * c < ans2:
        return 1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22

    if  sp11 + sp12 - d*h1 + e*ans1 > sp21 + sp22 - d*h2 + e*ans2:
        return -1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22
    else:
        if sp11 + sp12 - d*h1 + e*ans1 < sp21 + sp22 - d*h2 + e*ans2:
            return 1, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22
        else:
            return 0, h1, h2, ans1, ans2, sp11, sp12, sp21, sp22

#%%

def load_tests(file='.\P4b-data\\test_for_task1.txt'):
    complete_database = []
    i = 0
    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            #if i > 10:
            #    break
            line = line.strip()
            line = line.split(' ')
            complete_database.append(line[0])
            complete_database.append(line[1])
            i += 1

    return complete_database

tests = load_tests()

#%%

def resolve_tests(tests, a=0.345, b=3.5, c=1.908, d=0.2, e=0.2, debug=False):
    score = 0
    for i in range(0, len(tests), 2):
        res = resolve_pair(tests[i], tests[i+1], a, b, c)
        if res[0] == 1:
            score += 1
        if res[0] == 0:
            score += 0.5
        if debug:
            print(tests[i], tests[i+1], res)

    return (score*2) / len(tests)

#%%

k = 400
def optimize_resolve(a, b, c, d, e):
    return resolve_tests(tests[k*2:k*3], a, b, c, d, e)

#%%

resolve_tests(tests[k*11:k*20], debug=True)

#%%

resolve_tests(tests[k:k*2], debug=True)

#resolve_tests(["rować", "napęd"], debug=True)

#%%

# Bounded region of parameter space
pbounds = {'a': (0, 1), 'b': (0,10), 'c': (0,10), 'd': (0,1), 'e': (0,1)}

optimizer = BayesianOptimization(
    f=optimize_resolve,
    pbounds=pbounds,
)

optimizer.maximize(
    init_points=25,
    n_iter=300,
)

print(optimizer.max)

#%%
