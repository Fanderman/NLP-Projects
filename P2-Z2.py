import itertools
import bayes_opt
from bayes_opt import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import re


def load_tags(file='supertags.txt'):
    tags = {}
    with open(file, 'r', encoding='utf8') as corpus:
        for line in corpus:
            key, value = line.split()
            tags[key.lower()] = value.lower()

    return tags


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


def load_bases(file='superbazy.txt'):
    base = {}
    with open(file, 'r', encoding='utf8') as corpus:
        for line in corpus:
            key, value = line.split()
            base[key] = value

    return base


def create_tag_2grams(grams, tags):
    taggrams = {}
    for gram in grams:
        if gram in tags:
            tag = tags[gram]
            if tag not in taggrams:
                taggrams[tag] = {}
            for continuation in grams[gram]:
                target = continuation[0]
                if target in tags:
                    target = tags[target]
                    if target not in taggrams[tag]:
                        taggrams[tag][target] = 0
                    taggrams[tag][target] += continuation[1]

    return taggrams


def create_tag_3grams(grams, tags):
    taggrams = {}
    for gram in grams:
        if gram in tags:
            tag = tags[gram]
            if tag not in taggrams:
                taggrams[tag] = {}
            for continuation in grams[gram]:
                target1 = continuation[0][0]
                target2 = continuation[0][1]
                if target1 in tags and target2 in tags:
                    target1 = tags[target1]
                    target2 = tags[target2]
                    if (target1, target2) not in taggrams[tag]:
                        taggrams[tag][(target1, target2)] = 0
                    taggrams[tag][(target1, target2)] += continuation[1]

    return taggrams


def gram_value(word1, word2, grams, base, tag_val, b):
    if word1 in grams:
        gram = grams[word1]
        total = 0
        found = False
        for par in gram:
            total += par[1]
            if str(par[0][0]).strip() == word2:
                found = par[1]

        if found:
            return found/total

    if tag_val > 0:
        n_word1 = word1
        if word1 in base:
            n_word1 = base[word1]
        n_word2 = word2
        if word2 in base:
            n_word2 = base[word2]
        return gram_value(n_word1, n_word2, grams, base, 0, b) * b

    return 0


def tag_value(word1, word2, tags, taggrams):
    if word1 not in tags or word2 not in tags:
        return 0

    tag1 = tags[word1]
    tag2 = tags[word2]
    if tag2 not in taggrams[tag1]:
        return 0

    total = sum(taggrams[tag1].values())
    return taggrams[tag1][tag2]/total


def calc(words, grams, tags, base, taggrams, a=0.8, b=0.5):
    perms = []
    for perm in itertools.permutations(words):
        total_value = 0
        falses = 0
        for i in range(len(perm)-1):
            word1 = str(perm[i]).strip()
            word2 = str(perm[i+1]).strip()

            tag_val = tag_value(word1, word2, tags, taggrams)
            gram_val = gram_value(word1, word2, grams, base, tag_val, b)

            if gram_val + tag_val == 0:
                falses += 1
            else:
                total_value += a * gram_val + (1-a) * tag_val

        perms.append((perm, (total_value, falses)))

    global size
    size = len(words)

    def conv(a):
        global size
        return size-a[1][1]+a[1][0]

    perms.sort(reverse=True, key=conv)
    #for i in range(5):
        #print(perms[i])

    return perms[0]


def load_words(file='learning_set.txt'):
    RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ\-]+')
    with open(file, 'r', encoding='utf8') as o_corpus:
        lines = o_corpus.readlines()
        data = []
        for line in lines:
            normalized_line = [word.lower() for word in re.findall(RE, line)]
            if len(normalized_line) > 0:
                data.append(normalized_line)
    return data


def score(original, created):
    distance = 0
    for i in range(len(original)):
        for j in range(len(created)):
            if original[i] == created[j]:
                distance += abs(j-i)
                break
    return -distance


def load_global_all():
    global learning_set
    learning_set = load_words()
    global grams
    grams = load_2grams(k=5)
    global tags
    tags = load_tags()
    global base
    base = load_bases()
    global taggrams
    taggrams = create_tag_2grams(grams, tags)


def learning(a, b):
    global learning_set
    global grams
    global tags
    global base
    global taggrams
    total = 0

    for line in learning_set:
        target = calc(line, grams, tags, base, taggrams, a, b)
        print(line, target)
        total += score(line, target[0])

    print(total)
    return total


def operate(words):
    grams = load_2grams(k=5)
    tags = load_tags()
    base = load_bases()
    taggrams = create_tag_2grams(grams, tags)
    print(calc(words, grams, tags, base, taggrams))
    print(words, score(words, calc(words, grams, tags, base, taggrams)[0]))


load_global_all()

pbounds = {'a': (0, 1), 'b': (0, 1)}

optimizer = bayes_opt.BayesianOptimization(
    f=learning,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

#logger1 = JSONLogger(path="./logs5.json")
#optimizer.subscribe(Events.OPTMIZATION_STEP, logger1)

optimizer.maximize(
    init_points=10,
    n_iter=10,
)


#words = ['wyjątkowo', 'ciekawy', 'okaz', 'sztuki', 'kulinarnej']
#words = ['zjadłem', 'pyszną', 'kanapkę', 'z', 'szynką', 'i', 'serem']
#words = ['to', 'był', 'mój', 'najlepszy', 'mecz']
#words = ['judyta', 'dała', 'wczoraj', 'stefanowi', 'czekoladki']
#words = ['babuleńka', 'miała', 'dwa', 'rogate', 'koziołki']
#words = ['wczoraj', 'wieczorem', 'spotkałem', 'pewną', 'piękną', 'kobietę']

#operate(words)


