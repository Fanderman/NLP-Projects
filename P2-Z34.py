import itertools
import re
import random


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


def load_tags(file='supertags.txt'):
    tags = {}
    with open(file, 'r', encoding='utf8') as corpus:
        for line in corpus:
            key, value = line.split()
            tags[key.lower()] = value.lower()

    return tags


def reverse_tags(tags, unigrams):
    reversed_tags = {}
    for word in tags:
        if tags[word] not in reversed_tags:
            reversed_tags[tags[word]] = []
        weight = 1
        if word in unigrams:
            weight += unigrams[word]
        reversed_tags[tags[word]].append((word, weight))

    return reversed_tags


def reverse_bi_tags(tags, bigrams):
    reversed_bi_tags = {}
    for start in bigrams:
        if start not in tags:
            continue
        for end, weight in bigrams[start]:
            if end not in tags:
                continue
            tag1 = tags[start]
            tag2 = tags[end]
            if (tag1, tag2) not in reversed_bi_tags:
                reversed_bi_tags[(tag1, tag2)] = []
            reversed_bi_tags[(tag1, tag2)].append(((start, end), weight))

    return reversed_bi_tags


def load_words(file='examples34.txt'):
    RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ\-]+')
    with open(file, 'r', encoding='utf8') as o_corpus:
        lines = o_corpus.readlines()
        data = []
        for line in lines:
            normalized_line = [word.lower() for word in re.findall(RE, line)]
            if len(normalized_line) > 0:
                data.append(normalized_line)
    return data


def assign_tag(word, reversed_tags):
    suf = word
    if len(word) > 2:
        suf = word[-2:]

    max_result = 0
    max_tag = 't1000'

    for tag in reversed_tags:
        counted = 0
        total = 0
        for word2 in reversed_tags[tag]:
            suf2 = word2[0]
            if len(suf2) > 2:
                suf2 = suf2[-2:]
            if suf == suf2:
                counted += 1
            total += 1

        if counted/total > max_result:
            max_tag = tag
            max_result = counted/total

    return max_tag


def uni_swap(words, tags, reversed_tags):
    new_phrase = []
    for word in words:
        if word in tags:
            tag = tags[word]
        else:
            tag = assign_tag(word, reversed_tags)
        total = 0
        for _, weight in reversed_tags[tag]:
            total += weight
        choice = random.randint(0, total)

        sub = ' '
        total = 0
        for gram, weight in reversed_tags[tag]:
            total += weight
            if choice <= total:
                sub = gram
                break

        new_phrase.append(sub)

    return new_phrase


def bi_swap(words, tags, reversed_tags, reversed_bi_tags, bigrams):
    new_phrase = []
    start = words[0]
    if start in tags:
        start_tag = tags[start]
    else:
        start_tag = assign_tag(start, reversed_tags)
    end = words[1]
    if end in tags:
        end_tag = tags[end]
    else:
        end_tag = assign_tag(end, reversed_tags)

    total = 0
    if (start_tag, end_tag) in reversed_bi_tags:
        for _, weight in reversed_bi_tags[(start_tag, end_tag)]:
            total += weight
        choice = random.randint(0, total)
        sub = ('', '')
        total = 0
        for gram, weight in reversed_bi_tags[(start_tag, end_tag)]:
            total += weight
            if choice <= total:
                sub = gram
                break
        new_phrase.append(sub[0])
        new_phrase.append(sub[1])

    else:
        for _, weight in reversed_tags[start_tag]:
            total += weight
        choice = random.randint(0, total)
        sub = ' '
        total = 0
        for gram, weight in reversed_tags[start_tag]:
            total += weight
            if choice <= total:
                sub = gram
                break
        new_phrase.append(sub)
        new_phrase.append("|")
        for _, weight in reversed_tags[end_tag]:
            total += weight
        choice = random.randint(0, total)
        sub = ' '
        total = 0
        for gram, weight in reversed_tags[end_tag]:
            total += weight
            if choice <= total:
                sub = gram
                break
        new_phrase.append(sub)

    start = end
    start_tag = end_tag

    for word in words[2:]:
        end = word
        if end in tags:
            end_tag = tags[end]
        else:
            end_tag = assign_tag(end, reversed_tags)

        total = 0
        sub = ' '
        if start in bigrams:
            for extension, weight in bigrams[start]:
                if extension in tags and tags[extension] == end_tag:
                    total += weight

            if total > 0:
                choice = random.randint(0, total)

            total = 0
            for extension, weight in bigrams[start]:
                if extension in tags and tags[extension] == end_tag:
                    total += weight
                    if choice <= total:
                        sub = extension
                        break

        if total == 0:
            new_phrase.append("|")
            for _, weight in reversed_tags[end_tag]:
                total += weight
            choice = random.randint(0, total)
            total = 0
            for gram, weight in reversed_tags[end_tag]:
                total += weight
                if choice <= total:
                    sub = gram
                    break

        new_phrase.append(sub)
        start = end
        start_tag = end_tag

    return new_phrase


def operate():
    tags = load_tags()
    unigrams = load_unigrams()
    bigrams = load_2grams()
    reversed_tags = reverse_tags(tags, unigrams)
    reversed_bi_tags = reverse_bi_tags(tags, bigrams)
    phrases = load_words()
    for phrase in phrases:
        print()
        print(phrase)
        print(uni_swap(phrase, tags, reversed_tags))
        print(bi_swap(phrase, tags, reversed_tags, reversed_bi_tags, bigrams))


operate()
