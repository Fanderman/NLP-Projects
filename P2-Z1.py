import itertools
from functools import reduce
import re
import random
import numpy
import glob
from pprint import pprint


def calculate_unknown(dictionary, test):
    unknowns = 0
    for word in test:
        if word not in dictionary:
            unknowns += 1

    return unknowns/len(test)


def normalize_text(file, bases):
    RE = re.compile(r'[a-zA-ZęóąśłżźćńĘÓĄŚŁŻŹĆŃ\-]+')
    with open(file, 'r', encoding='utf8') as o_corpus:
        lines = o_corpus.readlines()
        data = []
        for line in lines:
            normalized_line = [word.lower() for word in re.findall(RE, line)]
            based_line = []
            for word in normalized_line:
                if word in bases:
                    based_line.append(bases[word])
                else:
                    based_line.append(word)
            if len(based_line) > 0:
                data.append(based_line)
    return data


def create_dictionary(text):
    dictionary = {}
    for word in text:
        if word not in dictionary:
            dictionary[word] = 1
        else:
            dictionary[word] += 1

    return dictionary


def divide_text(text, p=0.05):
    rng = random.Random()
    training = text.copy()
    test = []
    rng.shuffle(training)
    for i in range(len(text)//round(1/p)):
        word = training.pop()
        test.append(word)
    return training, test


def load_bases(file='superbazy.txt'):
    base = {}
    with open(file, 'r', encoding='utf8') as corpus:
        for line in corpus:
            key, value = line.split()
            base[key] = value

    return base


def letter_chance(text):
    letters = {}
    total = 0
    for word in text:
        for letter in word:
            if letter not in letters:
                letters[letter] = 1
            else:
                letters[letter] += 1
            total += 1

    for letter in letters:
        letters[letter] = numpy.log(letters[letter]/total)

    return letters


def words_chance(text):
    words = {}
    total = 0
    for word in text:
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
        total += 1

    for word in words:
        words[word] = numpy.log(words[word] / total)

    return words


def twograms_chance(text):
    grams = {}
    total = 0
    for line in text:
        start = line[0]
        for word in line[1:]:
            if (start, word) not in grams:
                grams[(start, word)] = 1
            else:
                grams[(start, word)] += 1
            total += 1
            start = word

    for gram in grams:
        grams[gram] = numpy.log(grams[gram] / total)

    return grams


def threegrams_chance(text):
    grams = {}
    total = 0
    for line in text:
        if len(line) > 2:
            start = line[0]
            middle = line[1]
            for word in line[2:]:
                if (start, middle, word) not in grams:
                    grams[(start, middle, word)] = 1
                else:
                    grams[(start, middle, word)] += 1
                total += 1
                start = middle
                middle = word

    for gram in grams:
        grams[gram] = numpy.log(grams[gram] / total)

    return grams


def naive_bayes(text, letters, words, twograms, threegrams):
    letter_probabilities = {}
    for author in letters:
        letter_probabilities[author] = 0

    word_probabilities = {}
    for author in words:
        word_probabilities[author] = 0

    twogram_probabilities = {}
    for author in twograms:
        twogram_probabilities[author] = 0

    threegram_probabilities = {}
    for author in threegrams:
        threegram_probabilities[author] = 0

    for line in text:
        if len(line) > 2:
            start = line[0]
            middle = line[1]
            for word in line[2:]:
                for author in threegrams:
                    if (start, middle, word) not in threegrams[author]:
                        threegram_probabilities[author] += numpy.log(1/160000)
                    else:
                        threegram_probabilities[author] += threegrams[author][(start, middle, word)]
                start = middle
                middle = word

        if len(line) > 1:
            start = line[0]
            for word in line[1:]:
                for author in twograms:
                    if (start, word) not in twograms[author]:
                        twogram_probabilities[author] += numpy.log(1/180000)
                    else:
                        twogram_probabilities[author] += twograms[author][(start, word)]
                start = word

        for word in line:
            for author in words:
                if word not in words[author]:
                    word_probabilities[author] += numpy.log(1/200000)
                else:
                    word_probabilities[author] += words[author][word]

            for letter in word:
                for author in letters:
                    if letter not in letters[author]:
                        letter_probabilities[author] += numpy.log(1/1000000)
                    else:
                        letter_probabilities[author] += letters[author][letter]


    for author in letters:
        letter_probabilities[author] = -1/letter_probabilities[author]
        word_probabilities[author] = -1/word_probabilities[author]
        twogram_probabilities[author] = -1/twogram_probabilities[author]
        threegram_probabilities[author] = -1/threegram_probabilities[author]

    t_letter = sum(letter_probabilities.values())
    t_word = sum(word_probabilities.values())
    t_twogram = sum(twogram_probabilities.values())
    t_threegram = sum(threegram_probabilities.values())

    t_total = t_letter + t_word + t_twogram + t_threegram

    for author in letters:
        letter_probabilities[author] = letter_probabilities[author] * (1/t_letter)
        word_probabilities[author] = word_probabilities[author] * (1/t_word)
        twogram_probabilities[author] = twogram_probabilities[author] * (1/t_twogram)
        threegram_probabilities[author] = threegram_probabilities[author] * (1/t_threegram)

    t_t = {}
    for author in letters:
        t_t[author] = (t_letter/t_total) * letter_probabilities[author] + (t_word/t_total) * word_probabilities[author] \
                      + (t_twogram/t_total) * twogram_probabilities[author] + (t_threegram/t_total) * threegram_probabilities[author]

    return t_t


def operate(file):
    bases = load_bases()
    text = normalize_text(file, bases)
    training, test = divide_text(reduce(list.__add__, text))
    dictionary = create_dictionary(training)
    unknown = calculate_unknown(dictionary, test)
    dictionary = create_dictionary(reduce(list.__add__, text))
    bayes_letters = letter_chance(reduce(list.__add__, text))
    bayes_words = words_chance(reduce(list.__add__, text))
    bayes_2grams = twograms_chance(text)
    bayes_3grams = threegrams_chance(text)
    #print(bayes_letters)
    #print(sorted(bayes_3grams.items(), key=lambda t: t[1], reverse=True)[:5])
    #print(max(bayes_3grams.items(), key=lambda t: t[1]))
    #print(bayes_2grams)
    #print(bayes_3grams)
    #print(unknown)
    return bayes_letters, bayes_words, bayes_2grams, bayes_3grams


def prepare():
    orzeszkowa = operate('korpus_orzeszkowej.txt')
    prus = operate('korpus_prusa.txt')
    sienkiewicz = operate('korpus_sienkiewicza.txt')

    letters = {'orzeszkowa': orzeszkowa[0], 'prus': prus[0], 'sienkiewicz': sienkiewicz[0]}
    words = {'orzeszkowa': orzeszkowa[1], 'prus': prus[1], 'sienkiewicz': sienkiewicz[1]}
    twograms = {'orzeszkowa': orzeszkowa[2], 'prus': prus[2], 'sienkiewicz': sienkiewicz[2]}
    threegrams = {'orzeszkowa': orzeszkowa[3], 'prus': prus[3], 'sienkiewicz': sienkiewicz[3]}

    return letters, words, twograms, threegrams


def test(bases, letters, words, twograms, threegrams, file='testy1/test_orzeszkowej.txt'):
    text = normalize_text(file, bases)
    return naive_bayes(text, letters, words, twograms, threegrams)


letters, words, twograms, threegrams = prepare()
bases = load_bases()

correct = 0
total = 0
for path in glob.glob('testy1/*'):
    filename = path.split('/')[-1]
    output = test(bases, letters, words, twograms, threegrams, filename)
    print('{} -> {}'.format(filename, max(output.items(), key=lambda t: t[1])[0]))
    if filename[12] == max(output.items(), key=lambda t: t[1])[0][0]:
        correct += 1
    total += 1
    pprint(output)

print(correct/total)
