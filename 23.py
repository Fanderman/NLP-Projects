import random

k = 10


def a(file='3grams'):

    dictionary = {}
    i = 0

    with open(file, 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            if i % 100000 == 0:
                print(i)
            line = line.strip()
            line = line.split(' ')
            if int(line[0]) >= k:
                if line[1] not in dictionary:
                    dictionary[line[1]] = []
                dictionary[line[1]].append((line[2:], int(line[0])))
            else:
                break
            i += 1

    rng = random.Random()
    start_word = rng.choice(list(dictionary.items()))
    print(start_word)

    phrase = ''
    phrase += str(start_word[0]) + ' '
    word = start_word[0]

    while word in dictionary:
        gram = dictionary[word]
        #print(gram)
        total = 0
        for par in gram:
            total += par[1]
        choice = rng.randint(0, total)

        cur = 0
        for par in gram:
            if choice <= cur + par[1]:
                choice = par[0]
                break
            cur += par[1]

        for words in choice:
            phrase += str(words) + ' '

        word = choice[-1]

    print(phrase)


a(file='3grams')
