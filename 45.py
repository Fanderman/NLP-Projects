import itertools

k = 5


def a(words):

    dictionary = {}
    i = 0
    with open('2grams', 'r', encoding='utf8') as base_vectors_lines:
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

    perms = []
    for perm in itertools.permutations(words):
        total_value = 0
        falses = 0
        for i in range(len(perm)-1):
            word = str(perm[i]).strip()

            if word not in dictionary:
                falses += 1

            else:
                gram = dictionary[word]
                total = 0
                found = False
                for par in gram:
                    total += par[1]
                    if str(par[0][0]).strip() == str(perm[i+1]).strip():
                        found = par[1]

                if not found:
                    word1 = word
                    word2 = str(perm[i+1]).strip()

                    for d1 in range(0, -5, -1):
                        for d2 in range(0, -5, -1):
                            if len(word1)+d1 > 1 and len(word2)+d2 > 1:
                                if d1 == 0:
                                    new_base = word1
                                else:
                                    new_base = word1[:d1]
                                if d2 == 0:
                                    new_con = word2
                                else:
                                    new_con = word2[:d2]
                                if new_base in dictionary:
                                    gram = dictionary[new_base]
                                    for par in gram:
                                        if str(par[0][0]).strip() == new_con:
                                            found = True

                    if not found:
                        falses += 1
                    else:
                        total_value += 0.000000001

                else:
                    total_value += found/total

        perms.append((perm, (total_value, falses)))

    global size
    size = len(words)

    def conv(a):
        global size
        return size-a[1][1]+a[1][0]

    perms.sort(reverse=True, key=conv)
    for i in range(5):
        print(perms[i])

#words = ['wyjątkowo', 'ciekawy', 'okaz', 'sztuki', 'kulinarnej']
#words = ['zjadłem', 'pyszną', 'kanapkę', 'z', 'szynką', 'i', 'serem']
#words = ['to', 'był', 'mój', 'najlepszy', 'mecz']
#words = ['judyta', 'dała', 'wczoraj', 'stefanowi', 'czekoladki']
words = ['babuleńka', 'miała', 'dwa', 'rogate', 'koziołki']
#words = ['wczoraj', 'wieczorem', 'spotkałem', 'pewną', 'piękną', 'kobietę']

a(words)

