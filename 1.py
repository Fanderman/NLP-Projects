signs = ['.', '(', ')', ';', '"', '[', ']', ',', '?', '!', ':', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def make_dict():
    dictionary = {}
    i = 0

    with open('train_shuf.txt', 'r', encoding='utf8') as base_vectors_lines:
        for line in base_vectors_lines:
            if i % 100000 == 0:
                print(i)
            line = line.split(' ')
            for base in line:
                base = base.lower()
                for sign in signs:
                    base = base.replace(sign, ' ')
                base = base.strip()
                base = base.split(' ')
                for word in base:
                    if word not in dictionary:
                        dictionary[word] = 1
                    else:
                        dictionary[word] += 1

            i += 1

    output = open('dictionary.txt', 'w', encoding='utf8')
    for word in dictionary:
        if dictionary[word] > 6:
            output.write(word + "\n")
    output.close()


def sort_dict():
    inpt = open('dictionary.txt', 'r', encoding='utf8')
    words = []
    for word in inpt:
        words.append(word)
    words.sort()

    output = open('sorted_dictionary.txt', 'w', encoding='utf8')
    for word in words:
        output.write(word)
    output.close()


def maxmatch(phrase, dictionary):
    tokens = []
    max_length = 30

    i = 0
    length = len(phrase)

    while i < length:
        used_length = i + max_length
        if used_length > length:
            used_length = length

        found = False
        while used_length > i:
            if phrase[i:used_length] in dictionary:
                tokens.append(phrase[i:used_length])
                i = used_length
                found = True
            else:
                used_length -= 1

        if not found:
            tokens.append(phrase[i:i+1])
            i += 1

    return tokens


best_tokens = []
max_value = []


def maxsquare(phrase, dictionary):
    global best_tokens
    global max_value
    best_tokens = []
    max_value = []
    for j in range(len(phrase)+1):
        best_tokens.append([])
        max_value.append(0)

    def minsquare2(phrase, dictionary, i=0):
        global best_tokens
        global max_value

        def value(tested_tokens):
            val = 0
            for word in tested_tokens:
                val += len(word) * len(word)
            return val

        max_length = 30
        length = len(phrase)
        tokens = []

        local_best_tokens = []
        local_max_value = 0

        used_length = i + 1
        while used_length < i+max_length and used_length <= length:

            if phrase[i:used_length] in dictionary:
                tokens.append(phrase[i:used_length])

                if len(best_tokens[used_length]) == 0:
                    minsquare2(phrase, dictionary, used_length)

                new_tokens = tokens.copy()
                for token in best_tokens[used_length]:
                    new_tokens.append(token)

                if value(new_tokens) > local_max_value:
                    local_max_value = value(new_tokens)
                    local_best_tokens = new_tokens.copy()

                tokens.pop()

            used_length += 1

        best_tokens[i] = local_best_tokens
        max_value[i] = local_max_value

    minsquare2(phrase, dictionary, 0)
    return best_tokens[0]


def b(method=1, limit=10000):
    i = 0
    dictionary = {}
    dictionary_file = open('sorted_dictionary.txt', 'r', encoding='utf8')
    for line in dictionary_file:
        dictionary[line.strip()] = line.strip()

    with open('train_shuf.txt', 'r', encoding='utf8') as base_vectors_lines:

        total_correct = 0
        total_size = 0

        for line in base_vectors_lines:
            if i > limit:
                break
            print(line)

            phrase = ''
            line = line.split(' ')
            for base in line:
                base1 = base.lower()
                for sign in signs:
                    base1 = base1.replace(sign, '')
                base1 = base1.strip()
                phrase += base1

            tokens = []
            for base in line:
                base = base.lower()
                for sign in signs:
                    base = base.replace(sign, ' ')
                base = base.strip()
                base = base.split(' ')
                for word in base:
                    if len(word) > 0:
                        tokens.append(word)

            print(phrase)
            print(tokens)
            if method == 1:
                tested_tokens = maxmatch(phrase, dictionary)
            if method == 2:
                tested_tokens = maxsquare(phrase, dictionary)
            print(tested_tokens)

            correct = 0
            size = len(tokens)
            j = 0
            x = 0
            while j < size - x:
                if tokens[j] in tested_tokens:
                    tested_tokens.remove(tokens[j])
                    tokens.remove(tokens[j])
                    j -= 1
                    x += 1
                    correct += 1
                j += 1

            print(correct)
            print(size)

            total_correct += correct
            total_size += size
            i += 1

        print(str(total_correct/total_size*100) + "%")


b(method=2)
