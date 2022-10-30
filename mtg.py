import random


def create_n_grams(corpus, n):
    """Create a dictionary of n-grams and their frequencies."""
    corpus_dict = {}
    # padding the words with n-1 spaces
    words = [" "] * (n - 1) + corpus + [" "] * (n - 1)
    for idx in range(len(words) - n + 1):
        key = tuple()
        for i in range(n):
            index = idx + i
            key += (words[index],)
        if key in corpus_dict:
            corpus_dict[key] += 1
        else:
            corpus_dict[key] = 1
    return corpus_dict


def get_max_freq(n_grams, word_set, prefix, predicted_words):
    """Find the maximum frequency of a word that follows a prefix."""
    for word in word_set:
        combo = prefix + [word]
        pred_key = tuple(combo)
        word_count = n_grams.get(pred_key, 0)
        predicted_words[word] = word_count
    return max(predicted_words.values())


def find_predicted_word(n_grams, prefix, corpus, n):
    """Find the most likely word(s) to follow a sentence prefix."""
    predicted_words = {}
    word_set = set(corpus)
    # find the word with the highest probability. In the case of a tie, return all words
    max_value = get_max_freq(n_grams, word_set, prefix, predicted_words)
    if max_value == 0:
        # perform backoff
        while max_value == 0:
            if n == 0:
                # return unknown word if backoff is not possible
                return "UNK"
            predicted_words = {}
            n -= 1
            n_grams = create_n_grams(corpus, n)
            prefix = prefix[1:]
            max_value = get_max_freq(n_grams, word_set, prefix, predicted_words)
    # give all words with the highest probability
    max_keys = [k for k, v in predicted_words.items() if v == max_value]
    if len(max_keys) == 1:
        return max_keys[0]
    return max_keys


def finish_sentence(sentence, n, corpus, deterministic=False):
    """Finish a sentence using n-grams and Markov assumption."""
    corpus_dict = create_n_grams(corpus, n)

    SEN_LEN = 10

    while (
        not ("." in sentence or "?" in sentence or "!" in sentence)
        and len(sentence) < SEN_LEN
    ):
        prefix = sentence[-n + 1 :]
        if deterministic:
            predicted_word = find_predicted_word(corpus_dict, prefix, corpus, n)
            # in case of ties, choose the word that appears first in the corpus
            if isinstance(predicted_word, list):
                {word: corpus.index(word) for word in predicted_word}
                predicted_word = min(predicted_word, key=corpus.index)
        else:
            # do a random choice based on the probability distribution
            population = [
                key for key in corpus_dict.keys() if list(key)[: n - 1] == prefix
            ]
            while len(population) == 0:
                if n == 0:
                    # return unknown word if backoff is not possible
                    sentence.append("UNK")
                    return sentence
                n -= 1
                corpus_dict = create_n_grams(corpus, n)
                prefix = prefix[1:]
                population = [
                    key for key in corpus_dict.keys() if list(key)[: n - 1] == prefix
                ]
            total_num = sum([corpus_dict[key] for key in population])
            weights = [corpus_dict[key] / total_num for key in population]
            predicted_word = random.choices(population, weights)[0][-1]
        sentence.append(predicted_word)
    return sentence


if __name__ == "__main__":
    finish_sentence(
        ["I", "am"],
        2,
        ["I", "am", "just", "a", "training", "set", "."],
        deterministic=True,
    )
