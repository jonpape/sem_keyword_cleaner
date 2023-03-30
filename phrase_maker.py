import nltk
from nltk.corpus import wordnet as wn

# Define a function to generate related phrases for a given word
def generate_phrases(word):
    # Get the synsets (conceptual synonyms) for the word
    synsets = wn.synsets(word)

    # Get the lemmas (words with the same meaning) for each synset
    lemmas = []
    for synset in synsets:
        lemmas.extend(synset.lemmas())

    # Get the names (strings) of each lemma
    names = [lemma.name().replace('_', ' ') for lemma in lemmas]

    # Generate phrases using the names and the original word
    phrases = []
    for name in names:
        tokens = nltk.word_tokenize(name)
        if len(tokens) >= 2 and len(tokens) <= 7:
            phrases.append(name)
        for i in range(len(tokens)):
            if i == 0:
                continue
            phrase = word + ' ' + ' '.join(tokens[:i])
            if len(tokens[:i]) >= 2 and len(tokens[:i]) <= 7:
                phrases.append(phrase)
    return phrases

# Define a list of seed words related to student loans
words = ['student', 'loan', 'debt', 'interest', 'repayment', 'default']

# Generate a list of phrases for each seed word
phrases = []
for word in words:
    phrases.extend(generate_phrases(word))

# Select a random sample of 200 phrases
import random
random.seed(42)  # Set the random seed for reproducibility
selected_phrases = random.sample(phrases, 20)

# Print the selected phrases
print(selected_phrases)
