# How to build Word2vec from scratch (Skip-gram with negative sampling)
This project implements [Word2Vec](https://arxiv.org/pdf/1301.3781) from scratch using only NumPy. Here I will try to explain as simply as possible how Word2Vec works.

For a long time, NLP researchers were trying to find ways to represent words as numbers. A simple approach was creating one-hot encodings of words in the dictionary, but the issue with that was the one-hot encodings did not capture the relationship between words. By relationship, I mean words that appear in the same context. For example, "king" and "queen" often appear in similar contexts, but "banana" and "pencil" do not.

Word2Vec solves this by creating embeddings of words where the embedding is meaningful and captures the semantic relationship between them. Imagine a 2D space where the words "school", "class", and "teacher" appear close together, but far away from words like "engine", "transmission", and "brakes", which refer to car parts. These groupings are what make the embeddings useful.

In this specific implementation, we let the model read the entire corpus and create pairs of words that appear close to each other in the corpus, given a window. So pairs are formed from a center word and all the surrounding words, either before or after the center word. This allows the model to guess the context words given a center word, and the fancy way of saying this is Skip-Gram.

We also utilize a concept called negative sampling, which means that every time we see a real (center, context) pair, we also sample k words that did not appear near the center word, and we try to push those further away in the embedding space. This discourages the model from placing unrelated words close together. However, we can’t just pick these negative words randomly from a uniform distribution. That’s because very frequent words like "the" or "is" would show up as negatives too often, which doesn't make sense. These words already appear everywhere, so they shouldn’t be treated like rare or niche words.

To fix that, we create what's called a unigram distribution. We count how many times each word appears in the corpus and then raise each count to the power of 0.75 to make the distribution smoother. Why 0.75? Because if the word "the" appears 100 times and "cpu" appears 35 times, their difference is 65. But if we raise both to the power of 0.75, their difference shrinks to about 17.2. This makes the frequent words a little less dominant and gives rare words more of a chance to be chosen as negatives.

When we train the model, we use a loss function, often called J, which tells us how badly the model is doing. For each (center, context) pair, we calculate the dot product between the center word vector and the context word vector. We pass that through a sigmoid to get a probability that these two words belong together. We do the same for each of the k negative samples. For those, we want the model to say that they do not belong together, so we take the negative of the dot product before applying sigmoid.

The total loss is the negative log of the probability that the true pair is correct, plus the log of the probability that all the negative samples are incorrect. The model then updates the embeddings using stochastic gradient descent to make the real pairs more likely and the negative pairs less likely. Over time, this pulls similar words closer together in the embedding space and pushes unrelated words apart.

By the end of training, you can do cool things like word analogies. For example:

"king" is to "man" as "woman" is to "queen"

This works because the model has learned consistent vector offsets that represent relationships like gender or royalty.

This whole project was implemented without any machine learning libraries or frameworks. It’s just raw NumPy and math, and it was a lot of fun to build.

## High-level Overview 

