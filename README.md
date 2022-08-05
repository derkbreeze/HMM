# HMM

This is my personal implementation of hidden markov model, a generative model in statistical machine learning. HMM is trained in an unsupervised way, the code implements the forward-backward algorithm to calculate the marginal probabilities of the states at any time step given the partial/full observations, the Baum-Welch algorithm is utilized to estimate the initial, transition and emission probability distributions.
For this example, the observations were spaces and letters but the code is generic enough that it could work with any sequence of observations and hidden states. All of the probabilities discussed below will be in log-space.

The inference of HMM is implemented as a Viterbi algorithm(Dynamic Programming)

## Dependencies
Python 3.x

## `load_corpus(path)` ##

This function is specific to this example and is used to read in the sequence of observations. It will read the file at the path, clean it up by only keeping only letters and single spaces, and also convert everything to lowercase. It will return the string of the cleaned up input.

## `load_probabilities(path)` ##

This function loads the pickle file at the specified path, which contains a tuple of dictionaries. The first dictionary contains the initial state probabilities and maps the integer *i* to the probability for the *i* th state. The second dictionary contains the transition probabilities and maps the integer *i* to a second dictionary which maps the integer *j* to the probability of transitioning from state *i* to state *j*. Finally, the third dictionary contains the emission probabilities and maps the integer *i* to a dictionary which maps the observation to the probability of seeing the observation in state *i*.

## `class HMM` ##
Intialize the `HMM` class with the tuple of dictionaries returned by `load_probabilities`.

    p = load_probabilities("prob_vector.pickle")
    h = HMM(p)

### Re-estimating HMM parameters
The function `update(sequence, cutoff_value)` repeatedly re-estimates the HMM parameters on the input `sequence` until the increase in log probability is less than the `cutoff_value`.

Thus, the overall process for re-estimating a HMM based on an observation sequence `s` is:

    p = load_probabilities("prob_vector.pickle")
    h = HMM(p)
    h.update(s, epsilon)

### Probability of a Sequence of Observations ###

The `forward` function runs the forward algorithm and returns a list of dictionaries, and the `forward_probability` function takes this output and returns the actual probability. For some sequence s:

    h.forward_probability(h.forward(s))

You can also use the backward algorithm to get the same result:

    h.backward_probability(h.backward(s), s)
