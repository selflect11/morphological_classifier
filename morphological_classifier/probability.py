#-*- coding: utf-8 -*-
import math
import random
import warnings
import array
from operator import itemgetter
from collections import defaultdict, Counter
from functools import reduce
from abc import ABCMeta, abstractmethod

_NINF = float('-inf')


class ProbDistI:
    SUM_TO_ONE = True

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prob(self, sample):
        """
        :type sample: any
        :rtype: float
        """
        pass

    def logprob(self, sample):
        """
        Return the base 2 logarithm of the probability for a given sample.

        :type sample: any
        :rtype: float
        """
        p = self.prob(sample)
        return (math.log(p, 2) if p != 0 else _NINF)

    @abstractmethod
    def max(self):
        """
        Return the sample with the greatest probability.  If two or
        more samples have the same probability, return one of them;
        which sample is returned is undefined.

        :rtype: any
        """
        pass

    @abstractmethod
    def samples(self):
        """
        Return a list of all samples that have nonzero probabilities.
        Use prob to find the probability of each sample.
        :rtype: list
        """
        pass

    # Subclasses should define more efficient implementations of this,
    # whenever possible.
    def generate(self):
        """
        Return a randomly selected sample from this probability distribution.
        The probability of returning each sample samp is equal to
        self.prob(samp).
        """
        p = random.random()
        p_init = p
        for sample in self.samples():
            p -= self.prob(sample)
            if p <= 0: return sample
        # allow for some rounding error:
        if p < .0001:
            return sample
        # we *should* never get here
        if self.SUM_TO_ONE:
            warnings.warn("Probability distribution %r sums to %r; generate()"
                          " is returning an arbitrary sample." % (self, p_init-p))
        return random.choice(list(self.samples()))


class FreqDist(Counter):
    """
    A frequency distribution records the number of times each outcome of
    an experiment has occurred.  For example, a frequency distribution
    could be used to record the frequency of each word type in a
    document.  Formally, a frequency distribution can be defined as a
    function mapping from each sample to the number of times that
    sample occurred as an outcome.
    """

    def __init__(self, samples=None):
        """
        If samples is given, then the frequency distribution will be initialized with the count of each object in samples;

        :type samples: Sequence
        """
        Counter.__init__(self, samples)

        # Cached number of samples in this FreqDist
        self._N = None

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this FreqDist.  For the number of unique
        sample values (or bins) with counts greater than zero, use
        FreqDist.B().

        :rtype: int
        """
        if self._N is None:
            self._N = sum(self.values())
        return self._N

    def __setitem__(self, key, val):
        """
        Override Counter.__setitem__() to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).__setitem__(key, val)

    def __delitem__(self, key):
        """
        Override Counter.__delitem__() to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).__delitem__(key)

    def update(self, *args, **kwargs):
        """
        Override Counter.update() to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).update(*args, **kwargs)

    def setdefault(self, key, val):
        """
        Override Counter.setdefault() to invalidate the cached N
        """
        self._N = None
        super(FreqDist, self).setdefault(key, val)

    def B(self):
        """
        Return the total number of sample values (or "bins") that
        have counts greater than zero.  For the total
        number of sample outcomes recorded, use FreqDist.N().
        (FreqDist.B() is the same as len(FreqDist).)
        :rtype: int
        """
        return len(self)

    def hapaxes(self):
        """
        Return a list of all samples that occur once (hapax legomena)

        :rtype: list
        """
        return [item for item in self if self[item] == 1]

    def Nr(self, r, bins=None):
        return self.r_Nr(bins)[r]

    def r_Nr(self, bins=None):
        """
        Return the dictionary mapping r to Nr, the number of samples with frequency r, where Nr > 0.

        :type bins: int
        :param bins: The number of possible sample outcomes.  bins
            is used to calculate Nr(0).  In particular, Nr(0) is
            bins-self.B().  If bins is not specified, it
            defaults to self.B() (so Nr(0) will be 0).
        :rtype: int
        """
        _r_Nr = defaultdict(int)
        for count in self.values():
            _r_Nr[count] += 1

        # Special case for Nr[0]:
        _r_Nr[0] = bins - self.B() if bins is not None else 0

        return _r_Nr

    def _cumulative_frequencies(self, samples):
        """
        If no samples are specified, all counts are returned, starting
        with the largest.

        :type samples: any
        :rtype: list(float)
        """
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    # slightly odd nomenclature freq() if FreqDist does counts and ProbDist does probs,
    # here, freq() does probs
    def freq(self, sample):
        """
        :type sample: any
        :rtype: float
        """
        n = self.N()
        if n == 0:
            return 0
        return self[sample] / n

    def max(self):
        """
        Return the sample with the greatest number of outcomes in this
        frequency distribution.  If two or more samples have the same
        number of outcomes, return one of them; which sample is
        returned is undefined.  If no outcomes have occurred in this
        frequency distribution, return None.

        :rtype: any or None
        """
        if len(self) == 0:
            raise ValueError('A FreqDist must have at least one sample before max is defined.')
        return self.most_common(1)[0][0]

    # Mathematical operations
    def __add__(self, other):
        return self.__class__(super(FreqDist, self).__add__(other))

    def __sub__(self, other):
        return self.__class__(super(FreqDist, self).__sub__(other))

    def __or__(self, other):
        """
        Union is the maximum of value in either of the input counters.
        >>> FreqDist('abbb') | FreqDist('bcc')
        FreqDist({'b': 3, 'c': 2, 'a': 1})
        """
        return self.__class__(super(FreqDist, self).__or__(other))

    def __and__(self, other):
        """
        Intersection is the minimum of corresponding counts.
        >>> FreqDist('abbb') & FreqDist('bcc')
        FreqDist({'b': 1})
        """
        return self.__class__(super(FreqDist, self).__and__(other))

    def __le__(self, other):
        if not isinstance(other, FreqDist):
            raise_unorderable_types("<=", self, other)
        return set(self).issubset(other) and all(self[key] <= other[key] for key in self)

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    __ge__ = lambda self, other: not self <= other or self == other
    __lt__ = lambda self, other: self <= other and not self == other
    __gt__ = lambda self, other: not self <= other

    def __repr__(self):
        maxlen = 10
        items = ['{0!r}: {1!r}'.format(*item) for item in self.most_common(maxlen)]
        if len(self) > maxlen:
            items.append('...')
        return 'FreqDist({{{0}}})'.format(', '.join(items))


class ConditionalFreqDist(defaultdict):
    """
    A collection of frequency distributions for a single experiment
    run under different conditions.  Conditional frequency
    distributions are used to record the number of times each sample
    occurred, given the condition under which the experiment was run.
    For example, a conditional frequency distribution could be used to
    record the frequency of each word (type) in a document, given its
    length.  Formally, a conditional frequency distribution can be
    defined as a function that maps from each condition to the
    FreqDist for the experiment under that condition.
    Conditional frequency distributions are typically constructed by
    repeatedly running an experiment under a variety of conditions,
    and incrementing the sample outcome counts for the appropriate
    conditions.  For example, the following code will produce a
    conditional frequency distribution that encodes how often each
    word type occurs, given the length of that word type:
        >>> from nltk.probability import ConditionalFreqDist
        >>> from nltk.tokenize import word_tokenize
        >>> sent = "the the the dog dog some other words that we do not care about"
        >>> cfdist = ConditionalFreqDist()
        >>> for word in word_tokenize(sent):
        ...     condition = len(word)
        ...     cfdist[condition][word] += 1

    An equivalent way to do this is with the initializer:
        >>> cfdist = ConditionalFreqDist((len(word), word) for word in word_tokenize(sent))

    The frequency distribution for each condition is accessed using
    the indexing operator:
        >>> cfdist[3]
        FreqDist({'the': 3, 'dog': 2, 'not': 1})
        >>> cfdist[3].freq('the')
        0.5
        >>> cfdist[3]['dog']
        2

    When the indexing operator is used to access the frequency
    distribution for a condition that has not been accessed before,
    ConditionalFreqDist creates a new empty FreqDist for that
    condition.
    """
    def __init__(self, cond_samples=None):
        """
        Initializes the count for every sample, under every condition,
        as zero.

        :type cond_samples: Sequence of (condition, sample) tuples
        """
        defaultdict.__init__(self, FreqDist)

        if cond_samples:
            for (cond, sample) in cond_samples:
                self[cond][sample] += 1

    # useful?
    def __reduce__(self):
        kv_pairs = ((cond, self[cond]) for cond in self.conditions())
        return (self.__class__, (), None, None, kv_pairs)

    def conditions(self):
        return list(self.keys())

    def N(self):
        """
        Return the total number of sample outcomes that have been
        recorded by this ConditionalFreqDist.

        :rtype: int
        """
        return sum(fdist.N() for fdist in self.values())



class DictionaryProbDist(ProbDistI):
    """
    A probability distribution whose probabilities are directly
    specified by a given dictionary.  The given dictionary maps
    samples to probabilities.
    """
    def __init__(self, prob_dict=None, log=False, normalize=False):
        """
        Construct a new probability distribution from the given
        dictionary, which maps values to probabilities (or to log
        probabilities, if log is true).  If normalize is
        true, then the probability values are scaled by a constant
        factor such that they sum to 1.
        If called without arguments, the resulting probability
        distribution assigns zero probability to all values.
        """

        self._prob_dict = (prob_dict.copy() if prob_dict is not None else {})
        self._log = log

        # Normalize the distribution, if requested.
        if normalize:
            if len(prob_dict) == 0:
                raise ValueError('A DictionaryProbDist must have at least one sample ' +
                             'before it can be normalized.')
            if log:
                value_sum = sum_logs(list(self._prob_dict.values()))
                if value_sum <= _NINF:
                    logp = math.log(1.0/len(prob_dict), 2)
                    for x in prob_dict:
                        self._prob_dict[x] = logp
                else:
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] -= value_sum
            else:
                value_sum = sum(self._prob_dict.values())
                if value_sum == 0:
                    p = 1.0/len(prob_dict)
                    for x in prob_dict:
                        self._prob_dict[x] = p
                else:
                    norm_factor = 1.0/value_sum
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] *= norm_factor

    def prob(self, sample):
        if self._log:
            return (2**(self._prob_dict[sample]) if sample in self._prob_dict else 0)
        else:
            return self._prob_dict.get(sample, 0)

    def logprob(self, sample):
        if self._log:
            return self._prob_dict.get(sample, _NINF)
        else:
            if sample not in self._prob_dict: return _NINF
            elif self._prob_dict[sample] == 0: return _NINF
            else: return math.log(self._prob_dict[sample], 2)

    def max(self):
        if not hasattr(self, '_max'):
            self._max = max((p,v) for (v,p) in self._prob_dict.items())[1]
        return self._max

    def samples(self):
        return self._prob_dict.keys()


class LidstoneProbDist(ProbDistI):
    """
    The Lidstone estimate for the probability distribution of the
    experiment used to generate a frequency distribution.  The
    "Lidstone estimate" is parameterized by a real number *gamma*,
    which typically ranges from 0 to 1.  The Lidstone estimate
    approximates the probability of a sample with count *c* from an
    experiment with *N* outcomes and *B* bins as
    c+gamma)/(N+B*gamma).  This is equivalent to adding
    *gamma* to the count for each bin, and taking the maximum
    likelihood estimate of the resulting frequency distribution.
    """
    SUM_TO_ONE = False
    def __init__(self, freqdist, gamma=0.5, bins=None):
        """
        :type freqdist: FreqDist
        :type gamma: float
        :param gamma: A real number used to parameterize the
            estimate.  The Lidstone estimate is equivalent to adding
            *gamma* to the count for each bin, and taking the
            maximum likelihood estimate of the resulting frequency
            distribution.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            bins is not specified, it defaults to freqdist.B().
        """
        if (bins == 0) or (bins is None and freqdist.N() == 0):
            name = self.__class__.__name__
            raise ValueError('A %s probability distribution ' % name +
                             'must have at least one bin.')
        if (bins is not None) and (bins < freqdist.B()):
            name = self.__class__.__name__
            raise ValueError('\nThe number of bins in a %s distribution ' % name +
                             '(%d) must be greater than or equal to\n' % bins +
                             'the number of bins in the FreqDist used ' +
                             'to create it (%d).' % freqdist.B())
        self._freqdist = freqdist
        self._gamma = float(gamma)
        self._N = self._freqdist.N()

        if bins is None:
            bins = freqdist.B()
        self._bins = bins

        self._divisor = self._N + bins * gamma
        if self._divisor == 0.0:
            # In extreme cases we force the probability to be 0,
            # which it will be, since the count will be 0:
            self._gamma = 0
            self._divisor = 1

    def freqdist(self):
        return self._freqdist

    def prob(self, sample):
        c = self._freqdist[sample]
        return (c + self._gamma) / self._divisor

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def discount(self):
        gb = self._gamma * self._bins
        return gb / (self._N + gb)


##//////////////////////////////////////////////////////
## Adding in log-space.
##//////////////////////////////////////////////////////

# If the difference is bigger than this, then just take the bigger one:
_ADD_LOGS_MAX_DIFF = math.log(1e-30, 2)

def add_logs(logx, logy):
    """
    Given two numbers logx = *log(x)* and logy = *log(y)*, return
    *log(x+y)*.  Conceptually, this is the same as returning
    log(2**(logx)+2**(logy)), but the actual implementation
    avoids overflow errors that could result from direct computation.
    """
    if (logx < logy + _ADD_LOGS_MAX_DIFF):
        return logy
    if (logy < logx + _ADD_LOGS_MAX_DIFF):
        return logx
    base = min(logx, logy)
    return base + math.log(2**(logx-base) + 2**(logy-base), 2)

def sum_logs(logs):
    return (reduce(add_logs, logs[1:], logs[0]) if len(logs) != 0 else _NINF)
