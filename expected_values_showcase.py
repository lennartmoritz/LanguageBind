import numpy as np
from functools import lru_cache
import math
# This file is not used in practice.
# It was created to showcase the expeceted value calculations for my thesis
# Therefore it just contains more comments and is made more compact.


def calc_expected_r_at_k(k, N, M):
    """
    Calculates R@k for a random baseline retrieval system.
    Parameters:
    k:  Rank
    N:  Result set size
    M:  List with number of matches per retrieval query in the experiment
    """
    # expected_r_at_k =  1 - P(no matches in first k results)
    i_values = np.arange(0, k)
    products = np.array([np.prod((N - m - i_values) / (N - i_values)) for m in M])
    expected_r_at_k = 1 - np.average(products)
    return expected_r_at_k


def calc_expected_mrr(N, M):
    """
    Calculates the expected value of Mean Reciprocal Rank (MRR) for a random baseline retrieval system
    Parameters:
    N: Result set size
    M:  List with number of matches per retrieval query in the experiment
    """
    # M[i] contains the number of relevant results in the i-th query, among N total results
    # k is the rank of the first relevant retrieval.
    # Let P(first relevant at k) be the chance to pick the first relevant item at the k-th position
    # Then P(first relevant at k) is:
    #   (Possibilities to pick k-1 irrelevant items) *
    #   (Possibilites to pick 1 relevant item) / (Possibilities to pick any k items)
    # ERR[i] = Sum from k=1 to k=(N - M[i] + 1): 1/k * P(first relevant at k)

    relevant_counts = {
        unique_match_amount: sum(entry == unique_match_amount for entry in M)
        for unique_match_amount in set(M)
    }
    assert len(M) == sum(relevant_counts.values())

    # Initialize Mean Reciprocal Rank (MRR)
    MRR = 0.0
    for rel_count in relevant_counts:
        # Initialize Expected Reciprocal Rank (ERR)
        ERR = 0.0
        # Loop over the potential ranks for the first relevant item
        for k in range(1, N - rel_count + 1 + 1):  # maximum k = (N - rel_count + 1)
            # Calculate the logarithm of the required products
            log_p_irrelevant = np.sum(np.log(N - rel_count - np.arange(k - 1)))
            # log of (N - M[0]) * (N - M[0] - 1) * ... * (N - M[0] - (k - 2))
            log_p_relevant = np.log(rel_count)
            log_p_total = np.sum(np.log(N - np.arange(k)))  # log(N * (N - 1) * ... * (N - (k - 1)))
            # Using log properties to calculate the sum
            probability = np.exp(log_p_irrelevant + log_p_relevant - log_p_total)
            ERR += (1 / k) * probability
        MRR += relevant_counts[rel_count] * ERR
    MRR /= sum(relevant_counts.values())
    return MRR


def calc_expected_mAP(N, M):
    """
    Calculates the expected value of Mean Reciprocal Rank (MRR) for a random baseline retrieval system
    Parameters:
    N: Result set size
    M:  List with number of matches per retrieval query in the experiment
    """

    def expected_average_precision(N, m):
        """Calculate the expected Average Precision for a random retrieval system.
        Parameters:
        m: Relevant matches for this retrieval query
        """
        @lru_cache(maxsize=None)
        def recursive_precision(start, N, m, depth=1):
            """Recursive function to calculate the sum of precisions.
            Parameters:
            start: First valid position of relevant item
            depth: Recursion depth
            """
            if depth > m:
                return 0.0, 1
            sum_precisions = 0.0
            runs = 0
            end = N - m + depth + 1
            for i in range(start, end):
                # Calculate precision at this position
                precision = depth / i
                # Continue recursively
                sub_sum, found_runs = recursive_precision(i + 1, N, m, depth + 1)
                # Add the precision for this rank as many times as it was used in a combination
                sum_precisions += found_runs * precision + sub_sum
                runs += found_runs
            return sum_precisions, runs

        if m == 0 or N == 0 or m > N:
            return 0.0
        sum_precisions, found_runs = recursive_precision(1, N, m)
        # Calculate the number combinations of selecting m relevant items out of N
        num_combinations = math.comb(N, m)
        # Divide the sum of precisions by the number of combinations and by m
        return sum_precisions / (num_combinations * m)

    relevant_counts = {
        unique_match_amount: sum(entry == unique_match_amount for entry in M)
        for unique_match_amount in set(M)
    }
    # Initialize mean Average Precision (mAP)
    mAP = 0.0
    for rel_count in relevant_counts:
        AP = expected_average_precision(N, rel_count)
        mAP += relevant_counts[rel_count] * AP
    mAP /= sum(relevant_counts.values())
    return mAP
