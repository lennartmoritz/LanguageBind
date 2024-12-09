import numpy as np
from sklearn.metrics import average_precision_score
from easydict import EasyDict
from functools import lru_cache
import math
from tqdm.auto import tqdm


def calculate_map(cosine_sim_matrix, ground_truth):
    """
    Calculate mean average precision (MAP) for retrieval.

    :param cosine_sim_matrix: NxM matrix of cosine similarities.
    :param ground_truth: List of lists with ground truth target indices for each query.
    :return: MAP score.
    """
    n_queries = cosine_sim_matrix.shape[0]
    y_true = np.zeros_like(cosine_sim_matrix)

    for i, targets in enumerate(ground_truth):
        for target in targets:
            y_true[i, target] = 1

    average_precisions = []
    for i in range(n_queries):
        average_precisions.append(average_precision_score(y_true[i], cosine_sim_matrix[i]))

    return np.mean(average_precisions)


def calculate_top_k_accuracy(cosine_sim_matrix, ground_truth, k):
    """
    Calculate top-K accuracy for retrieval.

    :param cosine_sim_matrix: NxM matrix of cosine similarities.
    :param ground_truth: List of lists with ground truth target indices for each query.
    :param k: Number of top elements to consider for accuracy calculation.
    :return: Top-K accuracy score.
    """
    n_queries = cosine_sim_matrix.shape[0]
    correct_count = 0

    for i, targets in enumerate(ground_truth):
        top_k_indices = np.argsort(cosine_sim_matrix[i])[::-1][:k]
        if any(target in top_k_indices for target in targets):
            correct_count += 1

    return 100 * correct_count / n_queries


def calculate_mean_median_rank(cosine_sim_matrix, ground_truth):
    """
    Calculate mean rank (MnR) and median rank (MdR) of correct results.

    :param cosine_sim_matrix: NxM matrix of cosine similarities.
    :param ground_truth: List of lists with ground truth target indices for each query.
    :return: mean rank and median rank.
    """
    ranks = []
    reciprocal_ranks = []

    for i, targets in enumerate(ground_truth):
        # Sort indices of similarities in descending order
        sorted_indices = np.argsort(cosine_sim_matrix[i])[::-1]
        ranks_for_current_query = []
        for target in targets:
            rank = np.where(sorted_indices == target)[0][0] + 1  # rank is 1-based
            ranks.append(rank)
            ranks_for_current_query.append(rank)
        reciprocal_rank = 1.0 / min(ranks_for_current_query)
        reciprocal_ranks.append(reciprocal_rank)

    mean_rank = np.mean(ranks) if ranks else float('nan')
    median_rank = np.median(ranks) if ranks else float('nan')
    mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else float('nan')

    return mean_rank, median_rank, mean_reciprocal_rank


def compute_metrics(cosine_sim_matrix, ground_truth):
    """
    Compute retrieval metrics including R@1, R@5, R@10, mAP, Mean Rank (MnR), and Median Rank (MdR).

    :param cosine_sim_matrix: NxM matrix of cosine similarities.
    :param ground_truth: List of lists with ground truth target indices for each query.
    :return: Dictionary with R1, R5, R10, mAP, MeanR, and MedianR scores.
    """
    metrics = {}

    # Calculate mAP
    metrics['mAP'] = calculate_map(cosine_sim_matrix, ground_truth)

    # Calculate top-K accuracies
    for k in [1, 5, 10]:
        metrics[f'R{k}'] = calculate_top_k_accuracy(cosine_sim_matrix, ground_truth, k)

    # Calculate Mean and Median Rank
    mean_rank, median_rank, mrr = calculate_mean_median_rank(cosine_sim_matrix, ground_truth)
    metrics['MeanR'] = mean_rank
    metrics['MedianR'] = median_rank
    metrics['MR'] = median_rank
    metrics['MRR'] = mrr

    return metrics


def calc_rand_retrieval_chance(s2c_true_matches, c2s_true_matches):
    """
    Return the metrics expected values for random retrieval success of sentence2chunk,
    chunk2sentence, chunk2chunk (aligned) and sentence2sentence (aligned, ideal) settings.

    Args:
    s2c_true_matches:          List of Lists with match ids for sentence to chunk retrieval
    c2s_true_matches:          List of Lists with match ids for chunk to sentence retrieval
    """
    chances = {}
    n_chunks = len(c2s_true_matches)
    n_sents = len(s2c_true_matches)

    chances["s2c_chance"] = get_chance_summary(result_set_size=n_chunks, true_matches_array=s2c_true_matches)
    chances["c2s_chance"] = get_chance_summary(result_set_size=n_sents, true_matches_array=c2s_true_matches)
    chances["c2c_chance"] = get_chance_summary(result_set_size=n_chunks, true_matches_array=[[1]] * n_chunks)
    chances["s2s_chance"] = get_chance_summary(result_set_size=n_sents, true_matches_array=[[1]] * n_sents)

    chances = EasyDict(chances)
    return chances


def get_chance_summary(result_set_size, true_matches_array):
    """
    Parameters:
    result_set_size:        Result set size
    true_matches_array:     List of Lists with match ids for query to result retrieval
    """
    matches_per_query = [len(match_ids) for match_ids in true_matches_array]
    chance_summary = {}

    chance_summary["mAP"] = calc_expected_mAP(N=result_set_size, M=matches_per_query)

    for k in [1, 5, 10]:
        chance_summary[f"R{k}"] = calc_expected_r_at_k(k=k, N=result_set_size, M=matches_per_query)

    chance_summary["MeanR"] = calc_expected_mean_rank(N=result_set_size)
    chance_summary["MedianR"] = chance_summary["MeanR"]  # same as MeanR
    chance_summary["MRR"] = calc_expected_mrr(N=result_set_size, M=matches_per_query)
    chance_summary["hits_per_query"] = sum(matches_per_query) / len(matches_per_query)
    return chance_summary


def calc_expected_r_at_k(k, N, M):
    """
    Parameters:
    k:  Rank
    N:  Result set size
    M:  Array of matches per Query
    """
    i_values = np.arange(0, k)
    products = np.array([np.prod((N - m - i_values) / (N - i_values)) for m in M])
    expected_r_at_k = 1 - np.average(products)
    return expected_r_at_k


def calc_expected_mrr(N, M):
    """
    Calculates the expected value of Mean Reciprocal Rank (MRR) for a random baseline retrieval system
    Parameters:
    N: Result set size
    M: Array of relevant matches per Query
    """
    # M[i] contains the number of relevant results in the i-th query, among N total results
    # k is the rank of the first relevant retrieval.
    # Let P(first relevant at k) be the chance to pick the first relevant item at the k-th position
    # Then P(first relevant at k) is: (Possibilities to pick k-1 irrelevant items) * (Possibilites to pick 1 relevant item) / (Possibilities to pick any k items) # noqa 
    # ERR = Sum from k=1 to k=(N - M[0] + 1): 1/k * P(first relevant at k)

    relevant_counts = {unique_match_amount: sum(entry == unique_match_amount for entry in M) for unique_match_amount in set(M)} # noqa 
    assert len(M) == sum(relevant_counts.values())

    # Initialize Mean Reciprocal Rank (MRR)
    MRR = 0.0
    for rel_count in tqdm(relevant_counts, leave=False, desc="MRR calculation"):

        # Initialize Expected Reciprocal Rank (ERR)
        ERR = 0.0

        # Loop over the potential ranks for the first relevant item
        for k in range(1, N - rel_count + 1 + 1):  # maximum k = (n - rel_count + 1)

            # Calculate the logarithm of the required products
            log_p_irrelevant = np.sum(np.log(N - rel_count - np.arange(k - 1)))  # log of (N - M[0]) * (N - M[0] - 1) * ... * (N - M[0] - (k - 2)) # noqa 
            log_p_relevant = np.log(rel_count)  # log(M[0])
            log_p_total = np.sum(np.log(N - np.arange(k)))  # log(N * (N - 1) * ... * (N - (k - 1)))

            # Using log properties to calculate the sum
            probability = np.exp(log_p_irrelevant + log_p_relevant - log_p_total)

            ERR += (1 / k) * probability

        MRR += relevant_counts[rel_count] * ERR
    MRR /= sum(relevant_counts.values())
    return MRR


def calculate_combinations(N, m):
    """Calculate the number combinations of selecting m relevant items out of N using numpy."""
    if m > N:
        return 0
    combinations = math.comb(N, m)
    return combinations


@lru_cache(maxsize=None)
def recursive_precision(start, N, m, depth=1):
    """Recursive function to calculate the sum of precisions."""
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
        # print(f"Pos: {i}, hits: {depth}, Precision (x{found_runs}):", precision)

    return sum_precisions, runs


def expected_average_precision(N, m):
    """Calculate the expected Average Precision for a random retrieval system."""
    if m == 0 or N == 0 or m > N:
        return 0.0

    sum_precisions, found_runs = recursive_precision(1, N, m)
    num_combinations = calculate_combinations(N, m)
    assert num_combinations == found_runs, f"Mismatch num_combinations ({num_combinations}) != found_runs({found_runs}), difference ({num_combinations - found_runs})"
    # print("Combinations", num_combinations)

    # Divide the sum of precisions by the number of combinations and by m
    return sum_precisions / (num_combinations * m)


def calc_expected_mAP(N, M):
    """
    Calculates the expected value of Mean Reciprocal Rank (MRR) for a random baseline retrieval system
    Parameters:
    N: Result set size
    M: Array of relevant matches per Query
    """
    relevant_counts = {unique_match_amount: sum(entry == unique_match_amount for entry in M) for unique_match_amount in set(M)} # noqa
    assert len(M) == sum(relevant_counts.values())

    # Initialize mean Average Precision (mAP)
    mAP = 0.0
    for rel_count in tqdm(relevant_counts, leave=False, desc="mAP calculation"):

        # Initialize Expected Reciprocal Rank (ERR)
        AP = expected_average_precision(N, rel_count)
        assert AP <= 1, f"Invalid AP detected: ({AP}), N={N}, rel_count={rel_count}"  # remove

        mAP += relevant_counts[rel_count] * AP
    mAP /= sum(relevant_counts.values())
    assert mAP <= 1, f"Invalid mAP detected: ({mAP})"  # remove
    return mAP


def calc_expected_mean_rank(N):
    """
    Calculate the expected mean rank of relevant results for a list of queries,
    assuming a random ordering. The result is always the average value of all available
    ranks in the result set. This is true regardless of the number of relevant items
    due to the symmetry of probabilities when averaging over all possible cases. This
    result is therefore also the expected median rank.
    Parameters:
    N: Total number of items in the result set.
    """
    expected_mean = (N + 1) / 2
    return expected_mean


if __name__ == "__main__":
    # Example usage
    cosine_sim_matrix = np.random.rand(5, 10)  # Replace with actual cosine similarity matrix
    ground_truth = [[1, 3], [2, 5], [0], [4, 7], [6]]  # Replace with actual ground truth
    cosine_sim_matrix = np.array([
        [0.9, 0.4, 0.3],
        [0.9, 0.4, 0.3],
        [0.9, 0.4, 0.3]])
    ground_truth = [[1, 2], [2], [0]]
    ground_truth = np.arange(cosine_sim_matrix.shape[0]).reshape(-1, 1)
    # rankings: 2,3,3,1
    print(cosine_sim_matrix[0, :4])

    # Calculate MAP
    map_score = calculate_map(cosine_sim_matrix, ground_truth)
    print(f'MAP Score: {map_score}')
    mnr, mdr, mrr = calculate_mean_median_rank(cosine_sim_matrix, ground_truth)
    print(f"Mean: {round(mnr, 2)} \t Median: {mdr} \t MRR: {round(mrr, 3)}")

    # Calculate Top-k Accuracy
    top_k_accuracy = {k: calculate_top_k_accuracy(cosine_sim_matrix, ground_truth, k) for k in [1, 5, 10]}
    for k, acc in top_k_accuracy.items():
        print(f'Top-{k} Accuracy: {acc}')
