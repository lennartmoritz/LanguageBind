import numpy as np
from sklearn.metrics import average_precision_score


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
