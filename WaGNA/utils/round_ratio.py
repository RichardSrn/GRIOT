import torch
import numpy as np


def round_ratio(to_round: torch.Tensor, matrix_to_match: torch.Tensor):
    """
    Round the values of vector_to_round, such that the ratio of 0/1 is the same as in matrix_to_match.
    To do so, we first need to find the ratio of 0/1 in matrix_to_match, and that to find the correct threshold
    to round vector_to_round such that is has the same ratio.
    :param to_round: matrix or vector of real values mainly between 0 and 1.
    :param matrix_to_match: matrix of binary features
    :return: matrix_rounded or vector_rounded
    """
    return to_round
    # return torch.round(to_round)

    # if len(to_round.size()) == 1:
    #     vector_to_round = to_round.clone()
    #     matrix_to_match = matrix_to_match.clone()
    #
    #     # remove the first column of matrix_to_match
    #     matrix_to_match = matrix_to_match[:, 1:]
    #
    #     # check that there are only 0, 1 and torch.nan in matrix_to_match
    #     assert torch.sum(matrix_to_match == 0) + \
    #            torch.sum(matrix_to_match == 1) + \
    #            torch.sum(torch.isnan(matrix_to_match)) \
    #            == matrix_to_match.numel()
    #
    #     # find the ratio of 0/1 in matrix_to_match (careful to nan)
    #     ratio = torch.sum(matrix_to_match == 0) / (matrix_to_match.numel() - torch.sum(torch.isnan(matrix_to_match)))
    #
    #     # find the threshold to round vector_to_round
    #     threshold = np.quantile(vector_to_round.detach().numpy(), ratio)
    #
    #     # round vector_to_round
    #     vector = torch.where(vector_to_round > threshold,
    #                          torch.ones_like(vector_to_round),
    #                          torch.zeros_like(vector_to_round))
    #
    #     return vector
    #
    # elif len(to_round.size()) == 2:
    #     matrix_to_round = to_round.clone()
    #     matrix_to_match = matrix_to_match.clone()
    #
    #     # keep the first column of matrix_to_round as labels
    #     labels = matrix_to_round[:, 0].clone()
    #
    #     # remove the first column of matrix_to_round
    #     matrix_to_round = matrix_to_round[:, 1:]
    #     # remove the first column of matrix_to_match
    #     matrix_to_match = matrix_to_match[:, 1:]
    #
    #     # check that there are only 0, 1 and torch.nan in matrix_to_match
    #     assert torch.sum(matrix_to_match == 0) + \
    #            torch.sum(matrix_to_match == 1) + \
    #            torch.sum(torch.isnan(matrix_to_match)) \
    #            == matrix_to_match.numel()
    #
    #     # find the ratio of 0/1 in matrix_to_match (careful to nan)
    #     ratio = torch.sum(matrix_to_match == 0) / (matrix_to_match.numel() - torch.sum(torch.isnan(matrix_to_match)))
    #
    #     # find the threshold to round vector_to_round
    #     threshold = np.quantile(matrix_to_round.detach().numpy(), ratio)
    #
    #     # round vector_to_round
    #     matrix_rounded = torch.where(matrix_to_round > threshold,
    #                                  torch.ones_like(matrix_to_round),
    #                                  torch.zeros_like(matrix_to_round))
    #
    #     # add the labels back to the matrix
    #     matrix_rounded = torch.cat((labels.unsqueeze(1), matrix_rounded), dim=1)
    #
    #     return matrix_rounded
