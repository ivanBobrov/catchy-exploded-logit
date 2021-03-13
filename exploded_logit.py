import torch
import math


# noinspection PyMethodOverriding
class ExplodedLogitTransformation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores, order):

        # The last dimension of the scores tensor is the size of the resulting
        # matrix. We shouldn't touch other dimensions to support batching. The
        # matrix (last two dimensions) should be square
        size = scores.shape[-1]

        # Transforming tensor into the matrix to support multi-dimensions,
        # adding extra dimension as last dimension and repeating all values for
        # that new dimension. Finally returning to the initial shape with new
        # dimension.
        output = scores \
            .view(-1, size) \
            .unsqueeze(-1) \
            .repeat(1, 1, size) \
            .view(scores.shape + (size,))

        # Building inverse mask to disable values that do not participate
        # in further rounds of exploded logit. The values to be discarded
        # are set to -math.inf. The mask is saved to the context to be applied
        # in the backward pass.
        ctx.inverse_mask = ExplodedLogitTransformation.inverse_mask(scores.shape, order, fill_value=-math.inf)

        # Finally summing up repeated values with the mask
        return output + ctx.inverse_mask

    @staticmethod
    def backward(ctx, grad_outputs):

        # Obtaining direct mask from inverse. Useful values are marked with 1
        # and values that correspond to other rounds of the exploded logit
        # model are NaNs. The mask has the same shape as the gradient.
        mask = ctx.inverse_mask * 0 + 1

        # Masking gradient outputs with element-wise product. Values that
        # do not participate in calculation are marked as Nans, all other
        # values are multiplied by 1
        grad_inputs = grad_outputs * mask

        # Summing up gradients by the added dimension and ignoring NaNs.
        # The order vector doesn't need the gradient, thus we provide None
        # as the result
        return grad_inputs.nansum(dim=-1), None

    @staticmethod
    def inverse_mask(shape, order, fill_value=1.):

        # Fold size is a cumulative size of all other dimensions.
        # This is required to support batching.
        # Mask size is the shape for the last two dimensions
        # of the outputs
        fold_size = math.prod(shape)
        mask_size = shape[-1]

        # Making tensor of the same shape as the scores tensor but flatten
        # with only two last dimensions. We need one extra value in the last
        # dimension to be able to mask the last value.
        mask = torch.zeros((fold_size, mask_size + 1), dtype=torch.float64, requires_grad=False)

        # Filling the values for the border value. All values in a row before
        # this order value considered as masked.
        mask[torch.arange(fold_size), order.flatten()] = fill_value

        # Cumulative sum over the rows of the last dimension to fill the values
        # after the order index to the specified value. We no longer need
        # the last column
        mask = mask.cumsum(dim=-1)[:, :-1]

        # Restoring the shape to match the input of the function
        return mask.view(shape + (mask_size,))


class ExplodedLogit(torch.nn.Module):
    def __init__(self, tracks_number, features_number):
        super(ExplodedLogit, self).__init__()
        self.tracks_number = tracks_number
        self.linear = torch.nn.Linear(features_number, 1)

    def forward(self, x):
        """
        Input to the module is 3-d tensor of [tracks, features].
        The output is 1-d tensor of rankings

        :param x: 3d tensor of shape (number_of_tracks, features_count)
        :return: 1d tensor of final rankings
        """

        # We map each track features to a single-valued score
        # The result is of shape (number_of_tracks, 1) nad viewed as 1d tensor
        scores = self.linear(x)

        # Now we need to do a softmax on a whole set
        result = scores
        for _ in range(self.tracks_number):
            idx = torch.argmax(scores)
            slice = self.delete_row(scores, idx)
            result = torch.cat((result, slice), dim=1)

        return result

    @staticmethod
    def delete_row(tensor, index):
        size = tensor.shape[0]
        mask = torch.ones(size)
        mask[index] = torch.tensor(1e-46).log()
        # matrix = torch.eye(size)
        # mask = torch.cat((matrix[:index], matrix[index+1:]))
        return torch.mm(tensor, mask.unsqueeze(0))


if __name__ == "__main__":
    ExplodedLogitTransformation.inverse_mask(
        (2, 2, 3, 4),
        torch.tensor([[[[1, 2, 3, 4], [2, 1, 3, 4], [4, 2, 3, 1]],
                       [[1, 2, 3, 4], [2, 1, 3, 4], [4, 2, 3, 1]]],
                      [[[1, 2, 3, 4], [2, 1, 3, 4], [4, 2, 3, 1]],
                       [[1, 2, 3, 4], [2, 1, 3, 4], [4, 2, 3, 1]]]])
    )
