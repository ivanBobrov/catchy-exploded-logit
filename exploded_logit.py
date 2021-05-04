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


class ExplodedLogitLoss(torch.nn.Module):

    def __init__(self, loss_type='nll', reduction='mean', top_n=None):
        super().__init__()
        self.loss_type = loss_type
        self.top_n = top_n

        if self.loss_type == 'bce':
            self.loss_function = torch.nn.BCELoss(reduction=reduction)
        elif self.loss_type == 'nll':
            self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise ValueError("Loss type '{0}' not supported".format(self.loss_type))

    def forward(self, scores, order):
        if self.loss_type == 'bce':
            matrix_of_rounds = ExplodedLogitTransformation.apply(scores, order)
            soft_maxed = torch.nn.functional.softmax(matrix_of_rounds, dim=-2)
            target = self.build_target(order)
            return self.loss_function.forward(soft_maxed, target)
        if self.loss_type == 'nll':
            matrix_of_rounds = ExplodedLogitTransformation.apply(scores, order)
            target = torch.argsort(order)

            if self.top_n is not None:
                # Cutting of lowest places in tournament
                matrix_of_rounds = matrix_of_rounds[..., :self.top_n]
                target = target[..., :self.top_n]

            if len(matrix_of_rounds.shape) == 2:
                # In case of two dimensions loss function expects first dim
                # as mini-batch, so we need to transpose
                return self.loss_function.forward(matrix_of_rounds.T, target)

            return self.loss_function.forward(matrix_of_rounds, target)
        else:
            raise ValueError("Loss type '{0}' not supported".format(self.loss_type))

    def build_target(self, order):
        fold_size = math.prod(order.shape)
        mask_size = order.shape[-1]
        target = torch.zeros((fold_size, mask_size), dtype=torch.float64, requires_grad=False)
        target[torch.arange(fold_size), order.flatten() - 1] = 1.
        return target.view(order.shape + (mask_size,))
