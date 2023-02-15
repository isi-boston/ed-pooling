from typing import Optional, Tuple
import torch

# See pretrained_transformer_mismatched_embedder.py from AllenAI for more info


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.
    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/master/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.
    The key reason this can't be done with basic torch functions
    is that we want to be able to use look-up tensors with an arbitrary number of dimensions
    (for example, in the coref model, we don't know a-priori how many spans we are looking up).
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ValueError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_span_select(
        target: torch.Tensor,
        spans: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns segmented spans in the target with respect to the provided span indices.
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    spans : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def average_span_extractor(embeddings, offsets):

    """
    embeddings : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    offsets : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.

    Example:
        embeddings = [
                       [[1, 1, 1],
                       [2, 2, 2]],

                      [[3, 3, 3],
                       [4, 4, 4]]
                     ]  # (batch_size=2, seq_len=2, embedding_size=3)
        offsets = [
                    [[0, 0],
                    [0, 1]],

                    [[0, 1],
                    [1, 1]]
                  ]  # (batch_size=2, num_spans=2, 2)
        return: [
                  [[1, 1, 1],        # average of embeddings[0][0] and embeddings[0][0]
                  [1.5, 1.5, 1.5]],  # average of embeddings[0][0] and embeddings[0][1]

                  [[3.5, 3.5, 3.5],  # average of embeddings[1][0] and embeddings[1][1]
                  [4, 4, 4]]         # average of embeddings[1][1] and embeddings[1][1]
                ]  # (batch_size=2, num_spans=2, embedding_size=3)
    """

    span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), offsets)
    span_mask = span_mask.unsqueeze(-1)
    span_embeddings *= span_mask  # zero out paddings

    span_embeddings_sum = span_embeddings.sum(2)
    span_embeddings_len = span_mask.sum(2)
    # Shape: (batch_size, num_orig_tokens, embedding_size)
    orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

    # All the places where the span length is zero, write in zeros.
    orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

    return orig_embeddings


def weighted_span_extractor(embeddings, offsets, weights):
    span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), offsets)
    span_mask = span_mask.unsqueeze(-1)
    span_embeddings *= span_mask  # zero out paddings
    weights = weights.unsqueeze(-1)
    span_embeddings = span_embeddings * weights
    span_embeddings = span_embeddings.sum(2)
    return span_embeddings


def attention_pooled_span_extractor(embeddings, offsets, attn, temperature):

    span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), offsets)
    span_mask = span_mask.unsqueeze(-1)

    weights = attn(span_embeddings)
    weights = weights / temperature
    weights = torch.where(span_mask, weights, torch.tensor(-float('inf')).to(weights.device))
    weights = torch.softmax(weights, dim=-2)

    span_embeddings = span_embeddings * weights
    span_embeddings = span_embeddings.sum(2)

    return span_embeddings


def last_token_span_extractor(embeddings, offsets):

    offsets = offsets.tolist()
    offsets = [[[o[-1], o[-1]] for o in current_offsets] for current_offsets in offsets]
    offsets = torch.LongTensor(offsets).to(embeddings.device)
    span_embeddings, _ = batched_span_select(embeddings.contiguous(), offsets)
    return span_embeddings.squeeze()


def first_token_span_extractor(embeddings, offsets):

    offsets = offsets.tolist()
    offsets = [[[o[0], o[0]] for o in current_offsets] for current_offsets in offsets]
    offsets = torch.LongTensor(offsets).to(embeddings.device)
    span_embeddings, _ = batched_span_select(embeddings.contiguous(), offsets)
    return span_embeddings.squeeze()
