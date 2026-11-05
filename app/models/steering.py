from contextlib import contextmanager


@contextmanager
def steer(model, layer, vector, alpha, scale):
    """
    Add alpha * scale * vector / ||vector|| to the residual stream at hidden_states[layer] for every token,
    while inside the `with` block. The hook is always removed on exit.
    "fixed" mode: scale = act_norm; "std" mode: scale = std of the projections on the emotion direction.

    Args:
        model: the model returned by load_model
        layer: hidden state index, as in the activations and vectors files (1..n_layers-1)
        vector: Tensor of shape [d], steering direction
        alpha: steering strength, as a multiple of scale
        scale: unit of the strength (act_norm or std of the emotion)
    """

    blocks = model.model.layers
    if not 1 <= layer < len(blocks):
        raise ValueError(f"layer must be in 1..{len(blocks) - 1}, got {layer}")

    # Compute the delta to add to the residual stream (the same for every token)
    delta = alpha * scale * vector / vector.norm()

    def hook(module, inputs, output):
        """
        Add delta to the residual stream (the first element of the output if it is a tuple).
        """

        if isinstance(output, tuple):
            h = output[0]
            return (h + delta.to(h.device, h.dtype),) + output[1:]
        return output + delta.to(output.device, output.dtype)

    # Register the hook on the specified layer (hidden_states[layer] is the output of blocks[layer - 1])
    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
