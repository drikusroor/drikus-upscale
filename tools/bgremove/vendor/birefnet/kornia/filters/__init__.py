"""Stand-in: BiRefNet imports kornia's laplacian for its training-only
gradient supervision branch, which is never reached in eval mode."""


def laplacian(*args, **kwargs):
    raise NotImplementedError('training-only code path')
