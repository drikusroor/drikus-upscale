"""Stand-in: BiRefNet inherits PyTorchModelHubMixin for Hub uploads only."""


class PyTorchModelHubMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
