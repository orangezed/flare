"""Preserve image inputs and feature outputs across encoder graph replay."""


class Features:
    compilation_variant = "replicated"

    def __init__(self, encoder, output):
        import torch
        self.encoder, self.output = encoder, output
        self.pixels = torch.empty_like(encoder.scratch.patches)

    def __call__(self):
        # The encoder reuses patch scratch for QKV and MLP work. Native image
        # publication likewise reloads pixels at the start of every graph.
        self.encoder.scratch.patches.copy_(self.pixels)
        self.output.copy_(self.encoder())
