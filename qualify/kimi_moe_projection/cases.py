"""Compare the prepared recipe with serial execution of the same projections."""

from ignition.recipes.kimi3.pointwise import rms_norm_columns


class MoEProjectionSchedule:
    def __init__(self, operation, source, reduction, mode):
        self.operation = operation
        self.source = source
        self.reduction = reduction
        self.mode = mode

    def __call__(self):
        op, source = self.operation, self.source
        if self.mode == "recipe":
            self.reduction(op(source))
            return
        op.router_projection(source)
        op.down(source)
        op.shared.start(source)
        op.route_gather()
        op.expert_reduce(op.experts())
        rms_norm_columns(op.latent_sum, op.latent_norm, op.up_input, op.eps,
                         begin=op.latent_begin)
        op.up(op.up_input)
        op.output.add_(op.shared.finish(source, op.output))
        self.reduction(op.output)
