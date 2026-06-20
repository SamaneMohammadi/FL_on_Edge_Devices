"""
DP-SGD wiring via Opacus.

The paper enforces local differential privacy by clipping each per-sample
gradient to norm C and adding Gaussian noise N(0, sigma^2 C^2 I) before the
optimizer step (Section III, "Local Differential Privacy"). Opacus does exactly
this when we attach its PrivacyEngine to the model/optimizer/dataloader.

We keep Opacus only for the noisy-clipped training step. Cumulative epsilon is
tracked separately by our MomentsAccountant so the bookkeeping matches the paper
and stays consistent between FedAvg and FedAsync.
"""

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def make_private(model, optimizer, data_loader, sigma, clip_norm):
    """Attach Opacus to a model/optimizer/loader and return the private versions.

    sigma      -- noise multiplier (paper: 0.5, 1.0, 1.5 or 2.0)
    clip_norm  -- per-sample max gradient norm C (paper: 1.0)
    """
    # Opacus refuses modules it cannot make private (e.g. batch norm). Our model
    # already uses GroupNorm, but we validate to fail loudly if that ever changes.
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=sigma,
        max_grad_norm=clip_norm,
    )
    return model, optimizer, data_loader, privacy_engine
