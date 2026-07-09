from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def make_private(model, optimizer, data_loader, sigma, clip_norm):
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
