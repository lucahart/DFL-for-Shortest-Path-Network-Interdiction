import pytest


torch = pytest.importorskip('torch')

from src.models.CalibratedPredictor import CalibratedPredictor


class Dummy(torch.nn.Module):
    def forward(self, x):
        # Return a constant vector of ones with shape [B,2]
        return torch.ones(x.size(0), 2)


def test_calibrated_predictor_scales_and_shifts():
    base = Dummy()
    model = CalibratedPredictor(base)
    target_s = 2.0
    # set parameters so that softplus(log_s)+eps = target_s
    model.log_s.data.fill_(torch.log(torch.exp(torch.tensor(target_s - model.eps)) - 1))
    model.b.data.fill_(1.0)
    x = torch.zeros(3, 5)
    out = model(x)
    expected = torch.full((3, 2), target_s * 1 + 1.0)
    assert torch.allclose(out, expected)


def test_return_all_outputs():
    base = Dummy()
    model = CalibratedPredictor(base)
    y, raw, s = model(torch.zeros(1, 4), return_all=True)
    assert raw.shape == y.shape == (1, 2)
    assert s > 0
