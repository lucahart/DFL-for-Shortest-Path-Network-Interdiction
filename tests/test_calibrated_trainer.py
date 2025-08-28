import pytest

pyepo = pytest.importorskip('pyepo')
torch = pytest.importorskip('torch')

from src.models.CalibratedPredictor import CalibratedPredictor
from src.models.CalibratedSPOTrainer import CalibratedSPOTrainer


class DummyBase(torch.nn.Module):
    def forward(self, x):
        return torch.ones(x.size(0), 2)


class DummyLoss(torch.nn.Module):
    def forward(self, pred, costs, sols, objs):
        return (pred - costs).pow(2).mean()


def test_calibrated_trainer_updates_params():
    base = DummyBase()
    model = CalibratedPredictor(base)
    opt_model = object()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = DummyLoss()
    trainer = CalibratedSPOTrainer(model, opt_model, optimizer, loss_fn)

    feats = torch.zeros(4, 3)
    costs = torch.zeros(4, 2)
    sols = torch.zeros(4, 2)
    objs = torch.zeros(4)
    dataset = torch.utils.data.TensorDataset(feats, costs, sols, objs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    before = model.b.clone()
    trainer.train_epoch(loader)
    assert not torch.equal(before, model.b)
