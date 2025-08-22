from .SPOTrainer import SPOTrainer


class HybridSPOTrainer(SPOTrainer):
    """Trainer that supports hybrid SPO+ loss with anchoring."""

    @staticmethod
    def VALID_METHODS():
        """Extend valid methods with 'hybrid'."""
        return ["hybrid"] + SPOTrainer.VALID_METHODS()

    @staticmethod
    def compute_loss(loss_criterion,
                    costs_pred,
                    costs,
                    sols,
                    objs,
                    method_name):
        """Compute loss including hybrid option."""
        if method_name == "hybrid":
            return loss_criterion(costs_pred, costs, sols, objs)
        return SPOTrainer.compute_loss(loss_criterion,
                                       costs_pred,
                                       costs,
                                       sols,
                                       objs,
                                       method_name)
