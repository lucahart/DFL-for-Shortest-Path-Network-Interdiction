from matplotlib.axes import Axes
from numpy import ndarray
from pyparsing import Optional
from .SPOTrainer import SPOTrainer
import matplotlib.pyplot as plt


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
    
    @staticmethod
    def vis_learning_curve(trainer: "SPOTrainer",
                        train_loss_log: ndarray[float],
                        train_regret_log: ndarray[float],
                        test_loss_log: ndarray[float] = None,
                        test_regret_log: ndarray[float] = None
                        ) -> None:
        
        fig, ax = plt.subplots(1, 3, figsize=(16,4))

        # Plot training regret and loss
        SPOTrainer.vis_learning_curve(trainer,
                                       train_loss_log,
                                       train_regret_log,
                                       test_loss_log,
                                       test_regret_log,
                                       ax=ax[0:2])
        
        # Plot cost convergence
        # alpha = 

