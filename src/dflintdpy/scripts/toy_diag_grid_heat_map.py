from dflintdpy.models.dgrid import DGrid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
import numpy as np

def main():
    m, n = 5, 5
    costs = np.random.rand(m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1))
    dgrid = DGrid(m, n, cost=costs)
    opt_model = ShortestPathGrb(graph=dgrid)
    shortest_path, objective = opt_model.solve()
    opt_model.visualize(colored_edges=shortest_path)

    dgrid.scale_diagonal_edges(scale=np.sqrt(2))
    opt_model_scaled = ShortestPathGrb(graph=dgrid)
    shortest_path_scaled, objective_scaled = opt_model_scaled.solve()
    opt_model_scaled.visualize(colored_edges=shortest_path_scaled)
    
    pass





if __name__ == "__main__":
    main()