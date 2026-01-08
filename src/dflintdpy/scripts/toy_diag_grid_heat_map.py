from dflintdpy.models.dgrid import DGrid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
import numpy as np


def single_shortest_path_example():
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

def shortest_path_heat_map():
    m, n = 5, 5
    n_costs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
    n_trials = 100
    costs = np.random.rand(n_trials, n_costs)
    arc_count = np.zeros(n_costs)
    for i in range(n_trials):
        dgrid = DGrid(m, n, cost=costs[i])
        # dgrid.scale_diagonal_edges(scale=np.sqrt(2))
        opt_model = ShortestPathGrb(graph=dgrid)
        shortest_path, objective = opt_model.solve()
        arc_count += shortest_path
    arc_count /= n_trials
    # Visualize heat map
    opt_model.setObj(arc_count)
    opt_model.visualize(heat_map=arc_count, title="Shortest Paths Heat Map")
    pass

# TODO: What happenes to the heat map if interdiction is applied?
def intd_shortest_path_heat_map():
    m, n = 5, 5
    n_costs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
    n_trials = 100
    costs = np.random.rand(n_trials, n_costs)
    intd_costs = np.random.rand(n_trials, n_costs)
    arc_count = np.zeros(n_costs)
    for i in range(n_trials):
        dgrid = DGrid(m, n, cost=costs[i])
        opt_model = ShortestPathGrb(graph=dgrid)
        intder = SymmetricInterdictor(
            dgrid,
            k=15,
            interdiction_cost=intd_costs[i],
        )
        intd, shortest_path, _ = intder.solve(versatile=False)
        # shortest_path, objective = opt_model.solve()
        arc_count += shortest_path
    arc_count /= n_trials
    # Visualize heat map
    opt_model.setObj(arc_count)
    opt_model.visualize(heat_map=arc_count, 
                        title="Interdicted Shortest Paths Heat Map")
    pass

def compare_shortest_path_heat_maps():
    shortest_path_heat_map()
    intd_shortest_path_heat_map()
    pass

# TODO: Learn the edges with DFL and show what edges are more over or underestimated

def main():
    # single_shortest_path_example()
    # shortest_path_heat_map()
    compare_shortest_path_heat_maps()
    pass

if __name__ == "__main__":
    main()