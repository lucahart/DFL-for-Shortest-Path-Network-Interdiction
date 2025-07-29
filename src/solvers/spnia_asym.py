# spnia_asym.py
import networkx as nx
import gurobipy as gp

from gurobipy import GRB
from typing import List
from numpy import ndarray
from copy import deepcopy

from src.models.ShortestPath import ShortestPath
from src.models.ShortestPathGrb import shortestPathGrb


class AsymmetricSPNI:
    """
    Asymmetric Shortest Path Network Interdiction (SPNI) model.
    This class implements the asymmetric SPNI model, which is a two-stage stochastic programming model
    for the shortest path interdiction problem with asymmetric costs and delays.

    Attributes:
    ----------
    """

    graph: 'ShortestPath' # The directed graph representing the network
    G: nx.DiGraph  # The directed graph representing the network
    budget: int  # The budget for the max-min knapsack problem
    true_costs: dict  # True costs of the edges in the graph
    true_delays: dict  # True delays of the edges in the graph
    est_costs: dict  # Estimated costs of the edges in the graph
    est_delays: dict  # Estimated delays of the edges in the graph
    theta: float # The maximum of the estimated delays

    def __init__(self, 
                 graph : 'ShortestPath', 
                 budget : int, 
                 true_costs: List[float] | ndarray[float], 
                 true_delays, 
                 est_costs, 
                 est_delays, 
                 lsd=10e-3):
        """
        Initialize the Asymmetric SPNI model.

        Parameters:
        -----------
        graph : ShortestPath
            The directed graph representing the network.
        budget : int
            The budget for interdictions.
        true_costs : List[float] | ndarray[float]
            True costs of the edges in the graph.
        true_delays : List[float] | ndarray[float]
            True delays of the edges in the graph.
        est_costs : List[float] | ndarray[float]
            Estimated costs of the edges in the graph.
        est_delays : List[float] | ndarray[float]
            Estimated delays of the edges in the graph.
        lsd : float, optional
            Least Significant Digit of the costs and interdictions.
            Choose a value that is small enough to not affect the solution, 
            but large enough to avoid numerical instability. Default is 10e-3.
        """

        # Store the graph and budget as given
        self.graph = deepcopy(graph)
        self.G = graph.graph
        self.budget = budget

        # Store true costs and delays as dictionaries indexed by edge
        self.true_costs = {e: true_costs[i] for i,e in enumerate(self.graph.arcs)}
        self.true_delays = {e: true_delays[i] for i,e in enumerate(self.graph.arcs)}
        self.est_costs = {e: est_costs[i] for i,e in enumerate(self.graph.arcs)}
        self.est_delays = {e: est_delays[i] for i,e in enumerate(self.graph.arcs)}

        # Compute theta as the maximum of the estimated delays
        longest_path = shortestPathGrb(self.graph)
        longest_path.setObj(-(self.true_costs + self.true_delays))
        self.theta = -longest_path.solve()[1]/lsd

    def build_spnia_L(self):
        """
        Return a Gurobi model of the optimistic SPNIA-L.
        
        Parameters:
        ----------
        G : networkx.DiGraph
            The directed graph representing the network.
        budget : int
            The budget for the max-min knapsack problem.
        true_c : dict
            True costs of the edges in the graph.
        true_d : dict
            True delays of the edges in the graph.
        est_c : dict
            Estimated costs of the edges in the graph.
        est_d : dict
            Estimated delays of the edges in the graph.
        theta : float, optional
            A parameter for the pessimistic model, if applicable.

        Returns:
        -------
        m : gurobipy.Model
            The Gurobi model for the optimistic SPNIA-L.
        x : gurobipy.Var
            Decision variables for the edges in the graph.
        """

        m  = gp.Model("SPNIA_L")
        x  = m.addVars(self.graph.arcs, vtype=GRB.BINARY, name="x")
        v  = m.addVars(self.graph.arcs, lb=0.0,        name="v")
        w  = m.addVars(self.graph.arcs, lb=0.0,        name="w")
        u  = m.addVars(self.graph.vertices, lb=-GRB.INFINITY, name="u")

        # objective: true path length
        m.setObjective(gp.quicksum(self.true_costs[e]*v[e] +
                                (self.true_costs[e]+self.true_delays[e])*w[e] for e in self.graph.arcs),
                    GRB.MAXIMIZE)

        s, t = 0, max(self.graph.vertices)
        for i in self.graph.vertices:
            m.addConstr(
                gp.quicksum((v[e]+w[e]) for e in self.graph.graph.out_edges(i)) -
                gp.quicksum((v[e]+w[e]) for e in self.graph.graph.in_edges(i))
                == (1 if i == s else -1 if i == t else 0),
                name=f"flow_{i}"
            )

        for i, j in self.graph.graph.edges:
            m.addConstr(u[i] - u[j] - self.est_delays[(i, j)]*x[(i, j)] <= self.est_costs[(i, j)],
                        name=f"dual_{i}_{j}")

        m.addConstr(u[t] - u[s] +
                    gp.quicksum(self.est_costs[e]*v[e] + (self.est_costs[e]+self.est_delays[e])*w[e]
                                for e in self.G.edges()) == 0, name="dual_link")

        for e in self.G.edges:
            m.addConstr(v[e] + x[e] <= 1, name=f"link1_{e}")
            m.addConstr(w[e] - x[e] <= 0, name=f"link2_{e}")

        m.addConstr(gp.quicksum(x[e] for e in self.G.edges) <= self.budget, name="budget")
        m.Params.OutputFlag = 0
        return m, x
    




    def build_spnia_LG(self):
        """
        Gurobi model of the pessimistic SPNIA-LG (no warm-start yet).
        
        Returns
        -------
        m : gurobipy.Model
            The Gurobi model for the pessimistic SPNIA-LG.
        x : gurobipy.Var
            Decision variables for the edges in the graph.
        v : gurobipy.Var
            Auxiliary variables for the edges in the graph.
        w : gurobipy.Var
            Auxiliary variables for the edges in the graph.
        u : gurobipy.Var
            Dual variables for the vertices in the graph.
        """

        m  = gp.Model("SPNIA_LG")
        x  = m.addVars(self.G.edges(), vtype=GRB.BINARY, name="x")
        v  = m.addVars(self.G.edges(), lb=0.0,        name="v")
        w  = m.addVars(self.G.edges(), lb=0.0,        name="w")
        u  = m.addVars(self.G.nodes(), lb=-GRB.INFINITY, name="u")

        s, t = 0, max(self.G.nodes())
        m.setObjective(
            u[s] - u[t] -
            self.theta*gp.quicksum(self.est_costs[e]*v[e] + (self.est_costs[e]+self.est_delays[e])*w[e]
                            for e in self.G.edges()),
            GRB.MAXIMIZE)

        for i in self.G.nodes():
            m.addConstr(
                gp.quicksum((v[e]+w[e]) for e in self.G.out_edges(i)) -
                gp.quicksum((v[e]+w[e]) for e in self.G.in_edges(i))
                == (1 if i == s else -1 if i == t else 0))

        for i, j in self.G.edges():
            m.addConstr(
                u[i] - u[j] - (self.theta*self.est_delays[(i, j)] + self.true_delays[(i, j)])*x[(i, j)]
                <=  self.theta*self.est_costs[(i, j)] + self.true_costs[(i, j)])

        for e in self.G.edges():
            m.addConstr(v[e] + x[e] <= 1)
            m.addConstr(w[e] - x[e] <= 0)

        m.addConstr(gp.quicksum(x[e] for e in self.G.edges()) <= self.budget)
        m.Params.OutputFlag = 0
        return m, x, v, w, u
    

    def solve_spnia_LG(self):
        """
        Two-step procedure of Bayrak & Bailey (2008).

        Returns
        -------
        x_star : dict
            Optimal decision vector from the max-min knapsack problem.
        z_star : float
            Optimal objective value.
        """
        # Step 1 – optimistic
        L, xL = self.build_spnia_L()
        L.optimize()
        z_star = L.ObjVal
        x_star = {e: xL[e].X for e in self.G.edges()}

        # Step 2 – pessimistic with warm-start and cut
        LG, xLG, v, w, u = self.build_spnia_LG()
        for e in self.G.edges():
            xLG[e].Start = x_star[e]

        # bounding cut
        s, t = 0, max(self.G.nodes())
        LG.addConstr(
            u[s] - u[t] -
            self.theta*gp.quicksum(self.est_costs[e]*v[e] + (self.est_costs[e]+self.est_delays[e])*w[e]
                            for e in self.G.edges()) <= z_star, name="warm_cut")

        LG.optimize()

        return {e: xLG[e].X for e in self.G.edges()}, LG.ObjVal
