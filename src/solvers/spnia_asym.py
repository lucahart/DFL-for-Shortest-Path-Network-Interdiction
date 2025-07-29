# spnia_asym.py
import networkx as nx
import gurobipy as gp
from copy import deepcopy
from gurobipy import GRB
from src.models.ShortestPath import ShortestPath


class AsymmetricSPNI:

    graph: 'ShortestPath' # The directed graph representing the network
    G: nx.DiGraph  # The directed graph representing the network
    budget: int  # The budget for the max-min knapsack problem
    true_costs: dict  # True costs of the edges in the graph
    true_delays: dict  # True delays of the edges in the graph
    est_costs: dict  # Estimated costs of the edges in the graph
    est_delays: dict  # Estimated delays of the edges in the graph

    def __init__(self, graph, budget, true_costs, true_delays, est_costs, est_delays):
        """
        Initialize the Asymmetric SPNI model.

        Parameters:
        -----------
        - graph: The directed graph representing the network.
        - budget: The budget for the max-min knapsack problem.
        - true_costs: True costs of the edges in the graph.
        - true_delays: True delays of the edges in the graph.
        - est_costs: Estimated costs of the edges in the graph.
        - est_delays: Estimated delays of the edges in the graph.
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
    




    def build_spnia_LG(G, budget, est_c, est_d, theta):
        """Gurobi model of the pessimistic SPNIA-LG (no warm-start yet)."""
        m  = gp.Model("SPNIA_LG")
        x  = m.addVars(G.edges(), vtype=GRB.BINARY, name="x")
        v  = m.addVars(G.edges(), lb=0.0,        name="v")
        w  = m.addVars(G.edges(), lb=0.0,        name="w")
        u  = m.addVars(G.nodes(), lb=-GRB.INFINITY, name="u")

        s, t = 0, max(G.nodes)
        m.setObjective(
            u[s] - u[t] -
            theta*gp.quicksum(est_c[e]*v[e] + (est_c[e]+est_d[e])*w[e]
                            for e in G.edges()),
            GRB.MAXIMIZE)

        for i in G.nodes:
            m.addConstr(
                gp.quicksum((v[e]+w[e]) for e in G.out_edges(i)) -
                gp.quicksum((v[e]+w[e]) for e in G.in_edges(i))
                == (1 if i == s else -1 if i == t else 0))

        for i, j in G.edges:
            m.addConstr(
                u[i] - u[j] - (theta*est_d[(i, j)] + est_d[(i, j)])*x[(i, j)]
                <=  theta*est_c[(i, j)] + est_c[(i, j)])

        for e in G.edges:
            m.addConstr(v[e] + x[e] <= 1)
            m.addConstr(w[e] - x[e] <= 0)

        m.addConstr(gp.quicksum(x[e] for e in G.edges) <= budget)
        m.Params.OutputFlag = 0
        return m, x, v, w, u

    def solve_spnia_LG(G, budget, true_c, true_d, est_c, est_d, theta):
        """Exactly the two-step procedure of Bayrak & Bailey (2008)."""
        # Step 1 – optimistic
        L, xL = build_spnia_L(G, budget, true_c, true_d, est_c, est_d)
        L.optimize()
        z_star = L.ObjVal
        x_star = {e: xL[e].X for e in G.edges}

        # Step 2 – pessimistic with warm-start and cut
        LG, xLG, v, w, u = build_spnia_LG(G, budget, est_c, est_d, theta)
        for e in G.edges:
            xLG[e].Start = x_star[e]

        # bounding cut
        s, t = 0, max(G.nodes)
        LG.addConstr(
            u[s] - u[t] -
            theta*gp.quicksum(est_c[e]*v[e] + (est_c[e]+est_d[e])*w[e]
                            for e in G.edges) <= z_star, name="warm_cut")

        LG.optimize()
        return {e: xLG[e].X for e in G.edges}, LG.ObjVal
