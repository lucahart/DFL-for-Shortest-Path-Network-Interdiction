# spnia_asym.py
import gurobipy as gp

from gurobipy import GRB
from typing import List
from numpy import ndarray
import numpy as np
from copy import deepcopy

from dflintdpy.models.graph import Graph
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb


class AsymmetricInterdictor:
    """
    Gurobi-based asymmetric shortest-path interdiction solver.

    The class models a leader-follower interdiction problem on a directed
    source-target graph. Each arc carries two versions of its data:

    - ``true_costs`` and ``true_delays`` describe the realized follower
      path length once interdictions are applied.
    - ``est_costs`` and ``est_delays`` describe the leader's estimated
      data used to build the optimistic and pessimistic relaxations.

    The implementation follows a two-stage Bayrak-Bailey-style workflow:

    1. Build and solve an optimistic model, ``SPNIA_L``, to obtain a
       candidate interdiction vector and an upper bound.
    2. Build a pessimistic model, ``SPNIA_LG``, warm-start it with the
       optimistic interdictions, add a bounding cut, and solve again.

    Attributes
    ----------
    ``graph``
        Deep-copied ``Graph`` instance owned by the solver. Its vertex
        labels, source, target, and arc order define every formulation and
        every returned interdiction vector.
    ``budget``
        Cardinality budget limiting how many arcs may be interdicted.
    ``true_costs``
        Dictionary mapping each arc to its realized base cost.
    ``true_delays``
        Dictionary mapping each arc to its realized interdiction delay.
    ``est_costs``
        Dictionary mapping each arc to its estimated base cost used during
        model construction.
    ``est_delays``
        Dictionary mapping each arc to its estimated interdiction delay used
        during model construction.
    ``theta``
        Positive scaling constant used by the pessimistic model. It is
        estimated from a helper shortest-path solve when possible and
        otherwise falls back to a summed-data approximation.
    ``_out_edges``
        Cached mapping from each vertex to the outgoing arcs incident to
        that vertex, stored in graph arc order.
    ``_in_edges``
        Cached mapping from each vertex to the incoming arcs incident to
        that vertex, stored in graph arc order.

    Methods
    -------
    ``__init__(graph, budget, true_costs, true_delays, est_costs,
    est_delays, lsd=...)``
        Normalizes list inputs to arrays, copies the graph, builds cached
        adjacency lists, stores true and estimated data as dictionaries,
        and estimates ``theta``.
    ``out_edges(node)``
        Return the cached outgoing-arc list for one vertex.
    ``in_edges(node)``
        Return the cached incoming-arc list for one vertex.
    ``build_spnia_L()``
        Assemble the optimistic ``SPNIA_L`` formulation with binary
        interdiction variables, auxiliary ``v`` and ``w`` variables,
        potential variables ``u``, flow conservation constraints, estimated
        dual constraints, linking constraints, and a budget row.
    ``build_spnia_LG()``
        Assemble the pessimistic ``SPNIA_LG`` formulation using ``theta``
        to combine estimated and realized data in its objective and arc
        constraints.
    ``solve_spnia_LG()``
        Execute the full two-stage solve: optimize the optimistic model,
        warm-start the pessimistic model from that solution, add a bounding
        cut, and solve the pessimistic model. If either stage reaches the
        time limit, return ``(None, None)``.
    ``solve()``
        Public wrapper around ``solve_spnia_LG`` that converts the final
        arc-keyed interdiction dictionary into a plain list ordered by
        ``graph.arcs`` and raises ``RuntimeError`` if the staged solve does
        not complete successfully.

    Typical use cases
    -----------------
    This solver is appropriate when:

    - the leader's estimated arc data differs from the true realized data,
    - interdictions are binary and budget-constrained,
    - the user needs either the built Gurobi formulations or the final
      interdiction decision itself, and
    - the graph's arc order is the canonical indexing shared by all cost,
      delay, and interdiction vectors.
    """

    graph: 'Graph' # The directed graph representing the network
    budget: int  # The budget for the max-min knapsack problem
    true_costs: dict  # True costs of the edges in the graph
    true_delays: dict  # True delays of the edges in the graph
    est_costs: dict  # Estimated costs of the edges in the graph
    est_delays: dict  # Estimated delays of the edges in the graph
    theta: float # The maximum of the estimated delays

    def __init__(self, 
                 graph : 'Graph', 
                 budget : int, 
                 true_costs: List[float] | ndarray[float], 
                 true_delays: List[float] | ndarray[float], 
                 est_costs: List[float] | ndarray[float], 
                 est_delays: List[float] | ndarray[float], 
                 lsd=10e-3):
        """
        Build an asymmetric interdictor around one graph instance.

        Parameters
        ----------
        graph : Graph
            Directed source-target graph defining the shared arc order for
            every cost, delay, and interdiction vector used by the solver.
        budget : int
            Maximum number of arcs that may be interdicted at once.
        true_costs : List[float] | ndarray[float]
            Realized base arc costs aligned with ``graph.arcs``.
        true_delays : List[float] | ndarray[float]
            Realized additive arc delays aligned with ``graph.arcs``.
        est_costs : List[float] | ndarray[float]
            Estimated base arc costs aligned with ``graph.arcs``.
        est_delays : List[float] | ndarray[float]
            Estimated additive arc delays aligned with ``graph.arcs``.
        lsd : float, optional
            Scaling term used when computing ``theta``. Smaller values make
            the resulting pessimistic-model scaling larger.

        Notes
        -----
        Construction performs four pieces of setup:

        1. Convert list inputs into NumPy arrays so vector arithmetic works
           consistently for ``theta`` estimation.
        2. Deep-copy the provided graph so later caller-side mutations do
           not alter the solver's internal state.
        3. Cache incoming and outgoing adjacency lists for each vertex.
        4. Store all true and estimated data as arc-keyed dictionaries and
           estimate ``theta`` from a helper shortest-path solve, with a
           summed-data fallback if that helper path fails.
        """

        # Convert input costs and delays to numpy arrays if they are lists
        if isinstance(true_costs, list):
            true_costs = np.array(true_costs)
        if isinstance(true_delays, list):
            true_delays = np.array(true_delays)
        if isinstance(est_costs, list):
            est_costs = np.array(est_costs)
        if isinstance(est_delays, list):
            est_delays = np.array(est_delays)

        # Store the graph and budget as given
        self.graph = deepcopy(graph)
        self.budget = budget

        # Pre-compute adjacency lists to avoid repeated filtering
        self._out_edges = {v: [] for v in self.graph.vertices}
        self._in_edges = {v: [] for v in self.graph.vertices}
        for i, j in self.graph.arcs:
            self._out_edges[i].append((i, j))
            self._in_edges[j].append((i, j))

        # Store true costs and delays as dictionaries indexed by edge
        self.true_costs = {e: true_costs[i] for i,e in enumerate(self.graph.arcs)}
        self.true_delays = {e: true_delays[i] for i,e in enumerate(self.graph.arcs)}
        self.est_costs = {e: est_costs[i] for i,e in enumerate(self.graph.arcs)}
        self.est_delays = {e: est_delays[i] for i,e in enumerate(self.graph.arcs)}

        # Compute theta as the maximum of the estimated delays
        try:
            longest_path = ShortestPathGrb(self.graph)
            longest_path.setObj(-(true_costs + true_delays))
            self.theta = -longest_path.solve()[1]/lsd
        except Exception:
            self.theta = sum(true_costs + true_delays)/lsd

    def out_edges(self, node):
        """
        Return the cached outgoing arcs of one vertex.

        Parameters
        ----------
        node : int
            Vertex label in ``self.graph.vertices``.

        Returns
        -------
        list[tuple[int, int]]
            Outgoing arcs stored in the same order as ``self.graph.arcs``.
        """
        return self._out_edges[node]

    def in_edges(self, node):
        """
        Return the cached incoming arcs of one vertex.

        Parameters
        ----------
        node : int
            Vertex label in ``self.graph.vertices``.

        Returns
        -------
        list[tuple[int, int]]
            Incoming arcs stored in the same order as ``self.graph.arcs``.
        """
        return self._in_edges[node]


    def build_spnia_L(self):
        """
        Build the optimistic asymmetric interdiction formulation.

        The optimistic model maximizes realized path length under the true
        arc data while linking the leader's interdiction variables to a dual
        representation based on the estimated arc data. It contains:

        - binary interdiction variables ``x``,
        - nonnegative path/dual-link variables ``v`` and ``w``,
        - unrestricted vertex potentials ``u``,
        - source-target flow conservation constraints,
        - one estimated-data dual constraint per arc,
        - one dual-link equality,
        - arc-wise linking constraints, and
        - one budget constraint.

        Returns
        -------
        m : gurobipy.Model
            Fully assembled optimistic ``SPNIA_L`` Gurobi model with output
            suppressed.
        x : gurobipy.tupledict
            Binary arc-indexed interdiction variables.
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

        s = self.graph.source
        t = self.graph.target
        for i in self.graph.vertices:
            m.addConstr(
                gp.quicksum((v[e]+w[e]) for e in self.out_edges(i)) -
                gp.quicksum((v[e]+w[e]) for e in self.in_edges(i))
                == (1 if i == s else -1 if i == t else 0),
                name=f"flow_{i}"
            )

        for i, j in self.graph.arcs:
            m.addConstr(u[i] - u[j] - self.est_delays[(i, j)]*x[(i, j)] <= self.est_costs[(i, j)],
                        name=f"dual_{i}_{j}")

        m.addConstr(u[t] - u[s] +
                    gp.quicksum(self.est_costs[e]*v[e] + (self.est_costs[e]+self.est_delays[e])*w[e]
                                for e in self.graph.arcs) == 0, name="dual_link")

        for e in self.graph.arcs:
            m.addConstr(v[e] + x[e] <= 1, name=f"link1_{e}")
            m.addConstr(w[e] - x[e] <= 0, name=f"link2_{e}")

        m.addConstr(gp.quicksum(x[e] for e in self.graph.arcs) <= self.budget, name="budget")
        m.Params.OutputFlag = 0
        return m, x


    def build_spnia_LG(self):
        """
        Build the pessimistic asymmetric interdiction formulation.

        The pessimistic model uses the scaling constant ``theta`` to combine
        estimated and realized data into a conservative second-stage model.
        It shares the same variable families as ``build_spnia_L`` but changes
        the objective and arc constraints to reflect the pessimistic bound.

        Returns
        -------
        m : gurobipy.Model
            Fully assembled pessimistic ``SPNIA_LG`` Gurobi model with output
            suppressed.
        x : gurobipy.tupledict
            Binary arc-indexed interdiction variables.
        v : gurobipy.tupledict
            Nonnegative auxiliary variables for non-interdicted arc use.
        w : gurobipy.tupledict
            Nonnegative auxiliary variables for interdicted arc use.
        u : gurobipy.tupledict
            Unrestricted vertex-potential variables.
        """

        m  = gp.Model("SPNIA_LG")
        x  = m.addVars(self.graph.arcs, vtype=GRB.BINARY, name="x")
        v  = m.addVars(self.graph.arcs, lb=0.0,        name="v")
        w  = m.addVars(self.graph.arcs, lb=0.0,        name="w")
        u  = m.addVars(self.graph.vertices, lb=-GRB.INFINITY, name="u")

        s = self.graph.source
        t = self.graph.target
        m.setObjective(
            u[s] - u[t] -
            self.theta*gp.quicksum(self.est_costs[e]*v[e] + (self.est_costs[e]+self.est_delays[e])*w[e]
                            for e in self.graph.arcs),
            GRB.MAXIMIZE)

        for i in self.graph.vertices:
            m.addConstr(
                gp.quicksum((v[e]+w[e]) for e in self.out_edges(i)) -
                gp.quicksum((v[e]+w[e]) for e in self.in_edges(i))
                == (1 if i == s else -1 if i == t else 0))

        for i, j in self.graph.arcs:
            m.addConstr(
                u[i] - u[j] - (self.theta*self.est_delays[(i, j)] + self.true_delays[(i, j)])*x[(i, j)]
                <=  self.theta*self.est_costs[(i, j)] + self.true_costs[(i, j)])

        for e in self.graph.arcs:
            m.addConstr(v[e] + x[e] <= 1)
            m.addConstr(w[e] - x[e] <= 0)

        m.addConstr(gp.quicksum(x[e] for e in self.graph.arcs) <= self.budget)
        m.Params.OutputFlag = 0
        return m, x, v, w, u
    

    def solve_spnia_LG(self):
        """
        Run the two-stage optimistic then pessimistic solve procedure.

        Stage 1 solves ``SPNIA_L`` and extracts an incumbent interdiction
        vector and objective bound. Stage 2 builds ``SPNIA_LG``, warm-starts
        its binary decision variables from stage 1, adds a bounding cut
        based on the optimistic objective, and solves the pessimistic model.

        Returns
        -------
        x_star : dict | None
            Arc-keyed interdiction decision returned by the pessimistic
            model. If either stage hits the configured time limit, the
            method returns ``None`` instead.
        z_star : float | None
            Objective value returned by the pessimistic model. If either
            stage hits the configured time limit, the method returns
            ``None`` instead.

        Notes
        -----
        Both stages currently use a 120-second Gurobi ``TimeLimit``.
        """
        # Step 1 – optimistic
        L, xL = self.build_spnia_L()
        L.setParam("TimeLimit", 120.0)
        L.optimize()
        if L.Status == GRB.TIME_LIMIT:
            print("Warning: Time limit reached during optimistic SPNIA-L solve.")
            return None, None
        z_star = L.ObjVal
        x_star = {e: xL[e].X for e in self.graph.arcs}

        # Step 2 – pessimistic with warm-start and cut
        LG, xLG, v, w, u = self.build_spnia_LG()
        for e in self.graph.arcs:
            xLG[e].Start = x_star[e]

        # bounding cut
        s = self.graph.source
        t = self.graph.target
        LG.addConstr(
            u[s] - u[t] -
            self.theta*gp.quicksum(self.est_costs[e]*v[e] + (self.est_costs[e]+self.est_delays[e])*w[e]
                            for e in self.graph.arcs) <= z_star, name="warm_cut")
        LG.setParam("TimeLimit", 120.0)

        LG.optimize()
        if LG.Status == GRB.TIME_LIMIT:
            print("Warning: Time limit reached during pessimistic SPNIA-LG solve.")
            return None, None

        return {e: xLG[e].X for e in self.graph.arcs}, LG.ObjVal
    
    def solve(self):
        """
        Solve the asymmetric interdiction problem and return arc-order data.

        Returns
        -------
        x_star : list[float]
            Final interdiction vector in ``self.graph.arcs`` order.
        z_star : float
            Final objective value returned by the pessimistic stage.

        Raises
        ------
        RuntimeError
            If ``solve_spnia_LG`` does not complete successfully and returns
            ``(None, None)``.

        Notes
        -----
        ``solve_spnia_LG`` returns an arc-keyed dictionary, while this public
        wrapper converts that dictionary into a plain list whose order matches
        ``self.graph.arcs``.
        """

        # Solve the SPNI problem using the two-step procedure
        x_star, z_star = self.solve_spnia_LG()

        # Raise an error if the solve did not complete successfully
        if x_star is None or z_star is None:
            raise RuntimeError("SPNI solve did not complete successfully.")

        return [x for x in x_star.values()], z_star
