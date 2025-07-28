# spnia_asym.py
import gurobipy as gp
from gurobipy import GRB

def build_spnia_L(G, budget, true_c, true_d, est_c, est_d, theta=None):
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
    v : gurobipy.Var
        Auxiliary variables for the edges in the graph.
    w : gurobipy.Var
        Auxiliary variables for the edges in the graph.
    u : gurobipy.Var
        Dual variables for the nodes in the graph.
    """
    
    m  = gp.Model("SPNIA_L")
    x  = m.addVars(G.edges(), vtype=GRB.BINARY, name="x")
    v  = m.addVars(G.edges(), lb=0.0,        name="v")
    w  = m.addVars(G.edges(), lb=0.0,        name="w")
    u  = m.addVars(G.nodes(), lb=-GRB.INFINITY, name="u")

    # objective: true path length
    m.setObjective(gp.quicksum(true_c[e]*v[e] +
                               (true_c[e]+true_d[e])*w[e] for e in G.edges()),
                   GRB.MAXIMIZE)

    s, t = 0, max(G.nodes)
    for i in G.nodes:
        m.addConstr(
            gp.quicksum((v[e]+w[e]) for e in G.out_edges(i)) -
            gp.quicksum((v[e]+w[e]) for e in G.in_edges(i))
            == (1 if i == s else -1 if i == t else 0),
            name=f"flow_{i}"
        )

    for i, j in G.edges:
        m.addConstr(u[i] - u[j] - est_d[(i, j)]*x[(i, j)] <= est_c[(i, j)],
                    name=f"dual_{i}_{j}")

    m.addConstr(u[t] - u[s] +
                gp.quicksum(est_c[e]*v[e] + (est_c[e]+est_d[e])*w[e]
                            for e in G.edges()) == 0, name="dual_link")

    for e in G.edges:
        m.addConstr(v[e] + x[e] <= 1, name=f"link1_{e}")
        m.addConstr(w[e] - x[e] <= 0, name=f"link2_{e}")

    m.addConstr(gp.quicksum(x[e] for e in G.edges) <= budget, name="budget")
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

# ---------- tiny demo (3-arc toy) ----------
if __name__ == "__main__":
    import networkx as nx
    G = nx.DiGraph()
    data = [(0,1, 4,2, 4.2,1.8),  # (i,j, true_c, true_d, est_c, est_d)
            (1,2, 3,5, 2.9,5.5),
            (0,2,10,2,10.3,1.7)]
    for i,j,c,d,chat,dhat in data:
        G.add_edge(i,j)
    true_c  = {(i,j): c  for i,j,c,_,_,_ in data}
    true_d  = {(i,j): d  for i,j,_,d,_,_ in data}
    est_c   = {(i,j): ch for i,j,_,_,ch,_ in data}
    est_d   = {(i,j): dh for i,j,_,_,_,dh in data}

    plan, val = solve_spnia_LG(G, budget=1,
                               true_c=true_c, true_d=true_d,
                               est_c=est_c,  est_d=est_d,
                               theta=1000)
    print("interdict arcs:", plan, "\nobjective:", val)
