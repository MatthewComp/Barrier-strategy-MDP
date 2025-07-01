import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import gamma, gammainc
from scipy.stats import gamma as gamma_dist
import csv

def policy_iteration(c, delta, lam, shape, risk_av, p, count):
    """
    Handles the main part of the iterative process of policy iteration.
    c: Vector of premiums
    delta: Discount factor
    lam: Rate parameter for gamma distribution
    shape: Shape parameter for gamma distribution
    risk_av: Risk aversion parameter
    p: Probability of no crash
    count: Number of iterations to perform.
    """
    # Create initial policy (all 1/2 in corresponding places)
    N = len(c)
    policy = np.eye(N, N, 1) / 2 + np.eye(N,N,-1) / 2
    policy[0][0] = 1/2
    policy[N-1][N-1] = 1/2

    for i in range(0,count):
        values = compute_value(policy, c, delta, lam, shape, risk_av, p)
        policy = next_policy(values, risk_av, p, lam, shape, c, delta, policy)

    return policy


def next_policy(values, risk_av, p, lam, shape, c, delta, prev_policy):
    """
    Given the previous set of values, use bellmans to find next policy.
    """
    prob_bound = Bounds(0, 1)
    N = len(c)
    next = np.zeros(N) # The next policy, to be updated below (just the probabilities)

    for i in range(0,N):
        # Minimize the bellman equation for each row in the policy
        a = find_prob(prev_policy, i) # Find the probability from policy
        result = minimize(bellman, a, (values, i, risk_av, p, lam, shape, c, delta), method="trust-constr", bounds = prob_bound)
        next[i] = result.x[0]

    policy = probs_to_policy(next, N, prev_policy)

    return policy

def probs_to_policy(probabilities, N, policy):
    """
    Takes the set of probabilities (pidown) and puts them into the matrix
    """
    # First place first/last row
    policy[0][0] = probabilities[0]
    policy[0][1] = 1 - probabilities[0]
    policy[N-1][N-2] = probabilities[N-1]
    policy[N-1][N-1] = 1 - probabilities[N-1]

    if N == 2:
        return policy

    for i in range(1,N-1):
        policy[i][i-1] = probabilities[i]
        policy[i][i+1] = 1 - probabilities[i] 

    return policy



def bellman(a, values, state, risk_av, p, lam, shape, c, delta):
    """
    The bellman operator, a is the value of alpha_i
    """
    N = len(c)
    b = 1 - a
    # First term of Bellman
    g = g_exp_gamma(state, risk_av, p, lam, shape, a, c)
    # The second term of Bellman
    if state == 0:
        b = delta * (values[0] * a + values[1] * b)
    elif state == N-1:
        b = delta * (values[N-2] * a + values[N-1] * b)
    else:
        b = delta * (values[state-1] * a + values[state+1] * b)
    # Return negative so scipy minimize gives maximiser
    return -(g + b)



def compute_value(policy, c, delta, lam, shape, risk_av, p):
    """
    Given a policy, compute the corresponding value from the sum.
    """
    N = len(c)  # Number of states
    A = np.eye(N,N) - delta * policy
    A_inverse = np.linalg.inv(A)
    g = g_vector(policy, risk_av, p, lam, shape, c)
    g = g.reshape([-1,1])
    
    V = A_inverse @ g
    return V

def g_exp_gamma(state, risk_av, p, lam, shape, a, c):
    """
    Calculates the immediate reward

    state: Current state
    risk_av: Risk aversion parameter
    p: Probability of not crashing
    lam: Rate parameter for gamma distribution
    shape: Shaper parameter for gamma distribution
    a: pidown_i, probability of not claiming in current state.
    c: Vector of premiums
    """
    prem = c[state]
    coeff = - np.exp(risk_av * prem) 
    L = quantile_gamma(a, lam, shape, p)

    if lam != risk_av:
        value_big_loss = (1-p) * np.power(lam/(lam-risk_av), shape) * gammainc(shape, (lam - risk_av)*L)
    else:
        value_big_loss = (1-p) * np.power(gamma * L, shape) / gamma(shape+1)
    
    return coeff * (1 - a + p + value_big_loss)

def g_vector(policy, risk_av, p, lam, shape, c):
    N = len(c)
    g = np.ones(N)
    for i in range(0,N):
        a = find_prob(policy, i)
        g[i] = g_exp_gamma(i, risk_av, p, lam, shape, a, c)

    return g

def find_prob(policy, state):
    """
    Find the "pidown" probablility in the policy
    """
    i = max(state - 1, 0)
    return policy[state][i]


def quantile_gamma(x, lam, shape, p):
    if x <= p:
        return 0
    else:
        arg = (x-p)/(1-p)
        gamma = gamma_dist(a = shape, scale = 1/lam) # Gamma distribution with given parameters
        return gamma.ppf(arg)
    
def recover_barrier(policy, lam, shape, p):
    N = len(policy)
    barriers = np.zeros(N)
    for i in range(0,N):
        a = find_prob(policy, i)
        barriers[i] = quantile_gamma(a, lam, shape, p)
    return barriers

def rates(lam, shape, p):
    """
    Deals with the input of rates based on the choice of principle
    """
    option = int(input("Inputting rates as: \n 1. Direct input \n 2. Expected value premium principle \n"))
    print("Input 'x' to finish")
    recent = 0
    i = 1
    c = []
    if option == 1:
        # Inputs are the premiums
        while True:
            recent = input(f"Enter premium for rate class {i}: ")
            i += 1
            if recent == "x":
                break
            else:
                c.append(recent)
    else:
        # In other rate classes inputs are rate loadings
        while True:
            recent = input(f"Enter rate loading for rate class {i}: ")
            i += 1
            if recent == "x":
                break
            else:
                c.append(float(recent) + 1)

    c = np.array(c)

    if option == 1:
        return c
    elif option == 2:
        # Note rates * (1-p) * shape / lam is the expected value
        return c * (1-p) * shape / lam
        



def main():
    # Take inputs
    risk_av = float(input("Risk aversion parameter: "))
    delta = float(input("Discount factor: "))
    p = float(input("Probability of no loss: "))
    lam = float(input("Rate parameter: "))
    shape = float(input("Shape parameter: "))
    c = rates(lam, shape, p)
    
    # Perform policy iteration and recover barriers
    policy = policy_iteration(c, delta, lam, shape, risk_av, p, 10)
    barriers = recover_barrier(policy, lam, shape, p)
    print(policy)
    print(barriers)

if __name__ == "__main__":
    main()