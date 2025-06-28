import numpy as np
import random
from scipy.optimize import minimize, Bounds
from scipy.special import gamma, gammainc
from scipy.stats import gamma as gamma_dist

def policy_iteration(initial, c, delta, lam, shape, risk_av, p, count):
    """
    Handles the main part of the iterative process of policy iteration.
    
    inital: Initial policy
    c: Vector of premiums
    delta: Discount factor
    lam: Rate parameter for gamma distribution
    shape: Shape parameter for gamma distribution
    risk_av: Risk aversion parameter
    p: Probability of no crash
    count: Number of iterations to perform.
    """
    for i in range(0,count):
        values = compute_value(initial, c, delta, lam, shape, risk_av, p)
        initial = next_policy(values, risk_av, p, lam, shape, c, delta, initial)

    return initial


def next_policy(values, risk_av, p, lam, shape, c, delta, prev):
    """
    Given the previous set of values, use bellmans to find next policy.
    """
    prob_bound = Bounds(0, 1)
    N = len(c)
    next = np.zeros(N) # The next policy, to be updated below

    for i in range(0,N):
        # Minimize the bellman equation for each row in the policy
        result = minimize(bellman, prev[i] , (values, i, risk_av, p, lam, shape, c, delta), method="trust-constr", bounds = prob_bound)
        next[i] = result.x[0]

    return next


def bellman(a, values, state, risk_av, p, lam, shape, c, delta):
    """
    The bellman operator, a is the value of alpha_i
    """
    N = len(c)
    b = 1 - a
    # First term of Bellman
    g = g_exp_gamma(state, risk_av, p, lam, shape, a, c)
    # The second term of Bellman
    if state == 1:
        b = delta * (values[0] * a + values[1] * b)
    elif state == N:
        b = delta * (values[N-2] * a + values[N-1] * b)
    else:
        b = delta * (values[state-2] * a + values[state] * b)
    # Return negative so scipy minimize gives maximiser
    return -(g + b)



def compute_value(A, c, delta, lam, shape, risk_av, p):
    """
    Given a policy, compute the corresponding value from the sum.
    A  = [a1, a2, ..., aN]
    """
    N = len(c)  # Number of states
    V = []
    for i in range(0,N):
        v = []
        # Generate 100 values and take average (MC simulation)
        for j in range(0, 50):
            losses = generate_gamma_losses(lam, shape, p, 20) # 50 Terms of the sum
            v.append(compute_sum(delta, i, A, c, losses, risk_av, p, lam, shape))

        # Append V with the estimate for V_i
        V.append(np.mean(v))

    return V

def compute_sum(delta, initial_state, A, c, losses, risk_av, p, lam, shape):
    sum = 0
    state = initial_state
    num = len(losses)

    # Find the finite horizon sum for num states
    for k in range(0,num):
        # Pass through a = alpha_i to g(i,alpha_i) (and all parameters)
        sum += np.power(delta, k) * g_exp_gamma(state, risk_av, p, lam, shape, A[state-1], c)
        # Update the state according to the policy, the loss in that period and the current state
        state = next_state(state, losses[k], A, lam, shape, p)

    return sum

def next_state(state, loss, A, lam, shape, p):
    """
    Given a loss and current state, determine next state
    """
    N = len(A)
    # quantile_gamma(alpha_i) = L_i the barrier
    if loss <= quantile_gamma(A[state-1], lam, shape, p):
        return max(1, state-1)
    else:
        return min(N, state+1)

def generate_gamma_losses(lam, shape, p, n):
    """
    Generate n loss variables according to mixed Gamma(shape, lambda) distribution
    """
    random.seed(2)
    losses = []
    for i in range(0,n):
        if np.random.binomial(1,1-p) == 0: # No loss made
            losses.append(0)
        else:
            losses.append(np.random.gamma(shape, 1/lam))

    return losses

def g_exp_gamma(state, risk_av, p, lam, shape, a, c):
    """
    Calculates the immediate reward

    state: Current state
    risk_av: Risk aversion parameter
    p: Probability of not crashing
    lam: Rate parameter for gamma distribution
    shape: Shaper parameter for gamma distribution
    a: alpha_i, probability of not claiming in current state.
    c: Vector of premiums
    """
    prem = c[state - 1]
    coeff = - np.exp(-risk_av * prem) 
    L = quantile_gamma(a, lam, shape, p)

    value_small_loss = 1 - a + p
    value_loss = (1-p) * np.power(lam/(lam-risk_av), shape) * gammainc(shape, (lam - risk_av)*L)
    
    return coeff * (value_small_loss + value_loss)


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
        barriers[i] = quantile_gamma(policy[i], lam, shape, p)
    return barriers

def main():
    delta = 0.95
    lam = 0.1
    risk_av = 2
    p=0.8
    shape = 5

    # Determine premiums using rate, expected value premium principle
    rates = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    c = rates *  (1-p) * shape / lam # note (1-p)*shape/lambda is the expected loss
    
    N = len(c)

    initial = np.ones(N) / 2

    policy = policy_iteration(initial, c, delta, lam, shape, risk_av, p, 100)
    barriers = recover_barrier(policy, lam, shape, p)
    print(policy)
    print(barriers) 
    

main()