# https://pythonhosted.org/PuLP/main/index.html
# https://towardsdatascience.com/linear-programming-and-discrete-optimization-with-python-using-pulp-449f3c5f6e99

# BASED ON: Deployed ARMOR Protection: The Application of a Game Theoretic Model for Security at the Los Angeles International Airport

import sys

# import library for solving MILP
import pulp
# without this line we would have to write every time pulp.lpSum instead of just lpSum
from pulp import lpSum

# defined in the paper
LARGE_POSITIVE_NUMBER = M = 1000000 # 1kk

# Problem class definition
class Problem(object):

    # Problem class constructor. This is executed when Problem(..., ..., ... ) is called
    def __init__(self, followersTypesCnt, followersStrategiesCnt, leaderStrategiesCnt, P, R, C):
        self.L = followersTypesCnt
        self.J = followersStrategiesCnt
        self.I = leaderStrategiesCnt

        # leader payoffs array
        self.R = R
        # follower payoffs array
        self.C = C
        # followers types probabilities list
        self.P = P
        
    # This is representation of Problem as a string, when someone calls for example print(problem)    
    def __str__(self):
        return ('''Bayesian Stackelberg game
                {} followers types
                {} followers strategies
                {} leader strategies'''.format(self.L,
                               self.J,
                               self.I)
                )

# The following functions are MILP objective and constraints
# to add objective function/constraint we just do milp += objective/constraint
# if function is bool expresion like lpSum(...) == 1, it is considered as constraint
# it is considered as objective function otherwise. If two or more objective functions are specified
# PuLP overrides them

# The arguments for the following functions are taken from local variables of function model()

# Objective function
# lpSum(p.P[l] * p.R[l][i][j] * z_s[l][i][j] for i in range(p.I) for j in range(p.J) for l in range(p.L) - this is in python called list comprehension
# shortcut for :
# resultList = []
# for l in range(p.L):
#     for i in range(p.I):
#         for j in range(p.J):
#             resultList.append(p.P[l] * p.R[l][i][j] * z_s[l][i][j])
#
# return lpSum(resultList)     
#
# **kwargs is python specific argument which is a dictionary of additionally passed arguments to a function, not used in our case       
def __objective_function(index, p, milp, z_s, **kwargs):
    milp += (lpSum(p.P[l] * p.R[l][i][j] * z_s[l][i][j] for i in range(p.I) for j in range(p.J) for l in range(p.L)), '({}) objective function'.format(index))

# constraint functions
# order the same as in the paper
def __sum_of_zs(index, p, milp, z_s, **kwargs):
    for l in range(p.L):
        milp += (lpSum(z_s[l][i][j] for i in range(p.I) for j in range(p.J)) == 1, '({}) sum of z[i, j] for given l ({}) == 1'.format(index, l))

def __zs_sum_by_j_lt_1(index, p, milp, z_s, **kwargs):
    for l in range(p.L):
        for i in range(p.I):
            milp += (lpSum(z_s[l][i][j] for j in range(p.J)) <= 1, '({}) sum of z for given i, l ({}, {}) <= 1'.format(index, i, l))

def __zs_sum_by_i_lt_1_gt_q(index, p, milp, z_s, q_s, **kwargs):
    for l in range(p.L):
        for j in range(p.J):
            milp += (lpSum(z_s[l][i][j] for i in range(p.I)) <= 1, '({}) sum of z for given j, l ({}, {}) <= 1'.format(index, j, l))
            milp += (lpSum(z_s[l][i][j] for i in range(p.I)) >= q_s[l][j], '({}) sum of z for given j, l ({}, {}) >= q'.format(index, j, l))

def __qs_sum_eq_1(index, p, milp, q_s, **kwargs):
    for l in range(p.L):
        milp += (lpSum(q_s[l][j] for j in range(p.J)) == 1, '({}) sum of q for given l ({}) == 1'.format(index, l))

def __most_complex_one(index, p, milp, z_s, q_s, a_s, **kwargs):
    for l in range(p.L):
        for j in range(p.J):
            milp += ((a_s[l] - lpSum(p.C[l][i][j] * lpSum(z_s[l][i][h] for h in range(p.J)) for i in range(p.I))) >= 0, '({}) j,l ({}, {}) <= complex'.format(index, j, l))
            milp += ((a_s[l] - lpSum(p.C[l][i][j] * lpSum(z_s[l][i][h] for h in range(p.J)) for i in range(p.I))) <= ((1 - q_s[l][j]) * M), '({}) j,l ({}, {}) <= complex <='.format(index, j, l))

def __zs_same_in_row(index, p, milp, z_s, **kwargs):
    for l in range(p.L):
        for i in range(p.I):
            milp += (lpSum(z_s[l][i][j] for j in range(p.J)) == lpSum(z_s[1][i][j] for j in range(p.J)), '({}) i,l ({}, {})zs same in row'.format(index, i, l))

# attach all functions to our model
__model = (
    __objective_function,
    __sum_of_zs,
    __zs_sum_by_j_lt_1,
    __zs_sum_by_i_lt_1_gt_q,
    __qs_sum_eq_1,
    __most_complex_one,
    __zs_same_in_row
)

def model(p: Problem):
    # dimensions: l x i x j
    # z - defines the desired probability of the leader's strategy
    z_s = pulp.LpVariable.dicts('z_%s_%s_%s',
                                (
                                    [i for i in range(p.L)],
                                    [i for i in range(p.J)],
                                    [i for i in range(p.I)]
                                ),
                                lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)

    # dimensions: l x j
    # q - defines which strategy is optimal for given follower
    q_s = pulp.LpVariable.dicts('q_%s_%s',
                                (
                                    [i for i in range(p.L)],
                                    [i for i in range(p.J)]
                                ),
                                cat=pulp.LpInteger, lowBound=0, upBound=1)

    # dimensions: l
    # a - defines maximal payoff for given follower
    a_s = pulp.LpVariable.dicts('a_%s',
                            (
                                [i for i in range(p.L)],
                            ),
                            cat=pulp.LpContinuous)                           

    # declare problem. pulp.LpMaximize says that we what no maximize objective function
    milp = pulp.LpProblem('DOBSS',
                          pulp.LpMaximize)
    for index, equations in enumerate(__model, 1):
        equations(**locals())

    return milp

# this class just keeps basic information about solution of MILP
# like milp itself and maximiezd value
class Solution:
    def __init__(self, p: Problem, milp: pulp.LpProblem):
        assert (milp.status == pulp.LpStatusNotSolved or
                milp.status == pulp.LpStatusOptimal), "problem is {}".format(
            pulp.LpStatus[milp.status])
        self.value = milp.objective.value()
        self.milp = milp

    def __str__(self):
        return str(self.milp.variablesDict())

def readLJI():
    l = int(input("Enter number of followers types: "))
    j = int(input("Enter number of strategies for a single follower: "))
    i = int(input("Enter number of strategies for a leader: "))

    # returning a triple (l, j, i)
    return l, j, i

# L - followers types count
# J - single follower strategies count
# I - leader strategies count
# returns array of payoffs l x i x j
def readPayoffsArray(payoffType, L, J, I):
    payoffsList = []

    for l in range(L):
        payoffsList.append([])
        for i in range(I):
            payoffsList[-1].append([])
            for j in range(J):
                payoffsList[-1][-1].append(float(input('Enter {} payoff. Follower of type {}, leader strategy {}, follower strategy {}: '.format(payoffType, l, i, j))))


    return payoffsList

def readProbabilities(P):
    probabs = []
    for p in range(P):
        probabs.append(float(input('Enter probability for follower {} [0..1]: '.format(p))))

    return probabs

def main():
    # read input
    L, J, I = readLJI()
    R = readPayoffsArray("leader", L, J, I)
    C = readPayoffsArray("follower", L, J, I)
    P = readProbabilities(L)

    # create object that stores the input
    problem = Problem(L, J, I, P, R, C)

    milp = model(problem)
    # print(model) # uncomment to print all equations in MILP problem definition

    # solve MILP
    milp.solve()
    # store solution
    sol = Solution(problem, milp)

    # maximized value
    print('Leader payoff: {}'.format(sol.value))

    z_res = {}
    a_res = {}
    for var in sol.milp.variables():
        if var.name[0] == 'z': 
            if var.value() != 0:
                i = var.name.split('_')[2]
                if i not in z_res:
                    z_res[i] = var.value()
        if var.name[0] == 'a':
            l = var.name.split('_')[1]
            a_res[l] = var.value()             

    for key in z_res:
        print('Leader should choose strategy {} with probability {}%'.format(key, z_res[key]*100))

    for key in a_res:
        print('Max follower {} payoff is {}'.format(key, a_res[key]))    

if __name__ == '__main__':
    sys.exit(main())    