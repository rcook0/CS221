import sys
from util import PriorityQueue
sys.setrecursionlimit(100000)

# Model

class TransportationProblem(object):
    def __init__(self, N):
        # N = number of blocks
        self.N = N
    def startState(self):
        return 1
    def isEnd(self, state):
        return state == self.N
    def succAndCost(self, state):
        # return a list of (action, newState, cost) triples
        result = []
        if state+1 <= self.N:
            result.append(('walk', state+1, 1))
        if state*2 <= self.N:
            result.append(('tram', 2*state, 2))
        return result

# Algorithms

def printSolution(solution):
    totalCost, history = solution
    print('Total cost: {}'.format(totalCost))
    for step in history:
        print(step)

def backtrackingSearch(problem):
    best = {
        'cost': float('inf'),
        'history': None
    }
    def recurse(state, history, totalCost):
        # at state, having undergone history, with totalCost so far
        if problem.isEnd(state):
            # update the best solution
            if totalCost<best['cost']:
                best['cost'] = totalCost
                best['history'] = history
            return
        for action, newState, cost in problem.succAndCost(state):
            recurse(newState, history+[(action, newState, cost)], totalCost+cost)
    recurse(problem.startState(), [], 0)
    return (best['cost'], best['history'])

def dynamicProgramming(problem):
    # state -> futureCost
    cache = {}
    def futureCost(state):
        # return best cost of reaching the end from state
        if problem.isEnd(state):
            return 0
        if state in cache:
            return cache[state]
        result = min([cost+futureCost(newState) for action, newState, cost in problem.succAndCost(state)])
        cache[state] = result
        return result
    return (futureCost(problem.startState()), [])

def uniformCostSearch(problem):
    frontier = PriorityQueue()
    frontier.update(problem.startState(), 0)
    while True:
        # move the top priority element from frontier to explored
        state, totalCost = frontier.removeMin()
        if problem.isEnd(state):
            return (totalCost, [])
        # update frontier according to state -> newState transitions
        for action, newState, cost in problem.succAndCost(state):
            frontier.update(newState, totalCost+cost)

# Main

problem = TransportationProblem(N=10000)
#printSolution(backtrackingSearch(problem))
printSolution(dynamicProgramming(problem))
printSolution(uniformCostSearch(problem))

#print(problem.succAndCost(2))
#print(problem.succAndCost(9))
