import sys
from util import PriorityQueue
sys.setrecursionlimit(100000)

# Model

class TransportationProblem(object):
    def __init__(self, N, weights):
        # N = number of blocks
        # weights: action -> cost dictionary
        self.N = N
        self.weights = weights
    def startState(self):
        return 1
    def isEnd(self, state):
        return state == self.N
    def succAndCost(self, state):
        # return a list of (action, newState, cost) triples
        result = []
        if state+1 <= self.N:
            result.append(('walk', state+1, self.weights['walk']))
        if state*2 <= self.N:
            result.append(('tram', 2*state, self.weights['tram']))
        return result

# Inference Algorithms

def printSolution(solution):
    totalCost, history = solution
    print('Total cost: {}'.format(totalCost))
    for step in history:
        print(step)

def dynamicProgramming(problem):
    # state -> futureCost
    cache = {}
    def futureCost(state):
        # return best cost of reaching the end from state
        if problem.isEnd(state):
            return 0
        if state in cache:
            return cache[state][0]
        result = min([(cost+futureCost(newState), action, newState, cost) \
            for action, newState, cost in problem.succAndCost(state)])
        cache[state] = result
        return result[0]
    
    # recover total cost
    totalCost = futureCost(problem.startState())

    # recover history
    history = []
    state = problem.startState()
    while not problem.isEnd(state):
        _, action, newState, cost = cache[state]
        history.append((action, newState, cost))
        state = newState

    return (totalCost, history)

def predict(N, weights):
    # inference: f: x -> y
    problem = TransportationProblem(N, weights)
    totalCost, history = dynamicProgramming(problem)
    return [action for action, newState, cost in history]

# Learning Algorithms

def structuredPerceptron(examples):
    # examples: (x, y) pairs where x is input (N) and y is output (sequence of actions)
    weights = {'walk': 0, 'tram': 0}
    for t in range(100):
        numMistakes = 0
        for N, trueActions in examples:
            predActions = predict(N, weights)
            if trueActions != predActions:
                numMistakes += 1
            for action in trueActions:
                weights[action] -= 1
            for action in predActions:
                weights[action] += 1
        print('Iteration: {} Mistakes: {} Weights: {}'.format(t, numMistakes, weights))
        if numMistakes == 0:
            break

# Main

def generateExamples():
    trueWeights = {'walk': 1, 'tram': 5}
    return [(N, predict(N, trueWeights)) for N in range(1, 200)]

examples = generateExamples()
for example in examples:
    print(example)

structuredPerceptron(examples)