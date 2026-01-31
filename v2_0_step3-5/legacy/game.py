class HalvingGame(object):
    def __init__(self, N):
        self.N = N
    
    # state = (player, number) where player = +1 (agent) or -1 (opponent)
    def startState(self):
        return (+1, self.N)
    
    def actions(self, state):
        return ['-', '/']
    
    def succ(self, state, action):
        player, number = state
        if action=='-':
            return (-player, number - 1)
        elif action=='/':
            return (-player, number // 2)
    
    def isEnd(self, state):
        player, number = state
        return number == 0
    
    def utility(self, state):
        assert self.isEnd(state)
        player, number = state
        return player * float('inf')
    
    def player(self, state):
        player, number = state
        return player

# policies

def simplePolicy(game, state):
    action = '-'
    print('simplePolicy: playing {}'.format(action))
    return action

def humanPolicy(game, state):
    while True:
        action = input('humanPolicy: please input action: ')
        if action in game.actions(state):
            return action

def minimaxPolicy(game, state):
    # recursively find (utility, bestAction)
    def recurse(state):
        if game.isEnd(state):
            return (game.utility(state), None)
        choices = [
            (recurse(game.succ(state, action))[0], action)
            for action in game.actions(state)
        ]
        if game.player(state) == +1:
            return max(choices)
        elif game.player(state) == -1:
            return min(choices)
    value, action = recurse(state)
    print('minimaxPolicy: action = {}, value = {}'.format(action, value))
    return action

# main loop

policies = {
    +1: humanPolicy,
    -1: minimaxPolicy
}

game = HalvingGame(15)

state = game.startState()

while not game.isEnd(state):
    print('='*10, 'state = {}'.format(state))
    player = game.player(state)
    policy = policies[player]
    action = policy(game, state)
    state = game.succ(state, action)

print('utility = {}'.format(game.utility(state)))