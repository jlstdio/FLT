

def updateMachine():
    self.train(epochs=self.metaData['epoch'])

    if self.finishRate.value < self.networkConfig['epoch']['RewardRate']:
        # assume that this device has better resource environment
        print(f'Client {self.client_internalId} is faster than others performing additional train')
        self.train(epochs=self.networkConfig['epoch']['RewardValue'])
        self.metaData['epoch'] += self.networkConfig['epoch']['RewardValue']
        print(f'Next time client {self.client_internalId} will perform ' + self.metaData['epoch'] + ' epochs')

    elif self.finishRate.value > self.networkConfig['epoch']['PenaltyRate']:
        # assume that this device is in limited resource environment
        self.metaData['epoch'] += self.networkConfig['epoch']['PenaltyValue']