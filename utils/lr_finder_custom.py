from torch_lr_finder import LRFinder

class LRFinderCustom:

    def __init__(self, nn_model, optimizer, criterion, device='cuda'):
        self.nn_model = nn_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_finder = None

    def get_lr_finder(self):
        self.lr_finder = LRFinder(self.nn_model,
                        self.optimizer,
                        self.criterion,
                        self.device)
        return self.lr_finder
    
    def find_lr(self, trainloader, end_lr=0.01, num_iter=500, step_mode='exp'):
        self.lr_finder.range_test(trainloader,
                                  end_lr=end_lr,
                                  num_iter=num_iter,
                                  step_mode=step_mode)


    def plot(self):
        self.lr_finder.plot()

    def reset(self):
        self.lr_finder.reset() 
