import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:

    def __init__(self, model, device, criterion, optimizer, batch_size, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
        self.train_loss_avg = []
        self.train_acc_avg = []
        self.epoch_train_loss = []
        self.epoch_train_acc = []
        self.epoch_test_loss = []
        self.epoch_test_acc = []

        self.misclassified_imgs = {}


    def get_train_stats(self):
        return list(map(lambda x: x.cpu().item(), self.train_loss_avg)), self.train_acc_avg
    
    def get_test_stats(self):
        return list(map(lambda x: x.cpu().item(), self.test_losses)), self.test_acc
    
    def train(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        loss_epoch = 0
        acc_epoch = 0 

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(data)

            loss = self.criterion(y_pred, target)
            self.train_losses.append(loss)
            loss_epoch += loss

            loss.backward()
            self.optimizer.step()


            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f' Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        
        loss_epoch /= len(train_loader.dataset)
        self.train_loss_avg.append(loss_epoch)
        self.train_acc_avg.append(100. * correct / len(train_loader.dataset))
        
        

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        idx = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))

    
    def train_resnet(self, trainloader):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Compute the training accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # compute and store training loss and accuracy
            self.train_losses.append(loss.item())
            self.train_acc.append(100. * correct / total)
        
        # print and store training loss and accuracy
        epoch_train_loss = sum(self.train_losses[-len(trainloader):])/ len(trainloader)
        epoch_train_acc = sum(self.train_acc[-len(trainloader):])/len(trainloader)

        self.epoch_train_loss.append(epoch_train_loss)
        self.epoch_train_acc.append(epoch_train_acc)

        print(f"training loss: {epoch_train_loss:.4f}, accuracy: {epoch_train_acc:.2f}, Optimzer LR: {self.optimizer.param_groups[0]['lr']}")


    def test(self, testloader):
        self.model.eval()
        test_losses = []
        test_accs = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Compute test accuracy
                _, predicted = outputs.max(1)
                total = targets.size(0)
                correct = predicted.eq(targets).sum().item()

                # Compute and store test loss and accuracy
                self.test_losses.append(loss.item())
                self.test_acc.append(100. * correct / total)
                test_losses.append(loss.item())
                test_accs.append(100. * correct / total)

            # Print and store test loss and accuracy
            test_loss = sum(test_losses) / len(testloader)
            test_acc = sum(test_accs) / len(testloader)
            print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.2f}")
            self.epoch_test_loss.append(test_loss)
            self.epoch_test_acc.append(test_acc)
        


    def get_misclassified_images(self, test_loader):
        self.model.eval()
        idx = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output  = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)

                for sample in range(data.shape[0]):
                    if (target[sample] != pred[sample]):
                        self.misclassified_imgs[idx] = [data[sample].cpu(), target[sample].cpu(), pred[sample].cpu()]
                    idx += 1
        return self.misclassified_imgs


def get_criterion_for_classification():
    return nn.CrossEntropyLoss()

def get_sgd_optimizer(model, lr=0.001, momentum=0.9, scheduler=False):
    optimizer = optim.SGD(model.parameters(),
                     lr=lr,
                     momentum=momentum)
    if scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4,
                                                    gamma=0.1)
        return optimizer, scheduler
    else:
        return optimizer
    
def get_adam_optimizer(model, lr=0.001, weight_decay=1e-2):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

