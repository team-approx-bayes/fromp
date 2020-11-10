import torch
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from torch.utils.data import TensorDataset, Dataset, Subset
import pickle
import gzip
from copy import deepcopy


class ToydataGenerator():
    def __init__(self, max_iter=5, num_samples=2000, option=0):

        self.offset = 5  # Offset when loading data in next_task()

        # Generate data
        if option == 0:
            # Standard settings
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 1:
            # Six tasks
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3], [1.65, 0.1],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1], [0.7, 0.25]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16], [0.14, 0.14],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22], [0.14, 0.14]]

        elif option == 2:
            # All std devs increased
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.12, 0.22], [0.24, 0.12], [0.07, 0.2], [0.16, 0.08], [0.08, 0.16],
                [0.12, 0.16], [0.16, 0.12], [0.08, 0.16], [0.24, 0.08], [0.08, 0.22]]

        elif option == 3:
            # Tougher to separate
            centers = [[0, 0.2], [0.6, 0.65], [1.3, 0.4], [1.6, -0.22], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.55], [1., 0.1], [1.7, -0.3], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 4:
            # Two tasks, of same two gaussians
            centers = [[0, 0.2], [0, 0.2],
                       [0.45, 0], [0.45, 0]]
            std = [[0.08, 0.22], [0.08, 0.22],
                   [0.08, 0.16], [0.08, 0.16]]

        else:
            # If new / unknown option
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        if option != 1 and max_iter > 5:
            raise Exception("Current toydatagenerator only supports up to 5 tasks.")

        self.X, self.y = make_blobs(num_samples*2*max_iter, centers=centers, cluster_std=std)
        self.X = self.X.astype('float32')
        h = 0.01
        self.x_min, self.x_max = self.X[:, 0].min() - 0.2, self.X[:, 0].max() + 0.2
        self.y_min, self.y_max = self.X[:, 1].min() - 0.2, self.X[:, 1].max() + 0.2
        self.data_min = np.array([self.x_min, self.y_min], dtype='float32')
        self.data_max = np.array([self.x_max, self.y_max], dtype='float32')
        self.data_min = np.expand_dims(self.data_min, axis=0)
        self.data_max = np.expand_dims(self.data_max, axis=0)
        xx, yy = np.meshgrid(np.arange(self.x_min, self.x_max, h),
                             np.arange(self.y_min, self.y_max, h))
        xx = xx.astype('float32')
        yy = yy.astype('float32')
        self.test_shape = xx.shape
        X_test = np.c_[xx.ravel(), yy.ravel()]
        self.X_test = torch.from_numpy(X_test)
        self.y_test = torch.zeros((len(self.X_test)), dtype=self.X_test.dtype)
        self.max_iter = max_iter
        self.num_samples = num_samples  # number of samples per task

        if option == 1:
            self.offset = 6
        elif option == 4:
            self.offset = 2

        self.cur_iter = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            x_train_0 = self.X[self.y == self.cur_iter]
            x_train_1 = self.X[self.y == self.cur_iter + self.offset]
            y_train_0 = np.zeros_like(self.y[self.y == self.cur_iter])
            y_train_1 = np.ones_like(self.y[self.y == self.cur_iter + self.offset])
            x_train = np.concatenate([x_train_0, x_train_1], axis=0)
            y_train = np.concatenate([y_train_0, y_train_1], axis=0)
            y_train = y_train.astype('int64')
            self.cur_iter += 1
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            return TensorDataset(x_train, y_train), TensorDataset(self.X_test, self.y_test)

    def full_data(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.max_iter):
            x_train_list.append(self.X[self.y == i])
            x_train_list.append(self.X[self.y == i+self.offset])
            y_train_list.append(np.zeros_like(self.y[self.y == i]))
            y_train_list.append(np.ones_like(self.y[self.y == i+self.offset]))
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_train = y_train.astype('int64')
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        return TensorDataset(x_train, y_train), TensorDataset(self.X_test, self.y_test)

    def reset(self):
        self.cur_iter = 0


class PermutedMnistGenerator():
    def __init__(self, max_iter=10, random_seed=0):
        # Open data file
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        # Define train and test data
        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.cur_iter = 0

        self.out_dim = 10           # Total number of unique classes
        self.class_list = range(10) # List of unique classes being considered, in the order they appear

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(range(0,10))

        self.sets = self.classes

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter+self.random_seed)
            perm_inds = np.arange(self.X_train.shape[1])

            # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
            if self.cur_iter > 0:
                np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]

            # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
            next_y_train = deepcopy(self.Y_train)

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]

            next_y_test = deepcopy(self.Y_test)

            self.cur_iter += 1

            next_x_train = torch.from_numpy(next_x_train)
            next_y_train = torch.from_numpy(next_y_train)
            next_x_test = torch.from_numpy(next_x_test)
            next_y_test = torch.from_numpy(next_y_test)
            return TensorDataset(next_x_train, next_y_train), TensorDataset(next_x_test, next_y_test)

    def reset(self):
        self.cur_iter = 0


class SplitMnistGenerator():
    def __init__(self):
        # Open data file
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        # Define train and test data
        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        # split MNIST
        task1 = [0, 1]
        task2 = [2, 3]
        task3 = [4, 5]
        task4 = [6, 7]
        task5 = [8, 9]
        self.sets = [task1, task2, task3, task4, task5]

        self.max_iter = len(self.sets)

        self.out_dim = 0        # Total number of unique classes
        self.class_list = []    # List of unique classes being considered, in the order they appear
        for task_id in range(self.max_iter):
            for class_index in range(len(self.sets[task_id])):
                if self.sets[task_id][class_index] not in self.class_list:
                    # Convert from MNIST digit numbers to class index number by using self.class_list.index(),
                    # which is done in self.classes
                    self.class_list.append(self.sets[task_id][class_index])
                    self.out_dim = self.out_dim + 1

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for task_id in range(self.max_iter):
            class_idx = []
            for i in range(len(self.sets[task_id])):
                class_idx.append(self.class_list.index(self.sets[task_id][i]))
            self.classes.append(class_idx)

        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            next_x_train = []
            next_y_train = []
            next_x_test = []
            next_y_test = []

            # Loop over all classes in current iteration
            for class_index in range(np.size(self.sets[self.cur_iter])):

                # Find the correct set of training inputs
                train_id = np.where(self.train_label == self.sets[self.cur_iter][class_index])[0]
                # Stack the training inputs
                if class_index == 0:
                    next_x_train = self.X_train[train_id]
                else:
                    next_x_train = np.vstack((next_x_train, self.X_train[train_id]))

                # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                next_y_train_interm = np.zeros((len(train_id)), dtype='int64')
                if class_index == 0:
                    next_y_train = next_y_train_interm
                else:
                    next_y_train_interm += 1
                    next_y_train = np.concatenate((next_y_train, next_y_train_interm), axis=0)

                # Repeat above process for test inputs
                test_id = np.where(self.test_label == self.sets[self.cur_iter][class_index])[0]
                if class_index == 0:
                    next_x_test = self.X_test[test_id]
                else:
                    next_x_test = np.vstack((next_x_test, self.X_test[test_id]))

                next_y_test_interm = np.zeros((len(test_id)), dtype='int64')
                if class_index == 0:
                    next_y_test = next_y_test_interm
                else:
                    next_y_test_interm += 1
                    next_y_test = np.concatenate((next_y_test, next_y_test_interm), axis=0)

            self.cur_iter += 1

            next_x_train = torch.from_numpy(next_x_train)
            next_y_train = torch.from_numpy(next_y_train)
            next_x_test = torch.from_numpy(next_x_test)
            next_y_test = torch.from_numpy(next_y_test)
            return TensorDataset(next_x_train, next_y_train), TensorDataset(next_x_test, next_y_test), self.sets[self.cur_iter-1]

    def reset(self):
        self.cur_iter = 0


class SplitCIFAR100:

    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.nr_classes = 100
        self.nr_classes_per_task = 10
        self.max_iter = self.nr_classes / self.nr_classes_per_task
        self.cur_iter = 0
        self.class_sets = [
            list(range(10, 20)),
            list(range(20, 30)),
            list(range(30, 40)),
            list(range(40, 50)),
            list(range(50, 60)),
            list(range(60, 70)),
            list(range(70, 80)),
            list(range(80, 90)),
            list(range(90, 100)),
            list(range(100, 110))
        ]

    def get_dims(self):
        # Get data input and output dimensions
        return len(self.train_dataset) / self.nr_classes_per_task, self.nr_classes_per_task

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            train_dataset = SplitDataSet(self.train_dataset, self.cur_iter, self.nr_classes,
                                         self.nr_classes_per_task)
            val_dataset = SplitDataSet(self.val_dataset, self.cur_iter, self.nr_classes,
                                       self.nr_classes_per_task)

            self.cur_iter += 1

        return train_dataset, val_dataset, self.class_sets[self.cur_iter-1]


class SplitDataSet(Dataset):

    def __init__(self, dataset, cur_iter, nr_classes, nr_classes_per_task):
        self.dataset = dataset
        self.cur_iter = cur_iter
        self.classes = [i for i in range(nr_classes)]

        targets = self.dataset.targets
        task_idx = torch.nonzero(torch.from_numpy(
            np.isin(targets, self.classes[nr_classes_per_task * self.cur_iter:
                                                       nr_classes_per_task * self.cur_iter
                                                       + nr_classes_per_task])))

        self.subset = Subset(self.dataset, task_idx)

    def __getitem__(self, index):
        img, target = self.subset[index]
        target = target - 10 * self.cur_iter

        return img, target

    def __len__(self):
        return len(self.subset)
