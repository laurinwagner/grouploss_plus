
from torch.utils.data.sampler import Sampler
import random
import numpy as np
def update_probs(l_inds,nb_classes,num_elements):
        lengths=np.zeros(nb_classes)
        for i,inds in enumerate(l_inds):
          if len(inds)>=num_elements:
            lengths[i]=len(inds)
          else:
            lengths[i]=0  
        probs=lengths/sum(lengths)
        return probs


class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl):
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []

        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        # add elements till every class has the same num of obs
        for inds in l_inds:
            n_els = self.max - len(inds) + 1  # take out 1?
            inds.extend(inds[:n_els])  # max + 1

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]

        # shuffle the order of classes
        random.shuffle(split_list_of_indices)
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerAdvanced(Sampler):





    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """
    def __init__(self, l_inds, num_classes, num_elements_class, num_neighbours, dict_class_distances, iterations_for_epoch):
        self.l_inds= list(map(lambda a: random.sample(a, len(a)),l_inds))
        self.num_classes = num_classes
        self.num_elements_class = num_elements_class
        self.num_neighbours=num_neighbours
        self.batch_size = self.num_classes * self.num_elements_class
        self.flat_list = []
        self.batch_list=[]
        self.iterations_for_epoch = iterations_for_epoch
        self.dict_class_distances = dict_class_distances
        self.max=-1
        self.nb_classes=len(l_inds)
        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        l_inds=self.l_inds
        l_inds=list(map(lambda a: random.sample(a, len(a)),l_inds))

        # add elements till every class has the same num of obs
        for inds in l_inds:
            n_els = self.max - len(inds) + 1  # take out 1?
            inds.extend(inds[:n_els])  # max + 1
        #initialize class sample probabilities
        probs=update_probs(l_inds, self.nb_classes, self.num_elements_class)
        #sample the batches 
        while(sum(probs>0)>(self.num_classes*self.num_neighbours+1)): #while there are enough unique classes left
            temp_list = []
            pivot_classes=[]
            other_class_indices=[]




            # get dissimilar pivot classes according to distance matrix
            x=np.random.choice(range(self.nb_classes),1,p=probs)[0] #select a class acording to probabilities 
            pivot_classes.append(x)
            #select num_classes-1 more distant classes so we end up with num_classes pivot classes
            for i in range(self.num_classes-1):
              neighbours=self.dict_class_distances[x][20:] #get a distant neighbour
              if (sum(probs[neighbours])!=0): #if there are distant neighbour classes with elements left
                probs_temp=probs[neighbours]/sum(probs[neighbours])#adjust how likely each one should be sampled
              else: #if no neighbour is left choose a random class according to probs
                neighbours=np.arange(self.nb_classes)
                probs_temp=probs
              x=np.random.choice(neighbours,1,p=probs_temp)[0] #sample class
              pivot_classes.append(x)



            # for each pivot class sample num_elements_class elements in temp list and remove the elements from the class
            for pivot_class_index in pivot_classes:
                pivot_class_indexes = l_inds[pivot_class_index]
                pivot_elements= pivot_class_indexes[:self.num_elements_class]
                l_inds[pivot_class_index]= pivot_class_indexes[self.num_elements_class:]#update the remaining elements
                temp_list.extend(pivot_elements) #add indexes to the batch

                # get the num_neighbours nearest neighbors of the pivot class
                probs=update_probs(l_inds, self.nb_classes,self.num_elements_class) #update probabilities after pivot classes have removed elements
                neighbours=self.dict_class_distances[pivot_class_index][:10] # get a close neighbour close neighbour to pivot classes
                if (sum(probs[neighbours]>0)>=self.num_neighbours): # if enough neighbours are left....
                  probs_temp=probs[neighbours]/sum(probs[neighbours])
                else:
                  neighbours=np.arange(self.nb_classes)
                  probs_temp=probs
                other_class_indices=np.random.choice(neighbours,self.num_neighbours,replace=False, p=probs_temp)
                #other_class_indices = random.sample(list(self.dict_class_distances[pivot_class_index][:10]),self.num_neighbours)


                  # for each neighbor, sample num_elements_class elements of it in the temp list
                for class_index in other_class_indices:
                    class_indexes = l_inds[class_index]
                    class_elements= class_indexes[:self.num_elements_class]
                    l_inds[class_index]= class_indexes[self.num_elements_class:]
                    temp_list.extend(class_elements)
                #update probs
                probs=update_probs(l_inds, self.nb_classes, self.num_elements_class) #update the probs

            #append the batch
            self.batch_list.append(temp_list)
        #shuffle batches
        random.shuffle(self.batch_list)
        self.flat_list = [item for sublist in self.batch_list for item in sublist]

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class CombineSamplerSuperclass(Sampler):
   """
   l_inds (list of lists)
   cl_b (int): classes in a batch
   n_cl (int): num of obs per class inside the batch
   """
   def __init__(self, l_inds, num_classes, num_elements_class, dict_superclass, iterations_for_epoch):
       self.l_inds = l_inds
       self.num_classes = num_classes
       self.num_elements_class = num_elements_class
       self.flat_list = []
       self.iterations_for_epoch = 10
       self.dict_superclass = dict_superclass
   def __iter__(self):
       self.flat_list = []
       for ii in range(int(self.iterations_for_epoch)):
           temp_list = []
           # randomly sample the superclass
           superclass = random.choice(list(self.dict_superclass.keys()))
           list_of_potential_classes = self.dict_superclass[superclass]
           # randomly sample k classes for the superclass
           classes = random.sample(list_of_potential_classes, self.num_classes)
           # get the n objects for each class
           for class_index in classes:
               # classes are '141742158611' etc instead of 1, 2, 3, ..., this should be fixed by finding a mapping between two types of names
               class_ = self.l_inds[class_index]
               # check if the number of elements is >= self.num_elements_class
               if len(class_) >= self.num_elements_class:
                   elements = random.sample(class_, self.num_elements_class)
               else:
                   elements = random.choices(class_, k=self.num_elements_class)
               temp_list.extend(elements)
           # shuffle the temp list
           random.shuffle(temp_list)
           self.flat_list.extend(temp_list)
       return iter(self.flat_list)
   def __len__(self):
       return len(self.flat_list)


class CombineSamplerSuperclass2(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, num_classes, num_elements_class, dict_superclass, iterations_for_epoch):
        self.l_inds = l_inds
        self.num_classes = num_classes
        self.num_elements_class = num_elements_class
        self.flat_list = []
        self.batch_list=[]
        self.iterations_for_epoch = iterations_for_epoch
        self.dict_superclass = dict_superclass

    def __iter__(self):
        self.flat_list = []
        for ii in range(int(self.iterations_for_epoch)):
            temp_list = []

            # randomly sample the superclass
            superclass_1 = random.choice(list(self.dict_superclass.keys()))
            list_of_potential_classes_1 = self.dict_superclass[superclass_1]

            superclass_2 = superclass_1
            while superclass_2 == superclass_1:
                superclass_2 = random.choice(list(self.dict_superclass.keys()))
            list_of_potential_classes_2 = self.dict_superclass[superclass_2]

            # randomly sample k classes for the superclass
            classes = random.sample(list_of_potential_classes_1, self.num_classes // 2)
            classes_2 = random.sample(list_of_potential_classes_2, self.num_classes // 2)
            classes.extend(classes_2)

            # get the n objects for each class
            for class_index in classes:
                # classes are '141742158611' etc instead of 1, 2, 3, ..., this should be fixed by finding a mapping between two types of names
                class_ = self.l_inds[class_index]
                # check if the number of elements is >= self.num_elements_class
                if len(class_) >= self.num_elements_class:
                    elements = random.sample(class_, self.num_elements_class)
                else:
                    elements = random.choices(class_, k=self.num_elements_class)
                temp_list.extend(elements)

            # shuffle the temp list
            random.shuffle(temp_list)
            self.batch_list.append(temp_list)
        random.shuffle(self.batch_list)

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)