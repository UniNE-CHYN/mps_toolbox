import numpy as np

class connected_components:
    def __init__(self, image):
        self.data = image
        self.labels = np.zeros_like(image)
        self.nx = image.shape[0]
        self.ny = image.shape[1]
        self.eq_set = DisjointSet()
        self.label_counter = 0
                    
    def neighbors_are_equal(self, x,y):
        if x-1 < 0 or y-1 < 0:
            return False
        else:
            return self.data[x-1,y] == self.data[x, y-1]
        
    def neighbors_have_same_label(self, x,y):
        if x-1 < 0 or y-1 < 0:
            return False
        else:
            return self.labels[x-1,y] == self.labels[x, y-1]
        
    def west_is_equal(self, x, y):
        if x-1 < 0:
            return False
        else:
            return self.data[x,y] == self.data[x-1,y]
    def north_is_equal(self, x, y):
        if y-1 < 0:
            return False
        else:
            return self.data[x,y] == self.data[x,y-1]
        
    def add_equivalence_relation_for_neighbors(self, x,y):
        label_x = self.labels[x-1,y]
        label_y = self.labels[x, y-1]
        if (self.eq_set.find_set(label_x) != self.eq_set.find_set(label_y)):
            self.eq_set.union(self.eq_set.find_set(label_x), self.eq_set.find_set(label_y))
        
    def new_label(self):
        self.label_counter += 1
        self.eq_set.make_set(self.label_counter)
        return self.label_counter
    
    def fill_label_array(self):
        for x in np.arange(self.nx):
            for y in np.arange(self.ny):
                if self.neighbors_are_equal(x, y):
                    if self.west_is_equal(x,y):
                        if self.neighbors_have_same_label(x,y):
                            self.labels[x,y] = self.labels[x-1,y]
                        else:
                            self.labels[x,y] = min(self.labels[x-1,y], self.labels[x, y-1])
                            self.add_equivalence_relation_for_neighbors(x,y)
                    else:
                        self.labels[x,y] = self.new_label()
                else:
                    if self.west_is_equal(x,y):
                        self.labels[x,y] = self.labels[x-1, y]
                    elif self.north_is_equal(x,y):
                        self.labels[x,y] = self.labels[x, y-1]
                    else:
                        self.labels[x,y] = self.new_label()
                        
        for x in np.arange(self.nx):
            for y in np.arange(self.ny):
                self.labels[x,y] = self.eq_set.find_set_minimal_representative(self.labels[x,y])
    def get_label_array(self):
        return self.labels

class DisjointSet:
    def __init__(self):
        self.set_of_sets = set({})
        
    def make_set(self, element_x):
        """creates a new set whose only member (and thus representative)
            is x. Since the sets are disjoint, we require that x not already be in some other
        set.
        """
        if self.find_set(element_x) is None:
            self.set_of_sets.add(frozenset([element_x]))
        else:
            print("x is already in some other set")
        
    def union(self, set_x,set_y):
        new_set = set_x.union(set_y)
        self.set_of_sets.add(new_set)
        self.set_of_sets.remove(set_x)
        self.set_of_sets.remove(set_y)
        
    def find_set(self, element_x):
        for set_ in self.set_of_sets:
            if element_x in set_:
                return set_
        return None
    
    def find_set_minimal_representative(self, element_x):
        S = self.find_set(element_x)
        return min(S)


def _test2():
	categ = categories(image_list[0])
	print(categ)
	print(percent_of_categories(image_list[0], categ))
	ind = indicator_variable(image_list[0], 2)

def _test():
    data = np.array([5 , 9 , 7, 7, 9, 7, 7, 7, 7])
    cc = connected_components(data)
    cc.fill_label_array()
    print(cc.labels)
    print(cc.eq_set.set_of_sets)

    S = DisjointSet()
    S.make_set(5)
    S.make_set(4)
    S.make_set(3)
    print(S.set_of_sets)
    S.union(S.find_set(3), S.find_set(4))

    print(S.set_of_sets)
    print(S.find_set_minimal_representative(4))
    print(S.find_set_minimal_representative(5))
