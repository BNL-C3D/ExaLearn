import inspect
import numpy as np
import cosmo_data

def calc_budget_per_class(total_budget, weights):
    r'''
        given the total budget and weights (as prob), 
        return a list of budgets 

        Note: len(weights) == num of classes
    '''

    n = len(weights)
    y = np.asarray(weights)
    y = y/y.sum()
    y = np.floor(total_budget * y)
    residue = total_budget - y.sum()
    while residue > 0 :
        residue -= 1
        y[np.random.randint(n)] += 1
    
    return [int(_) for _ in y]

def sample_given_budget_excluding(budget_per_class, data_set, exclusion, rnd_seed=None):
    r'''
        sample data points given the budget per class except the ones in the
        exclusion.

        `data_set`: should support [] operator and returns a tuple of (x, y)
                    where y is the class id. For example, pytorch.utils.data.Dataset type
                    or its children.


        return: a list (ids) of data points.
    '''
    rnd_state = None
    if not hasattr(data_set, "__getitem__"):
        print("ERROR: ", "data_set should has [] operator")
        return

    if rnd_seed:
        rnd_state = np.random.get_state()
        np.random.seed(rnd_seed)

    idx = np.arange(len(data_set))
    np.random.shuffle(idx)
    exclusion = set(exclusion)
    rst = []

    tot_bgt = np.asarray(budget_per_class).sum()
    for i in idx:
        if i not in exclusion:
            _, y = data_set[i]
            if budget_per_class[y] > 0 :
                budget_per_class[y] -= 1
                rst.append(int(i))
                tot_bgt -= 1
                if tot_bgt == 0 : break
    
    if rnd_state: np.random.set_state(rnd_state)
    if tot_bgt > 0 : print("WARNING: ", inspect.currentframe().f_code.co_name,
                           " cannot fulfill all the classes. ",
                           budget_per_class)
    return rst

if __name__ == '__main__':
    x = [1.51937983, 4.8979591 , 2.29166665, 2.98666665, 4.40449431, 1.47639485]
    print("with weights =", x)
    print("and budget = ", 100)
    print("budget per class is: ")
    bgt = calc_budget_per_class(100, x)
    print(bgt)
    data_dir = "/home/yren/data/cosmo_data/npy" 
    train_data = cosmo_data.Cosmo3D(data_dir, transform=cosmo_data.np_norm)
    exclusion = [3, 12, 31, 153]
    ans = sample_given_budget_excluding(bgt, train_data, exclusion)
    print("selected idx:")
    print(ans)

