from itertools import combinations, islice


def bootstrap_index_generator_nostaticVal(numbers: int, length: int, num_iters: int):
    if length>len(numbers):
        print('error #$#$$#$&%')
    perms = islice(combinations(numbers, length),num_iters)
    permutation_sets = list(perms)
    list_2 = []
    for ii in range(len(permutation_sets)):
        list_2.append(tuple([id for id in list(numbers) if id not in permutation_sets[ii]]))
    return permutation_sets, list_2
