def fixed_one_factor(fixed_indexs):
    length = len(fixed_indexs)

    fixed_factors = []
    for idx in fixed_indexs:
        full_factors = [i for i in range(length)]
        full_factors.remove(idx)
        fixed_factors.append(full_factors)
    return fixed_factors

def find_index_from_factors(factors, dataset):
    factor_dict = {}
    sampled_idx = []

    for i, classes in enumerate(dataset.latents_classes):
        factor_dict[classes.tobytes()] = i
    for factor in factors:
        sampled_idx.append(factor_dict[factor.tobytes()])
    return sampled_idx