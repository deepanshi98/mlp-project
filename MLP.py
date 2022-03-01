import min2net
from min2net.preprocessing.MBDMNIST.v16Cut2 import time_domain


min2net.utils.load_raw('MBDMNIST16Cut2')
time_domain.subject_dependent_setting(k_folds=2,
                                        pick_smp_freq=100, 
                                        bands=[8, 30], 
                                        order=5, 
                                        save_path='datasets', 
                                        num_class = 10
                                        )