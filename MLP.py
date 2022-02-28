import min2net
from min2net.preprocessing import MBD2117


# min2net.utils.load_raw('MBD2117')
MBD2117.time_domain.subject_dependent_setting(k_folds=5,
                                                 pick_smp_freq=100, 
                                                 bands=[8, 30], 
                                                 order=5, 
                                                 save_path='datasets')