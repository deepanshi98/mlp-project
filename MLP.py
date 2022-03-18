#%%
import min2net
from min2net.preprocessing.MBDEPV1 import time_domain


min2net.utils.load_raw("MBDEPV1")
time_domain.subject_dependent_setting(
    k_folds=1,
    pick_smp_freq=128,
    bands=[4, 50],
    order=5,
    save_path="datasets",
    num_class=10,
)
#%%
# from min2net.utils import DataLoader
# loader = VisualDataLoader(dataset='MBDEPV1',
#                     train_type='subject_dependent',
#                     subject=1,
#                     data_format='NDTC',
#                     data_type='visual',
#                     dataset_path='datasets',
#                     num_class = 10,
#                     )
# #%%
# # load dataset
# EEG_train, MNIST_train, y_train = loader.load_train_set(fold=1)
# EEG_val, MNIST_val, y_val = loader.load_val_set(fold=1)
# EEG_test, MNIST_test, y_test = loader.load_test_set(fold=1)

# from min2net.model import BrainVizNet
# model = BrainVizNet(epochs = 200, input_shape=(1,200,2), num_class=10, monitor='val_loss', shuffle=True)
# model.fit(EEG_train, MNIST_train, y_train, EEG_val, MNIST_val, y_val)
# Y, evaluation = model.predict(EEG_test, MNIST_test, y_test)
# # %%
