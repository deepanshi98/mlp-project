#%%
import min2net
from min2net.preprocessing.MBDMNIST.v16Cut2 import time_domain, visual


min2net.utils.load_raw('MBDMNIST16Cut2')
visual.subject_dependent_setting(k_folds=1,
                                pick_smp_freq=100, 
                                bands=[8, 30], 
                                order=5, 
                                save_path='datasets', 
                                num_class = 10
)
#%%
from min2net.utils import VisualDataLoader
loader = VisualDataLoader(dataset='MBDMNIST16Cut2', 
                    train_type='subject_dependent', 
                    subject=1, 
                    data_format='NDTC',
                    data_type='visual', 
                    dataset_path='datasets',
                    num_class = 10,
                    )
#%%
# load dataset
EEG_train, MNIST_train, y_train = loader.load_train_set(fold=1)
EEG_val, MNIST_val, y_val = loader.load_val_set(fold=1)
EEG_test, MNIST_test, y_test = loader.load_test_set(fold=1)

from min2net.model import BrainVizNet
model = BrainVizNet(epochs = 200, input_shape=(1,200,2), num_class=10, monitor='val_loss', shuffle=True)
model.fit(EEG_train, MNIST_train, y_train, EEG_val, MNIST_val, y_val)
Y, evaluation = model.predict(EEG_test, MNIST_test, y_test)
# %%
