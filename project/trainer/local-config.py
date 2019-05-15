img_rows, img_cols = 320, 320
# img_rows_half, img_cols_half = 160, 160
channel = 3
batch_size = 8 
epochs = 1000
patience = 50
num_bgs_per_fg = 1
training_fraction = 0.8
num_samples = 27
num_train_samples = 22
# num_samples - num_train_samples
num_valid_samples = 5
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2
skip_crop = True
reuse_backgrounds = True
composite_backgrounds = False
loss_ratio = .5 #mix between alpha-loss and compositional-loss

##############################################################
# Set your paths here

bucket = 'secret-compass-237117-mlengine'

# path to provided foreground images
fg_path = '../data/fg/'

# path to provided alpha mattes
a_path = '../data/mask/'

# Path to background images (MSCOCO)
bg_path = '../data/bg/'

# Path to folder where you want the composited images to go
out_path = '../data/merged/'

train_names_path = '../data/train_names.txt'

valid_names_path = '../data/valid_names.txt'

vgg16_weights_remote_path = 'gs://%s/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5' % bucket

vgg16_weights_local_path = './cache/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

fg_names_path = '../data/fg_names.txt'

bg_names_path = '../data/bg_names.txt'
