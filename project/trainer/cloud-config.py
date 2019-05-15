img_rows, img_cols = 320, 320
# img_rows_half, img_cols_half = 160, 160
channel = 4
batch_size = 8
epochs = 1000
patience = 50
num_bgs_per_fg = 1
training_fraction = 0.8
num_samples = 368
num_train_samples = 295
# num_samples - num_train_samples
num_valid_samples = 73
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2
skip_crop = True
reuse_backgrounds = True
w_l = 0.5
#if composite_backgrounds = False, w_l should be set to 1 (don't use compositional loss w/o compositing)
composite_backgrounds = True


##############################################################
# Set your paths here

bucket = 'secret-compass-237117-mlengine'

# path to provided foreground images
fg_path = 'gs://%s/data/fg/' % bucket

# path to provided alpha mattes
a_path = 'gs://%s/data/mask/' % bucket

# Path to background images (MSCOCO)
bg_path = 'gs://%s/data/bg/' % bucket

# Path to folder where you want the composited images to go
out_path = 'data/merged/'

train_names_path = 'gs://%s/data/train_names.txt' % bucket

valid_names_path = 'gs://%s/data/valid_names.txt' % bucket

vgg16_weights_remote_path = 'gs://%s/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5' % bucket

vgg16_weights_local_path = './cache/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

fg_names_path = 'gs://%s/data/fg_names.txt' % bucket

bg_names_path = 'gs://%s/data/bg_names.txt' % bucket 
