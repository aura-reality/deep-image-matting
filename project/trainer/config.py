img_rows, img_cols = 320, 320
# img_rows_half, img_cols_half = 160, 160
channel = 3
batch_size = 8
epochs = 50
patience = 10
num_bgs_per_fg = 10
training_fraction = 0.8

num_samples = 50880
num_train_samples = 40704
num_valid_samples = 10176

unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2
skip_crop = False
add_noise = True

reuse_backgrounds = True
composite_backgrounds = True
loss_ratio = .5 #mix between alpha-loss and compositional-loss

epochs_per_dataset = 15 # should generally be 1

#*********
env = 'remote' #'local' or 'remote'
#************

if env == 'remote':

	bucket = 'secret-compass-237117-mlengine-us-west-1'
	# path to provided foreground images
	fg_base_path = 'gs://%s/all_data/fg/' % bucket 
	# path to provided alpha mattes
	a_base_path = 'gs://%s/all_data/mask/' % bucket
	# Path to background images (MSCOCO)
	bg_base_path = 'gs://%s/all_data/bg/' % bucket
	# Path to folder where you want the composited images to go
	out_path = 'data/merged/'
	train_names_path = 'gs://%s/all_data/ten_bgs/train_names.txt' % bucket
	valid_names_path = 'gs://%s/all_data/ten_bgs/valid_names.txt' % bucket
	vgg16_weights_remote_path = 'gs://%s/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5' % bucket
	vgg16_weights_local_path = './cache/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	fg_names_path = 'gs://%s/all_data/ten_bgs/fg_names.txt' % bucket
	bg_names_path = 'gs://%s/all_data/ten_bgs/bg_names.txt' % bucket 
	#checkpoint_models_path = 'gs://%s/models/checkpoints' % bucket

if env == 'local': 

	bucket = 'secret-compass-237117-mlengine-us-west-1'
	# path to provided foreground images
	fg_base_path = '../all_data/fg/'
	# path to provided alpha mattes
	a_base_path = '../all_data/mask/'
	# Path to background images (MSCOCO)
	bg_base_path = '../all_data/bg/'
	# Path to folder where you want the composited images to go
	out_path = '../all_data/merged/'
	train_names_path = '../all_data/train_names.txt'
	valid_names_path = '../all_data/valid_names.txt'
	vgg16_weights_remote_path = 'gs://%s/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5' % bucket
	vgg16_weights_local_path = '../models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	fg_names_path = '../all_data/fg_names.txt'
	bg_names_path = '../all_data/bg_names.txt'
	#checkpoint_models_path = '../models'

