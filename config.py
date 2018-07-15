# Name of the training dataset and location of all datasets
dataset_name = 'textures_all'
data_folder = './data/'

# Path to folder with checkpoints
saves_folder = './saves/'

# If this is set to False, the training will be skipped.
# One image will be generated using existing weights instead (checkpoint needed).
train = True

# width and height of images
w, h = 512, 512

# Dimension of random vector, needed to generate images
# Typically should be set to 100
z_dimension = 100

# Range for random values, for z dimension vector
z_dim_range = [-0.5, 0.5]

# generator's and  discriminator filters,
# these are multiplied on each layer to upsample and downsample images
generator_filters = 64
discriminator_filters = 64

# Alpha for LeakyRelu activation
alpha = 0.2

# NOT YET IMPLEMENTED
# Dropout enabled or disabled
# Droput rate (if enabled)
# e.g '0.1' droput rate would drop out 10% of input units
dropout = True
dropout_rate = 0.2

# Smoothing enabled or disabled
# Multiplier (?)
# Other values should be tested
smoothing = True
zero_smoothing = [0., 0.3]
one_smoothing = [0.7, 1.2]

# Learning Rate
d_lr = 0.0002
g_lr = 0.0002

# Momentum, pretty important variable,
# other values should be tested as well
d_beta1 = 0.5
g_beta1 = 0.5

# training_steps is number of training iterations with random training samples, taken from dataset
batch_size = 8
training_steps = 1000000