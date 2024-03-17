## Transformer architecture
N_HEAD = 32
N_LAYER = 8
N_EMBD = 256
# N_KV_HEAD = 4
N_FF = N_EMBD * 4

PATCH_SIZE = 4

## Diffusion architecture
T_MAX = 400
BETA_MIN = 1e-4
BETA_MAX = 0.02
SCHEDULE = "linear"

## Data and training
DATASET = "MNIST"
BATCH_SIZE = 64
LR = 1e-3

IMG_CHANNELS = 1 if DATASET == "MNIST" else 3
IMG_SIZE = 28 if DATASET == "MNIST" else 32

## Batch/step-level hyperparameters
EPOCHS = 100
SAVE_EVERY = int(EPOCHS * 0.05)  # save model every x epochs
GENERATE_EVERY = int(EPOCHS * 0.05)  # generate image from model every x epochs

### ––––TEMP–––– ###
SAVE_EVERY = 1
GENERATE_EVERY = 1
### ––––TEMP–––– ###

## Step-level hyperparameters
PRINT_EVERY = 100  # print loss every x steps

## Model loading
CHECKPOINT = True  # False
LOAD_EPOCH = 29  # None
START_EPOCH = LOAD_EPOCH if LOAD_EPOCH is not None else 0

MODEL_NAME = (  ### lacks prefix such as "dit_cfg", "ada_ln_patel_dit_cfg", added in individual notebooks
    f"{N_LAYER}_LAYERs_{N_HEAD}_HEADs_{N_EMBD}_EMBD_DIM_{T_MAX}_TMAX_{DATASET}"
)

N_CLASS = 10
LABEL = 0  ## when generating images of a single digit

UNCOND_LABEL = (
    N_CLASS  ### unconditional token. Note this will yield n_class <- n_class + 1
)
## cannot use -1, which gives nn.Embedding error
LABEL_DROPOUT = 0.2

TRAIN_GUIDANCE = 5  ## for regular CFG, set both to 5. If Patel CFG loss, use train guidance 1 and sample 3
SAMPLE_GUIDANCE = 5

#### 5-10 all look good
# 0 = unconditional
# 1 = conditional
# > 1 = guidance

path = "./"
