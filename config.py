# ── Project-wide configuration ────────────────────────────────────────────

# Data paths
CSV_PATH      = '../data/maestro-v3.0.0/maestro-v3.0.0.csv'
BASE_FOLDER   = '../data/maestro-v3.0.0/'

# Piano-roll settings
SAMPLE_RATE   = 16        # frames per second
SEQ_LENGTH    = 100       # time steps per training window
N_PITCHES     = 128       # MIDI pitch range

# Model hyperparameters
HIDDEN_SIZE   = 256
LATENT_SIZE   = 64
N_HEAD        = 8
N_LAYERS      = 3
DROPOUT       = 0.1

# Training settings
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 50
WARMUP_EPOCHS = 40        # KL annealing warmup (VAE only)

# Generation settings
GEN_STEPS     = 500       # time steps to generate
TOPK          = 3         # notes active per time step during export