set -e

# Springs 2D
expyrun configs/track/sinetra.yml --video.scenario springs_2d --video.seed $@

# Hydra Flow
expyrun configs/track/sinetra.yml --video.scenario hydra_flow --video.seed $@
