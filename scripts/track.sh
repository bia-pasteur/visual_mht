set -e

# Springs 2D
$RUN_KOFT_ENV expyrun configs/track/sinetra.yml --video.scenario springs_2d --video.seed $@

# Hydra Flow
$RUN_KOFT_ENV expyrun configs/track/sinetra.yml --video.scenario hydra_flow --video.seed $@
