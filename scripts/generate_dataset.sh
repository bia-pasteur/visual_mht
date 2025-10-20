set -e

export SINETRA=~/workspace/pasteur/SINETRA/dataset/
export TRASE_IN=~/data/pasteur/hydra/movies/ali/doi_10_5061_dryad_h9w0vt4q3__v20230925/


# Train datasets
# expyrun configs/build_dataset/sinetra.yml --video.scenario hydra_flow --video.seed 111
# expyrun configs/build_dataset/sinetra.yml --video.scenario springs_2d --video.seed 111
# expyrun configs/build_dataset/sinetra.yml --video.scenario springs_2d --video.seed 222

# expyrun configs/build_dataset/trasein.yml --video.video_id contrxn-1
# expyrun configs/build_dataset/trasein.yml --video.video_id contrxn-2
# expyrun configs/build_dataset/trasein.yml --video.video_id contrxn-3

# Validation datasets
expyrun configs/build_validation_dataset/sinetra/hydra_flow_111.yml
expyrun configs/build_validation_dataset/sinetra/springs_2d_111.yml
expyrun configs/build_validation_dataset/sinetra/springs_2d_222.yml

# expyrun configs/build_validation_dataset/trasein/contrxn-1.yml
# expyrun configs/build_validation_dataset/trasein/contrxn-2.yml
# expyrun configs/build_validation_dataset/trasein/contrxn-3.yml

