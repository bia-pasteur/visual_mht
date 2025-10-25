set -e

expyrun configs/train/trasein.yml --training.epochs 300
expyrun configs/train/sinetra.yml --training.epochs 300

# Train with ResNet (~ similar performance)
# expyrun configs/train/trasein.yml --training.epochs 300 --model.backbone ResNet32
# expyrun configs/train/sinetra.yml --training.epochs 300 --model.backbone ResNet32

# What about a smaller model
# expyrun configs/train/trasein.yml --training.epochs 300 --model.backbone PreActResNet20
# expyrun configs/train/sinetra.yml --training.epochs 300 --model.backbone PreActResNet20

# Let's see the impact of pseudo tracking augmentation
expyrun configs/train/trasein.yml --training.epochs 300 --dataset.delta_t 0
expyrun configs/train/sinetra.yml --training.epochs 300 --dataset.delta_t 0


# Let's see the impact of random augmentation
expyrun configs/train/trasein.yml --training.epochs 300 --dataset.elastic.prob 0.0 --dataset.affine.prob 0.0 -dataset.erase.prob 0.0 --dataset.motion_blur.prob 0.0 --dataset.blur_shot_noise 0.0 --dataset.jitter 0.0
expyrun configs/train/sinetra.yml --training.epochs 300 --dataset.elastic.prob 0.0 --dataset.affine.prob 0.0 -dataset.erase.prob 0.0 --dataset.motion_blur.prob 0.0 --dataset.blur_shot_noise 0.0 --dataset.jitter 0.0
