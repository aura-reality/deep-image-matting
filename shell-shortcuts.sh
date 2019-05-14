# Usage: Put the following in ~/.bash_profile
#
# export DEEP_MATTING_HOME="/path/to/deep-image-matting"
# . $DEEP_MATTING_HOME/shell-shortcuts.sh

alias activate_dm="cd $DEEP_MATTING_HOME && source venv/bin/activate"

# Accepts the job id as the first argument
tensorboard_remote() {
    tensorboard --logdir="gs://secret-compass-237117-mlengine/$1/logs"
}
