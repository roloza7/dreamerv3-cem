#!/bin/bash

# Use:
# ./autoslurm_v2.sh -n[ame] (run name) -r[esume] -o[vercap]
# will always request an a40 and 6 cores, and use the ei-lab partition by default

RESUME=false
PARTITION="ei-lab"
LOGDIR="logs/"
NAME="null"
CONFIG="null"
CONFIG_PATH="config.yaml"

RED=$'\e[0;31m'
GREEN=$'\e[0;32m'
NC=$'\e[0m'

while getopts "n:roc:" arg; do
    if [[ $OPTARG =~ ^- ]]; then
        echo "optarg '${OPTARG}' cannot start with '-'"
        exit 1
    fi
    case $arg in
        n)
            NAME=$OPTARG
            ;;
        r)
            RESUME=true
            ;;
        o)
            PARTITION="overcap"
            ;;
        c)
            CONFIG=$OPTARG
            ;;
        \?)
            echo "Invalid option: $arg"
            exit 1
            ;;
    esac
done

if [ "$NAME" == "null" ]; then
    echo "Please provide a name for the run with --name. (So we can identify the run to resume later)"
    exit 1
fi

REALLOGDIR=$(realpath $LOGDIR)

echo "Absolute path to be sent to slurm: ${GREEN}${REALLOGDIR}${NC}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}"

if [[ $CONFIG == "null" ]]; then
    echo "${RED}Please provide a config to use with -c${NC}"
    exit 1
fi

CONFIGS=$(grep "^$CONFIG:" $CONFIG_PATH | cut -d' ' -f2-)

if [[ -z $CONFIGS ]]; then
    echo "${RED}Config not found in config.yaml${NC}"
    exit 1
fi

echo "Will be using configs: ${GREEN}${CONFIGS}${NC}"

if $RESUME; then
    LOGDIR=$(find logs/${NAME}-* -maxdepth 0 -type d -printf "%T@ %p\n" | sort -nr | head -n 1 | awk '{print $2}')
    uuid=$(basename "$LOGDIR")
    echo "Resuming run with logdir: ${GREEN}$LOGDIR${NC} from uuid: ${GREEN}$uuid${NC}"
else
    uuid=$(uuidgen | tr -d '-' | cut -c 1-8)
    LOGDIR="logs/${NAME}-$uuid"
    mkdir "$LOGDIR"
    echo "Starting run from scratch with logdir: ${GREEN}$LOGDIR${NC} from uuid ${GREEN}$uuid${NC}"
fi

echo "Will be using partition: ${GREEN}${PARTITION}${NC}"

export LOGDIR=${LOGDIR}
export NAME=${NAME}
export CONFIGS=${CONFIGS}
sbatch --error "$REALLOGDIR/stderr.out" --output "$REALLOGDIR/stdout.out" --partition $PARTITION slurm/entrypoint.sh