#!/bin/bash

# Use:
# ./autoslurm_v2.sh --name (run name) [--resume] [--overcap]
# will always request an a40 and 6 cores, and use the ei-lab partition by default

RESUME=false
PARTITION="ei-lab"
LOGDIR="./logs"
NAME="null"

GETNAME=false

GREEN=$'\e[0;32m'
NC=$'\e[0m'

for arg in "$@"; do
    if [[ "$arg" == "--name" ]]; then
        GETNAME=true
        continue
    fi
    if $GETNAME; then
        NAME="$arg"
        GETNAME=false
        continue
    fi
    if [[ "$arg" == "--resume" ]]; then
        RESUME=true
        continue
    fi
    if [[ "$arg" == "--overcap" ]]; then
        PARTITION="overcap"  
        continue
    fi
    echo "Unknown argument: $arg"
    exit 1
done

if [ "$NAME" == "null" ]; then
    echo "Please provide a name for the run with --name. (So we can identify the run to resume later)"
    exit 1
fi

if $RESUME; then
    LOGDIR=$(find ./logs/${NAME}-* -maxdepth 0 -type d -printf "%T@ %p\n" | sort -nr | head -n 1 | awk '{print $2}')
    uuid=$(basename "$LOGDIR")
    echo "Resuming run with logdir: ${GREEN}$LOGDIR${NC} from uuid: ${GREEN}$uuid${NC}"
else
    uuid=$(uuidgen | tr -d '-')
    LOGDIR="./logs/${NAME}-$uuid"
    mkdir "$LOGDIR"
    echo "Starting run from scratch with logdir: ${GREEN}$LOGDIR${NC} from uuid ${GREEN}$uuid${NC}"
fi

REALLOGDIR=$(realpath $LOGDIR)

export LOGDIR=${LOGDIR}
export NAME=${NAME}
# sbatch --error "$REALLOGDIR/stderr.out" --output "$REALLOGDIR/stdout.out" --partition $PARTITION slurm/entrypoint.sh

echo $NAME
