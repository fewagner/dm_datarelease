#!/bin/sh
# execution script for SLURM cluster 

# Calculation times are roughly (min):
# a 10 
# b 10 
# c 20
# ab 30 
# bc 60
# ac 60
# abc 120


for which in a b c ab bc ac abc
do
  for (( idx=0; idx<=5; idx++))
  do
    echo "Submitting job ${which} nmbr ${idx}"
    JOB_NAME="Limit_${which}_${idx}"
    SCRIPT_PATH="/users/felix.wagner/dm_datarelease/lhlimitcalc.py"
    CONTAINER_PATH="/users/felix.wagner/cait_v1_1_0_latest.sif"
    OUTPUT_FILE="/users/felix.wagner/outputs/${JOB_NAME}.out"
    SBATCH_OPTIONS=" -c 1 --mem=4G --output=${OUTPUT_FILE} --job-name=${JOB_NAME} --time=480"
    SINGULARITY_OPTIONS=" -c -B /eos/ -H /users/felix.wagner"
    PYTHON_OPTIONS=" -u "
    CMD_ARGUMENTS=" -i ${idx} -n 40 -w ${which} -p /users/felix.wagner/dm_datarelease/"

    sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"
  done
done

exit 0
