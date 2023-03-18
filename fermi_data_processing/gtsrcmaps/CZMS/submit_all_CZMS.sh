#!/bin/bash

COUNT=1
WORK_DIR=/work/submit/yitians/fermi/fermi-prob-prog/fermi_data_processing/process_CZMS

for TMPL_CODE in 'bs' '0w' '5q' '9l' '9z' 'bh' 'ba' '22' '23' 'ch' '24' '26' '2c' '2e' 'c4' '2g' '2i' '2p' '2q' '2r' '2s' '2t' '2u' '2v' '2w' '2y' '2z' '30' '31' '32' '3j' '3z' '5s' '67' '6e' '6n' '6z' '70' '71' '7c' '7n' '7o' '7p' '7t' '7u' '80' '8f' '8l' '8t' '93' 'au' 'ay' 'bf' 'cg' 'cv' 'd1' 'dy' '0a' '0b' '0c' '0g' '0t' '0u' '0v' '14' '1c' '1d' '1e' '1i' '1j' '1k' '1nB' '1oB' '1pB' '1qB' '1rB' '1sB' '1xB' '1yB' '1zB'; do
    for TMPL_TYPE in 'pi0' 'bremss' 'ICS'; do

        RUN_NAME="$TMPL_TYPE"_"$TMPL_CODE"
        
        sed "s|__REPLACE_RUN_NAME__|$RUN_NAME|" $WORK_DIR/submit_base.sh > $WORK_DIR/slurm/$RUN_NAME.sh
        
        echo -n "$COUNT: "
        
        sbatch $WORK_DIR/slurm/$RUN_NAME.sh
        #echo "$RUN_NAME"
        
        ((COUNT=COUNT+1))
        
    done
done