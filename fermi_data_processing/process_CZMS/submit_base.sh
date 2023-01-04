#!/bin/bash
#
#SBATCH --job-name=__REPLACE_RUN_NAME__
#
#SBATCH --time=5:00:00 
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=yitians@mit.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --error=/work/submit/yitians/fermi/fermi-prob-prog/fermi_data_processing/process_CZMS/slurm/sbatch.%x.%j.err
#SBATCH --output=/work/submit/yitians/fermi/fermi-prob-prog/fermi_data_processing/process_CZMS/slurm/sbatch.%x.%j.out 

source /home/submit/yitians/.bashrc
conda activate fermitools

###############
## config
RUN_NAME=__REPLACE_RUN_NAME__

DATA_DIR=/data/submit/yitians/fermi/fermidata/exposure/pass8_573weeks
WORK_DIR=/work/submit/yitians/fermi/fermi-prob-prog/fermi_data_processing/process_CZMS

TEMPLATE_FILE=/data/submit/yitians/fermi/templates/CZMS/GALACTIC_DIFFUSE_EMISSION_MAPS_0p25deg/"$RUN_NAME"_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits
IN_FILE=$WORK_DIR/tmp/"$RUN_NAME"_in.fits
XML_FILE=$WORK_DIR/tmp/"$RUN_NAME".xml
OUT_FILE=/data/submit/yitians/fermi/templates/CZMS_exposed/573weeks_ultracleanveto_bestpsf/"$RUN_NAME".fits
###############

cd $WORK_DIR

## pre-process
python $WORK_DIR/pre_process_CZMS.py $TEMPLATE_FILE $IN_FILE

## build xml
mkdir -p $WORK_DIR/tmp
sed "s|__REPLACE_WITH_TEMPLATE__|$IN_FILE|" $WORK_DIR/../base.xml > $XML_FILE

## run
OPTS="srcmdl=$XML_FILE \
bexpmap=$DATA_DIR/exposure-edge-ultracleanveto-bestpsf.fits \
expcube=$DATA_DIR/livetime_ultracleanveto_bestpsf.fits \
evtype=32 \
chatter=4"

punlearn gtsrcmaps
time gtsrcmaps $OPTS \
    cmap=$DATA_DIR/ccube_ultracleanveto_bestpsf_halfsky1.fits \
    outfile=$OUT_FILE irfs="P8R3_ULTRACLEANVETO_V3" # not necessary to do halfsky2
    
## post-process
python $WORK_DIR/post_process_CZMS.py $OUT_FILE