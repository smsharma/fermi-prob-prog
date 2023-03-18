###############
## config
RUN_NAME=$1
TEMPLATE=$2
IRFS="P8R3_ULTRACLEANVETO_V3"

###############

echo "running on computer: $COMPUTER_NAME."
## directories
if [ "$COMPUTER_NAME" = "erebus" ]; then
    DATA_DIR=/zfs/tslatyer/fermidata/exposure/pass8_573weeks
    OUT_DIR=/zfs/yitians/fermi/fermi-prob-prog/data/exposed_templates
    WORK_DIR=/zfs/yitians/fermi/fermi-prob-prog/fermi_data_processing
elif [ "$COMPUTER_NAME" = "submit" ]; then
    DATA_DIR=/work/submit/yitians/fermi/data/pass8_573weeks
    OUT_DIR=/work/submit/yitians/fermi/fermi-prob-prog/data/exposed_templates/CZMS
    WORK_DIR=/work/submit/yitians/fermi/fermi-prob-prog/fermi_data_processing
else
    echo "unknown computer."
    return
fi

## build xml
mkdir -p $WORK_DIR/tmp
XML_FILE=$WORK_DIR/tmp/$RUN_NAME.xml
sed "s|REPLACE_WITH_TEMPLATE|$TEMPLATE|" $WORK_DIR/base.xml > $XML_FILE

## run
if [ "$COMPUTER_NAME" = "erebus" ]; then
    conda activate fermi
elif [ "$COMPUTER_NAME" = "submit" ]; then
    conda activate fermitools
fi

OPTS="srcmdl=$XML_FILE \
bexpmap=$DATA_DIR/exposure-edge-ultracleanveto-bestpsf.fits \
expcube=$DATA_DIR/livetime_ultracleanveto_bestpsf.fits \
evtype=32 \
chatter=4"

OUT_FILE=$OUT_DIR/$RUN_NAME"_ultracleanveto_bestpsf_halfsky1.fits"

punlearn gtsrcmaps

gtsrcmaps $OPTS \
    cmap=$DATA_DIR/ccube_ultracleanveto_bestpsf_halfsky1.fits \
    outfile=$OUT_FILE irfs=$IRFS
# not necessary to do halfsky2
    
## post process
#python $WORK_DIR/post_process.py $OUT_FILE

