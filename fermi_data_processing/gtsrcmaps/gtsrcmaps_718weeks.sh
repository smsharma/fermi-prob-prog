###############
## config
RUN_NAME=test
#TEMPLATE=/zfs/nrodd/GalPropMap/Models/ccwapibrem.fit
TEMPLATE=/zfs/tslatyer/galactic/fermi/diffusemodel/gll_iem_v02_P6_V11_DIFFUSE.fit
IRFS="P8R3_ULTRACLEANVETO_V3"

###############

## directories
DATA_DIR=/zfs/tslatyer/fermidata/exposure/pass8_718weeks
OUT_DIR=/zfs/yitians/fermi/fermi-prob-prog/data/exposed_templates/test
mkdir -p $OUT_DIR
WORK_DIR=/zfs/yitians/fermi/fermi-prob-prog/fermi_data_processing

## build xml
mkdir -p $WORK_DIR/tmp
XML_FILE=$WORK_DIR/tmp/$RUN_NAME.xml
sed "s|REPLACE_WITH_TEMPLATE|$TEMPLATE|" $WORK_DIR/base.xml > $XML_FILE

## run
conda activate fermi

OPTS="srcmdl=$XML_FILE \
bexpmap=$DATA_DIR/REBIN_exposure_edge_ultracleanveto_bestpsf_part0_joined.fits \
expcube=$DATA_DIR/REBIN_livetime_ultracleanveto_bestpsf_part0.fits \
evtype=32 \
chatter=4"

OUT_FILE=$OUT_DIR/$RUN_NAME"_ultracleanveto_bestpsf_halfsky1.fits"

punlearn gtsrcmaps

gtsrcmaps $OPTS \
    cmap=$DATA_DIR/REBIN_ccube_ultracleanveto_bestpsf_part0_halfsky1.fits \
    outfile=$OUT_FILE irfs=$IRFS
# not necessary to do halfsky2
    
## post process
python $WORK_DIR/post_process.py $OUT_FILE