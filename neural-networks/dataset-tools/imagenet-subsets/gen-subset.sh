#!/bin/bash

function printUsage() {
	echo "$0 <imgs-list-file>"
}

function realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

if [ $# -lt 1 ]; then
	printUsage
	exit 1
fi

IMGS_OUT_DIR=../../inputs/ILSVRC2012_val_100
IMGS_LIST_FILE=$(realpath $1)

mkdir -p $IMGS_OUT_DIR
rm $IMGS_OUT_DIR/*

while IFS= read -r img_path;
do
	cp $img_path $IMGS_OUT_DIR;
done < "$IMGS_LIST_FILE"

echo "`ls -l $IMGS_OUT_DIR | wc -l` images successfully copied to $IMGS_OUT_DIR"
