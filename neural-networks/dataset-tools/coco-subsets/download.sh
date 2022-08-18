#!/bin/bash

function printUsage() {
	echo "$0 <urls-file>"
}

function realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

if [ $# -lt 1 ]; then
	printUsage
	exit 1
fi

URLS_FILE=$(realpath $1)
IMGS_OUT_DIR=../../inputs/rand_coco_subset_100

mkdir -p $IMGS_OUT_DIR
cd $IMGS_OUT_DIR

while IFS= read -r url;
do
	curl -O $url;
done < "$URLS_FILE"

echo "`ls -l $IMGS_OUT_DIR | wc -l` images successfully downloaded to $IMGS_OUT_DIR"
