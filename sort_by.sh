#!/bin/bash
###############################################################################
# Sort the generated descriptions in the 'generations' folder after           #
# various metrics:                                                            #
# * ./sort_by.sh bleu                                                         #
# * ./sort_by.sh co_distance                                                  #
# * ./sort_by.sh rg_precision                                                 #
# * ./sort_by.sh rg_number                                                    #
# * ./sort_by.sh cs_precision                                                 #
# * ./sort_by.sh cs_recall                                                    #
###############################################################################


scored_files=()

find_by_bleu() {
    for file in generations/*; do
        scored_files+="'$(sed '/## Content Plan/q' $file | head -n -3 | tail -n 1 | grep -Po '\d+\.\d+') $(basename $file)' "
    done
}
find_by_co_distance() {
    for file in generations/*; do
        scored_files+="'$(sed '/## Content Plan/q' $file | head -n -3 | tail -n 2 | head -n 1 | grep -Po '\d+\.\d+%') $(basename $file)' "
    done
}
find_by_rg_precision() {
    for file in generations/*; do
        scored_files+="'$(sed '/## Content Plan/q' $file | head -n -3 | tail -n 3 | head -n 1 | grep -Pom 1 '\d+\.\d+%') $(basename $file)' "
    done
}
find_by_rg_number() {
    for file in generations/*; do
        scored_files+="'$(sed '/## Content Plan/q' $file | head -n -3 | tail -n 3 | head -n 1 | grep -Pom 1 '\d+\.\d+$') $(basename $file)' "
    done
}
find_by_cs_precision() {
    for file in generations/*; do
        scored_files+="'$(sed '/## Content Plan/q' $file | head -n -3 | tail -n 4 | head -n 1 | grep -Po '\d+\.\d+%' | xargs | awk '{print $1}') $(basename $file)' "
    done
}
find_by_cs_recall() {
    for file in generations/*; do
        scored_files+="'$(sed '/## Content Plan/q' $file | head -n -3 | tail -n 4 | head -n 1 | grep -Po '\d+\.\d+%$') $(basename $file)' "
    done
}

case $1 in
    bleu)
        find_by_bleu
    ;;
    co_distance)
        find_by_co_distance
    ;;
    rg_precision)
        find_by_rg_precision
    ;;
    rg_number)
        find_by_rg_number
    ;;
    cs_precision)
        find_by_cs_precision
    ;;
    cs_recall)
        find_by_cs_recall
    ;;
esac

echo $scored_files | xargs -n 1 | sort -hr
