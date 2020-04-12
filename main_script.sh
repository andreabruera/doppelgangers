#!/bin/bash

TRAINING_MODE=$1
NUMBER_AT_A_TIME=$2
S=1
NOVELS_FOLDER=/import/cogsci/andrea/github/novel_aficionados_dataset/novels
TRAINING_FOLDER=${TRAINING_MODE}_training
mkdir -p ${TRAINING_FOLDER}
echo 'Created folder: '${TRAINING_FOLDER}
for novel in $(ls ${NOVELS_FOLDER});
    do
    CURRENT_FOLDER=${TRAINING_FOLDER}/${novel}
    mkdir ${CURRENT_FOLDER}
    cp -r ${NOVELS_FOLDER}/${novel}/test_files/* ${CURRENT_FOLDER}
    for filename in $(ls ${CURRENT_FOLDER}/novel);
    do
        if [[ ${filename} == *no_header* ]]; 
        then
            echo ${filename}
            BOOK_NUMBER=${filename/'_no_header.txt'/''}
        fi
    done

    echo 'Starting with novel ' ${novel} 
    echo 'Number: ' ${BOOK_NUMBER}
    python3 experiment.py --on ${TRAINING_MODE}  --folder ${CURRENT_FOLDER} --number ${BOOK_NUMBER} --write_to_file & 
    S=$(($S+1))
    if (($S<=${NUMBER_AT_A_TIME}));
        then
        echo 'novel number '${S}' from the current batch'
    else
        S=1
        wait
    fi
done

python3 tests.py --folder ${TRAINING_MODE}_training
#wait
