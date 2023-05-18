#!/bin/bash
EXP=$3
LENGTH=16
for RUN in 0 1 2 3 4 5 6 7 8 9
do
    for N_DRAFT in 128
    do
        for CTEMP in 6.0
        do
            python sample_vqgan_transformer_videos.py --base $1 \
            --gpt_ckpt $2 --exp_name $EXP --vid_c_temp $CTEMP --total_length $LENGTH --vid_n_steps $N_DRAFT \
            --context_size $LENGTH --step_size $LENGTH --verbose --dataset ucf101 --no_phase --n_sample 2048 --run $RUN --batch_size 16 --save_videos \
            --decoding_strategy maskgit \
            --save_codemap --save_n 5
            python measure_fvd_with_numpy.py --batch_size 16 --compute_fvd \
                --np_file results/${EXP}/numpy_files_${LENGTH}/ucf101/VID_n_steps${N_DRAFT}_temp1.0_ctemp${CTEMP}linear_maskgit_cosine_no_phase_run${RUN}.npy \
                --data_path datasets/vqgan_data/ucf_128 --train --image_folder --sequence_length $LENGTH --n_sample 2048 --resolution 128
        done
    done
    for CTEMP in 6.0
    do
        for N_DRAFT in 128
        do
            for M in 4
            do
                for STEPS in 4
                do
                    for TEMP in 0.7
                    do
                        python draft_and_revise_videos.py --base $1 \
                            --gpt_ckpt $2 --exp_name $EXP --total_length $LENGTH \
                            --n_revise $STEPS --M $M --revise_t $TEMP \
                            --np_draft results/${EXP}/numpy_files_${LENGTH}/ucf101/VID_n_steps${N_DRAFT}_temp1.0_ctemp${CTEMP}linear_maskgit_cosine_no_phase_run${RUN}_codemap.npy \
                            --context_size $LENGTH --step_size $LENGTH --verbose --dataset ucf101 --no_phase --n_sample 2048 --run $RUN --batch_size 16 --save_videos --save_n 5
                        python measure_fvd_with_numpy.py --batch_size 16 --compute_fvd \
                            --np_file results/${EXP}/numpy_files_${LENGTH}/ucf101/VID_dnr_nd${N_DRAFT}_dt0.0_nr${STEPS}_rt${TEMP}_M${M}_ctemp${CTEMP}_run${RUN}.npy \
                            --data_path datasets/vqgan_data/ucf_128 --train --image_folder --sequence_length $LENGTH --n_sample 2048 --resolution 128
                    done
                done
            done
        done
    done
done
