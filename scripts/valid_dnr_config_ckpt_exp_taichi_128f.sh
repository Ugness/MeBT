#!/bin/bash
EXP=$3
LENGTH=128
for RUN in 0 1 2 3 4
do
    for N_DRAFT in 32
    do
        for CTEMP in 4.0
        do
            apython sample_vqgan_transformer_videos.py --base $1 \
            --gpt_ckpt $2 --exp_name $EXP --vid_c_temp $CTEMP --total_length $LENGTH --vid_n_steps $N_DRAFT \
            --context_size $LENGTH --step_size $LENGTH --verbose --dataset taichi --no_phase --n_sample 512 --run $RUN --batch_size 4 --save_videos \
            --decoding_strategy maskgit --top_k 32 \
            --save_codemap --bootstrap 64 --save_n 20
            apython measure_fvd_with_numpy.py --batch_size 16 --compute_fvd \
                --np_file results/${EXP}/numpy_files_${LENGTH}/taichi/VID_n_steps${N_DRAFT}_k32_temp1.0_ctemp${CTEMP}linear_maskgit_cosine_no_phase_run${RUN}.npy \
                --data_path datasets/vqgan_data/taichi_fvd --image_folder --sequence_length $LENGTH --n_sample 512 --resolution 128
        done
    done
    for CTEMP in 4.0
    do
        for N_DRAFT in 32
        do
            for M in 4
            do
                for STEPS in 2
                do
                    for TEMP in 0.1
                    do
                        python draft_and_revise_videos.py --base $1 \
                            --gpt_ckpt $2 --exp_name $EXP --total_length $LENGTH \
                            --n_revise $STEPS --M $M --revise_t $TEMP \
                            --np_draft results/${EXP}/numpy_files_${LENGTH}/taichi/VID_n_steps${N_DRAFT}_k32_temp1.0_ctemp${CTEMP}linear_maskgit_cosine_no_phase_run${RUN}_codemap.npy \
                            --context_size $LENGTH --step_size $LENGTH --verbose --dataset taichi --no_phase --n_sample 512 --run $RUN --batch_size 4 --save_videos --save_n 20 --save_codemap
                        apython measure_fvd_with_numpy.py --batch_size 16 --compute_fvd \
                            --np_file results/${EXP}/numpy_files_${LENGTH}/taichi/VID_dnr_nd${N_DRAFT}_dt0.0_nr${STEPS}_rt${TEMP}_M${M}_ctemp${CTEMP}_run${RUN}.npy \
                            --data_path datasets/vqgan_data/taichi_fvd --image_folder --sequence_length $LENGTH --n_sample 512 --resolution 128
                    done
                done
            done
        done
    done
done
