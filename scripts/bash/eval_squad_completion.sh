#!/bin/bash

# Use float32 dtype for V100 GPUs
dtype="float32"

## leval env
MODELS=(
	google/gemma-3-4b-it
	# sarvamai/sarvam-1
	# meta-llama/Llama-3.1-8B-Instruct
	# openai/gpt-oss-20b
	# google/gemma-3-12b-it
	# google/gemma-3-27b-it
)

## lmeval-phimm env
# MODELS=(
# microsoft/Phi-4-mini-instruct
# microsoft/Phi-4-multimodal-instruct
# /home/aiscuser/ankunchu/llmspeech/models/phi4-silica-7b-text-dpo-orig_then_msr-0419
# )

export IGB_RENDER_CHAT_TEMPLATE=1
export IGB_RENDER_CHAT_TEMPLATE_NAME="gemma3"

# LANGS=(hi ml en)

OUTDIR="/output/llmspeech/igb_xquad_in_gen"
RESDIR="$OUTDIR/results"
LOGDIR="$OUTDIR/log"

mkdir -p $RESDIR $LOGDIR

for model in "${MODELS[@]}"
do 

	### for specific tasks
	# task_str="$(printf "igb_xquad_in_gen_%s " "${LANGS[@]}")"
	# task_str="${task_str% }"   # remove trailing space
	# echo "$task_str"

	task_str=igb_xquad_in_gen

	echo "***** Running tasks: $task_str for model: $model"

	model_str=$(echo "$model" | sed 's#/#__#g')

	accelerate launch -m lm_eval --model hf \
		--model_args pretrained=$model,dtype="$dtype",trust_remote_code=True \
		--tasks $task_str	\
		--batch_size 4 \
		--limit 100 \
		--output_path $RESDIR \
		--log_samples \
		--write_out \
		--seed 42 > $LOGDIR/${model_str}_igb_xquad_in_gen.log 2>&1
done