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

# Download and setup Dakshina dataset
DAKSHINA_DATASET_DIR="/input/data/dakshina_gen/dakshina_dataset_v1.0"
DAKSHINA_TAR_URL="https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"

# Download dataset if not already present
if [ ! -d "$DAKSHINA_DATASET_DIR" ]; then
    echo "Downloading Dakshina dataset..."
    mkdir -p "/input/data/dakshina_gen"
    wget -O "/input/data/dakshina_gen/dakshina_dataset_v1.0.tar" "$DAKSHINA_TAR_URL"
    
    echo "Extracting dataset..."
    tar -xf "/input/data/dakshina_gen/dakshina_dataset_v1.0.tar" -C "/input/data/dakshina_gen"
    
    echo "Dataset downloaded and extracted to: $DAKSHINA_DATASET_DIR"
else
    echo "Dakshina dataset already exists at: $DAKSHINA_DATASET_DIR"
fi

# Set environment variable for the task to find the dataset
export DAKSHINA_DATASET_PATH="$DAKSHINA_DATASET_DIR"

# LANGS=(hi bn gu kn ml mr pa ta te ur si ne)

OUTDIR="/output/llmspeech/dakshina_gen"
RESDIR="$OUTDIR/results"
LOGDIR="$OUTDIR/log"

mkdir -p $RESDIR $LOGDIR

for model in "${MODELS[@]}"
do 

	### for specific tasks
	# task_str="$(printf "dakshina_%s_in " "${LANGS[@]}")"
	# task_str="${task_str% }"   # remove trailing space
	# echo "$task_str"

	# Run all dakshina tasks (both _in and _latn for all languages)
	task_str=dakshina_gen

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
		--seed 42 > $LOGDIR/${model_str}_dakshina_gen.log 2>&1
done
