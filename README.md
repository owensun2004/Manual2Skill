# Manual2Skill

This is the official repo for the paper "Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly Using Vision-Language Models."
It contains code for the 2 critical sections of Manual2Skill: (1) VLM-Guided Hierachical Assembly Graph Generation, and (2) Per-step Assembly Pose Estimation
Planning and Execution

![Demo Preview](assets/demo.gif) <!-- Replace with actual image path -->

---
## Installation

### Prerequisites
- CUDA

### Setup
1. Clone repository:
  ```bash
  git clone https://github.com/owensun2004/Manual2Skill.git
  cd Manual2Skill
  ```

2. Install dependencies:
```bash
conda create -n manual python=3.11
conda activate manual
pip install -r requirements.txt

# for blender rendering of variations in pre-assembly scene
wget https://download.blender.org/release/Blender3.6/blender-3.6.19-linux-x64.tar.xz
tar -xf blender-3.6.19-linux-x64.tar.xz
cd blender-3.6.19-linux-x64
sudo ln -s $(pwd)/blender /usr/local/bin/blender
cd ..
rm blender-3.6.19-linux-x64.tar.xz
```

3. OpenAI API Key
```bash
export OPENAIKEY="your-api-key"
```

## VLM-Guided Hierachical Assembly Graph Generation
This section includes scripts for running VLM inference on the 102 furniture, generating variations in pre-assembly scene, ablation studies, and different evaluation metrics for VLM generation results

```bash
cd VLM_assembly_graph_gen
```

### Data Preparation
To access the data for the 102 furniture items' manuals and pre-assembly scenes, simply download and unzip the ZIP folder
```bash
mkdir data
gdown https://drive.google.com/uc?id=1hPesH_zd_NMd842JGaXaUxkviLU2Th4L
unzip data.zip -d ./data
```

### VLM inference
To generate assembly graphs for each furniture, we need to run inference on the VLM. Please note that inferencing on each furniture item will approximately take 15 images and 1700 words as input, and will take 1-2 minutes to obtain the output assembly graph. To run VLM inference on all 102 furniture, you can use the following command. 
```bash
python inference/run.py 
```
This will output to a folder under `outputs` named with the current time of the run, which contains all the furniture items' predicted assembly graphs. Under each furniture, you will see a `tree.json` which stores the predicted assembly graph in a nested array.

You can also change some parameters such as the temperature, model, furniture items, and input manual type. For example, to run the assembly graph generation for the first two furniture items (Bench/applaro and Bench/applaro_2) using input manuals without numbers:

```bash
python inference/run.py --start 0 --end 2 --temperature 0.1 --model o1 --prompt_type not_numbered --debug

# --start and --end values from 0 to 102
# --temperature ranges from 0 to 1
# --model includes gpt-4o, gpt-4.5, o1, o3
# --prompt_type includes numbered and not_numbered
# Use --debug to see the full VLM inputs and outputs, this will save additional .txt and .json files for each stage of each furniture under the output folder of the current run
```

### Evaluate
First, make sure to change to the `eval` directory.

Assuming you are currently under `VLM_assembly_graph_gen`, type this command:
```bash
cd eval
```

To test the success rates of your generated assembly graphs, copy the name of your inference output folder to `--tree_dir`. For example, suppose your most recent inference output folder is named `2025_05_01_193302`, then you can type this command:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd) python manual_generation/test_accuracy.py --data_json ../data/main_data.json --parts_dir ../data/parts --part_features_pkl resources/features_dgcnn_1024_102.pkl --tree_dir 2025_05_01_193302

# --data_json, --parts_dir, --part_features_pkl are mandatory arguments
# --tree_dir specifies the generated assembly graphs to evaluate. It can be set to ours (which evaluates the assembly graph results reported in the paper's Table I), 'singlestep', 'geocluster' (which are the baselines), or the name of your custom generated assembly graph folder.
# --difficulty can be set to 'easy' for evaluating furniture with only 2-4 parts, 'medium' for 5-6 parts, 'hard' for 7-8 parts, 'impossible' for 9-19 parts, and 'all' for 2-19 parts
# To the predicted assembly graphs for each individual furniture, use --debug
```

Sometimes, you may encouter `json.decoder.JSONDecodeError: Extra data`, this is due to the VLM's limitations, which may sometimes output an incorrect json file. In this case, it is helpful to set the `--debug` parameter to see which furniture's assembly graph contains a wrong `tree.json` file, and edit the file accordingly.

Disclaimer: Because of the complexity of the prompts and the multi-stage VLM querying theme, the VLM may output different results even with the same inputs. This may result in slightly different success rates compared to the metrics reported in the paper. We expect the success rate to further increase as better VLMs are introduced.
### Pre-assembly scene generations
To generate variations of pre-assembly scenes, such as rotating and shuffling different parts, we must use blender. First, go back to `VLM_assembly_graph_gen` directory (assuming you are currently under `VLM_assembly_graph_gen/eval`):
```bash
cd ..
```

Then type this command:
```bash
python scene_gen/generator.py --rand_translate true --rand_rotate true

# if you want to randomly shuffle furniture parts, set --rand_translate true
# if you want to randomly rotate furniture parts in-place, set --rand_rotate true
# if you leave these two arguments empty, the generated scenes will be the same as the original scene_annotated.png provided in the data/preassembly_scenes folder
```

This will create `scene_rot.png` and `scene_rot_annotated.png` under the `data/preassembly_scenes` for each furniture item. Please note that since the rotations and translations are performed randomly, each run will produce different scenes.

To test the VLM's assembly graph generation performance for these scene variations, simply use the `--scene_type` argument for VLM inference:
```bash
python inference/run.py --scene_type not_original
```
And then use the same procedures for evaluation and getting the success rate as mentioned above.

## Per-step Assembly Pose Estimation

### Training

### Inference & Evaluation





