## OpenVoice-cli

**This fork, does not generate voice from text, it only uses the 2nd stage of voice2voice. Therefore, you need to have a sample and a voice already prepared**

[Paper](https://arxiv.org/abs/2312.01479) |
[Website](https://research.myshell.ai/open-voice) 

## About

The second stage of OpenVoice "Tone color extractor" is used, via console or python scripts.

Feel free to make PRs or use the code for your own needs

## Demo

https://github.com/daswer123/OpenVoice-cli/assets/22278673/7b4255eb-7797-4370-825a-81f2c67c8f90

## Changelog

You can keep track of all changes on the [release page](https://github.com/daswer123/OpenVoice-cli/releases)

## TODO
- [x] Batch generation via console
- [x] Possibility to use inference import through code

## Installation

Simple installation :

```bash
pip install openvoice-cli
```

**Attention is used Pytorch version 1.13.1 higher unfortunately can not, waiting for an update from the authors of openvoice**

This will install all the necessary dependencies, including a **CPU support only** version of PyTorch

I recommend that you install the **GPU version** to improve processing speed ( up to 3 times faster )

Read the end of the README to learn how to install.

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install openvoice-cli
pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

### Linux
```bash
python -m venv venv
source venv\bin\activate
pip install openvoice-cli
pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

## Usage

This tool supports both single file and batch processing for audio tone color conversion using a reference audio file. Below are the commands for each mode of operation:

### Single File Processing

```bash
python -m openvoice_cli single -i INPUT -r REF [-o OUTPUT] [-d DEVICE]
```

### Batch Processing

```bash
python -m openvoice_cli batch -id INPUT_DIR -rf REF_FILE [-od OUTPUT_DIR] [-d DEVICE] [-of OUTPUT_FORMAT]
```

### Options

#### Common Options

- `-h`, `--help`:  
  Show this help message and exit.

#### Single File Processing Options

- `-i INPUT`, `--input INPUT` (mandatory):  
  Path to the input audio file.

- `-r REF`, `--ref REF` (mandatory):  
  Path to the reference audio file for tone color extraction.

- `-o OUTPUT`, `--output OUTPUT`:  
  Designate the output path for the converted audio file. By default, the output will be saved as "out.wav" in the current directory.

- `-d DEVICE`, `--device DEVICE`:  
  Specify the device to use for processing; defaults to 'cpu'. Can be set to a CUDA device with 'cuda:0' if supported and desired.

#### Batch Processing Options

- `-id INPUT_DIR`, `--input_dir INPUT_DIR` (mandatory):  
  Input directory containing audio files to process.

- `-rf REF_FILE`, `--ref_file REF_FILE` (mandatory):  
  Reference audio file path.

- `-od OUTPUT_DIR`, `--output_dir OUTPUT_DIR`:  
  Output directory for converted audio files. Defaults to "outputs".

- `-d DEVICE`, `--device DEVICE`:  
  Specify the processing device. Defaults to 'cuda' if available.

- `-of OUTPUT_FORMAT`, `--output_format OUTPUT_FORMAT`:  
  Output file format (e.g., ".wav"). Defaults to ".wav".

### Example Commands via console

**Single file processing**

```bash
python -m openvoice_cli single -i ./test/test.wav -r ./test/ref.wav -o ./test/ready.wav
```

**Batch processing**

```bash
python -m openvoice_cli batch -id ./test/input_folder -rf ./test/ref.wav -od ./test/output_folder -of .mp3
```

### Example via Python Code

For integrating the audio tone color conversion capabilities into your Python code, you can import and use the `tune_one` and `tune_batch` functions provided by the `openvoice_cli`. Here are some examples on how to invoke these functions in a Python script:

**Single File Conversion**

```python
from openvoice_cli import tune_one

# Set parameters for single file processing
input_file = 'path_to_input.wav'
ref_file = 'path_to_reference.wav'
output_file = 'path_to_output.wav'
device = 'cpu'  # or 'cuda:0' for GPU processing

# Convert the tone color of a single audio file
tune_one(input_file=input_file, ref_file=ref_file, output_file=output_file, device=device)
```

**Batch Processing**

```python
from openvoice_cli import tune_batch

# Set parameters for batch processing
input_dir = 'path_to_input_directory'
ref_file = 'path_to_reference.wav'
output_dir = 'path_to_output_directory'
device = 'cuda'  # or 'cpu' for CPU processing
output_format = '.wav'  # could be .mp3 or other formats

# Convert the tone color of multiple audio files in a directory
output = tune_batch(input_dir=input_dir, ref_file=ref_file, output_dir=output_dir, device=device, output_format=output_format)
```

In these examples:
- Replace `'path_to_input.wav'`, `'path_to_reference.wav'`, and `'path_to_output.wav'` with the actual file paths for your input, reference, and output audio files respectively.
- Replace `'path_to_input_directory'` and `'path_to_output_directory'` with the actual directories containing your input audio files and where you want the converted files to be saved.
- The `device` parameter allows you to specify whether to perform processing using the CPU (`'cpu'`) or GPU (`'cuda:0'`). Ensure that your environment supports CUDA before attempting to use GPU acceleration.

## License
This repository is licensed under MIT License

Original repository is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which prohibits commercial usage. **MyShell reserves the ability to detect whether an audio is generated by OpenVoice**, no matter whether the watermark is added or not.
