import os
import argparse
from tqdm import tqdm
from openvoice_cli.downloader import download_checkpoint
from openvoice_cli.api import ToneColorConverter
import openvoice_cli.se_extractor as se_extractor
import glob

def tune_one(input_file,ref_file,output_file,device):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    ckpt_converter = os.path.join(checkpoints_dir, 'converter')

    if not os.path.exists(ckpt_converter):
        os.makedirs(ckpt_converter, exist_ok=True)
        download_checkpoint(ckpt_converter)

    device = device

    tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))

    source_se, _ = se_extractor.get_se(input_file, tone_color_converter, vad=True)
    target_se, _ = se_extractor.get_se(ref_file, tone_color_converter, vad=True)

    # Ensure output directory exists and is writable
    output_dir = os.path.dirname(output_file)
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Run the tone color converter
    tone_color_converter.convert(
        audio_src_path=input_file,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_file,
    )

def tune_batch(input_dir, ref_file, output_dir=None, device='cpu', output_format='.wav'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    ckpt_converter = os.path.join(checkpoints_dir, 'converter')

    if not os.path.exists(ckpt_converter):
        os.makedirs(ckpt_converter, exist_ok=True)
        download_checkpoint(ckpt_converter)

    tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))

    target_se, _ = se_extractor.get_se(ref_file, tone_color_converter, vad=True)

    # Use default output directory 'out' if not provided
    if output_dir is None:
        output_dir = os.path.join(current_dir, 'out')
    os.makedirs(output_dir, exist_ok=True)

    # Check for any audio files in the input directory (wav, mp3, flac) using glob
    audio_extensions = ('*.wav', '*.mp3', '*.flac')
    audio_files = []
    for extension in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, extension)))
    
    for audio_file in tqdm(audio_files,"Tune file",len(audio_files)):
        # Extract source SE from audio file
        source_se, _ = se_extractor.get_se(audio_file, tone_color_converter, vad=True)

        # Run the tone color converter
        filename_without_extension = os.path.splitext(os.path.basename(audio_file))[0]
        output_filename = f"{filename_without_extension}_tuned{output_format}"
        output_file = os.path.join(output_dir, output_filename)
        
        tone_color_converter.convert(
            audio_src_path=audio_file,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_file,
        )
        print(f"Converted {audio_file} to {output_file}")

    return output_dir

def main_single(args):
    tune_one(input_file=args.input, ref_file=args.ref, output_file=args.output, device=args.device)

def main_batch(args):
    output_dir = tune_batch(
        input_dir=args.input_dir,
        ref_file=args.ref_file,
        output_dir=args.output_dir,
        device=args.device,
        output_format=args.output_format
    )
    print(f"Batch processing complete. Converted files are saved in {output_dir}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the tone color of audio files using a reference audio.')
    
    # Create subparsers for single and batch processing
    subparsers = parser.add_subparsers(help='commands', dest='command')
    
    # Single file conversion arguments
    single_parser = subparsers.add_parser('single', help='Process a single file')
    single_parser.add_argument('-i', '--input', help='Input audio file path', required=True)
    single_parser.add_argument('-r', '--ref', help='Reference audio file path', required=True)
    single_parser.add_argument('-o', '--output', default="out.wav", help='Output path for converted audio file')
    single_parser.add_argument('-d', '--device', default="cpu", help='Device to use (e.g., "cuda:0" or "cpu")')
    single_parser.set_defaults(func=main_single)

    # Batch processing arguments
    batch_parser = subparsers.add_parser('batch', help='Process a batch of files in a directory')
    batch_parser.add_argument('-id', '--input_dir', help='Input directory containing audio files to process', required=True)
    batch_parser.add_argument('-rf', '--ref_file', help='Reference audio file path', required=True)
    batch_parser.add_argument('-od', '--output_dir', help='Output directory for converted audio files', default="outputs")
    batch_parser.add_argument('-d', '--device', default="cuda", help='Device to use')
    batch_parser.add_argument('-of', '--output_format', default=".wav", help='Output file format (e.g., ".wav")')
    batch_parser.set_defaults(func=main_batch)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
