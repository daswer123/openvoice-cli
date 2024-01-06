import os
import argparse
from openvoice_cli.downloader import download_checkpoint
from openvoice_cli.api import ToneColorConverter
import openvoice_cli.se_extractor as se_extractor

def main(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    ckpt_converter = os.path.join(checkpoints_dir, 'converter')

    if not os.path.exists(ckpt_converter):
        os.makedirs(ckpt_converter, exist_ok=True)
        download_checkpoint(ckpt_converter)

    device = args.device

    tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))

    source_se, _ = se_extractor.get_se(args.input, tone_color_converter, vad=True)
    target_se, _ = se_extractor.get_se(args.ref, tone_color_converter, vad=True)

    # Ensure output directory exists and is writable
    output_dir = os.path.dirname(args.output)
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Run the tone color converter
    tone_color_converter.convert(
        audio_src_path=args.input,
        src_se=source_se,
        tgt_se=target_se,
        output_path=args.output,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the tone color of an audio file using a reference audio.')
    parser.add_argument('-i', '--input', help='Input audio file path', required=True)
    parser.add_argument('-r', '--ref', help='Reference audio file path', required=True)
    parser.add_argument('-d', '--device', default="cpu", help='Device to use (e.g., "cuda:0" or "cpu")')
    parser.add_argument('-o', '--output', default="out.wav", help='Output path for converted audio file')

    args = parser.parse_args()
    main(args)
