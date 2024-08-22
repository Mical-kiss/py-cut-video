from get_voice import get_voice
from splice_voice import splice_voice
from splice_video import splice_video

import argparse
parser = argparse.ArgumentParser(description='命令行参数示例')
parser.add_argument('--input', '-i', type=str, help='输入文件')
args = parser.parse_args()

get_voice(filename = args.input)
splice_voice(filename = args.input)
splice_video(filename = args.input)

