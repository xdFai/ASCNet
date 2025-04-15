"""
生成 .h5 数据文件
"""
import argparse
from dataset import prepare_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building the training patch database")
    parser.add_argument("--gray", default=True, action='store_true', help='prepare grayscale database instead of RGB')
    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=64, help="Patch size")
    parser.add_argument("--stride", "--s", type=int, default=40, help="Size of stride")
    parser.add_argument("--max_number_patches", "--m", type=int, default=180,
    # parser.add_argument("--max_number_patches", "--m", type=int, default=18,
                        help="Maximum number of patches")
    parser.add_argument("--aug_times", "--a", type=int, default=2,
                        help="How many times to perform data augmentation")
    # Dirs
    parser.add_argument("--trainset_dir", type=str, default=r"D:\SCI\03-SCI\dataset\All", help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default=r"D:\SCI\03-SCI\dataset\All_16\image",
                        help='path of validation set')
    parser.add_argument("--valset_dirty_dir", type=str, default=r"D:\SCI\03-SCI\dataset\All_16\nosie",
                        help='path of validation set')
    args = parser.parse_args()


    print("\n### Building databases ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    prepare_data(args.trainset_dir, args.valset_dir, args.valset_dirty_dir, args.patch_size, args.stride, args.max_number_patches,
                 aug_times=args.aug_times, gray_mode=args.gray)
