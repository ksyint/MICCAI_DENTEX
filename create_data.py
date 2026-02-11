"""
Create example data files for demonstration
"""

import os
import random


def create_example_data_files(output_dir='./data', num_classes=10, 
                              num_labeled=100, num_unlabeled=1000, 
                              num_test=200):
    """
    Create example data list files
    
    Args:
        output_dir: directory to save data files
        num_classes: number of classes
        num_labeled: number of labeled samples
        num_unlabeled: number of unlabeled samples
        num_test: number of test samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    labeled_file = os.path.join(output_dir, 'labeled.txt')
    with open(labeled_file, 'w') as f:
        for i in range(num_labeled):
            label = random.randint(0, num_classes - 1)
            f.write(f'tooth_labeled_{i:04d}.jpg\t{label}\n')
    
    print(f'Created labeled data file: {labeled_file}')
    print(f'  - {num_labeled} labeled samples')
    
    unlabeled_file = os.path.join(output_dir, 'unlabeled.txt')
    with open(unlabeled_file, 'w') as f:
        for i in range(num_unlabeled):
            f.write(f'tooth_unlabeled_{i:04d}.jpg\n')
    
    print(f'Created unlabeled data file: {unlabeled_file}')
    print(f'  - {num_unlabeled} unlabeled samples')
    
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for i in range(num_test):
            label = random.randint(0, num_classes - 1)
            f.write(f'tooth_test_{i:04d}.jpg\t{label}\n')
    
    print(f'Created test data file: {test_file}')
    print(f'  - {num_test} test samples')
    
    print('\nExample data files created successfully!')
    print(f'Directory: {output_dir}')
    print('\nNote: These are just data list files.')
    print('You need to prepare actual images in the specified paths.')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create example data files')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='directory to save data files')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--num_labeled', type=int, default=100,
                        help='number of labeled samples')
    parser.add_argument('--num_unlabeled', type=int, default=1000,
                        help='number of unlabeled samples')
    parser.add_argument('--num_test', type=int, default=200,
                        help='number of test samples')
    
    args = parser.parse_args()
    
    create_example_data_files(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        num_labeled=args.num_labeled,
        num_unlabeled=args.num_unlabeled,
        num_test=args.num_test
    )
