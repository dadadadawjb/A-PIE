import argparse
import itertools
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='the path to the collected data')
    parser.add_argument('--grasp', action='store_true', help='whether contain grasp data')
    args = parser.parse_args()

    fns = sorted(list(itertools.chain(*[list(Path(args.dataset_path).glob('*/*/*.npz'))])))

    num_data = len(fns)
    avg_point = 0
    if args.grasp:
        num_grasp = 0
        num_fail_grasp = 0
        num_none_success_grasp = 0
        num_half_success_grasp = 0
        num_both_success_grasp = 0
        avg_score = 0
        avg_fail_score = 0
        avg_none_success_score = 0
        avg_half_success_score = 0
        avg_both_success_score = 0
    for fn in fns:
        if args.grasp:
            data = np.load(fn, allow_pickle=True)
        else:
            data = np.load(fn)

        pc = data['point_cloud']
        avg_point += pc.shape[0]

        if args.grasp:
            grasp = data['grasp']

            num_grasp += grasp.shape[0]
            num_fail_grasp += grasp[grasp[:, -1] == -1].shape[0]
            num_none_success_grasp += grasp[grasp[:, -1] == 0].shape[0]
            num_half_success_grasp += grasp[grasp[:, -1] == 1].shape[0]
            num_both_success_grasp += grasp[grasp[:, -1] == 2].shape[0]

            avg_score += np.sum(grasp[:, 0])
            avg_fail_score += np.sum(grasp[grasp[:, -1] == -1, 0])
            avg_none_success_score += np.sum(grasp[grasp[:, -1] == 0, 0])
            avg_half_success_score += np.sum(grasp[grasp[:, -1] == 1, 0])
            avg_both_success_score += np.sum(grasp[grasp[:, -1] == 2, 0])
    avg_point = avg_point / num_data
    if args.grasp:
        avg_score = avg_score / num_grasp if num_grasp > 0 else None
        avg_fail_score = avg_fail_score / num_fail_grasp if num_fail_grasp > 0 else None
        avg_none_success_score = avg_none_success_score / num_none_success_grasp if num_none_success_grasp > 0 else None
        avg_half_success_score = avg_half_success_score / num_half_success_grasp if num_half_success_grasp > 0 else None
        avg_both_success_score = avg_both_success_score / num_both_success_grasp if num_both_success_grasp > 0 else None
    
    print('num_data:', num_data)
    print('avg_point:', avg_point)
    if args.grasp:
        print('num_grasp:', str(num_fail_grasp) + ' + ' + str(num_none_success_grasp) + ' + ' + str(num_half_success_grasp) + ' + ' + str(num_both_success_grasp) + ' = ' + str(num_grasp))
        print('avg_score:', '( ' + str(avg_fail_score) + ', ' + str(avg_none_success_score) + ', ' + str(avg_half_success_score) + ', ' + str(avg_both_success_score) + ' ) = ' + str(avg_score))
