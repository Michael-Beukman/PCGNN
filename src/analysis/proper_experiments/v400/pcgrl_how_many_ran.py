def main():
    dic = {
        'wide': [
            'logs/pipelines/all_pcgrl/pcgrl_smb_wide_5.mscluster41.144549.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_wide_4.mscluster40.144548.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_wide_3.mscluster18.144536.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_wide_2.mscluster39.144547.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_wide_1.mscluster14.144391.out',
        ],
    'turtle': [
            'logs/pipelines/all_pcgrl/pcgrl_smb_turtle_5.mscluster34.144546.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_turtle_3.mscluster26.144544.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_turtle_4.mscluster32.144545.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_turtle_1.mscluster24.144542.out',
            'logs/pipelines/all_pcgrl/pcgrl_smb_turtle_2.mscluster25.144543.out',
    ]
    }

    for key, li_of_files in dic.items():
        steps = []
        for l in li_of_files:
            with open(l, 'r') as f:
                lines = [l.strip() for l in f.readlines()]
                lines = [int(l.split(' ')[0]) for l in lines if ' timesteps' in l][-1]
                steps.append(lines)
            K = [f'{s:1.2e}' for s in steps]
        print(f"For {key:<10}, average number of timesteps = {sum(steps) / len(steps):1.3e} and steps = {K}")
if __name__ == '__main__':
    main()