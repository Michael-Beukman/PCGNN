import copy
import datetime
import pprint
import re
from typing import Dict, List
import glob
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import bz2

METHOD_NAME = 'PCGNN (Ours)'



def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def clean_name(name: str) -> str:
    """
    Changes a word/few words to lower case and separated by _ instead of spaces.
    Args:
        name (str): 

    Returns:
        str:
    """
    return '_'.join(name.split(' ')).lower()



def mypartial(func, *args):
    """Attempts to replicate functools.partial, but return the name as well.

    Args:
        func (function): The function to use.
    """
    
    def new_func(*temp_args):
        return func(*args, *temp_args)
    
    new_func.__name__ = func.__name__
    return new_func

def _bold_extreme_values(data, format_string="%.2f", max_=True, rows=None):
    # Thanks: https://flopska.com/highlighting-pandas-to_latex-output-in-bold-face-for-extreme-values.html
    def X(s):
        return float(s.split(' (')[0])
    temp = rows.map(X)
    current = X(data)
    if max_:
        good = current == temp.max()
    else:
        good = current == temp.min()
    if good:
        return "\\textbf{" + data + "}"
    else:
        return data

def bold_pandas_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """This simply adds textbf to the extrema values of a table. Usually max, instead if the column name contains 'time'

    Args:
        df (pd.DataFrame):

    Returns:
        pd.DataFrame:
    """
    df = df.T
    for col in df.columns.get_level_values(0).unique():
        df[col] = df[col].apply(lambda data : _bold_extreme_values(data, max_='time' not in col.lower(), rows = df[col]))
    df = df.T
    return df


def get_only_solvable_metrics(li: List[float], solvabilities: List[int]) -> List[float]:
    """This simply returns a list where all unsolvable levels are not taken into consideration.
        Precisely, if len(li) == len(solvabilities), then we return li[solvabilities > 0]
        Otherwise we assume it was generated by calculating all the pairwise values.

    Args:
        li (List[float]): [description]
        solvabilities (List[int]): [description]

    Returns:
        List[float]: Corrected list
    """
    N = len(solvabilities)
    if len(li) == N:
        return np.array(li)[np.array(solvabilities) > 0]
    
    elif len(li) == N * (N - 1) // 2:
        clean_list = []
        I = 0
        for i in range(N):
            for j in range(N)[i+1:]:
                # the diversities were calculated between
                # i and j.
                if solvabilities[i] > 0 and solvabilities[j] > 0:
                    # only now we can use this datapoint.
                    clean_list.append(li[I])
                I += 1
        return clean_list
    elif 100 < len(li) < N * (N - 1) // 2:
        return li
    else:
        raise Exception(f"Invalid lengths, {len(li)} {N}")

def do_statistical_tests_and_get_df(main_dic: Dict[str, Dict[str, float]], stats_dic: Dict[str, Dict[str, List[float]]], default_moniker: str) -> pd.DataFrame:
    """This does some statistical tests as well as return a dataframe that can easily be exported to Latex.

    Args:
        main_dic (Dict[str, Dict[str, float]]): [description]
        stats_dic (Dict[str, Dict[str, List[float]]]): [description]
        default_moniker (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    pprint.pprint(stats_dic)
    # what tests do we have?
    # - Wilcoxon
    # - Mann Whitney
    # - t test
    from scipy.stats import wilcoxon, ttest_ind, friedmanchisquare, mannwhitneyu, shapiro

    dic_for_stats_df = copy.deepcopy(stats_dic)
    dic_for_stats_df_temp = copy.deepcopy(stats_dic)

    for moniker, dic_of_vals in stats_dic.items():
        for metric_key, other_method in dic_of_vals.items():
            mode = 'two-sided'
            if metric_key == 'Generation Time (s)':
                print("HEY one sided")
                mode = 'less'

            neat_baseline = stats_dic[default_moniker][metric_key]
            _, p_is_normal = shapiro(other_method)
            
            _, p_is_normal_2 = shapiro(neat_baseline)
            if moniker == default_moniker:
                dic_for_stats_df[moniker][metric_key] = f"- | Pn={np.round(p_is_normal, 2):<5}"
                dic_for_stats_df_temp[moniker][metric_key] = {'d': 0, 'p_to_use': 1, 'is_t_test': True}
            else:
                t_t, p_t = ttest_ind(neat_baseline, other_method, equal_var=False, alternative=mode)
                u_m, p_m = mannwhitneyu(neat_baseline, other_method, alternative=mode)
                cohens_d = ((np.mean(other_method) - np.mean(neat_baseline)) / np.std(neat_baseline))
                # Pw={np.round(p_w, 2):<6} | 
                dic_for_stats_df[moniker][metric_key] = f"Pt={np.round(p_t, 2):5.2f} | Pm={np.round(p_m, 2):5.2f} | Pn={np.round(p_is_normal, 2):5.2f} | D={np.round(cohens_d, 2):5.2f}"
                

                # Use the mann whitney test if the data is not normal, else do.
                if p_is_normal < 0.05 or p_is_normal_2 < 0.05:
                    is_t_test = False
                    correct_p_to_use = p_m
                else:
                    correct_p_to_use = p_t
                    is_t_test = True
                
                dic_for_stats_df_temp[moniker][metric_key] = {'d': cohens_d, 'p_to_use': correct_p_to_use, 'is_t_test': is_t_test}

                
                # if correct_p_to_use

    # Now do finner test thingy.
    for metric_key in main_dic[default_moniker]:
        list_for_this_key = []
        for name in dic_for_stats_df_temp:
            T = dic_for_stats_df_temp[name][metric_key]
            list_for_this_key.append((T['p_to_use'], T['d'], name))
        
        # sort ascending, remove our method here
        list_for_this_key = sorted(list_for_this_key)[:-1]
        gamma = 0.95
        # print("GAMMAS: ",[1 - gamma**((i+1)/(len(list_for_this_key)-1)) for i, p in enumerate(list_for_this_key)])
        # find i
        Xs = [i for i, p in enumerate(list_for_this_key) if p[0] > 1 - gamma**((i+1)/(len(list_for_this_key)-1))]
        i = (min(Xs) - 1) if len(Xs) else len(list_for_this_key) - 1
        names_to_reject = [list_for_this_key[j][2] for j in range(i+1)]
        print(f"For key = {metric_key:<50}, Xs = {Xs} we have pis = {np.round([p[0] for p in list_for_this_key], 2)} and i = {i}. Hence we reject {names_to_reject}.")
        for name in main_dic:
            if not dic_for_stats_df_temp[name][metric_key]['is_t_test']:
                main_dic[name][metric_key] = "+" + main_dic[name][metric_key]
        for j in range(i+1):
            p, d, name = list_for_this_key[j]
            v = main_dic[name][metric_key]

            if p < 0.05:
                s = ""
                if p < 0.01:
                    s += "*"
                    if p < 0.001:
                        s += "*"
                if abs(d) >= 0.8: s+= "\dagger"
                if '$' in v:
                    re.sub('\$(.*?)\$', r'$\mathbf{\1}$', v)
                v = r"\textbf{" + v + "}"
                v = "$" + v  + "^{" + s + "}"+ "$"

            main_dic[name][metric_key] = v

    df = pd.DataFrame(main_dic)
    def make_cols_good(df):
        cols = list(df.columns)
        i = cols.index(default_moniker)
        cols = [default_moniker] + cols[:i] + cols[i+1:]
        df = df[cols]
        return df

    df = make_cols_good(df)
    
    df_stats = pd.DataFrame(dic_for_stats_df)
    df_stats = make_cols_good(df_stats)
    return df, df_stats


def mysavefig(*args, **kwargs):
    args = list(args)
    kwargs['dpi'] = 400
    plt.savefig(*args, **kwargs)
    if 0:
        args[0] = args[0].split(".png")[0] + ".pdf"
        plt.savefig(*args, **kwargs)
        args[0] = args[0].split(".pdf")[0] + ".eps"
        plt.savefig(*args, **kwargs)
    
    
def my_fig_savefig(fig, *args, **kwargs):
    args = list(args)
    kwargs['dpi'] = 400
    fig.savefig(*args, **kwargs)
    args[0] = args[0].split(".png")[0] + ".pdf"
    fig.savefig(*args, **kwargs)
    if 0:
        args[0] = args[0].split(".pdf")[0] + ".eps"
        fig.savefig(*args, **kwargs)
        
    
def get_latest_folder(s):
    # Returns the folder that is latest, which will be determined by the value of the date in the path, hence sorting is valid.
    return max(glob.glob(s))



# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data
