import sys, os
sys.path.insert(0, os.path.abspath('..'))
import re
from library.utils import StatisticResult, statistic_test
from collections import defaultdict
import pandas as pd
from typing import final
from library.RecRunner import NameType, RecRunner
from library.constants import METRICS_PRETTY, RECS_PRETTY, experiment_constants, CITIES_PRETTY,DATA,UTIL
import argparse
import app_utils

LATEX_HEADER = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{underscore}
\usepackage[margin=0.5in]{geometry}
\usepackage{booktabs}
\begin{document}
"""

LATEX_FOOT = r"""
\end{document}"""

argparser =argparse.ArgumentParser()
app_utils.add_cities_arg(argparser)
app_utils.add_base_recs_arg(argparser)
app_utils.add_final_recs_arg(argparser)
args = argparser.parse_args()
# cities = ['lasvegas', 'phoenix']
# base_recs = [ 'usg','geosoca','geomf',]
# final_recs = ['geodiv']
final_rec_list_size = experiment_constants.K
rr=RecRunner(args.base_recs[0],args.final_recs[0],args.cities[0],experiment_constants.N,final_rec_list_size,DATA)

metrics_k = experiment_constants.METRICS_K
final_recs_metrics= defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
base_recs_metrics= defaultdict(lambda: defaultdict(lambda: defaultdict()))
latex_table = ""
main_metrics = ['precision','recall','gc','ild','pr','epc']
def get_metrics_renamed_order(METRICS_PRETTY_k):
    METRICS_PRETTY_k = [METRICS_PRETTY_k[v] for v in main_metrics]
    return METRICS_PRETTY_k
def get_metrics_pretty_k(metric_k:int) -> dict:
    return {k: v+f'@{metric_k}' for k, v in METRICS_PRETTY.items()}
def df_format(df_unf:pd.DataFrame,metric_k):
    # METRICS_PRETTY_k = {k: v+f'@{metric_k}' for k, v in METRICS_PRETTY.items()}
    METRICS_PRETTY_k = get_metrics_pretty_k(metric_k)
    df_unf=df_unf[main_metrics].rename(columns=METRICS_PRETTY_k)
    # print(df_unf)
    return df_unf

def get_base_name(base_name):
  return RECS_PRETTY[base_name]

def get_final_name(base_name,final_name):
  return get_base_name(base_name)+'+'+RECS_PRETTY[final_name]
  
for city in args.cities:
  for base_rec in args.base_recs:
    rr.city = city
    rr.base_rec = base_rec
    metrics = rr.load_metrics(
        base=True, name_type=NameType.PRETTY, METRICS_KS=metrics_k)
    for metric_k in metrics_k:
      base_recs_metrics[city][base_rec][metric_k] = pd.DataFrame(
          metrics[metric_k])
      base_recs_metrics[city][base_rec][metric_k]=df_format(base_recs_metrics[city][base_rec][metric_k],metric_k)
    
    for final_rec in args.final_recs:
      rr.final_rec = final_rec
      metrics = rr.load_metrics(
          base=False, name_type=NameType.PRETTY, METRICS_KS=metrics_k)
      for metric_k in metrics_k:
        final_recs_metrics[city][base_rec][final_rec][metric_k] = pd.DataFrame(
            metrics[metric_k])
        final_recs_metrics[city][base_rec][final_rec][metric_k] =df_format(final_recs_metrics[city][base_rec][final_rec][metric_k],metric_k)
num_metrics = 6
num_columns= num_metrics+1
latex_table_header= """
\\begin{{tabular}}{{{}}}
""".format('l'*(num_columns))
latex_table_footer= r"""
\end{tabular}
"""

top_count = defaultdict(lambda:defaultdict(int))
for count1, city in enumerate(args.cities):
  if count1 == 0:
    latex_table += '\\toprule\n'
  latex_table += '\\multicolumn{{{}}}{{l}}{{{}}}\\\\\n'.format((num_columns),CITIES_PRETTY[city])
  latex_table += '\\bottomrule\n'
  for metric_k in metrics_k:
    dfs = []
    names_recs_in_order = []
    for base_rec in args.base_recs:
      current_metrics = {}
      current_metrics[get_base_name(base_rec)] = base_recs_metrics[city][base_rec][metric_k]
      names_recs_in_order.append(get_base_name(base_rec))
      for final_rec in args.final_recs:
        current_metrics[get_final_name(base_rec,final_rec)] = final_recs_metrics[city][base_rec][final_rec][metric_k]
        names_recs_in_order.append(get_final_name(base_rec,final_rec))
      df = pd.concat(current_metrics, axis=1)
      # print(df)
      dfs.append(df)
    df = pd.concat(dfs,axis=1)
    # print(df)

    df_reordered = df.reorder_levels([1,0],axis=1)
    # print(df_reordered)
    df_reordered_means = df_reordered.mean()
    top_methods = df_reordered.mean().reset_index().set_index('level_1').groupby('level_0').idxmax().to_dict()[0]
    df_top2_methods = df_reordered.copy()
    for k, v in top_methods.items():
      df_top2_methods=df_top2_methods.drop((k,v),axis=1)
    top2_methods = df_top2_methods.mean().reset_index().set_index('level_1').groupby('level_0').idxmax().to_dict()[0]
    highlight_elements = defaultdict(list)
    metrics_og_name = {v: k for k,v in get_metrics_pretty_k(metric_k).items()}
    # print(metrics_og_name)
    names_recs_to_og = {}
    for base_rec in args.base_recs:
        names_recs_to_og[get_base_name(base_rec)] = base_rec
        for final_rec in args.final_recs:
          names_recs_to_og[get_final_name(base_rec,final_rec)] = (base_rec,final_rec)
    for k,v in top_methods.items():
      top1_values = df_reordered[k,v]
      v_top2 = top2_methods[k]
      top2_values =df_reordered[k,v_top2]
      statistic_result= statistic_test(top1_values,top2_values,0.05)
      if statistic_result == StatisticResult.GAIN:
        highlight_elements[k].extend([v])
      elif statistic_result == StatisticResult.TIE:
        highlight_elements[k].extend([v,v_top2])
      elif statistic_result == StatisticResult.LOSS:
        highlight_elements[k].extend([v_top2])
      for hige in highlight_elements[k]:
          top_count[metrics_og_name[k]][names_recs_to_og[hige]] += 1
        
    
    df_reordered_means=df_reordered_means.map(lambda x: f'{x:.4f}')
    for metric, methods in highlight_elements.items():
      for method in methods:
        df_reordered_means.at[metric,method] = '\\textbf{{{}}}'.format(df_reordered_means.at[metric,method])
    table_df_result : pd.DataFrame = df_reordered_means.unstack(level=0)
    table_df_result=table_df_result[get_metrics_renamed_order(get_metrics_pretty_k(metric_k))]
    table_df_result=table_df_result.reindex(names_recs_in_order)

    table_df_result_latex = table_df_result.to_latex(header=True,escape=False)
    table_df_result_latex=re.sub('\{\}','Algorithm',table_df_result_latex)
    table_df_result_latex = table_df_result_latex.split('\n')[:-2][2:]
    table_df_result_latex = '\n'.join(table_df_result_latex)
    latex_table+=table_df_result_latex+'\n'
    # raise SystemError
    


table_top_count = pd.DataFrame.from_dict(top_count).fillna(0).astype(int)
table_top_count=table_top_count[main_metrics]
table_top_count.columns = [METRICS_PRETTY[i] for i in table_top_count.columns]
recs_order = []
new_names = {}
for base_rec in args.base_recs:
    recs_order.append(base_rec)
    new_names[base_rec] = get_base_name(base_rec)
    for final_rec in args.final_recs:
        recs_order.append((base_rec,final_rec))
        new_names[(base_rec,final_rec)] = get_final_name(base_rec,final_rec)
    
table_top_count = table_top_count.reindex(recs_order)
table_top_count=table_top_count.rename(index=new_names)
table_top_count = pd.concat([table_top_count,table_top_count.sum(axis=1)],axis=1)
table_top_count = table_top_count.rename(columns={0:'Total'})
table_top_count_latex = table_top_count.to_latex()

latex_table = LATEX_HEADER+latex_table_header+latex_table+latex_table_footer+ '\n'+table_top_count_latex+LATEX_FOOT
with open(DATA+'/'+UTIL+'benchmark_table.tex','w') as f:
  f.write(latex_table)
os.system(
  f"cd {DATA+'/'+UTIL} && latexmk -pdf -interaction=nonstopmode benchmark_table.tex"
)

