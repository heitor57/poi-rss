
from numpy.core import numeric
from lib.utils import StatisticResult, statistic_test
import scipy.stats
from collections import defaultdict
import pandas as pd
import sys, os
from typing import final
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import NameType, RecRunner
# rr=RecRunner("usg","geocat","madison",80,20,"../data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()
import inquirer
from lib.constants import METRICS_PRETTY, RECS_PRETTY, experiment_constants, CITIES_PRETTY

# questions = [
#   inquirer.Checkbox('city',
#                     message="City to use",
#                     choices=experiment_constants.CITIES,
#                     ),
#   inquirer.Checkbox('baser',
#                     message="Base recommender",
#                     choices=list(RecRunner.get_base_parameters().keys()),
#                     ),
#   inquirer.Checkbox('finalr',
#                     message="Final recommender",
#                     choices=list(RecRunner.get_final_parameters().keys()),
#                     ),
# ]

# answers = inquirer.prompt(questions)
# cities = answers['city']
# base_recs = answers['baser']
# final_recs = answers['finalr']

# cities = ['lasvegas', 'phoenix']
cities = ['lasvegas']
base_recs = ['geomf', 'usg']
# base_recs = ['geomf']
final_recs = ['geocat']
# print(cities)
# print(base_recs)
# print(final_recs)
final_rec_list_size = 20
rr=RecRunner(base_recs[0],final_recs[0],cities[0],80,final_rec_list_size,"../data")

# rr.print_latex_vert_cities_metrics_table(cities=city)
metrics_k = experiment_constants.METRICS_K
final_recs_metrics= defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
base_recs_metrics= defaultdict(lambda: defaultdict(lambda: defaultdict()))
latex_table = ""
num_metrics = None
for city in cities:
  for base_rec in base_recs:
    rr.city = city
    rr.base_rec = base_rec
    metrics = rr.load_metrics(
        base=True, name_type=NameType.PRETTY, METRICS_KS=metrics_k)
    for metric_k in metrics_k:
      METRICS_PRETTY_k = {k:v+f'@{metric_k}' for k,v in METRICS_PRETTY.items()}
      base_recs_metrics[city][base_rec][metric_k] = pd.DataFrame(
          metrics[metric_k]).rename(columns=METRICS_PRETTY_k)
  # rr.final_rec_list_size = final_rec_list_size
    
    for final_rec in final_recs:
      rr.final_rec = final_rec
      metrics = rr.load_metrics(
          base=False, name_type=NameType.PRETTY, METRICS_KS=metrics_k)
      for metric_k in metrics_k:
        METRICS_PRETTY_k = {k: v+f'@{metric_k}' for k, v in METRICS_PRETTY.items()}
        final_recs_metrics[city][base_rec][final_rec][metric_k] = pd.DataFrame(
            metrics[metric_k]).rename(columns=METRICS_PRETTY_k)
        
num_metrics = 6
latex_table_header= """
\\begin{{tabular}}{{{}}}
""".format('c'*num_metrics)
latex_table_footer= r"""
\end{tabular}
"""
for city in cities:
  latex_table += r'\toprule\n'
  latex_table += '\\multicolumn{{{}}}{{l}}{}\\\\\n'.format(num_metrics,CITIES_PRETTY[city])
  latex_table += r'\bottomrule\n'
  for metric_k in metrics_k:
    dfs = []
    for base_rec in base_recs:
      current_metrics = {}
      current_metrics[RECS_PRETTY[base_rec]] = base_recs_metrics[city][base_rec][metric_k].drop(columns='user_id')
      for final_rec in final_recs:
        current_metrics[RECS_PRETTY[base_rec]+'+'+RECS_PRETTY[final_rec]] = final_recs_metrics[city][base_rec][final_rec][metric_k].drop(columns='user_id')
      df = pd.concat(current_metrics, axis=1)
      dfs.append(df)
    df = pd.concat(dfs)

    df_reordered = df.reorder_levels([1,0],axis=1)
    top_methods = df_reordered.mean().reset_index().set_index('level_1').groupby('level_0').idxmax().to_dict()[0]
    df_top2_methods = df_reordered.copy()

    print(df_top2_methods)
    for k, v in top_methods.items():
      print(k,v)
      df_top2_methods=df_top2_methods.drop((k,v),axis=1)
    top2_methods = df_top2_methods.mean().reset_index().set_index('level_1').groupby('level_0').idxmax().to_dict()[0]
    print(top_methods)
    print(top2_methods)
    highlight_elements = defaultdict(list)
    for k,v in top_methods.items():
      top1_values = df_reordered[k,v]
      v_top2 = top2_methods[k]
      top2_values =df_reordered[k,v_top2]
      statistic_result= statistic_test(top1_values,top2_values,0.05)
      if statistic_test == StatisticResult.GAIN:
        highlight_elements[k].extend([v])
      elif statistic_test == StatisticResult.TIE:
        highlight_elements[k].extend([v,top2_values])
      else:
        highlight_elements[k].extend([top2_values])
        
    table_df_result = df_reordered.mean().unstack(level=0)
    table_df_result_latex = table_df_result.to_latex(header=False)
    table_df_result_latex = table_df_result_latex.split('\n')[:-2][2:]
    table_df_result_latex = '\n'.join(table_df_result_latex)
    latex_table+=table_df_result_latex+'\n'
    # raise SystemError
    
latex_table = latex_table_header+latex_table+latex_table_footer
print(latex_table)