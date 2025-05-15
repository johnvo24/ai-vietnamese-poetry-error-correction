import pandas as pd
from utils import data_helper
import re

def get_step_structure_score(error_poem, step_content):
  step = None
  structure_score = 0
  actionability_score = 0
  try:
    step = data_helper.parse_step(step_str=step_content)
    structure_score = 1
    if step['error'] == 'CONTEXT':
      if step['action'] == '' and step['replace'] == '' and step['line'] == '' and step['index'] == '': actionability_score = 1
    elif step['error'] in ['SE', 'RE', 'TE', 'ME', 'IE']:
      poem_match = re.search(r"<sop>(.*?)<eop>", error_poem, re.DOTALL)
      poem = poem_match.group(1).strip() if poem_match else ""
      lines = poem.split("\n")
      line = re.sub(r'[^a-zA-ZÀ-Ỹà-ỹ\s]', '', lines[int(step['line'])-1].strip())
      replace = re.sub(r'[^a-zA-ZÀ-Ỹà-ỹ\s]', '', step['replace'].strip())
      if replace in line and replace.split()[0] == line.split()[int(step['index']) - 1]: actionability_score = 1
  except Exception as e:
    pass
  
  return {
    'error': step['error'] if step and step['error'] else None,
    'structure_score': structure_score,
    'actionability_score': actionability_score,
  }

# df = pd.read_csv('data/generated_data/test_data_0.csv')
def get_avg_structure_score(df: pd.DataFrame):
  total_structure_score = 0
  total_actionability_score = 0
  errors = {'CONTEXT': 0, 'SE': 0, 'RE': 0, 'TE': 0, 'ME': 0, 'IE': 0}
  for _, samples in df.iterrows():
    scores = get_step_structure_score(samples['error_poem'], samples['step_content'])
    if scores['error'] and scores['actionability_score'] > 0: errors[scores['error']] += 1
    total_structure_score += scores['structure_score']
    total_actionability_score += scores['actionability_score']

  return {
    'errors': errors,
    'avg_structure_score': total_structure_score/len(df),
    'avg_actionability_score': total_actionability_score/len(df),
  }

def filter_high_structure_score(df: pd.DataFrame):
  filtered_rows = []
  for _, row in df.iterrows():
    scores = get_step_structure_score(row['error_poem'], row['step_content'])
    if scores['structure_score'] == 1 and scores['actionability_score'] == 1:
      filtered_rows.append(row)
  return pd.DataFrame(filtered_rows)