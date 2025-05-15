import pandas as pd
import data_helper
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

df = pd.read_csv('data/generated_data/test_data_0.csv')
total_structure_score = 0
total_actionability_score = 0
errors = {'CONTEXT': 0, 'SE': 0, 'RE': 0, 'TE': 0, 'ME': 0, 'IE': 0}
for index, samples in df.iterrows():
  scores = get_step_structure_score(samples['error_poem'], samples['step_content'])
  if scores['error'] and scores['actionability_score'] > 0: errors[scores['error']] += 1
  if scores['error'] == 'SE': print(samples['step_content'])
  total_structure_score += scores['structure_score']
  total_actionability_score += scores['actionability_score']
print(errors)
print(total_structure_score/len(df))
print(total_actionability_score/len(df))

# print(get_step_structure_score("""<sop> Nhà tao ăn cưới con người,
# Cớ sao chú nói những lời chay ma?
# Có đường mi cút thật xa,
# Mi còn đứng đó ắt là rùi trâu. <eop> <reasoning_memory> Tóm tắt ngữ cảnh: Bài thơ thể hiện sự tức giận, mỉa mai của người nói đối với một người khác vì đã nói những lời không phù hợp trong một dịp vui (ăn cưới). <eois>""", """<error> TE
# <desc> Lỗi thanh điệu dòng 3. Tiếng thứ 6 ""xa"" là thanh bằng, trong khi yêu cầu là thanh bằng. Điều này phá vỡ luật bằng trắc của thơ lục bát, gây khó chịu cho người đọc.
# <reason> Cần thay thế từ ""xa"" ở vị trí thứ 6 của dòng 3 bằng một từ thanh trắc để đảm bảo luật thơ.
# <action> rui xa <replace> rùi trâu <line> 4 <index> 7
# <effect> Đã sửa lỗi thanh điệu ở dòng 3, tiếng thứ 6, giúp câu thơ trở nên hài hòa hơn.
# <eois>"""))