import re
import json
import csv


def format_step(step: dict, is_last_step: bool) -> str:
  """
  Convert step (dict) to string.
  """
  return (
    f"<error> {step['error']} "
    f"<desc> {step['desc']} "
    f"<reason> {step['reason']} "
    f"<action> {step['action']} "
    f"<replace> {step['replace']} "
    f"<line> {step['line']} "
    f"<index> {step['index']} "
    f"<effect> {step['effect']} "
    f"{'<eos>' if is_last_step else '<eois>'}"
  )


def parse_step(step_str: str) -> dict:
  """
  Parse tokens string to dict.
  """
  fields = ["error", "desc", "reason", "action", "replace", "line", "index", "effect", "end_token"]

  step = {}
  # Regex để match từng trường theo pattern <field> content
  pattern = r"<(\w+)>(?:(.*?))?(?=<\w+>|$)"
  matches = re.findall(pattern, step_str, re.S)
  # Vòng lặp parsing
  for field, content in matches:
    if field == "eois" or field == "eos":
      step["end_token"] = f"<{field.strip()}>"
      continue
    if field in fields:
      step[field] = content.strip()
  # Kiểm tra nếu thiếu bất kỳ trường nào và ném lỗi
  missing_fields = [field for field in fields if field not in step]
  if missing_fields:
    raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
  
  return step

def convert_jsonl_to_csv(jsonl_path, csv_path):
  """
  Đọc file .jsonl chứa các entry reasoning steps và ghi ra file CSV phẳng để dễ xem/chỉnh sửa.
  """
  with open(jsonl_path, 'r', encoding='utf-8') as infile, open(csv_path, 'w', encoding='utf-8', newline='') as outfile:
    fieldnames = ['poem', 'step_index', 'error', 'desc', 'reason', 'action', 'replace', 'line', 'index', 'effect', 'end_token']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for line in infile:
      data = json.loads(line)
      poem = data.get("poem", "")
      steps = data.get("steps", [])
      for idx, step in enumerate(steps):
        row = {
          "poem": poem,
          "step_index": idx,
          "error": step.get("error", ""),
          "desc": step.get("desc", ""),
          "reason": step.get("reason", ""),
          "action": step.get("action", ""),
          "replace": step.get("replace", ""),
          "line": step.get("line", ""),
          "index": step.get("index", ""),
          "effect": step.get("effect", ""),
          "end_token": step.get("end_token", "")
        }
        writer.writerow(row)
    print(f"✅ Đã xuất dữ liệu CSV tại: {csv_path}")

def filter_reasoning_memory(self, sample, max_reasoning_memory):
  parts = sample.split('<eois>')
  if len(parts) <= max_reasoning_memory:
    return sample
  return '<eois>'.join(parts[:1] + parts[-max_reasoning_memory:])

def apply_edit_poem(poem: str, action: str, replace: str, line: int, index: int) -> str:
    lines = poem.strip().split("\n")
    target_line = lines[line - 1].split()

    replace_tokens = replace.strip().split()
    action_tokens = action.strip().split()

    start = index - 1
    end = start + len(replace_tokens)

    if target_line[start:end] == replace_tokens:
        new_line = target_line[:start] + action_tokens + target_line[end:]
        lines[line - 1] = " ".join(new_line)
    else:
        raise ValueError("Cụm từ ở vị trí chỉ định không khớp với 'replace'.")

    return "\n".join(lines)