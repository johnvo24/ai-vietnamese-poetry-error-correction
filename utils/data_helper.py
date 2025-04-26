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


def parse_step(step_str: str, is_last_step: bool) -> dict:
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


## TEST 0
# step = {
#   "error": "TE",
#   "desc": "Từ ngữ chưa tạo hình ảnh rõ ràng.",
#   "reason": "Cần thay cụm từ mơ\n hồ bằng hình ảnh cụ thể.",
#   "action": "sáng rõ",
#   "replace": "mờ xa",
#   "line": "0",
#   "index": "4",
#   "effect": "Câu thơ trở nên trực quan và sống động hơn.",
#   "end_token": 'eois'
# }
# step_str = format_step(step, is_last_step=False)
# print(step_str)
# parsed_step = parse_step(step_str, is_last_step=False)
# print(parsed_step)