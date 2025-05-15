import google.generativeai as genai

class Gemini():
  def __init__(self, api_key):
    self.name = "gemini-1.5-flash"
    genai.configure(api_key=api_key)
    self.model = genai.GenerativeModel("gemini-1.5-flash")
    
  def __generate__(self, prompt: str):
    result = self.model.generate_content(f"{prompt}")
    return result.text