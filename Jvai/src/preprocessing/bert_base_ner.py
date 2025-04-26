from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizer
from transformers import pipeline

class BertBaseNER():
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def get_named_entity(self, text):
        ner_results = self.ner_pipeline(text)
        for entity in ner_results:
            print(f"Word: {entity['word']}, Label: {entity['entity']}, Start: {entity['start']}, End: {entity['end']}, Score: {entity['score']:.4f}")
        return ner_results
    
    def replace_entities_with_tokens(self, text):
        ner_results = self.ner_pipeline(text)

        for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
            entity_label = entity['entity']
            if 'PER' in entity_label:
                token = 'PERSON '
            elif 'ORG' in entity_label:
                token = 'ORGANIZTION '
            elif 'LOC' in entity_label:
                token = 'LOCATION '
            else:
                token = 'ENTITY '
            text = text[:entity['start']] + token + text[entity['end']:]
        return text

# bbner = BertBaseNER()
# text = bbner.replace_entities_with_tokens("Resident Evil Village is a first person survival horror and the sequel to Resident Evil 7: Biohazard. The game maintains elements from previous Resident Evil games with players having to scavenge environments for items and manage their resources. However, it adds more action-oriented gameplay, with higher enemy counts and a greater emphasis on combat.")
# print(text)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print(tokenizer.tokenize(text))