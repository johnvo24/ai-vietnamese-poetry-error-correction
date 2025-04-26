import pandas as pd
import re

class JDataPreprocessor():
    def __init__(self):
        self.VN_CHARS = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'
        self.VN_lOWERCASE_CHARS = 'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz'
        self.EN_CHARS = 'a-zA-Z'
        self.EN_LOWERCASE_CHARS = 'a-z'
        self.STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    def read_data(self, file_path='', drop_duplicates_from=[], drop_column_value={}, columns_selected=[], read_size=None):
        print(f"[JV] Reading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # _note_ > Drop duplicates
        if drop_duplicates_from: df.drop_duplicates(subset=drop_duplicates_from, keep='first', inplace=True)
        # _note_ > Drop column drop_column_value.column has value as drop_column_value.value
        if drop_column_value: df.drop(df[df[drop_column_value.column] == drop_column_value.value].index)
        # _note_ > Get the columns needed to solve the problem
        if columns_selected: df = df[columns_selected]
        # _note_ > Drop rows containing NaN values
        df = df.dropna()[:read_size] if read_size else df.dropna()
        print(f"[JV] Raw data shape: {df.shape}")
        return df
    
    # Normalize text ------------------------------------------< JOHN TAG
    def normalize_text(self, text, regex_chars='', regex_spec_chars='', stopwords=[]):
        # regex_chars -> represent letters that need to be kept in the sentences
        # regex_spec_chars -> represents special characters that need to be kept in the sentences
        if regex_spec_chars: 
            text = re.sub(rf'[^0-9{regex_chars}\{regex_spec_chars}]', ' ', text.lower())
            # _note_ > loop to handle special characters
            for ch in regex_spec_chars:
                if ch in text:
                    list = text.split(ch)
                    list = [' '.join(item.split()) for item in list if item]
                    list = [item for item in list if (item != '' and item != ' ')]
                    text = f'{ch} '.join(list + [''])

            text = text.replace('-', ' -')
        else: text = re.sub(rf'[^0-9{regex_chars}]', ' ', text.lower())
        # _note_ > Drop stop words
        if stopwords: text = [word for word in text.split() if word not in stopwords]
        else: text = text.split()
        return ' '.join(text)