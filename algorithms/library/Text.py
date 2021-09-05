from collections import defaultdict
class Text:
    def __init__(self):
        self.txt = defaultdict(str)
    def __len__(self):
        return len(self.txt)
    def __getitem__(self, key):
        return self.txt[key]
    def __setitem__(self, key, value):
        self.txt[key] = value
    def __add__(self,string):
        self.txt[max(list(self.txt.keys()),default=0)+1] = string
        return self
    def __str__(self):
        string = ''
        for line, value in self.txt.items():
            string+=value+'\n'
        return string
