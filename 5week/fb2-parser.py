import sys
import glob
import os
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import pickle
from tqdm import tqdm

tag_a_clean = re.compile('<a.*/a>', re.U)
tag_clean = re.compile('<[^>]*>', re.U)
brackets_clean = re.compile('[«»]')
long_dash_clean = re.compile('—')
email_clean = re.compile('[^\s]*@[^\s]*')
clean_text = ''
c = Counter()
for file in glob.glob(os.path.join(sys.argv[1], '*.fb2')):
    print('\n' + file)
    try:
        with open(file, encoding='utf-8') as f:
            for line in tqdm(f):
                stripped = line.strip()
                if stripped[:3] == '<p>' and stripped[-4:] == '</p>':
                    stripped = tag_a_clean.sub('', stripped)
                    stripped = tag_clean.sub('', stripped)
                    stripped = brackets_clean.sub('', stripped)
                    stripped = long_dash_clean.sub(' ', stripped)
                    stripped = email_clean.sub('', stripped)
                    c.update(word_tokenize(stripped.lower(), language='english'))
    except UnicodeDecodeError:
        print("Cannot read!")
with open('vocab.pkl', 'wb') as f:
    pickle.dump(c, f)
print(len(c))
