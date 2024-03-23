import re
from text.english import english_to_lazy_ipa, english_to_ipa2, english_to_lazy_ipa2

def cjke_cleaners2(text):
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text