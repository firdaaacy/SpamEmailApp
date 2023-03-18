import string
import re
import os
current_path = os.getcwd()

def Cleaning(text):
  result = []
  noises=[]
  noises += [i for i in string.punctuation ]
  noises += [i for i in range(10)]
  text = text.replace("\r", " ")
  text = text.replace("\n", " ")
  text = str(re.sub("http : .* /",'', str(text)))
  cleaned = text
    
  for noise in noises :
    cleaned = cleaned.replace(str(noise), " ")
  result.append(cleaned)
  return result

def readFileTxt(file_path):
  with open(file_path, encoding='utf-8', errors='ignore') as f:
    content = f.read()
    return content 

def caseFolding(Texts):
  return [i.lower() for i in Texts]

def Tokenization(doc):
  res=doc[0].split()
  return res

def stopWordRemoval(doc):
  path_doc = os.path.join(current_path, 'stopwords.txt')
  stoplist = (readFileTxt(path_doc)).split()
  res = []
  
  # for doc in Documents:
  #   res.append([term for term in doc if term not in stoplist])
  return [term for term in doc if term not in stoplist]

def untokenized(Documents) :
  res = [ ' '.join(text) for text in Documents]
  return res
