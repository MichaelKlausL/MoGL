# This is a simple example of using ChemDataExtractor to extract chemical entities from text
from chemdataextractor import Document

text = """
The reaction of benzene with nitric acid produces nitrobenzene.
Aspirin (acetylsalicylic acid) is widely used as an analgesic.
"""

doc = Document(text)

for cem in doc.cems:
    print(cem.text)
