from urllib import request
from re import split, sub
from pathlib import Path

ASSETS_FOLDER_PATH = Path(__file__).resolve().parent.parent / "assets"

# Helpers
def get_short_story():
	ASSETS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
	file_path = ASSETS_FOLDER_PATH / "the-verdict.txt"

	if file_path.exists():
		raw_text = file_path.read_text(encoding="utf-8")
		return raw_text 

	url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
	request.urlretrieve(url, file_path)

	with open(file_path, "r", encoding="utf-8") as f:
		raw_text = f.read()

	return raw_text

class SimpleTokenizerV1:
	def __init__(self, vocab):
		self.str_to_int = vocab
		self.int_to_str = {i:s for s, i in vocab.items()}

	def encode(self, text):
		preprocessed = split(r'([,.:;?_!"()\']|--|\s)', text)
		preprocessed = [item.strip() for item in preprocessed if item.strip()]
		ids = [self.str_to_int[s] for s in preprocessed]
		return ids

	def decode(self, ids):
		text = " ".join([self.int_to_str[i] for i in ids])
		text = sub(r'\s+([,.?!"()\'])', r'\1', text)
		return text


# Text work
def short_story():
	raw_text = get_short_story()

	print("Total number of characters:", len(raw_text))
	print(raw_text[:99])

def text_split():
	text = "Hello, world. This, is a test."
	result = split(r'([,.]|\s)', text)
	result = [item for item in result if item.strip()]
	print(result)

def text_split_2():
	text = "Hello, world. Is this-- a test?"
	result = split(r'([,.:;?_!"()\']|--|\s)', text)
	result = [item.strip() for item in result if item.strip()]
	print(result)

def short_story_2():
	raw_text = get_short_story()

	preprocessed = split(r'([,.:;?_!"()\']|--|\s)', raw_text)
	preprocessed = [item.strip() for item in preprocessed if item.strip()]
	print(len(preprocessed))
	print(preprocessed[:30])

def short_story_tokens():
	raw_text = get_short_story()

	preprocessed = split(r'([,.:;?_!"()\']|--|\s)', raw_text)
	preprocessed = [item.strip() for item in preprocessed if item.strip()]

	all_words = sorted(set(preprocessed))
	vocab_size = len(all_words)
	vocab = {token:integer for integer, token in enumerate(all_words)}

	# for i, item in enumerate(vocab.items()):
		# print(item)

	tokenizer = SimpleTokenizerV1(vocab)
	text = """It's the last he painted, you know,"
	       Mrs. Gisburn said with pardonable pride."""
	ids = tokenizer.encode(text)
	print(ids)
	print(tokenizer.decode(ids))

short_story_tokens()