import re
import sys
import numpy as np
from utils import check_consistency

class VQAEval:
	def __init__(self):
		self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
		self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}
		self.articles     = ['a',
							 'an',
							 'the'
							]


		self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
		self.commaStrip   = re.compile("(\d)(\,)(\d)")
		self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']


        
	def evaluate(self, prediction, candidate_answer_list, question=None, metric="em"):
		assert isinstance(candidate_answer_list, list)
		if prediction == '':
			return 0
		if prediction[-1] == '.':
			prediction = prediction[:-1]
		candidate_answer_list = [ans.replace('\n', ' ').replace('\t', ' ').strip().lower() for ans in candidate_answer_list]
		assert isinstance(prediction, str), prediction
		prediction = prediction.replace('\n', ' ').replace('\t', ' ').strip().lower()
		gtAnswers = [self.processDigitArticle(self.processPunctuation(inText)) for inText in candidate_answer_list]
		resAns = self.processDigitArticle(self.processPunctuation(prediction))
		if metric == "em":
			appear_count = gtAnswers.count(resAns)
		elif metric == "turbo":
			turbo_eval_result = {}
			appear_count = {}
			for unique_gt in set(gtAnswers):
				turbo_eval_result[unique_gt] = check_consistency({"question": question, "prediction": resAns, "groundtruth": unique_gt}, "vqa")
			for unique_gt, is_true in turbo_eval_result.items():
				if is_true == 1:
					appear_count[unique_gt] = gtAnswers.count(unique_gt)
			appear_count = max(appear_count.values()) if len(appear_count) > 0 else 0

		if len(candidate_answer_list) > 1:
			acc = min(1.0, appear_count / 3) # acc = 1 if pred matches gt at least 3 times
		else:
			acc = 1 if prediction == candidate_answer_list[0] else 0
		# print(resAns, candidate_answer_list)
		return acc



	def processPunctuation(self, inText):
		outText = inText
		for p in self.punct:
			if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
				outText = outText.replace(p, '')
			else:
				outText = outText.replace(p, ' ')
		outText = self.periodStrip.sub("",
									  outText,
									  re.UNICODE)
		return outText

	def processDigitArticle(self, inText):
		outText = []
		tempText = inText.lower().split()
		for word in tempText:
			word = self.manualMap.setdefault(word, word)
			if word not in self.articles:
				outText.append(word)
			else:
				pass
		for wordId, word in enumerate(outText):
			if word in self.contractions:
				outText[wordId] = self.contractions[word]
		outText = ' '.join(outText)
		return outText

def compute_vqa_acc(predicted_answer_li, gt_answer_li, question_li=None, mode="em"):
    acc_li = []
    VQAEvalTool = VQAEval()
    for i in range(len(gt_answer_li)):
        gt_answer = gt_answer_li[i]
        predict_answer = str(predicted_answer_li[i])
        question = question_li[i] if question_li is not None else None
        assert isinstance(predict_answer, str), print(predict_answer, type(predict_answer))
        assert isinstance(gt_answer, list), print(gt_answer, type(gt_answer))
        acc = VQAEvalTool.evaluate(predict_answer, gt_answer, question, mode)
        acc_li.append(acc)
    return np.average(acc_li)


import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_f1(predicted_answer_li, gt_answer_li):
    f1_li = []
    for i in range(len(gt_answer_li)):
        gt_answer = gt_answer_li[i]
        predict_answer = str(predicted_answer_li[i])
        # print(predict_answer, gt_answer)
        assert isinstance(predict_answer, str)
        assert isinstance(gt_answer, list)
        f1 = metric_max_over_ground_truths(f1_score, predict_answer, gt_answer)
        f1_li.append(f1)
    return np.average(f1_li)



