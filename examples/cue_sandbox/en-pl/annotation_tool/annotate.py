import spacy
import pandas as pd
import re
import logging
from typing import Dict
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import os

logging.basicConfig(level=logging.INFO)


class Attributes:
    def __init__(self):
        self.types = {
            'SpGender': ['<sp:feminine>', '<sp:masculine>'],
            'IlGender': ['<il:feminine>', '<il:masculine>', '<il:mixed>'],
            'IlNumber': ['<singular>', '<plural>'],
            'Formality': ['<formal>', '<informal>']
        }
        self.attribute_list = list(self.types.keys())
        self.type_list = [a for b in self.types.values() for a in b]
        self.reverse_map = {
            '<sp:feminine>': '<sp:masculine>',
            '<sp:masculine>': '<sp:feminine>',
            '<il:feminine>': '<il:masculine>',
            '<il:masculine>': '<il:feminine>',
            '<il:mixed>': '<il:feminine>',
            '<singular>': '<plural>',
            '<plural>': '<singular>',
            '<formal>': '<informal>',
            '<informal>': '<formal>',
            '': ''
        }

    def identify_from_type(self, attr_type):
        for attr, types in self.types.items():
            if attr_type in types:
                return attr
        logging.error(f"Tried to identify a type which does not exist: {attr_type}")

    @staticmethod
    def types_to_str(types):
        types = {k: v if v is not None else '' for k, v in types.items()}
        return f"{types['SpGender']},{types['IlGender']},{types['IlNumber']},{types['Formality']}"

    def sort_group(self, group):
        pattern = []
        for attrib in self.types.keys():
            for type_ in self.types[attrib]:
                if type_ in group:
                    pattern.append(type_)
        return ' '.join(pattern)


class Detector:
    def __init__(self, path_to_root):
        try:
            nlp = spacy.load('pl_spacy_model_morfeusz_big')
            self.nlp = nlp
        except ValueError:
            assert hasattr(self, 'nlp')
        with open(os.path.join(path_to_root, 'data/stopwords'), 'r') as f:
            stopwords = f.read().splitlines()
        self.stopwords = stopwords
        self.attribs = Attributes()

    def parse_sentence(self, sentence):
        try:
            parsed = self.nlp(sentence)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.info(e)
            return ""
        return parsed

    def initialise_types(self):
        return {k: None for k in self.attribs.attribute_list}

    def calculate_type_agreement(self, sents, en_sents, attr_type):
        # Used for evaluation.
        rev_type = [self.attribs.reverse_map[x] for x in attr_type]
        # A list of bools depending on whether ith sentence agreed to the ith type
        correct = [self.verify_context(sents[i], en_sents[i], attr_type[i]) for i in tqdm(range(len(sents)))]
        incorrect = [self.verify_context(sents[i], en_sents[i], rev_type[i]) for i in tqdm(range(len(sents)))]

        corr = {
            x: np.sum(np.array(correct)
                      & np.array([t_ in self.attribs.types[x] for t_ in attr_type]))
            for x in self.attribs.attribute_list
        }
        incorr = {
            x: np.sum(np.array(incorrect)
                      & np.array([t_ in self.attribs.types[x] for t_ in rev_type]))
            for x in self.attribs.attribute_list
        }
        logging.info(f"Correct assignments:   {corr}\nIncorrect assignments: {incorr}")
        return corr, incorr

    def verify_context(self, sentence: str, en_sentence: str, predicted_type: str) -> bool:
        """Verify whether the predicted type matches actual types detected from the given sentences.

        :param sentence: raw sentence in Polish.
        :param en_sentence: raw sentence in English.
        :param predicted_type: type predicted by the model.
        :return: True if type matches actual type, else False.
        """
        attr = self.attribs.identify_from_type(predicted_type)
        return self.predict_types(sent_pair={'pl': sentence, 'en': en_sentence})[attr] == predicted_type

    def predict_types(self, sent_pair) -> Dict[str, str]:
        """
        Go through the sentence and identify markers for all phenomena. return them in a list.
        Markers:
        SpGender:   <sp:feminine> or <sp:masculine>
        IlGender:   <il:feminine>, <il:masculine> or <il:mixed>
        IlNumber:   <singular> or <plural>
        Formality:  <formal> or <informal>
        Assumption: if no type is returned for a given phenomenon, then the sentence is ambivalent w.r.t. the phenomenon

        :param sent_pair:   input data
        :return:            a dictionary of types.
        """
        parsed_pl = self.parse_sentence(sent_pair['pl'])
        types = self.initialise_types()
        # 1. Check SpGender.
        types = self.check_speaker_gender(parsed_pl, types=types)
        # 2. Check formality. If sentence is matched as formal, then return the right features and quit.
        # Lemma suggests formal addressing and no_det makes sure that there are no determinants (e.g. lady vs this lady)
        types, sent_is_formal = self.check_if_formal(parsed_pl, sent_pair['en'], types)
        if sent_is_formal:
            return types

        # 3. If sentence did not match as formal, then keep looking for other interlocutor features.
        # If found, annotate sentence as informal.
        types = self.check_interlocutor(parsed_pl, types)
        return types

    def annotate(self, data):
        en, pl = data['en'].tolist(), data['pl'].tolist()
        annotations = []
        for en_, pl_ in tqdm(zip(en, pl)):
            types = self.predict_types({'en': en_, 'pl': pl_})
            annotations.append(self.attribs.types_to_str(types))
        return annotations

    def check_speaker_gender(self, sentence, types=None):
        if types is None:
            types = self.initialise_types()

        for token in sentence:
            token_feats = token._.feats.split(':')
            head_feats = token.head._.feats.split(':')
            if token.head.pos_ not in ['NOUN', 'VERB', 'ADJ']: continue

            if 'sg' in token_feats and 'pri' in token_feats:
                # Past tense and future tense verbs
                if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                    types = self.gender_check(token.head, types, 'SpGender')

                # Nouns
                if token.head.pos_ == 'NOUN' and 'inst' in head_feats:
                    if token.dep_ in ['aux:clitic', 'cop']:
                        if self.no_adp(sentence, token.i, token.head.i):
                            if token.head.lemma_.lower() not in self.stopwords:
                                types = self.gender_check(token.head, types, 'SpGender')

                # Adjectives
                if token.head.pos_ == 'ADJ':
                    if token.dep_ in ['aux:clitic', 'aux:pass', 'cop', 'obl:cmpr', 'obl']:
                        types = self.gender_check(token.head, types, 'SpGender')
        return types

    def check_interlocutor(self, sentence, types):
        for token in sentence:
            token_feats = token._.feats.split(':')
            head_feats = token.head._.feats.split(':')
            for number in ('sg', 'pl'):
                if number in head_feats and 'sec' in head_feats:
                    if token.head.pos_ in ['VERB', 'PRON']:
                        types['IlNumber'] = '<singular>' if number == 'sg' else '<plural>'
                        types['Formality'] = '<informal>'
                        if token.pos_ == 'ADJ' and number in token_feats:
                            if token.dep_ in ['xcomp:pred', 'nsubj', 'conj', 'nsubj', 'iobj', 'xcomp',
                                              'amod', 'vocative', 'obl:cmpr']:
                                types = self.gender_check(token, types, 'IlGender')

                        if token.pos_ == 'NOUN':
                            if token.dep_ == 'vocative' or (token.dep_ in ['appos', 'obj'] and 'voc' in token_feats):
                                ner = [a.text for a in sentence.ents]
                                if token.orth_ not in ner:
                                    types = self.gender_check(token, types, 'IlGender')
            continue_check = False
            # Your/yours
            if token.lemma_.lower() == 'twój':
                types['IlNumber'] = '<singular>'
                types['Formality'] = '<informal>'
            if token.lemma_.lower() == 'wasz':
                types['IlNumber'] = '<plural>'
                types['Formality'] = '<informal>'
            for number in ('sg', 'pl'):
                if 'sec' in token_feats and number in token_feats:
                    if not (token.orth_ == 'ś' and sentence[token.i - 1].orth_ in ['czym', 'kim']):
                        types['IlNumber'] = '<singular>' if number == 'sg' else '<plural>'
                        types['Formality'] = '<informal>'
                        continue_check = True
            if continue_check:
                # Past tense and future tense verbs
                if token.head.pos_ == 'VERB' and token.dep_ in ['aux:clitic', 'aux', 'aux:pass']:
                    types = self.gender_check(token.head, types, 'IlGender')
                # Nouns
                if token.head.pos_ == 'NOUN':
                    if token.dep_ in ['aux:clitic', 'cop']:
                        if self.no_adp(sentence, token.i, token.head.i):
                            if token.head.lemma_.lower() not in self.stopwords:
                                types = self.gender_check(token.head, types, 'IlGender')
                # Adjectives
                if token.head.pos_ == 'ADJ':
                    # First 3 come from SpGender, obl:cmpr is "takiemu jak ty"
                    if token.dep_ in ['aux:clitic', 'aux:pass', 'cop', 'obl:cmpr', 'obl']:
                        types = self.gender_check(token.head, types, 'IlGender')
        return types

    def check_if_formal(self, sentence, src_sentence, types):
        for token in sentence:
            if token.orth_.lower() == 'proszę' and not re.findall(r'please|ask', src_sentence.lower()):
                types['Formality'] = '<formal>'

            if token.lemma_.lower() in ['pan', 'pani'] \
                    and self.no_det(sentence, token) \
                    and self.no_appos(sentence, token) \
                    and self.no_title(src_sentence):
                types['Formality'] = '<formal>'
                # Check gender of interlocutor
                types = self.gender_check(token, types, 'IlGender')
                # Check number of interlocutor
                number = re.findall(r'sg|pl', token._.feats)[0]
                assert number in ['sg', 'pl']
                types['IlNumber'] = '<singular>' if number == 'sg' else '<plural>'
                return types, True

            elif token.lemma_.lower() == 'pański':
                types['Formality'] = '<formal>'
                types['IlNumber'] = '<singular>'
                types['IlGender'] = '<il:masculine>'
                return types, True

            if token.lemma_ == 'państwo' and self.no_det(sentence, token) and self.no_nation(src_sentence):
                types['Formality'] = '<formal>'
                types['IlNumber'] = '<plural>'
                types['IlGender'] = '<il:mixed>'
                return types, True
        return types, False

    @staticmethod
    def gender_check(token, types, attr):
        assert attr in ['SpGender', 'IlGender']
        prefix = 'il' if attr == 'IlGender' else 'sp'
        if re.findall(r'm[123]', token._.feats):
            types[attr] = f'<{prefix}:masculine>'
        if 'f' in token._.feats.split(':'):
            types[attr] = f'<{prefix}:feminine>'
        return types

    @staticmethod
    def no_title(en_sentence):
        if re.findall(r"lad(ies|y)|gentlem[ea]n|(^| )(sir|mr[ .]|mrs[ .]|ms[ .]|herr)|"
                      r"lord|master|messieurs|dames|monsieur|madam[e ]|ma'am", en_sentence.lower()):
            return False
        return True

    @staticmethod
    def no_det(sentence, token):
        """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
        for t in sentence:
            if t.head == token and t.dep_ == 'det':
                return False
        return True

    @staticmethod
    def no_appos(sentence, token):
        """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
        for t in sentence:
            if t.head == token and t.dep_ == 'appos' \
                    and 'gen' not in t._.feats.split(':'):
                return False
        return True

    @staticmethod
    def no_nation(sentence):
        if re.findall('(countr|nation|land|state|kingdom|realm|econom|elsewhere|rule)|\b', sentence.lower()):
            return False
        return True

    @staticmethod
    def no_adp(parsed, i, j):
        for x in range(i, j):
            if parsed[x].pos_ == 'ADP' and parsed[x].head == parsed[j]:
                return False
        return True

def evaluate(attribs, detector, src_file, ref_file, cxt_file, mark_file, hyp_file):
    def read_from_file(filename):
        with open(filename) as f:
            return f.read().splitlines()

    attrib_counts = {x: [[], []] for x in attribs.attribute_list}
    srcs, refs, hyps, cxts, marks = (read_from_file(x) for x in [src_file, ref_file, hyp_file, cxt_file, mark_file])

    for reference, hypothesis, marking, context in zip(refs, hyps, marks, cxts):
        attrib = attribs.identify_from_type(marking)
        attrib_counts[attrib][0] += [reference]
        attrib_counts[attrib][1] += [hypothesis]
    agr_corr_hyp, agr_incorr_hyp = detector.calculate_type_agreement(hyps, srcs, marks)
    for att in attribs.attribute_list:
        result = agr_corr_hyp[att] / (agr_corr_hyp[att] + agr_incorr_hyp[att]) * 100
        logging.info(f"Result for attribute {att}: {result}")


def compute_pr(answers, golden_answers):
    stats = {
        'false_positive': 0,
        'false_negative': 0,
        'true_positive': 0,
        'true_negative': 0
    }
    for predicted, actual in zip(answers, golden_answers):
        predicted = predicted.split(',')
        actual = actual.split(',')
        stats['true_positive'] += sum([actual[i] == predicted[i] == '' for i in range(len(predicted))])
        stats['true_positive'] += sum([actual[i] == predicted[i] and actual[i] != '' for i in range(len(predicted))])
        stats['false_negative'] += sum([actual[i] != '' and predicted[i] == '' for i in range(len(predicted))])
        stats['false_positive'] += sum([actual[i] == '' and predicted[i] != '' for i in range(len(predicted))])

    precision = stats['true_positive'] / (stats['true_positive'] + stats['false_positive']) * 100
    recall = stats['true_positive'] / (stats['true_positive'] + stats['false_negative']) * 100
    logging.info(f"{precision = }; {recall = }")
    return (2 * precision * recall) / (precision + recall)


def evaluate_on_sample(prefix: str, detector: Detector) -> float:
    annotations = annotate_corpus(prefix, detector)
    with open(f"{prefix}.gold") as f:
        gold_annotations = f.read().splitlines()

    return compute_pr(annotations, gold_annotations)


def annotate_corpus(prefix: str, detector: Detector) -> None:
    # Typical annotation: '<sp:feminine>,<il:masculine>,<plural>,<formal>'
    data = read_from_file(prefix, config='pretrain')
    annotations = detector.annotate(data)
    with open(f'{prefix}.cxt', 'w+') as f:
        f.write("\n".join(annotations) + "\n")
    return annotations

def read_from_file(filename, config='pretrain'):
    with open(filename + '.en', 'r') as s, open(filename + '.pl', 'r') as t:
        df = pd.concat([pd.DataFrame(s.read().splitlines()),
                        pd.DataFrame(t.read().splitlines())], axis=1, join='outer', ignore_index=True)
    df.columns = ['en', 'pl']
    if config != 'pretrain':
        with open(filename + '.cxt', 'r') as c:
            lines = c.read().splitlines()
            df = pd.concat([df, pd.DataFrame(lines)], axis=1, join='outer', ignore_index=True)
        df.columns = ['en', 'pl', 'cxt']
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--hyp', required=True)
    parser.add_argument('--cxt', required=True)
    parser.add_argument('--mark', required=True)
    parser.add_argument('--path_to_root', default='examples/cue_sandbox/en-pl/annotation_tool')

    args = parser.parse_args()

    d = Detector(args.path_to_root)
    a = Attributes()
    # Evaluate on a sample
    f1 = evaluate_on_sample(os.path.join(args.path_to_root, 'data/detector_dev/detector_sample'), d)

    assert f1 > 99, "Threshold not reached; quitting"
    logging.info('--- Threshold reached. Annotating corpus...')
    evaluate(attribs=a,
             detector=d,
             src_file=args.src,
             ref_file=args.ref,
             hyp_file=args.hyp,
             cxt_file=args.cxt,
             mark_file=args.mark)
