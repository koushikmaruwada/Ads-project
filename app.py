# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib
import re
from collections import defaultdict
from math import fabs

# Optional model imports (handled gracefully if not installed)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

app = Flask(__name__)
CORS(app)

# -----------------------
# Config
# -----------------------
DEFAULT_BIAS_THRESHOLD = 0.60  # safer default threshold

# -----------------------
# Lexicons & helpers
# -----------------------
# Single-word bias tokens (expand as needed)
BIAS_WORDS = {
    'obviously','clearly','undeniable','shocking','scandal','corrupt','traitor',
    'disgraceful','pathetic','hero','villain','miracle','conspiracy','agenda',
    'radical','extremist','alarmist','outrageous','biased','prejudiced','incompetent',
    'loathe','hate','despicable','unbelievable','attack','critics','scaremongering',
    'propaganda','fabricated','false','lies',
    # suggested additions for news framing
    'unfair','escalating','escalate','protect','negatively','tension','tensions','strong','advantage'
}

# Multi-word biased phrases (longer phrases first will be prioritized)
BIAS_PHRASES = {
    'unfair competitive edge',
    'unfair competitive advantage',
    'negatively impact',
    'take strong measures',
    'escalating trade tensions'
}

# Sentiment helper lexicons (lightweight)
POS_WORDS = {'good','positive','benefit','successful','improve','strong','best','hope','support','advantage','win','gain','promising','effective'}
NEG_WORDS = {'bad','negative','harm','fail','weak','worse','loss','problem','risk','danger','concern','criticize','attack','fear','angry'}

# VADER analyzer (if available)
vader_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

# Zero-shot pipeline (if available)
zero_shot_pipeline = None
if pipeline is not None:
    try:
        zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        zero_shot_pipeline = None  # model may not be downloadable in the environment

# -----------------------
# Text processing helpers
# -----------------------
def split_into_sentences(text):
    text = re.sub(r'\s+', ' ', (text or '')).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def tokenize_words(text):
    return re.findall(r"[A-Za-z0-9']+", (text or '').lower())

def compute_sentiment_score_lexicon(words):
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    return (pos - neg) / max(1, len(words))

def compute_bias_word_ratio(words):
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in BIAS_WORDS)
    return hits / max(1, len(words))

def compute_rule_bias(article_text, flagged_sections_count=0):
    sents = split_into_sentences(article_text)
    words = tokenize_words(article_text)
    total_sents = max(1, len(sents))

    bias_ratio = compute_bias_word_ratio(words)
    sentiment = compute_sentiment_score_lexicon(words)
    flagged_ratio = flagged_sections_count / total_sents

    w_bias, w_sent, w_flag = 0.55, 0.30, 0.15
    sent_component = fabs(sentiment)
    raw = (w_bias * bias_ratio) + (w_sent * sent_component) + (w_flag * flagged_ratio)
    score = max(0.0, min(1.0, raw))
    return {
        'bias_score': round(score, 4),
        'bias_word_ratio': round(bias_ratio, 6),
        'sentiment_score_lexicon': round(sentiment, 6),
        'flagged_ratio': round(flagged_ratio, 6),
        'sentence_count': len(sents),
        'word_count': len(words)
    }

def compute_vader_sentiment(article_text):
    if not vader_analyzer:
        return None
    vs = vader_analyzer.polarity_scores(article_text)
    return vs  # contains 'neg','neu','pos','compound'

def zero_shot_bias_check(text):
    if not zero_shot_pipeline:
        return None
    candidate_labels = ["biased", "neutral", "opinionated", "informative"]
    out = zero_shot_pipeline(text, candidate_labels, multi_label=False)
    return {'label': out['labels'][0], 'score': float(out['scores'][0])}

def sentence_differences(a1, a2):
    s1 = split_into_sentences(a1)
    s2 = split_into_sentences(a2)
    sm = difflib.SequenceMatcher(None, s1, s2)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        diffs.append({
            'type': tag,
            'from_indices': [i1, i2],
            'to_indices': [j1, j2],
            'from_sentences': s1[i1:i2],
            'to_sentences': s2[j1:j2]
        })
    return diffs

# -----------------------
# Improved biased-word detector (phrase-first + skip tokens inside matched phrases)
# -----------------------
def compute_biased_words(article_text):
    """
    Detect multi-word phrases first, then single-word tokens.
    When a phrase matches in a sentence, tokens covered by that phrase
    are marked as consumed for that sentence so single-word matches inside
    the phrase are not reported separately.
    Returns a dict: { token_or_phrase: {count, sentences:[idx...], examples:[sentences...]} }
    """
    sentences = split_into_sentences(article_text)
    found = defaultdict(lambda: {'count': 0, 'sentences': [], 'examples': []})

    # Prepare phrase regex (longest-first to avoid partial overlap)
    phrases = sorted(BIAS_PHRASES, key=lambda s: -len(s))
    phrase_pattern = None
    if phrases:
        escaped_phrases = [re.escape(p) for p in phrases]
        phrase_pattern = re.compile(r'\b(' + '|'.join(escaped_phrases) + r')\b', flags=re.IGNORECASE)

    word_pattern = re.compile(r"[A-Za-z0-9']+")

    for idx, sent in enumerate(sentences):
        low_sent = sent.lower()
        consumed_spans = []  # list of (start,end) in lowercased sentence that are covered by phrase matches

        # 1) phrase matches
        if phrase_pattern:
            for m in phrase_pattern.finditer(low_sent):
                phrase = m.group(0).lower()
                entry = found[phrase]
                entry['count'] += 1
                if idx not in entry['sentences']:
                    entry['sentences'].append(idx)
                    entry['examples'].append(sent.strip())
                consumed_spans.append((m.start(), m.end()))

        # helper: check if a token span overlaps consumed spans
        def is_consumed(start, end):
            for a, b in consumed_spans:
                # overlap if start < b and end > a
                if start < b and end > a:
                    return True
            return False

        # 2) single-word matches (skip tokens inside consumed phrase spans)
        for m in word_pattern.finditer(low_sent):
            token = m.group(0)
            if token in BIAS_WORDS:
                if is_consumed(m.start(), m.end()):
                    # token occurs inside an already-matched phrase for this sentence -> skip
                    continue
                entry = found[token]
                entry['count'] += 1
                if idx not in entry['sentences']:
                    entry['sentences'].append(idx)
                    entry['examples'].append(sent.strip())

    # convert to normal dict
    return {k: {'count': v['count'], 'sentences': v['sentences'], 'examples': v['examples']} for k, v in found.items()}

# -----------------------
# More robust zero-shot: run on sentences and average
# -----------------------
def zero_shot_sentence_score(text, max_sentences=60):
    """
    Run zero-shot-classification on each sentence and return an aggregated bias score in [0,1].
    Returns dict: {'score': float, 'per_sentence': [ {sent, label, score, contribution}, ... ] }
    If pipeline unavailable returns None.
    """
    if not zero_shot_pipeline:
        return None

    sents = split_into_sentences(text)
    if not sents:
        return {'score': 0.0, 'per_sentence': []}

    # Limit number of sentences processed to avoid huge cost
    if len(sents) > max_sentences:
        # sample beginning, middle, end for representation
        keep = []
        third = max(1, max_sentences // 3)
        keep.extend(sents[:third])
        mid = len(sents) // 2
        half = max(1, third // 2)
        keep.extend(sents[mid-half: mid+half])
        keep.extend(sents[-third:])
        sents = keep

    candidate_labels = ["biased", "neutral", "opinionated", "informative"]

    total_score = 0.0
    count = 0
    debug_per_sentence = []
    for sent in sents:
        try:
            out = zero_shot_pipeline(sent, candidate_labels, multi_label=False)
            lbl = out['labels'][0].lower()
            sc = float(out['scores'][0])
        except Exception:
            # if pipeline errors on a sentence, skip it
            continue

        # Map label to a per-sentence bias contribution:
        # - 'biased' -> sc (0..1)
        # - 'opinionated' -> sc * 0.7 (less strong)
        # - 'neutral'/'informative' -> 0.0
        if lbl == 'biased':
            contribution = sc
        elif lbl == 'opinionated':
            contribution = sc * 0.7
        else:
            contribution = 0.0

        total_score += contribution
        count += 1
        debug_per_sentence.append({'sent': sent, 'label': lbl, 'score': round(sc, 4), 'contribution': round(contribution, 4)})

    if count == 0:
        return {'score': 0.0, 'per_sentence': debug_per_sentence}
    aggregated = total_score / count  # average per-sentence contribution
    return {'score': round(aggregated, 4), 'per_sentence': debug_per_sentence}

# -----------------------
# Flask routes
# -----------------------
@app.route('/')
def home():
    return '✅ Flask bias analyzer running. POST /analyze with article1 & article2 (model optional: zero_shot|vader|rule).'

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    a1 = data.get('article1', '')
    a2 = data.get('article2', '')
    # default model: zero_shot
    model_choice = (data.get('model') or 'zero_shot').lower()
    flagged_A = int(data.get('flagged_A_count', 0) or 0)
    flagged_B = int(data.get('flagged_B_count', 0) or 0)
    threshold = float(data.get('threshold', DEFAULT_BIAS_THRESHOLD))

    if not a1 or not a2:
        return jsonify({'error': 'Please provide both article1 and article2.'}), 400

    # 1) sentence differences (kept for potential frontend use; may be empty if you removed diff UI)
    diffs = sentence_differences(a1, a2)

    # 2) rule-based scores
    ruleA = compute_rule_bias(a1, flagged_A)
    ruleB = compute_rule_bias(a2, flagged_B)

    response = {
        'differences': diffs,
        'rule_based': {'article1': ruleA, 'article2': ruleB},
    }

    # 3) model-based processing
    if model_choice == 'vader':
        if not vader_analyzer:
            return jsonify({'error': 'VADER not available on server. Install vaderSentiment.'}), 500
        vaderA = compute_vader_sentiment(a1)
        vaderB = compute_vader_sentiment(a2)
        response['vader'] = {'article1': vaderA, 'article2': vaderB}
        finalA = (ruleA['bias_score'] + abs(vaderA.get('compound', 0))) / 2.0
        finalB = (ruleB['bias_score'] + abs(vaderB.get('compound', 0))) / 2.0
        response['final'] = {'article1': {'bias_score': round(finalA,4)}, 'article2': {'bias_score': round(finalB,4)}}

    elif model_choice == 'zero_shot':
        if not zero_shot_pipeline:
            # fallback: return rule-based and indicate missing pipeline
            response['warning'] = 'Zero-shot pipeline not available. Returning rule-based scores.'
            response['final'] = {'article1': {'bias_score': ruleA['bias_score']}, 'article2': {'bias_score': ruleB['bias_score']}}
        else:
            # sentence-level zero-shot scoring (more robust)
            zsA = zero_shot_sentence_score(a1)
            zsB = zero_shot_sentence_score(a2)

            scoreA = zsA['score'] if isinstance(zsA, dict) else float(zsA or 0.0)
            scoreB = zsB['score'] if isinstance(zsB, dict) else float(zsB or 0.0)

            # record debug info
            response['zero_shot'] = {
                'article1': zsA,
                'article2': zsB
            }

            # Combine sentence-level zero-shot with rule-based score (weighted average)
            # Give more weight to zero-shot for semantic judgement:
            weight_zs = 0.7
            weight_rule = 0.3
            finalA = weight_zs * scoreA + weight_rule * ruleA['bias_score']
            finalB = weight_zs * scoreB + weight_rule * ruleB['bias_score']

            response['final'] = {'article1': {'bias_score': round(finalA, 4)}, 'article2': {'bias_score': round(finalB, 4)}}

            # Helpful debug prints in server logs for tuning
            try:
                print("ZERO_SHOT DEBUG ARTICLE1:", response['zero_shot']['article1'])
                print("ZERO_SHOT DEBUG ARTICLE2:", response['zero_shot']['article2'])
            except Exception:
                pass

    else:
        # rule-only
        response['final'] = {'article1': {'bias_score': ruleA['bias_score']}, 'article2': {'bias_score': ruleB['bias_score']}}

    # 4) verdict labels
    finalA = response['final']['article1']['bias_score']
    finalB = response['final']['article2']['bias_score']
    def label(score): return 'biased' if score >= threshold else 'not_biased'
    response['verdict'] = {
        'article1_label': label(finalA),
        'article2_label': label(finalB),
        'which_more_biased': ('article1' if finalA > finalB else ('article2' if finalB > finalA else 'equal'))
    }

    # 5) biased words / phrases detection
    response['biased_words'] = {
        'article1': compute_biased_words(a1),
        'article2': compute_biased_words(a2)
    }

    return jsonify(response), 200

if __name__ == '__main__':
    
    # For local development:
    app.run(host='127.0.0.1', port=5000, debug=True)
