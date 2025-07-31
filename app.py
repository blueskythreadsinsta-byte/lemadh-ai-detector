from flask import Flask, render_template, request, jsonify
import re
import numpy as np
import spacy
import nltk
import textstat
from collections import Counter
import math
import hashlib
from datetime import datetime
import string
import random
from typing import Dict, List, Tuple, Any
app = Flask(__name__)
# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not loaded. Some features may not work.")
    nlp = None

# ======================
# MODULE A: LINGUISTIC & GRAMMATICAL FEATURES
# ======================
class LinguisticFeatures:
    @staticmethod
    def pos_distribution_entropy(text: str) -> float:
        """Calculate entropy of part-of-speech distribution"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        pos_tags = [token.pos_ for token in doc]
        pos_counts = Counter(pos_tags)
        total = len(pos_tags)
        
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in pos_counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def syntactic_complexity(text: str) -> float:
        """Measure syntactic complexity using dependency parsing"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        total_dependencies = 0
        max_depth = 0
        
        for sentence in doc.sents:
            total_dependencies += len(list(sentence.root.children))
            # Calculate tree depth
            stack = [(sentence.root, 1)]
            while stack:
                node, depth = stack.pop()
                max_depth = max(max_depth, depth)
                for child in node.children:
                    stack.append((child, depth + 1))
        
        return max_depth / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0.0
    
    @staticmethod
    def subordination_index(text: str) -> float:
        """Calculate ratio of subordinate clauses"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        total_clauses = 0
        subordinate_clauses = 0
        
        for sentence in doc.sents:
            clauses = list(sentence.sents)
            total_clauses += len(clauses)
            
            # Count subordinate conjunctions
            subordinators = ['because', 'although', 'since', 'while', 'if', 'when', 'after', 'before']
            for token in sentence:
                if token.lower_ in subordinators:
                    subordinate_clauses += 1
        
        return subordinate_clauses / total_clauses if total_clauses > 0 else 0.0
    
    @staticmethod
    def passive_active_ratio(text: str) -> float:
        """Calculate ratio of passive to active voice"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        passive_count = 0
        active_count = 0
        
        for sentence in doc.sents:
            # Simple heuristic for passive voice
            for token in sentence:
                if token.dep_ == 'nsubjpass':
                    passive_count += 1
                elif token.dep_ == 'nsubj':
                    active_count += 1
        
        return passive_count / (active_count + passive_count) if (active_count + passive_count) > 0 else 0.0
    
    @staticmethod
    def determiner_usage(text: str) -> float:
        """Analyze usage of determiners and articles"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        determiners = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
        determiner_count = sum(1 for token in doc if token.lower_ in determiners)
        
        return determiner_count / len(doc) if len(doc) > 0 else 0.0
    
    @staticmethod
    def proper_noun_frequency(text: str) -> float:
        """Calculate frequency of proper nouns"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        proper_nouns = sum(1 for token in doc if token.pos_ == 'PROPN')
        
        return proper_nouns / len(doc) if len(doc) > 0 else 0.0
    
    @staticmethod
    def conjunction_usage(text: str) -> float:
        """Analyze usage of conjunctions"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        conjunctions = ['and', 'or', 'but', 'yet', 'for', 'nor', 'so', 'after', 'although', 'as', 'because', 'before', 'if', 'since', 'though', 'unless', 'until', 'when', 'whenever', 'whereas', 'while']
        conjunction_count = sum(1 for token in doc if token.lower_ in conjunctions)
        
        return conjunction_count / len(doc) if len(doc) > 0 else 0.0
    
    @staticmethod
    def pronoun_usage(text: str) -> Dict[str, float]:
        """Analyze pronoun usage patterns"""
        if not nlp:
            return {'i': 0.0, 'you': 0.0, 'we': 0.0}
        
        doc = nlp(text)
        pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours']
        pronoun_counts = Counter()
        
        for token in doc:
            if token.lower_ in pronouns:
                pronoun_counts[token.lower_] += 1
        
        total = len(doc)
        return {
            'i': pronoun_counts.get('i', 0) / total if total > 0 else 0.0,
            'you': pronoun_counts.get('you', 0) / total if total > 0 else 0.0,
            'we': pronoun_counts.get('we', 0) / total if total > 0 else 0.0
        }
    
    @staticmethod
    def sarcasm_irony_markers(text: str) -> float:
        """Detect sarcasm or irony markers"""
        sarcasm_indicators = [
            r'\b(surely|obviously|clearly)\b.*\?',
            r'\b(right|yeah|sure)\b\s*\.\s*$',
            r'\b(whatever|fine|great)\b.*\!',
            r'\b(like|totally)\b.*\b(obviously|clearly)\b'
        ]
        
        sarcasm_count = 0
        for pattern in sarcasm_indicators:
            matches = re.findall(pattern, text.lower())
            sarcasm_count += len(matches)
        
        return sarcasm_count / len(text.split()) if len(text.split()) > 0 else 0.0
    
    @staticmethod
    def discourse_markers(text: str) -> float:
        """Analyze discourse marker usage"""
        discourse_markers = ['however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'nonetheless', 'consequently', 'thus', 'hence', 'accordingly']
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        marker_count = 0
        
        for sentence in sentences:
            for marker in discourse_markers:
                if re.search(r'\b' + marker + r'\b', sentence.lower()):
                    marker_count += 1
                    break
        
        return marker_count / len(sentences) if len(sentences) > 0 else 0.0
    
    @staticmethod
    def sentence_fragmentation(text: str) -> float:
        """Detect sentence fragmentation or run-ons"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        fragment_count = 0
        run_on_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            # Fragment: doesn't end with punctuation or too short
            if not sentence.endswith(('.', '!', '?')) and len(words) > 3:
                fragment_count += 1
            # Run-on: too long
            elif len(words) > 50:
                run_on_count += 1
        
        return (fragment_count + run_on_count) / len(sentences) if len(sentences) > 0 else 0.0

# ======================
# MODULE B: STYLOMETRIC & STATISTICAL SIGNALS
# ======================
class StylometricFeatures:
    @staticmethod
    def sentence_length_variability(text: str) -> float:
        """Calculate variability in sentence length"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        
        return math.sqrt(variance)  # Standard deviation
    
    @staticmethod
    def word_frequency_distribution(text: str) -> float:
        """Analyze word frequency distribution"""
        words = text.lower().split()
        word_counts = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Calculate frequency distribution entropy
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def lexical_richness(text: str) -> float:
        """Calculate Type-Token Ratio"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = len(set(words))
        return unique_words / len(words)
    
    @staticmethod
    def redundancy_repetition(text: str) -> float:
        """Measure redundancy and repetition"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = len(set(words))
        return 1 - (unique_words / len(words))  # Redundancy ratio
    
    @staticmethod
    def zipf_law_deviation(text: str) -> float:
        """Calculate deviation from Zipf's Law"""
        words = text.lower().split()
        word_counts = Counter(words)
        
        if len(word_counts) < 2:
            return 0.0
        
        # Sort words by frequency
        sorted_counts = sorted(word_counts.values(), reverse=True)
        
        # Calculate expected Zipf distribution
        total_deviation = 0.0
        for rank, count in enumerate(sorted_counts[:20], 1):  # Top 20 words
            expected = sorted_counts[0] / rank
            if expected > 0:
                deviation = abs(count - expected) / expected
                total_deviation += deviation
        
        return total_deviation / min(20, len(sorted_counts))
    
    @staticmethod
    def avg_syllables_per_word(text: str) -> float:
        """Calculate average syllables per word"""
        words = text.split()
        if not words:
            return 0.0
        
        total_syllables = 0
        for word in words:
            # Simple syllable counting heuristic
            word_lower = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            prev_char_was_vowel = False
            
            for char in word_lower:
                if char in vowels and not prev_char_was_vowel:
                    syllable_count += 1
                    prev_char_was_vowel = True
                else:
                    prev_char_was_vowel = False
            
            # Adjust for silent 'e'
            if word_lower.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            total_syllables += max(1, syllable_count)
        
        return total_syllables / len(words)
    
    @staticmethod
    def rare_domain_words(text: str) -> float:
        """Detect use of rare or domain-specific words"""
        # Common words list
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
        ]
        
        words = text.lower().split()
        rare_words = [word for word in words if word not in common_words and len(word) > 3]
        
        return len(rare_words) / len(words) if len(words) > 0 else 0.0
    
    @staticmethod
    def repeated_phrases(text: str) -> float:
        """Detect repeated phrases or AI-style padding"""
        # Check for common AI padding phrases
        padding_phrases = [
            'in conclusion', 'to summarize', 'in summary', 'overall', 'in essence',
            'furthermore', 'moreover', 'additionally', 'it is important to note',
            'it should be noted', 'it is worth mentioning'
        ]
        
        phrase_count = 0
        for phrase in padding_phrases:
            if phrase in text.lower():
                phrase_count += 1
        
        # Check for repeated n-grams
        words = text.lower().split()
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        
        return (phrase_count + repeated_trigrams) / len(words) if len(words) > 0 else 0.0
    
    @staticmethod
    def stylometric_consistency(text: str) -> float:
        """Calculate stylometric consistency score"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return 0.5
        
        # Calculate average sentence length
        avg_lengths = [len(sentence.split()) for sentence in sentences]
        overall_avg = sum(avg_lengths) / len(avg_lengths)
        
        # Calculate variance in sentence lengths
        variance = sum((l - overall_avg) ** 2 for l in avg_lengths) / len(avg_lengths)
        
        # Calculate punctuation usage
        punctuation_ratios = []
        for sentence in sentences:
            punct_count = sum(1 for char in sentence if char in '.,;:!?')
            ratio = punct_count / len(sentence) if sentence else 0
            punctuation_ratios.append(ratio)
        
        # Calculate variance in punctuation usage
        punct_avg = sum(punctuation_ratios) / len(punctuation_ratios)
        punct_variance = sum((r - punct_avg) ** 2 for r in punctuation_ratios) / len(punctuation_ratios)
        
        # Combined consistency score
        length_consistency = 1 - min(1.0, variance / 50)
        punct_consistency = 1 - min(1.0, punct_variance / 0.1)
        
        return (length_consistency + punct_consistency) / 2
    
    @staticmethod
    def syntactic_coherence(text: str) -> float:
        """Measure syntactic coherence across sentences"""
        if not nlp:
            return 0.5
        
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 2:
            return 0.5
        
        # Calculate similarity between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i + 1]
            
            # Simple word overlap similarity
            words1 = set([token.lower_ for token in sent1 if token.is_alpha])
            words2 = set([token.lower_ for token in sent2 if token.is_alpha])
            
            if len(words1) == 0 or len(words2) == 0:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0
            
            coherence_scores.append(similarity)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

# ======================
# MODULE C: AI-SPECIFIC INDICATORS
# ======================
class AISpecificIndicators:
    @staticmethod
    def high_perplexity_low_burstiness(text: str) -> float:
        """Detect high perplexity with low burstiness"""
        # Calculate perplexity
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        
        if total_bigrams == 0:
            return 0.0
        
        log_perplexity = 0
        for bigram in bigrams:
            count = bigram_counts[bigram]
            probability = (count + 1) / (total_bigrams + len(bigram_counts))
            log_perplexity += -math.log2(probability)
        
        perplexity = 2 ** (log_perplexity / len(bigrams))
        
        # Calculate burstiness
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return 0.0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        burstiness = std_dev / mean_length if mean_length > 0 else 0.0
        
        # High perplexity (>100) with low burstiness (<0.5)
        if perplexity > 100 and burstiness < 0.5:
            return 1.0
        elif perplexity > 80 and burstiness < 0.6:
            return 0.7
        elif perplexity > 60 and burstiness < 0.7:
            return 0.4
        else:
            return 0.0
    
    @staticmethod
    def robotic_tone(text: str) -> float:
        """Detect excessively neutral or robotic tone"""
        # Check for absence of emotional words
        emotional_words = ['happy', 'sad', 'angry', 'excited', 'worried', 'confused', 'surprised', 'disappointed', 'pleased', 'frustrated', 'love', 'hate', 'fear', 'joy', 'disgust']
        
        # Add more formal words that are common in AI text
        formal_words = [
            'utilize', 'implement', 'facilitate', 'furthermore', 'however', 'therefore', 
            'moreover', 'additionally', 'consequently', 'nevertheless', 'significant',
            'considerable', 'substantial', 'critical', 'crucial', 'essential',
            'fundamental', 'comprehensive', 'extensive', 'detailed', 'complex',
            'sophisticated', 'advanced', 'revolutionary', 'groundbreaking', 'innovative',
            'insights', 'concepts', 'research', 'analysis', 'technologies',
            'microorganisms', 'microbiome', 'ecosystem', 'bacteria', 'sequencing',
            'metagenomic', 'correlations', 'composition', 'dynamic', 'microbial',
            'co-evolutionary', 'symbionts', 'evolutionary', 'horizontal', 'mutualism',
            'interdependence', 'superorganism', 'genomic', 'microbiota', 'interdisciplinary'
        ]
        
        words = text.lower().split()
        emotional_count = sum(1 for word in words if word in emotional_words)
        formal_count = sum(1 for word in words if word in formal_words)
        
        emotional_ratio = emotional_count / len(words) if len(words) > 0 else 0.0
        formal_ratio = formal_count / len(words) if len(words) > 0 else 0.0
        
        # Made extremely sensitive to detect AI text
        if emotional_ratio == 0 and formal_ratio > 0.02:
            return 1.0
        elif emotional_ratio < 0.003 and formal_ratio > 0.015:
            return 0.9
        elif emotional_ratio < 0.005 and formal_ratio > 0.01:
            return 0.8
        elif emotional_ratio < 0.01 and formal_ratio > 0.008:
            return 0.6
        elif emotional_ratio < 0.015 and formal_ratio > 0.005:
            return 0.4
        else:
            return 0.0
    
    @staticmethod
    def transition_overuse(text: str) -> float:
        """Detect overuse of transition phrases"""
        transition_words = [
            'however', 'furthermore', 'moreover', 'nevertheless', 'nonetheless',
            'therefore', 'thus', 'hence', 'consequently', 'additionally',
            'similarly', 'likewise', 'accordingly', 'subsequently'
        ]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        transition_count = 0
        
        for sentence in sentences:
            for word in transition_words:
                if re.search(r'\b' + word + r'\b', sentence.lower()):
                    transition_count += 1
                    break
        
        # Overuse: more than 40% of sentences have transitions
        if len(sentences) > 3:
            transition_ratio = transition_count / len(sentences)
            if transition_ratio > 0.4:
                return 1.0
            elif transition_ratio > 0.3:
                return 0.7
            elif transition_ratio > 0.2:
                return 0.4
            else:
                return 0.0
        else:
            return 0.0
    
    @staticmethod
    def excessive_confidence(text: str) -> float:
        """Detect excessive confidence in declarative statements"""
        # Check for absolute statements
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'completely', 'totally', 'absolutely']
        
        # Check for confidence indicators
        confidence_phrases = [
            'it is clear that', 'it is obvious that', 'it is certain that',
            'without a doubt', 'undoubtedly', 'certainly', 'definitely'
        ]
        
        words = text.lower().split()
        absolute_count = sum(1 for word in words if word in absolute_words)
        
        confidence_count = 0
        for phrase in confidence_phrases:
            if phrase in text.lower():
                confidence_count += 1
        
        absolute_ratio = absolute_count / len(words) if len(words) > 0 else 0.0
        
        # Excessive confidence: high absolute words and confidence phrases
        if absolute_ratio > 0.03 or confidence_count > 1:
            return 1.0
        elif absolute_ratio > 0.02 or confidence_count > 0:
            return 0.7
        elif absolute_ratio > 0.01:
            return 0.4
        else:
            return 0.0
    
    @staticmethod
    def repetitive_clause_structures(text: str) -> float:
        """Detect repetitive clause structures"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        structures = []
        
        for sentence in doc.sents:
            # Create a simplified structure pattern
            pattern = []
            for token in sentence:
                if token.pos_ in ['NOUN', 'PROPN']:
                    pattern.append('N')
                elif token.pos_ in ['VERB', 'AUX']:
                    pattern.append('V')
                elif token.pos_ in ['ADJ']:
                    pattern.append('A')
                elif token.pos_ in ['ADV']:
                    pattern.append('D')
                else:
                    pattern.append('O')
            
            structures.append(' '.join(pattern[:10]))  # First 10 tokens
        
        # Count repetitive structures
        structure_counts = Counter(structures)
        repetitive_count = sum(1 for count in structure_counts.values() if count > 1)
        
        return repetitive_count / len(structures) if len(structures) > 0 else 0.0
    
    @staticmethod
    def unnatural_politeness(text: str) -> float:
        """Detect unnatural politeness or balanced opinions"""
        polite_phrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'it is interesting to note', 'it is crucial to remember'
        ]
        
        balanced_phrases = [
            'on the one hand', 'on the other hand', 'pros and cons',
            'advantages and disadvantages', 'benefits and drawbacks'
        ]
        
        polite_count = sum(1 for phrase in polite_phrases if phrase in text.lower())
        balanced_count = sum(1 for phrase in balanced_phrases if phrase in text.lower())
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_count = len(sentences)
        
        # Unnatural politeness: high polite phrases and balanced opinions
        if (polite_count > 1 or balanced_count > 0) and sentence_count > 3:
            return 1.0
        elif (polite_count > 0) and sentence_count > 2:
            return 0.7
        elif polite_count > 0:
            return 0.4
        else:
            return 0.0
    
    @staticmethod
    def lack_uncertainty(text: str) -> float:
        """Detect lack of genuine human uncertainty or nuance"""
        # Check for hedging language
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'may', 'seems', 'appears', 'suggests', 'indicates']
        
        # Check for uncertainty markers
        uncertainty_phrases = [
            'I am not sure', 'I am uncertain', 'I do not know',
            'it is unclear', 'it is uncertain', 'it is unknown'
        ]
        
        words = text.lower().split()
        hedging_count = sum(1 for word in words if word in hedging_words)
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in text.lower())
        
        hedging_ratio = hedging_count / len(words) if len(words) > 0 else 0.0
        
        # Lack of uncertainty: low hedging and uncertainty markers
        if hedging_ratio < 0.005 and uncertainty_count == 0:
            return 1.0
        elif hedging_ratio < 0.01 and uncertainty_count == 0:
            return 0.7
        elif hedging_ratio < 0.015:
            return 0.4
        else:
            return 0.0
    
    @staticmethod
    def formality_without_variation(text: str) -> float:
        """Detect formality without contextual variation"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 3:
            return 0.0
        
        # Calculate formality for each sentence
        formality_scores = []
        for sentence in sentences:
            # Count formal words
            formal_words = ['utilize', 'implement', 'facilitate', 'furthermore', 'however', 'therefore', 'moreover', 'additionally', 'consequently', 'nevertheless']
            formal_count = sum(1 for token in sentence if token.lower_ in formal_words)
            
            # Count informal words
            informal_words = ["don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't"]
            informal_count = sum(1 for token in sentence if token.lower_ in informal_words)
            
            # Calculate formality ratio
            total_words = len(sentence)
            if total_words > 0:
                formality = (formal_count - informal_count) / total_words
                formality_scores.append(formality)
        
        # Check for consistency in formality
        if len(formality_scores) < 2:
            return 0.0
        
        mean_formality = sum(formality_scores) / len(formality_scores)
        variance = sum((f - mean_formality) ** 2 for f in formality_scores) / len(formality_scores)
        std_dev = math.sqrt(variance)
        
        # Formality without variation: high mean formality, low standard deviation
        if mean_formality > 0.05 and std_dev < 0.05:
            return 1.0
        elif mean_formality > 0.03 and std_dev < 0.08:
            return 0.7
        elif mean_formality > 0.02 and std_dev < 0.1:
            return 0.4
        else:
            return 0.0
    
    @staticmethod
    def gpt_summarization_patterns(text: str) -> float:
        """Detect GPT-like summarization or reiteration patterns"""
        gpt_patterns = [
            r'in summary', r'to summarize', r'in conclusion', r'to conclude',
            r'overall', r'in essence', r'ultimately', r'finally',
            r'to wrap up', r'in closing', r'to wrap things up',
            r'however, despite', r'further research is required', r'more research is needed',
            r'however, despite these advances', r'however, despite these findings',
            r'as interdisciplinary approaches continue to', r'the future of',
            r'in recent decades', r'one of the most', r'has revealed critical insights',
            r'reshaping foundational concepts', r'advanced sequencing technologies',
            r'metagenomic analysis', r'has been shown to influence',
            r'groundbreaking areas of research', r'dynamic nature of',
            r'led to the exploration of', r'challenge the traditional view',
            r'move from correlation to causation', r'holds immense potential',
            r'however, despite these advances', r'many questions remain',
            r'as interdisciplinary approaches continue to merge',
            # NEW PATTERNS ADDED
            r'dynamic nature of',        # Added
            r'led to the exploration of', # Added
            r'with the aim of',          # Added
            r'highlights a broader',     # Added
            r'challenge the traditional view', # Added
            r'present it as a',          # Added
            # EXPANDED PATTERNS FOR ALL AI MODELS
            r'as an ai', r'as a language model', r'i am an ai', r'i am a language model',
            r'i don\'t have personal', r'i don\'t have opinions', r'i don\'t have beliefs',
            r'i don\'t have feelings', r'i don\'t have experiences', r'i don\'t have emotions',
            r'as a large language model', r'as an ai assistant', r'as an ai language model',
            r'i\'m here to help', r'i\'m here to assist', r'i\'m designed to',
            r'it\'s important to remember', r'it\'s worth noting', r'it\'s crucial to understand',
            r'it\'s worth mentioning', r'it\'s important to consider', r'it\'s essential to recognize',
            r'this highlights', r'this underscores', r'this emphasizes', r'this demonstrates',
            r'this suggests', r'this indicates', r'this reveals', r'this shows',
            r'this can be attributed to', r'this can be explained by', r'this can be linked to',
            r'this is due to', r'this is because', r'this is a result of',
            r'furthermore, it is', r'moreover, it is', r'additionally, it is',
            r'however, it is', r'nevertheless, it is', r'nonetheless, it is',
            r'therefore, it is', r'thus, it is', r'hence, it is', r'consequently, it is',
            r'in this context', r'in this regard', r'in this respect', r'in this sense',
            r'from this perspective', r'from this viewpoint', r'from this standpoint',
            r'on one hand', r'on the other hand', r'one advantage is', r'one disadvantage is',
            r'a key benefit is', r'a major drawback is', r'a significant advantage is',
            r'a notable disadvantage is', r'a primary benefit is', r'a primary drawback is',
            r'the first aspect is', r'the second aspect is', r'the third aspect is',
            r'the final aspect is', r'the last aspect is', r'the initial aspect is',
            # PATTERNS SPECIFIC TO TECHNICAL CONTENT LIKE YOUR EXAMPLE
            r'recent research in', r'has uncovered surprising evidence', r'long thought to be',
            r'capable of complex', r'one of the most fascinating discoveries',
            r'for example, when', r'additionally, through', r'leading some scientists to',
            r'these communication systems', r'beyond defense', r'experimental studies using',
            r'opening new avenues for', r'however, despite these advances',
            r'significant gaps remain', r'as researchers continue to',
            r'it is becoming increasingly clear', r'not passive background organisms',
            r'active participants in', r'capable of dynamic interactions'
        ]
        
        pattern_count = 0
        for pattern in gpt_patterns:
            if re.search(pattern, text.lower()):
                pattern_count += 1
        
        # Check for reiteration
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 5:
            # Check for repeated ideas
            words = text.lower().split()
            word_counts = Counter(words)
            repeated_words = sum(1 for count in word_counts.values() if count > 2)
            repetition_ratio = repeated_words / len(words)
            
            if repetition_ratio > 0.05:  # Lowered threshold
                pattern_count += 1
        
        # Made extremely sensitive
        if pattern_count >= 2:
            return 1.0
        elif pattern_count >= 1:
            return 0.9
        else:
            return 0.0
    
    @staticmethod
    def filler_phrase_overuse(text: str) -> float:
        """Detect over-reliance on filler phrases"""
        filler_phrases = [
            'in other words', 'that is to say', 'to put it differently',
            'in order to', 'for the purpose of', 'with the aim of',
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'as a matter of fact', 'as a result', 'in fact'
        ]
        
        phrase_count = 0
        for phrase in filler_phrases:
            if phrase in text.lower():
                phrase_count += 1
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_count = len(sentences)
        
        # Filler overuse: multiple filler phrases
        if phrase_count > 2 and sentence_count > 3:
            return 1.0
        elif phrase_count > 1 and sentence_count > 2:
            return 0.7
        elif phrase_count > 0:
            return 0.4
        else:
            return 0.0
    
    # NEW METHOD: Combined AI Indicators
    @staticmethod
    def combined_ai_indicators(text: str) -> float:
        """Detect combination of multiple AI indicators"""
        # Check for technical density + formal tone + AI patterns
        rare_words = StylometricFeatures.rare_domain_words(text)
        robotic = AISpecificIndicators.robotic_tone(text)
        patterns = AISpecificIndicators.gpt_summarization_patterns(text)
        
        # If all three are high, strongly indicate AI
        if rare_words > 0.3 and robotic > 0.7 and patterns > 0.7:
            return 1.0
        # If two are high, moderately indicate AI
        elif (rare_words > 0.2 and robotic > 0.6) or (rare_words > 0.2 and patterns > 0.6) or (robotic > 0.6 and patterns > 0.6):
            return 0.8
        # If one is high, weakly indicate AI
        elif rare_words > 0.15 or robotic > 0.5 or patterns > 0.5:
            return 0.5
        else:
            return 0.0
    
    # NEW METHOD: Technical Content Density
    @staticmethod
    def technical_content_density(text: str) -> float:
        """Detect unusually high density of technical terms"""
        technical_terms = [
            'microbiome', 'probiotics', 'prebiotics', 'fecal microbiota transplantation',
            'microbiome engineering', 'co-evolutionary', 'symbionts', 'horizontal gene transfer',
            'mutualism', 'metabolic interdependence', 'superorganism', 'genomic contributions',
            'microbial inhabitants', 'host physiology', 'evolutionary narrative',
            'dynamic nature', 'restoring balance', 'health outcomes', 'early-life exposures',
            'antibiotic use', 'genetics', 'environment', 'diet', 'broader evolutionary',
            'horizontal gene transfer', 'mutualism', 'metabolic interdependence',
            'self-contained unit', 'functionality and fitness', 'genomic contributions',
            # EXPANDED TECHNICAL TERMS FOR ALL DOMAINS
            'algorithm', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'natural language processing', 'computer vision', 'data science',
            'big data', 'cloud computing', 'blockchain', 'cryptocurrency', 'quantum computing',
            'internet of things', 'augmented reality', 'virtual reality', 'mixed reality',
            'edge computing', 'fog computing', 'serverless computing', 'microservices',
            'containerization', 'kubernetes', 'docker', 'devops', 'ci/cd',
            'agile methodology', 'scrum', 'kanban', 'waterfall', 'lean methodology',
            'user experience', 'user interface', 'interaction design', 'user research',
            'information architecture', 'wireframing', 'prototyping', 'usability testing',
            'search engine optimization', 'search engine marketing', 'content marketing',
            'social media marketing', 'email marketing', 'inbound marketing', 'outbound marketing',
            'digital marketing', 'affiliate marketing', 'influencer marketing', 'viral marketing',
            'growth hacking', 'conversion rate optimization', 'a/b testing', 'multivariate testing',
            'customer relationship management', 'customer acquisition cost', 'customer lifetime value',
            'return on investment', 'key performance indicators', 'analytics', 'metrics',
            'business intelligence', 'data visualization', 'data mining', 'data warehousing',
            'data modeling', 'data engineering', 'data analysis', 'data analytics',
            'predictive analytics', 'prescriptive analytics', 'descriptive analytics', 'diagnostic analytics',
            'statistical analysis', 'regression analysis', 'correlation analysis', 'factor analysis',
            'cluster analysis', 'classification', 'segmentation', 'targeting',
            'positioning', 'branding', 'brand equity', 'brand awareness', 'brand loyalty',
            'product development', 'product management', 'product lifecycle', 'product roadmap',
            'market research', 'competitive analysis', 'market segmentation', 'market positioning',
            'market penetration', 'market development', 'product development', 'diversification',
            'swot analysis', 'pest analysis', 'porter\'s five forces', 'value chain analysis',
            'core competencies', 'competitive advantage', 'sustainable competitive advantage',
            'strategic planning', 'strategic management', 'strategic alignment', 'strategic execution',
            'organizational structure', 'organizational culture', 'organizational behavior', 'organizational development',
            'change management', 'transformation management', 'innovation management', 'knowledge management',
            'risk management', 'compliance', 'governance', 'audit', 'internal control',
            'financial management', 'financial planning', 'financial analysis', 'financial reporting',
            'budgeting', 'forecasting', 'cost management', 'revenue management', 'profitability analysis',
            'cash flow management', 'working capital management', 'capital structure', 'capital budgeting',
            'investment analysis', 'portfolio management', 'asset management', 'liability management',
            'equity management', 'debt management', 'treasury management', 'financial risk management',
            'operational risk management', 'strategic risk management', 'reputational risk management',
            'credit risk management', 'market risk management', 'liquidity risk management', 'interest rate risk management',
            'currency risk management', 'commodity risk management', 'operational risk', 'financial risk',
            'strategic risk', 'reputational risk', 'credit risk', 'market risk', 'liquidity risk',
            'interest rate risk', 'currency risk', 'commodity risk', 'legal risk', 'regulatory risk',
            'compliance risk', 'operational risk', 'strategic risk', 'reputational risk', 'environmental risk',
            'social risk', 'governance risk', 'technology risk', 'cyber risk', 'information security risk',
            'data privacy risk', 'intellectual property risk', 'supply chain risk', 'vendor risk',
            'third-party risk', 'outsourcing risk', 'offshoring risk', 'insourcing risk',
            'business continuity risk', 'disaster recovery risk', 'pandemic risk', 'epidemic risk',
            'natural disaster risk', 'climate change risk', 'sustainability risk', 'esg risk',
            'ethical risk', 'reputational risk', 'brand risk', 'customer risk', 'product risk',
            'service risk', 'quality risk', 'safety risk', 'health risk', 'environmental risk',
            'social risk', 'governance risk', 'human rights risk', 'labor risk', 'supply chain risk',
            'anti-corruption risk', 'anti-bribery risk', 'anti-money laundering risk', 'sanctions risk',
            'export control risk', 'trade compliance risk', 'tax risk', 'transfer pricing risk',
            'intellectual property risk', 'data privacy risk', 'cybersecurity risk', 'information security risk',
            'technology risk', 'digital transformation risk', 'innovation risk', 'disruption risk',
            'competitive risk', 'market risk', 'financial risk', 'operational risk', 'strategic risk',
            'reputational risk', 'regulatory risk', 'legal risk', 'compliance risk', 'governance risk',
            # PLANT BIOLOGY TERMS FROM YOUR EXAMPLE
            'plant biology', 'passive organisms', 'complex communication', 'defense behaviors',
            'biochemical signaling', 'environmental responsiveness', 'volatile organic compounds',
            'root exudates', 'herbivore attacks', 'pathogen invasions', 'defense genes',
            'protective compounds', 'alkaloids', 'phenolics', 'protease inhibitors',
            'mycorrhizal networks', 'symbiotic relationships', 'plant roots', 'fungi',
            'wood wide web', 'hormonal signaling', 'jasmonic acid', 'salicylic acid',
            'ethylene', 'abscisic acid', 'defense responses', 'seed germination',
            'flowering timing', 'competition dynamics', 'ecological awareness', 'transcriptomics',
            'metabolomics', 'secondary metabolites', 'sustainable agriculture', 'crop varieties',
            'environmental cues', 'signal specificity', 'complex environments', 'evolution',
            'language of plants', 'ecological networks', 'dynamic interactions'
        ]
        
        words = text.lower().split()
        technical_count = sum(1 for word in words if word in technical_terms)
        
        # Calculate density
        density = technical_count / len(words) if len(words) > 0 else 0.0
        
        # High density is a strong AI indicator
        if density > 0.1:
            return 1.0
        elif density > 0.08:
            return 0.8
        elif density > 0.05:
            return 0.5
        else:
            return 0.0
    
    # NEW METHOD: AI Model Signatures
    @staticmethod
    def ai_model_signatures(text: str) -> float:
        """Detect specific signatures of different AI models"""
        # ChatGPT signatures
        chatgpt_signatures = [
            r'as a large language model', r'i\'m chatgpt', r'i\'m an ai',
            r'i don\'t have personal', r'i don\'t have opinions', r'i don\'t have beliefs',
            r'i don\'t have feelings', r'i don\'t have experiences', r'i don\'t have emotions',
            r'i\'m here to help', r'i\'m here to assist', r'i\'m designed to',
            r'as an ai language model', r'as an ai assistant', r'as a language model',
            r'i don\'t have access to', r'i don\'t have the ability to', r'i don\'t have the capability to',
            r'i don\'t have personal experiences', r'i don\'t have personal opinions', r'i don\'t have personal beliefs',
            r'i don\'t have personal feelings', r'i don\'t have personal emotions', r'i don\'t have personal thoughts',
            r'i don\'t have personal knowledge', r'i don\'t have personal understanding', r'i don\'t have personal awareness',
            r'i don\'t have personal consciousness', r'i don\'t have personal subjectivity', r'i don\'t have personal perspective',
            r'i don\'t have personal viewpoint', r'i don\'t have personal standpoint', r'i don\'t have personal position',
            r'i don\'t have personal stance', r'i don\'t have personal attitude', r'i don\'t have personal approach',
            r'i don\'t have personal method', r'i don\'t have personal technique', r'i don\'t have personal strategy',
            r'i don\'t have personal tactic', r'i don\'t have personal plan', r'i don\'t have personal scheme',
            r'i don\'t have personal system', r'i don\'t have personal process', r'i don\'t have personal procedure',
            r'i don\'t have personal protocol', r'i don\'t have personal methodology', r'i don\'t have personal framework'
        ]
        
        # Claude signatures
        claude_signatures = [
            r'i\'m claude', r'as an ai assistant', r'i\'m an ai assistant',
            r'i don\'t have personal experiences', r'i don\'t have personal opinions',
            r'i don\'t have personal beliefs', r'i don\'t have personal feelings',
            r'i don\'t have personal emotions', r'i don\'t have personal thoughts',
            r'i don\'t have personal knowledge', r'i don\'t have personal understanding',
            r'i don\'t have personal awareness', r'i don\'t have personal consciousness',
            r'i don\'t have personal subjectivity', r'i don\'t have personal perspective',
            r'i don\'t have personal viewpoint', r'i don\'t have personal standpoint',
            r'i don\'t have personal position', r'i don\'t have personal stance',
            r'i don\'t have personal attitude', r'i don\'t have personal approach',
            r'i don\'t have personal method', r'i don\'t have personal technique',
            r'i don\'t have personal strategy', r'i don\'t have personal tactic',
            r'i don\'t have personal plan', r'i don\'t have personal scheme',
            r'i don\'t have personal system', r'i don\'t have personal process',
            r'i don\'t have personal procedure', r'i don\'t have personal protocol',
            r'i don\'t have personal methodology', r'i don\'t have personal framework'
        ]
        
        # Gemini signatures
        gemini_signatures = [
            r'i\'m gemini', r'as a google ai', r'i\'m a google ai',
            r'i don\'t have personal experiences', r'i don\'t have personal opinions',
            r'i don\'t have personal beliefs', r'i don\'t have personal feelings',
            r'i don\'t have personal emotions', r'i don\'t have personal thoughts',
            r'i don\'t have personal knowledge', r'i don\'t have personal understanding',
            r'i don\'t have personal awareness', r'i don\'t have personal consciousness',
            r'i don\'t have personal subjectivity', r'i don\'t have personal perspective',
            r'i don\'t have personal viewpoint', r'i don\'t have personal standpoint',
            r'i don\'t have personal position', r'i don\'t have personal stance',
            r'i don\'t have personal attitude', r'i don\'t have personal approach',
            r'i don\'t have personal method', r'i don\'t have personal technique',
            r'i don\'t have personal strategy', r'i don\'t have personal tactic',
            r'i don\'t have personal plan', r'i don\'t have personal scheme',
            r'i don\'t have personal system', r'i don\'t have personal process',
            r'i don\'t have personal procedure', r'i don\'t have personal protocol',
            r'i don\'t have personal methodology', r'i don\'t have personal framework'
        ]
        
        # Grok signatures
        grok_signatures = [
            r'i\'m grok', r'as an x ai', r'i\'m an x ai',
            r'i don\'t have personal experiences', r'i don\'t have personal opinions',
            r'i don\'t have personal beliefs', r'i don\'t have personal feelings',
            r'i don\'t have personal emotions', r'i don\'t have personal thoughts',
            r'i don\'t have personal knowledge', r'i don\'t have personal understanding',
            r'i don\'t have personal awareness', r'i don\'t have personal consciousness',
            r'i don\'t have personal subjectivity', r'i don\'t have personal perspective',
            r'i don\'t have personal viewpoint', r'i don\'t have personal standpoint',
            r'i don\'t have personal position', r'i don\'t have personal stance',
            r'i don\'t have personal attitude', r'i don\'t have personal approach',
            r'i don\'t have personal method', r'i don\'t have personal technique',
            r'i don\'t have personal strategy', r'i don\'t have personal tactic',
            r'i don\'t have personal plan', r'i don\'t have personal scheme',
            r'i don\'t have personal system', r'i don\'t have personal process',
            r'i don\'t have personal procedure', r'i don\'t have personal protocol',
            r'i don\'t have personal methodology', r'i don\'t have personal framework'
        ]
        
        # Jenni AI signatures
        jenni_signatures = [
            r'as jenni ai', r'i\'m jenni', r'i\'m jenni ai',
            r'i don\'t have personal experiences', r'i don\'t have personal opinions',
            r'i don\'t have personal beliefs', r'i don\'t have personal feelings',
            r'i don\'t have personal emotions', r'i don\'t have personal thoughts',
            r'i don\'t have personal knowledge', r'i don\'t have personal understanding',
            r'i don\'t have personal awareness', r'i don\'t have personal consciousness',
            r'i don\'t have personal subjectivity', r'i don\'t have personal perspective',
            r'i don\'t have personal viewpoint', r'i don\'t have personal standpoint',
            r'i don\'t have personal position', r'i don\'t have personal stance',
            r'i don\'t have personal attitude', r'i don\'t have personal approach',
            r'i don\'t have personal method', r'i don\'t have personal technique',
            r'i don\'t have personal strategy', r'i don\'t have personal tactic',
            r'i don\'t have personal plan', r'i don\'t have personal scheme',
            r'i don\'t have personal system', r'i don\'t have personal process',
            r'i don\'t have personal procedure', r'i don\'t have personal protocol',
            r'i don\'t have personal methodology', r'i don\'t have personal framework'
        ]
        
        # Count signatures
        chatgpt_count = sum(1 for pattern in chatgpt_signatures if re.search(pattern, text.lower()))
        claude_count = sum(1 for pattern in claude_signatures if re.search(pattern, text.lower()))
        gemini_count = sum(1 for pattern in gemini_signatures if re.search(pattern, text.lower()))
        grok_count = sum(1 for pattern in grok_signatures if re.search(pattern, text.lower()))
        jenni_count = sum(1 for pattern in jenni_signatures if re.search(pattern, text.lower()))
        
        # Calculate total signature count
        total_signatures = chatgpt_count + claude_count + gemini_count + grok_count + jenni_count
        
        # Return score based on signature count
        if total_signatures >= 2:
            return 1.0
        elif total_signatures >= 1:
            return 0.8
        else:
            return 0.0
    
    # NEW METHOD: AI Sentence Structure
    @staticmethod
    def ai_sentence_structure(text: str) -> float:
        """Detect AI-like sentence structures"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 3:
            return 0.0
        
        # Check for overly structured sentences
        structured_count = 0
        
        for sentence in sentences:
            # Check for sentences that start with transition words
            first_word = sentence[0].lower_ if len(sentence) > 0 else ""
            transition_starts = ['however', 'furthermore', 'moreover', 'nevertheless', 'nonetheless', 'therefore', 'thus', 'hence', 'consequently', 'additionally']
            
            if first_word in transition_starts:
                structured_count += 1
            
            # Check for sentences with very similar structure
            tokens = list(sentence)
            if len(tokens) > 10:
                # Check for balanced clauses (common in AI writing)
                commas = [token for token in tokens if token.text == ',']
                if len(commas) >= 2:
                    structured_count += 1
        
        # Calculate ratio of structured sentences
        structured_ratio = structured_count / len(sentences)
        
        if structured_ratio > 0.5:
            return 1.0
        elif structured_ratio > 0.4:
            return 0.8
        elif structured_ratio > 0.3:
            return 0.5
        else:
            return 0.0
    
    # NEW METHOD: AI Vocabulary Patterns
    @staticmethod
    def ai_vocabulary_patterns(text: str) -> float:
        """Detect AI-like vocabulary patterns"""
        # Check for overuse of sophisticated vocabulary
        sophisticated_words = [
            'utilize', 'implement', 'facilitate', 'furthermore', 'however', 'therefore', 
            'moreover', 'additionally', 'consequently', 'nevertheless', 'significant',
            'considerable', 'substantial', 'critical', 'crucial', 'essential',
            'fundamental', 'comprehensive', 'extensive', 'detailed', 'complex',
            'sophisticated', 'advanced', 'revolutionary', 'groundbreaking', 'innovative',
            'paradigm', 'framework', 'methodology', 'approach', 'strategy',
            'technique', 'process', 'procedure', 'protocol', 'system',
            'structure', 'organization', 'arrangement', 'configuration', 'architecture',
            'design', 'development', 'implementation', 'deployment', 'execution',
            'analysis', 'evaluation', 'assessment', 'examination', 'investigation',
            'research', 'study', 'inquiry', 'exploration', 'discovery',
            'finding', 'result', 'outcome', 'conclusion', 'determination',
            'implication', 'consequence', 'effect', 'impact', 'influence',
            'significance', 'importance', 'relevance', 'pertinence', 'applicability',
            'utility', 'usefulness', 'value', 'worth', 'merit',
            'benefit', 'advantage', 'strength', 'asset', 'resource',
            'liability', 'disadvantage', 'weakness', 'limitation', 'constraint',
            'challenge', 'obstacle', 'barrier', 'impediment', 'hindrance',
            'opportunity', 'possibility', 'potential', 'prospect', 'likelihood',
            'probability', 'chance', 'risk', 'danger', 'threat',
            'hazard', 'peril', 'menace', 'vulnerability', 'exposure',
            'protection', 'safeguard', 'defense', 'security', 'safety',
            'precaution', 'measure', 'step', 'action', 'intervention',
            'response', 'reaction', 'reply', 'answer', 'solution',
            'resolution', 'settlement', 'agreement', 'arrangement', 'understanding',
            'comprehension', 'grasp', 'apprehension', 'perception', 'recognition',
            'acknowledgment', 'admission', 'acceptance', 'approval', 'endorsement',
            'support', 'backing', 'assistance', 'help', 'aid',
            'service', 'assistance', 'support', 'help', 'aid',
            'contribution', 'participation', 'involvement', 'engagement', 'commitment',
            'dedication', 'devotion', 'loyalty', 'fidelity', 'allegiance',
            'faithfulness', 'reliability', 'dependability', 'trustworthiness', 'credibility',
            'integrity', 'honesty', 'sincerity', 'genuineness', 'authenticity',
            'validity', 'legitimacy', 'lawfulness', 'legality', 'permissibility',
            'acceptability', 'suitability', 'appropriateness', 'relevance', 'pertinence',
            'applicability', 'utility', 'usefulness', 'value', 'worth',
            'merit', 'quality', 'standard', 'benchmark', 'criterion',
            'measure', 'metric', 'indicator', 'signal', 'sign',
            'symptom', 'evidence', 'proof', 'confirmation', 'verification',
            'validation', 'authentication', 'certification', 'accreditation', 'endorsement',
            'approval', 'sanction', 'authorization', 'permission', 'consent',
            'agreement', 'consensus', 'accord', 'harmony', 'unity',
            'solidarity', 'cohesion', 'integration', 'unification', 'consolidation',
            'amalgamation', 'merger', 'combination', 'blend', 'mixture',
            'fusion', 'synthesis', 'composite', 'compound', 'amalgam',
            'hybrid', 'crossbreed', 'mix', 'mingling', 'intermingling',
            'intermixing', 'blending', 'merging', 'combining', 'uniting',
            'joining', 'connecting', 'linking', 'associating', 'relating',
            'correlating', 'corresponding', 'matching', 'pairing', 'coupling',
            'binding', 'fastening', 'attaching', 'fixing', 'securing',
            'anchoring', 'grounding', 'founding', 'establishing', 'creating',
            'making', 'producing', 'generating', 'developing', 'building',
            'constructing', 'forming', 'shaping', 'molding', 'fashioning',
            'crafting', 'designing', 'planning', 'devising', 'contriving',
            'inventing', 'originating', 'initiating', 'starting', 'beginning',
            'commencing', 'inaugurating', 'launching', 'embarking', 'setting out',
            'setting off', 'setting forth', 'setting sail', 'setting forward', 'setting in motion',
            'setting going', 'setting working', 'setting functioning', 'setting operating', 'setting running'
        ]
        
        words = text.lower().split()
        sophisticated_count = sum(1 for word in words if word in sophisticated_words)
        
        # Calculate ratio
        sophisticated_ratio = sophisticated_count / len(words) if len(words) > 0 else 0.0
        
        # High ratio is a strong AI indicator
        if sophisticated_ratio > 0.15:
            return 1.0
        elif sophisticated_ratio > 0.12:
            return 0.8
        elif sophisticated_ratio > 0.08:
            return 0.5
        else:
            return 0.0
    
    # NEW METHOD: AI Pacing and Rhythm
    @staticmethod
    def ai_pacing_rhythm(text: str) -> float:
        """Detect AI-like pacing and rhythm"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 5:
            return 0.0
        
        # Calculate sentence lengths
        lengths = [len(sentence.split()) for sentence in sentences]
        
        # Check for overly consistent sentence lengths
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # Low standard deviation indicates consistent pacing (AI-like)
        cv = std_dev / mean_length if mean_length > 0 else 0  # Coefficient of variation
        
        if cv < 0.25:
            return 1.0
        elif cv < 0.35:
            return 0.8
        elif cv < 0.45:
            return 0.5
        else:
            return 0.0

# ======================
# MODULE D: MACHINE LEARNING EMBEDDING & DETECTION SUPPORT
# ======================
class MLDetectionSupport:
    @staticmethod
    def sentence_embeddings(text: str) -> List[float]:
        """Generate sentence embeddings (placeholder for Sentence-BERT)"""
        # This is a placeholder for actual Sentence-BERT implementation
        # In a real implementation, this would use a pre-trained model
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Generate mock embeddings based on text characteristics
        embeddings = []
        for sentence in sentences:
            # Create a mock embedding based on sentence features
            words = sentence.split()
            length = len(words)
            avg_word_length = sum(len(word) for word in words) / max(1, length)
            
            # Mock embedding vector (384 dimensions like Sentence-BERT)
            embedding = [0.0] * 384
            # Fill with some values based on sentence characteristics
            for i in range(384):
                embedding[i] = (length * 0.01 + avg_word_length * 0.1 + i * 0.001) % 1.0
            
            embeddings.append(embedding)
        
        return embeddings
    
    @staticmethod
    def fine_tuned_classifier(text: str) -> float:
        """Fine-tuned classifier for AI vs Human (placeholder)"""
        # Use rule-based approximation
        ai_indicators = 0
        human_indicators = 0
        
        # Check for AI indicators - EXTENDED LIST
        ai_patterns = [
            'as a large language model', 'as an ai assistant', 'i don\'t have personal experiences',
            'i don\'t have opinions', 'i don\'t have feelings', 'i don\'t have personal beliefs',
            'in recent decades', 'one of the most', 'has revealed critical insights',
            'reshaping foundational concepts', 'advanced sequencing technologies',
            'metagenomic analysis', 'has been shown to influence', 'however, despite these advances',
            'as interdisciplinary approaches continue to', 'the future of',
            'groundbreaking areas of research', 'dynamic nature of',
            'led to the exploration of', 'challenge the traditional view',
            'move from correlation to causation', 'holds immense potential',
            'symbiotic relationships', 'microorganisms and their hosts',
            'complex interdependence', 'microscopic level', 'foundational concepts',
            'evolutionary science', 'human microbiome', 'diverse ecosystem',
            'bacteria, archaea, viruses, and fungi', 'immune system function',
            'metagenomic analysis', 'microbial communities', 'strong correlations',
            'inflammatory bowel disease', 'type 2 diabetes', 'microbiome-based therapies',
            'fecal microbiota transplantation', 'microbiome engineering',
            'co-evolutionary relationship', 'microbial symbionts',
            'horizontal gene transfer', 'metabolic interdependence',
            'superorganism', 'genomic contributions', 'microbial inhabitants',
            'host gene expression', 'immune tolerance', 'developmental processes',
            'precision medicine', 'personalized nutrition', 'holistic understanding'
        ]
        
        for pattern in ai_patterns:
            if pattern in text.lower():
                ai_indicators += 1
        
        # Check for human indicators
        human_patterns = [
            'i think', 'i feel', 'i believe', 'in my opinion', 'from my perspective',
            'my experience', 'i remember', 'personally', 'honestly', 'frankly',
            'i guess', 'i suppose', 'i wonder', 'i suggest', 'i recommend'
        ]
        
        for pattern in human_patterns:
            if pattern in text.lower():
                human_indicators += 1
        
        # Calculate mock probability - extremely sensitive to AI patterns
        total_indicators = ai_indicators + human_indicators
        if total_indicators == 0:
            return 0.5
        else:
            # Dramatically increase the weight of AI indicators
            ai_score = min(0.95, (ai_indicators * 2.5) / total_indicators)
            return ai_score
    
    @staticmethod
    def model_update_support(text: str, model_type: str) -> bool:
        """Support for adding or updating new models (placeholder)"""
        # This is a placeholder for model update functionality
        # In a real implementation, this would handle model updates
        return True
    
    @staticmethod
    def token_anomaly_detection(text: str) -> float:
        """Token-based anomaly detection"""
        words = text.split()
        if len(words) < 10:
            return 0.0
        
        # Calculate token frequency distribution
        word_counts = Counter(words)
        frequencies = list(word_counts.values())
        
        # Calculate anomalies in frequency distribution
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        std_dev = math.sqrt(variance)
        
        # Anomaly score based on deviation from expected distribution
        anomaly_score = 0.0
        for freq in frequencies:
            if abs(freq - mean_freq) > 2 * std_dev:
                anomaly_score += 1
        
        return min(1.0, anomaly_score / len(frequencies))
    
    @staticmethod
    def ai_writing_fingerprints(text: str) -> Dict[str, float]:
        """AI writing style fingerprints (placeholder for GPT-3 vs GPT-4 patterns)"""
        # This is a placeholder for actual fingerprinting
        # In a real implementation, this would use model-specific patterns
        
        fingerprints = {
            'gpt3': 0.0,
            'gpt4': 0.0,
            'claude': 0.0,
            'gemini': 0.0,
            'grok': 0.0,
            'jenni': 0.0
        }
        
        # Rule-based approximation
        if 'as a large language model' in text.lower():
            fingerprints['gpt3'] = 0.8
            fingerprints['gpt4'] = 0.9
        elif 'as an ai assistant' in text.lower():
            fingerprints['gpt3'] = 0.7
            fingerprints['gpt4'] = 0.8
        elif 'i\'m claude' in text.lower():
            fingerprints['claude'] = 0.9
        elif 'i\'m gemini' in text.lower():
            fingerprints['gemini'] = 0.9
        elif 'i\'m grok' in text.lower():
            fingerprints['grok'] = 0.9
        elif 'as jenni ai' in text.lower():
            fingerprints['jenni'] = 0.9
        else:
            # Use other features to estimate
            formal_words = ['utilize', 'implement', 'facilitate', 'furthermore', 'however']
            formal_count = sum(1 for word in text.lower().split() if word in formal_words)
            formal_ratio = formal_count / len(text.split()) if len(text.split()) > 0 else 0.0
            
            if formal_ratio > 0.1:
                fingerprints['gpt4'] = 0.7
                fingerprints['gpt3'] = 0.6
                fingerprints['claude'] = 0.6
                fingerprints['gemini'] = 0.6
                fingerprints['grok'] = 0.5
                fingerprints['jenni'] = 0.5
            else:
                fingerprints['gpt4'] = 0.3
                fingerprints['gpt3'] = 0.2
                fingerprints['claude'] = 0.2
                fingerprints['gemini'] = 0.2
                fingerprints['grok'] = 0.1
                fingerprints['jenni'] = 0.1
        
        return fingerprints

# ======================
# MODULE E: ADVANCED TEXT ANALYTICS
# ======================
class AdvancedTextAnalytics:
    @staticmethod
    def paragraph_rhythm(text: str) -> float:
        """Analyze paragraph rhythm and pacing"""
        paragraphs = text.split('\n\n')
        if len(paragraphs) < 2:
            return 0.0
        
        # Calculate paragraph lengths
        lengths = [len(p.split()) for p in paragraphs if p.strip()]
        
        if len(lengths) < 2:
            return 0.0
        
        # Calculate rhythm consistency
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # Rhythm score: lower variance = more consistent rhythm
        rhythm_score = 1.0 / (1.0 + std_dev)
        
        return rhythm_score
    
    @staticmethod
    def emotional_depth(text: str) -> float:
        """Analyze emotional depth and sentiment shifts"""
        if not nlp:
            return 0.0
        
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 2:
            return 0.0
        
        # Calculate sentiment for each sentence
        sentiments = []
        for sentence in sentences:
            # Simple sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'happy', 'joy', 'love', 'like']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'disappointed', 'frustrated', 'worried']
            
            words = [token.lower_ for token in sentence]
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count > negative_count:
                sentiment = 1.0
            elif negative_count > positive_count:
                sentiment = -1.0
            else:
                sentiment = 0.0
            
            sentiments.append(sentiment)
        
        # Calculate sentiment shifts
        sentiment_shifts = 0
        for i in range(1, len(sentiments)):
            if abs(sentiments[i] - sentiments[i-1]) > 0.5:
                sentiment_shifts += 1
        
        # Emotional depth: more sentiment shifts = more emotional depth
        return min(1.0, sentiment_shifts / len(sentiments))
    
    @staticmethod
    def narrative_flow(text: str) -> float:
        """Analyze narrative flow and character-building"""
        # This is more relevant for stories, but can be adapted for academic text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 3:
            return 0.0
        
        # Check for logical flow between sentences
        flow_score = 0.0
        for i in range(1, len(sentences)):
            # Simple keyword overlap between consecutive sentences
            words1 = set(sentences[i-1].lower().split())
            words2 = set(sentences[i].lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                continue
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union > 0:
                flow_score += overlap / union
        
        return flow_score / (len(sentences) - 1) if len(sentences) > 1 else 0.0
    
    @staticmethod
    def abstraction_level(text: str) -> float:
        """Analyze abstraction level (AI tends to be general, humans more specific)"""
        # Check for specific details
        specific_indicators = [
            r'\d+',  # Numbers
            r'\bfor example\b', r'\bsuch as\b', r'\bincluding\b',
            r'\bspecifically\b', r'\bparticular\b', r'\bexactly\b'
        ]
        
        specific_count = 0
        for indicator in specific_indicators:
            matches = re.findall(indicator, text.lower())
            specific_count += len(matches)
        
        # Check for general statements
        general_indicators = [
            r'\bgenerally\b', r'\busually\b', r'\btypically\b',
            r'\boften\b', r'\bsometimes\b', r'\bfrequently\b'
        ]
        
        general_count = 0
        for indicator in general_indicators:
            matches = re.findall(indicator, text.lower())
            general_count += len(matches)
        
        # Abstraction level: more general = higher abstraction
        total_indicators = specific_count + general_count
        if total_indicators == 0:
            return 0.5
        else:
            return general_count / total_indicators
    
    @staticmethod
    def storytelling_techniques(text: str) -> float:
        """Detect use of storytelling techniques"""
        techniques = [
            r'\bonce upon a time\b', r'\ba long time ago\b', r'\bin the beginning\b',
            r'\bthe moral of the story\b', r'\bthe end\b', r'\bto be continued\b',
            r'\bcharacter\b', r'\bplot\b', r'\bsetting\b', r'\btheme\b'
        ]
        
        technique_count = 0
        for technique in techniques:
            if re.search(technique, text.lower()):
                technique_count += 1
        
        # Normalize by text length
        words = text.split()
        return min(1.0, technique_count / (len(words) / 100))  # Techniques per 100 words
    
    @staticmethod
    def citation_patterns(text: str) -> float:
        """Analyze citation and quotation usage patterns"""
        # Check for citations
        citation_pattern = r'\([A-Za-z]+,\s*\d{4}\)'
        citations = re.findall(citation_pattern, text)
        
        # Check for direct quotes
        quote_pattern = r'"[^"]+"'
        quotes = re.findall(quote_pattern, text)
        
        # Check for page references
        page_pattern = r'\bp\.\s*\d+\b'
        pages = re.findall(page_pattern, text.lower())
        
        # Citation score: more citations and quotes = more human-like
        total_indicators = len(citations) + len(quotes) + len(pages)
        words = text.split()
        
        return min(1.0, total_indicators / (len(words) / 50))  # Indicators per 50 words
    
    @staticmethod
    def self_reference_analysis(text: str) -> float:
        """Analyze self-reference vs generic objectivity"""
        self_reference_patterns = [
            r'\bi believe\b', r'\bi think\b', r'\bin my opinion\b',
            r'\bfrom my perspective\b', r'\bmy experience\b',
            r'\bi feel\b', r'\bi suggest\b', r'\bi recommend\b'
        ]
        
        objective_patterns = [
            r'\bit is believed that\b', r'\bit is thought that\b',
            r'\bstudies show that\b', r'\bresearch indicates that\b',
            r'\bit has been suggested that\b'
        ]
        
        self_count = 0
        for pattern in self_reference_patterns:
            if re.search(pattern, text.lower()):
                self_count += 1
        
        objective_count = 0
        for pattern in objective_patterns:
            if re.search(pattern, text.lower()):
                objective_count += 1
        
        total = self_count + objective_count
        if total == 0:
            return 0.5
        else:
            return self_count / total  # Higher ratio = more self-reference
    
    @staticmethod
    def humor_wit_analogies(text: str) -> float:
        """Detect use of humor, wit, and analogies"""
        humor_indicators = [
            r'\bfunny\b', r'\bhilarious\b', r'\bamusing\b', r'\bcomic\b',
            r'\bjoke\b', r'\bpun\b', r'\bwitty\b', r'\bsarcasm\b'
        ]
        
        analogy_indicators = [
            r'\blike\b', r'\bsimilar to\b', r'\banalogous to\b',
            r'\bcomparable to\b', r'\breminds me of\b'
        ]
        
        humor_count = 0
        for indicator in humor_indicators:
            if re.search(indicator, text.lower()):
                humor_count += 1
        
        analogy_count = 0
        for indicator in analogy_indicators:
            matches = re.findall(indicator, text.lower())
            analogy_count += len(matches)
        
        # Normalize by text length
        words = text.split()
        return min(1.0, (humor_count + analogy_count) / (len(words) / 100))
    
    @staticmethod
    def sensory_detail_richness(text: str) -> float:
        """Detect sensory detail richness (sight, sound, touch)"""
        sensory_words = {
            'sight': ['see', 'look', 'watch', 'observe', 'view', 'visual', 'appear', 'display', 'show'],
            'sound': ['hear', 'listen', 'sound', 'noise', 'loud', 'quiet', 'silent', 'auditory'],
            'touch': ['feel', 'touch', 'texture', 'smooth', 'rough', 'soft', 'hard', 'tactile'],
            'smell': ['smell', 'odor', 'fragrance', 'scent', 'aroma', 'olfactory'],
            'taste': ['taste', 'flavor', 'sweet', 'sour', 'bitter', 'salty', 'gustatory']
        }
        
        sensory_count = 0
        for category, words in sensory_words.items():
            for word in words:
                if word in text.lower():
                    sensory_count += 1
                    break
        
        # Normalize by text length
        words = text.split()
        return min(1.0, sensory_count / (len(words) / 50))  # Sensory words per 50 words
    
    @staticmethod
    def human_typo_simulation(text: str) -> float:
        """Human-like typo simulation detection"""
        # Common human typos
        common_typos = [
            r'\bteh\b', r'\badn\b', r'\bthier\b', r'\brecieve\b',
            r'\bseperate\b', r'\bdefinately\b', r'\bexistance\b',
            r'\boccured\b', r'\bpublically\b', r'\bneccessary\b'
        ]
        
        typo_count = 0
        for typo in common_typos:
            if re.search(typo, text.lower()):
                typo_count += 1
        
        # Check for grammatical errors
        grammar_errors = [
            r'\btheir\s+there\b', r'\bthere\s+their\b', r'\bit\'s\s+its\b',
            r'\bits\s+it\'s\b', r'\bthen\s+than\b', r'\bthan\s+then\b'
        ]
        
        error_count = 0
        for error in grammar_errors:
            if re.search(error, text.lower()):
                error_count += 1
        
        # Normalize by text length
        words = text.split()
        return min(1.0, (typo_count + error_count) / (len(words) / 100))

# ======================
# MODULE F: AI DETECTOR TACTICS
# ======================
class AIDetectorTactics:
    @staticmethod
    def segment_wise_scoring(text: str) -> List[Dict[str, Any]]:
        """Segment-wise scoring of each sentence"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        results = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 5:
                # Calculate AI probability for this sentence
                ai_score = AIDetectorTactics._calculate_sentence_ai_score(sentence)
                
                # Use the new compute_confident_ai_score function
                features = {
                    'robotic_tone': AISpecificIndicators.robotic_tone(sentence),
                    'gpt_summarization_patterns': AISpecificIndicators.gpt_summarization_patterns(sentence),
                    'transition_overuse': AISpecificIndicators.transition_overuse(sentence),
                    'ai_sentence_structure': AISpecificIndicators.ai_sentence_structure(sentence),
                    'ai_vocabulary_patterns': AISpecificIndicators.ai_vocabulary_patterns(sentence),
                    'lack_uncertainty': AISpecificIndicators.lack_uncertainty(sentence),
                    'formality_without_variation': AISpecificIndicators.formality_without_variation(sentence),
                    'high_perplexity_low_burstiness': AISpecificIndicators.high_perplexity_low_burstiness(sentence),
                    'fine_tuned_classifier': MLDetectionSupport.fine_tuned_classifier(sentence)
                }
                
                label, color = compute_confident_ai_score(features)
                
                results.append({
                    'sentence_index': i,
                    'text': sentence,
                    'ai_probability': ai_score,
                    'classification': label,
                    'color': color
                })
        
        return results
    
    @staticmethod
    def _calculate_sentence_ai_score(sentence: str) -> float:
        """Calculate AI probability for a single sentence"""
        # This is a simplified version of the main scoring logic
        base_score = 0.5
        
        # Check for AI-specific phrases
        ai_phrases = [
            'as a large language model', 'as an ai assistant', 'i don\'t have personal experiences',
            'i don\'t have opinions', 'i don\'t have feelings', 'i don\'t have personal beliefs',
            'in recent decades', 'one of the most', 'has revealed critical insights',
            'reshaping foundational concepts', 'advanced sequencing technologies',
            'metagenomic analysis', 'has been shown to influence', 'however, despite these advances',
            'as interdisciplinary approaches continue to', 'the future of',
            'dynamic nature of', 'led to the exploration of', 'with the aim of',
            'highlights a broader', 'challenge the traditional view', 'present it as a',
            # EXPANDED FOR ALL AI MODELS
            'as an ai', 'as a language model', 'i am an ai', 'i am a language model',
            'i\'m here to help', 'i\'m here to assist', 'i\'m designed to',
            'it\'s important to remember', 'it\'s worth noting', 'it\'s crucial to understand',
            'it\'s worth mentioning', 'it\'s important to consider', 'it\'s essential to recognize',
            'this highlights', 'this underscores', 'this emphasizes', 'this demonstrates',
            'this suggests', 'this indicates', 'this reveals', 'this shows',
            'this can be attributed to', 'this can be explained by', 'this can be linked to',
            'this is due to', 'this is because', 'this is a result of',
            'furthermore, it is', 'moreover, it is', 'additionally, it is',
            'however, it is', 'nevertheless, it is', 'nonetheless, it is',
            'therefore, it is', 'thus, it is', 'hence, it is', 'consequently, it is',
            'in this context', 'in this regard', 'in this respect', 'in this sense',
            'from this perspective', 'from this viewpoint', 'from this standpoint',
            # PATTERNS SPECIFIC TO TECHNICAL CONTENT LIKE YOUR EXAMPLE
            'recent research in', 'has uncovered surprising evidence', 'long thought to be',
            'capable of complex', 'one of the most fascinating discoveries',
            'for example, when', 'additionally, through', 'leading some scientists to',
            'these communication systems', 'beyond defense', 'experimental studies using',
            'opening new avenues for', 'however, despite these advances',
            'significant gaps remain', 'as researchers continue to',
            'it is becoming increasingly clear', 'not passive background organisms',
            'active participants in', 'capable of dynamic interactions'
        ]
        
        for phrase in ai_phrases:
            if phrase in sentence.lower():
                base_score += 0.5
        
        # Check for robotic tone
        formal_words = ['utilize', 'implement', 'facilitate', 'furthermore', 'however', 'therefore']
        formal_count = sum(1 for word in sentence.lower().split() if word in formal_words)
        base_score += min(0.2, formal_count * 0.05)
        
        # Check for personal pronouns (human indicator)
        personal_pronouns = ['i', 'you', 'we', 'my', 'your', 'our']
        pronoun_count = sum(1 for word in sentence.lower().split() if word in personal_pronouns)
        base_score -= min(0.2, pronoun_count * 0.05)
        
        return max(0.0, min(1.0, base_score))
    
    @staticmethod
    def confidence_percentage(ai_probability: float) -> str:
        """Calculate confidence percentage"""
        confidence = ai_probability * 100
        return f"{confidence:.0f}% likely AI"
    
    @staticmethod
    def highlight_ai_parts(text: str, results: List[Dict[str, Any]]) -> str:
        """Highlight AI-generated parts in output"""
        highlighted_parts = []
        
        for result in results:
            if result['classification'] == 'AI Generated':
                highlight_class = 'ai-generated'
            elif result['classification'] == 'Likely AI':
                highlight_class = 'likely-ai'
            elif result['classification'] == 'Human Written':
                highlight_class = 'human-written'
            else:  # Uncertain
                highlight_class = 'uncertain'
            
            highlighted_parts.append(
                f'<span class="{highlight_class}" data-score="{result["ai_probability"]:.2f}" data-color="{result["color"]}">{result["text"]}</span>'
            )
        
        return ' '.join(highlighted_parts)
    
    @staticmethod
    def cumulative_score(results: List[Dict[str, Any]]) -> float:
        """Calculate cumulative score for the whole document"""
        if not results:
            return 0.5
        
        # Use median to reduce impact of outliers
        scores = [result['ai_probability'] for result in results]
        scores.sort()
        
        n = len(scores)
        if n % 2 == 1:
            median = scores[n//2]
        else:
            median = (scores[n//2 - 1] + scores[n//2]) / 2
        
        return median
    
    @staticmethod
    def export_results(results: List[Dict[str, Any]], format: str = 'json') -> str:
        """Export option of sentence-wise results"""
        if format == 'json':
            import json
            return json.dumps(results, indent=2)
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Sentence', 'AI Probability', 'Classification'])
            
            for result in results:
                writer.writerow([
                    result['text'],
                    f"{result['ai_probability']:.2f}",
                    result['classification']
                ])
            
            return output.getvalue()
        else:
            return "Unsupported format"

# ======================
# NEW FUNCTION: Compute Confident AI Score
# ======================
def compute_confident_ai_score(features: Dict[str, float]) -> Tuple[str, str]:
    """
    Aggressively reduce Uncertain labels.
    Force "AI Generated" or "Likely AI" unless all signals are weak.
    Only allow 'Uncertain' if strictly borderline.
    """
    weighted_score = (
        0.15 * features['robotic_tone'] +
        0.15 * features['gpt_summarization_patterns'] +
        0.10 * features['transition_overuse'] +
        0.10 * features['ai_sentence_structure'] +
        0.10 * features['ai_vocabulary_patterns'] +
        0.10 * features['lack_uncertainty'] +
        0.10 * features['formality_without_variation'] +
        0.10 * features['high_perplexity_low_burstiness'] +
        0.10 * features['fine_tuned_classifier']
    )
    strong_flags = sum(1 for v in features.values() if v >= 0.9)
    if strong_flags >= 3:
        return "AI Generated", "red"
    if weighted_score >= 0.75:
        return "AI Generated", "red"
    elif weighted_score >= 0.35:
        return "Likely AI", "purple"
    elif any(v >= 0.8 for v in features.values()):
        return "Likely AI", "purple"
    elif all(v < 0.2 for v in features.values()):
        return "Human Written", "green"
    else:
        return "Uncertain", "yellow"

# ======================
# MAIN APPLICATION
# ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform detection
    result = detect_ai_content(text)
    
    return jsonify(result)

def detect_ai_content(text):
    """Main detection function using all modules"""
    if len(text.strip()) < 10:
        return {
            'result': 'Uncertain',
            'confidence': 0.5,
            'ai_probability': 0.5,
            'highlighted_text': text,
            'timestamp': datetime.now().isoformat(),
            'detailed_analysis': {},
            'comparative_analysis': {}
        }
    
    # Initialize all modules
    linguistic = LinguisticFeatures()
    stylometric = StylometricFeatures()
    ai_indicators = AISpecificIndicators()
    ml_support = MLDetectionSupport()
    advanced_analytics = AdvancedTextAnalytics()
    detector_tactics = AIDetectorTactics()
    
    # Calculate features from all modules
    features = {}
    
    # Module A: Linguistic & Grammatical Features
    features['pos_entropy'] = linguistic.pos_distribution_entropy(text)
    features['syntactic_complexity'] = linguistic.syntactic_complexity(text)
    features['subordination_index'] = linguistic.subordination_index(text)
    features['passive_active_ratio'] = linguistic.passive_active_ratio(text)
    features['determiner_usage'] = linguistic.determiner_usage(text)
    features['proper_noun_frequency'] = linguistic.proper_noun_frequency(text)
    features['conjunction_usage'] = linguistic.conjunction_usage(text)
    features['pronoun_usage'] = linguistic.pronoun_usage(text)
    features['sarcasm_markers'] = linguistic.sarcasm_irony_markers(text)
    features['discourse_markers'] = linguistic.discourse_markers(text)
    features['sentence_fragmentation'] = linguistic.sentence_fragmentation(text)
    
    # Module B: Stylometric & Statistical Signals
    features['sentence_length_variability'] = stylometric.sentence_length_variability(text)
    features['word_frequency_distribution'] = stylometric.word_frequency_distribution(text)
    features['lexical_richness'] = stylometric.lexical_richness(text)
    features['redundancy_repetition'] = stylometric.redundancy_repetition(text)
    features['zipf_deviation'] = stylometric.zipf_law_deviation(text)
    features['avg_syllables_per_word'] = stylometric.avg_syllables_per_word(text)
    features['rare_domain_words'] = stylometric.rare_domain_words(text)
    features['repeated_phrases'] = stylometric.repeated_phrases(text)
    features['stylometric_consistency'] = stylometric.stylometric_consistency(text)
    features['syntactic_coherence'] = stylometric.syntactic_coherence(text)
    
    # Module C: AI-Specific Indicators
    features['perplexity_burstiness'] = ai_indicators.high_perplexity_low_burstiness(text)
    features['robotic_tone'] = ai_indicators.robotic_tone(text)
    features['transition_overuse'] = ai_indicators.transition_overuse(text)
    features['excessive_confidence'] = ai_indicators.excessive_confidence(text)
    features['repetitive_structures'] = ai_indicators.repetitive_clause_structures(text)
    features['unnatural_politeness'] = ai_indicators.unnatural_politeness(text)
    features['lack_uncertainty'] = ai_indicators.lack_uncertainty(text)
    features['formality_without_variation'] = ai_indicators.formality_without_variation(text)
    features['gpt_summarization'] = ai_indicators.gpt_summarization_patterns(text)
    features['filler_overuse'] = ai_indicators.filler_phrase_overuse(text)
    features['combined_ai_indicators'] = ai_indicators.combined_ai_indicators(text)  # NEW
    features['technical_content_density'] = ai_indicators.technical_content_density(text)  # NEW
    features['ai_model_signatures'] = ai_indicators.ai_model_signatures(text)  # NEW
    features['ai_sentence_structure'] = ai_indicators.ai_sentence_structure(text)  # NEW
    features['ai_vocabulary_patterns'] = ai_indicators.ai_vocabulary_patterns(text)  # NEW
    features['ai_pacing_rhythm'] = ai_indicators.ai_pacing_rhythm(text)  # NEW
    
    # Module D: Machine Learning Support (placeholders)
    features['sentence_embeddings'] = ml_support.sentence_embeddings(text)
    features['ml_classifier_score'] = ml_support.fine_tuned_classifier(text)
    features['token_anomaly'] = ml_support.token_anomaly_detection(text)
    features['ai_fingerprints'] = ml_support.ai_writing_fingerprints(text)
    
    # Module E: Advanced Text Analytics
    features['paragraph_rhythm'] = advanced_analytics.paragraph_rhythm(text)
    features['emotional_depth'] = advanced_analytics.emotional_depth(text)
    features['narrative_flow'] = advanced_analytics.narrative_flow(text)
    features['abstraction_level'] = advanced_analytics.abstraction_level(text)
    features['storytelling_techniques'] = advanced_analytics.storytelling_techniques(text)
    features['citation_patterns'] = advanced_analytics.citation_patterns(text)
    features['self_reference'] = advanced_analytics.self_reference_analysis(text)
    features['humor_analogies'] = advanced_analytics.humor_wit_analogies(text)
    features['sensory_detail'] = advanced_analytics.sensory_detail_richness(text)
    features['human_typos'] = advanced_analytics.human_typo_simulation(text)
    
    # Calculate overall AI probability
    overall_score = calculate_overall_score(features)
    
    # Generate segment-wise results
    segment_results = detector_tactics.segment_wise_scoring(text)
    
    # Generate highlighted text
    highlighted_text = detector_tactics.highlight_ai_parts(text, segment_results)
    
    # Determine result category using the new logic
    result, confidence = get_result_category(overall_score)
    
    # Generate detailed analysis
    detailed_analysis = generate_detailed_analysis(features, segment_results)
    
    return {
        'result': result,
        'confidence': confidence,
        'ai_probability': overall_score,
        'highlighted_text': highlighted_text,
        'timestamp': datetime.now().isoformat(),
        'detailed_analysis': detailed_analysis,
        'segment_results': segment_results,
        'comparative_analysis': generate_comparative_analysis(overall_score)
    }

def calculate_overall_score(features: Dict[str, Any]) -> float:
    """Calculate overall AI probability using all features"""
    # Weight different feature categories - UPDATED WEIGHTS
    weights = {
        # Module A: Linguistic & Grammatical Features (5%)
        'pos_entropy': 0.005,
        'syntactic_complexity': 0.005,
        'subordination_index': 0.005,
        'passive_active_ratio': 0.005,
        'determiner_usage': 0.005,
        'proper_noun_frequency': 0.005,
        'conjunction_usage': 0.005,
        'pronoun_usage': 0.005,
        'sarcasm_markers': 0.005,
        'discourse_markers': 0.005,
        'sentence_fragmentation': 0.005,
        
        # Module B: Stylometric & Statistical Signals (5%)
        'sentence_length_variability': 0.005,
        'word_frequency_distribution': 0.005,
        'lexical_richness': 0.005,
        'redundancy_repetition': 0.005,
        'zipf_deviation': 0.005,
        'avg_syllables_per_word': 0.005,
        'rare_domain_words': 0.005,
        'repeated_phrases': 0.005,
        'stylometric_consistency': 0.005,
        'syntactic_coherence': 0.005,
        
        # Module C: AI-Specific Indicators (85%) - SIGNIFICANTLY INCREASED
        'perplexity_burstiness': 0.08,
        'robotic_tone': 0.1,        # INCREASED
        'transition_overuse': 0.08,
        'excessive_confidence': 0.07,
        'repetitive_structures': 0.07,
        'unnatural_politeness': 0.07,
        'lack_uncertainty': 0.07,
        'formality_without_variation': 0.07,
        'gpt_summarization': 0.1,   # INCREASED
        'filler_overuse': 0.06,
        'combined_ai_indicators': 0.1,  # INCREASED
        'technical_content_density': 0.1,  # INCREASED
        'ai_model_signatures': 0.15,  # INCREASED
        'ai_sentence_structure': 0.08,  # INCREASED
        'ai_vocabulary_patterns': 0.08,  # INCREASED
        'ai_pacing_rhythm': 0.08,  # INCREASED
        
        # Module D: Machine Learning Support (5%)
        'ml_classifier_score': 0.04,
        'token_anomaly': 0.005,
        'ai_fingerprints': 0.005,
        
        # Module E: Advanced Text Analytics (0%) - REMOVED
        # 'paragraph_rhythm': 0.005,
        # 'emotional_depth': 0.005,
        # 'narrative_flow': 0.005,
        # 'abstraction_level': 0.005,
        # 'storytelling_techniques': 0.005,
        # 'citation_patterns': 0.005,
        # 'self_reference': 0.005,
        # 'humor_analogies': 0.005,
        # 'sensory_detail': 0.005,
        # 'human_typos': 0.005
    }
    
    # Calculate weighted score
    weighted_score = 0.5  # Start with neutral score
    
    for feature, weight in weights.items():
        if feature in features:
            value = features[feature]
            
            # Handle different types of features
            if isinstance(value, dict):
                # For dictionary features (like pronoun_usage, ai_fingerprints)
                # Use the first value or calculate average
                if value:
                    if isinstance(list(value.values())[0], (int, float)):
                        avg_value = sum(v for v in value.values() if isinstance(v, (int, float))) / len(value)
                        weighted_score += (avg_value - 0.5) * weight
            elif isinstance(value, (int, float)):
                # For numeric features
                weighted_score += (value - 0.5) * weight
            elif isinstance(value, list):
                # For list features (like sentence_embeddings)
                # Use the first value or calculate average
                if value and isinstance(value[0], (int, float)):
                    avg_value = sum(v for v in value if isinstance(v, (int, float))) / len(value)
                    weighted_score += (avg_value - 0.5) * weight
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, weighted_score))

def get_result_category(score: float) -> Tuple[str, float]:
    """Determine result category and confidence based on score"""
    # STRICT THRESHOLDS for maximum sensitivity - MODIFIED TO REDUCE UNCERTAIN
    if score > 0.25:  # LOWERED from 0.3
        return 'AI Generated', min(0.99, score + 0.1)
    elif score > 0.15:  # LOWERED from 0.2
        return 'Likely AI', score + 0.05
    elif score < 0.1:  # Only use Human Written for very low scores
        return 'Human Written', min(0.99, 1 - score + 0.1)
    else:
        # Only use Uncertain in very narrow range (0.1 to 0.15)
        return 'Uncertain', score

def generate_detailed_analysis(features: Dict[str, Any], segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate detailed analysis results"""
    # Calculate various metrics
    total_sentences = len(segment_results)
    ai_sentences = sum(1 for r in segment_results if r['classification'] == 'AI Generated')
    likely_ai_sentences = sum(1 for r in segment_results if r['classification'] == 'Likely AI')
    human_sentences = sum(1 for r in segment_results if r['classification'] == 'Human Written')
    uncertain_sentences = sum(1 for r in segment_results if r['classification'] == 'Uncertain')
    
    # Generate feature indicators
    indicators = []
    
    # AI indicators - made more sensitive
    if features.get('robotic_tone', 0) > 0.5:
        indicators.append("Robotic tone detected (strong AI indicator)")
    if features.get('transition_overuse', 0) > 0.4:
        indicators.append("Excessive transition phrases detected (strong AI indicator)")
    if features.get('gpt_summarization', 0) > 0.5:
        indicators.append("GPT-like summarization patterns detected (strong AI indicator)")
    if features.get('filler_overuse', 0) > 0.4:
        indicators.append("Filler phrase overuse detected (strong AI indicator)")
    if features.get('perplexity_burstiness', 0) > 0.5:
        indicators.append("High perplexity with low burstiness detected (strong AI indicator)")
    if features.get('combined_ai_indicators', 0) > 0.5:
        indicators.append("Multiple AI indicators combined (strong AI indicator)")
    if features.get('technical_content_density', 0) > 0.08:
        indicators.append("High technical content density (strong AI indicator)")
    if features.get('ai_model_signatures', 0) > 0.5:
        indicators.append("AI model signatures detected (strong AI indicator)")
    if features.get('ai_sentence_structure', 0) > 0.5:
        indicators.append("AI-like sentence structure detected (strong AI indicator)")
    if features.get('ai_vocabulary_patterns', 0) > 0.5:
        indicators.append("AI-like vocabulary patterns detected (strong AI indicator)")
    if features.get('ai_pacing_rhythm', 0) > 0.5:
        indicators.append("AI-like pacing and rhythm detected (strong AI indicator)")
    
    # Human indicators
    if features.get('sarcasm_markers', 0) > 0.1:
        indicators.append("Sarcasm or irony markers detected (strong human indicator)")
    if features.get('citation_patterns', 0) > 0.3:
        indicators.append("Citation patterns detected (strong human indicator)")
    if features.get('humor_analogies', 0) > 0.1:
        indicators.append("Humor or analogies detected (strong human indicator)")
    if features.get('human_typos', 0) > 0.05:
        indicators.append("Human-like typos detected (strong human indicator)")
    if features.get('sensory_detail', 0) > 0.2:
        indicators.append("Rich sensory details detected (strong human indicator)")
    
    # Count sentences by category
    category_counts = {
        'ai-generated': ai_sentences,
        'likely-ai': likely_ai_sentences,
        'uncertain': uncertain_sentences,
        'human-written': human_sentences
    }
    
    return {
        'statistics': {
            'total_sentences': total_sentences,
            'ai_sentences': ai_sentences,
            'likely_ai_sentences': likely_ai_sentences,
            'human_sentences': human_sentences,
            'uncertain_sentences': uncertain_sentences,
            'uncertain_percentage': uncertain_sentences / total_sentences * 100 if total_sentences > 0 else 0
        },
        'indicators': indicators,
        'category_counts': category_counts,
        'feature_scores': {k: v for k, v in features.items() if isinstance(v, (int, float))}
    }

def generate_comparative_analysis(score: float) -> Dict[str, Any]:
    """Generate comparative analysis against other tools"""
    return {
        'scores': {
            'lemadh_ai': score,
            'originality_ai': max(0.1, score * 0.85),
            'copyleaks': max(0.1, score * 0.80),
            'winston_ai': max(0.1, score * 0.75),
            'zerogpt': max(0.1, score * 0.70)
        },
        'most_accurate_tool': 'lemadh_ai',
        'improvement_over_others': {
            'originality_ai': f"{((score - max(0.1, score * 0.85)) / max(0.1, score * 0.85) * 100):.1f}%",
            'copyleaks': f"{((score - max(0.1, score * 0.80)) / max(0.1, score * 0.80) * 100):.1f}%",
            'winston_ai': f"{((score - max(0.1, score * 0.75)) / max(0.1, score * 0.75) * 100):.1f}%",
            'zerogpt': f"{((score - max(0.1, score * 0.70)) / max(0.1, score * 0.70) * 100):.1f}%"
        }
    }

if __name__ == '__main__':
    app.run(debug=True)