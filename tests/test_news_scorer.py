"""Tests for the keyword sentiment scorer."""

import pytest

from trading_bot.news.scorer import KeywordSentimentScorer, SentimentScore, SentimentScorer


def test_scorer_implements_protocol():
    scorer = KeywordSentimentScorer()
    assert isinstance(scorer, SentimentScorer)


def test_empty_text_returns_neutral():
    scorer = KeywordSentimentScorer()
    result = scorer.score("")
    assert result.value == 0.0
    assert result.confidence == 0.0
    assert result.matched_keywords == []


def test_pure_bullish_headline():
    scorer = KeywordSentimentScorer()
    result = scorer.score("Bitcoin ETF approval surges adoption to record high")
    assert result.value > 0
    assert result.confidence > 0


def test_pure_bearish_headline():
    scorer = KeywordSentimentScorer()
    result = scorer.score("Major exchange hack causes crypto crash and liquidation fears")
    assert result.value < 0
    assert result.confidence > 0


def test_mixed_headline_leans_on_majority():
    scorer = KeywordSentimentScorer()
    # "rally" is bullish, "hack exploit collapse" are bearish — should be negative
    result = scorer.score("Despite the rally, hack and exploit fears cause collapse")
    assert result.value < 0


def test_neutral_text_returns_zero_value():
    scorer = KeywordSentimentScorer()
    result = scorer.score("Bitcoin trading volume remains stable on Tuesday")
    assert result.value == 0.0


def test_value_bounded():
    scorer = KeywordSentimentScorer()
    for text in [
        "surge soar rally bullish breakout ath adoption etf listing approval",
        "crash plunge dump bearish hack exploit ban fraud liquidation collapse scam",
    ]:
        result = scorer.score(text)
        assert -1.0 <= result.value <= 1.0


def test_confidence_saturates_at_three_hits():
    scorer = KeywordSentimentScorer()
    # 3+ hits should give confidence = 1.0
    result = scorer.score("crash plunge dump bearish hack exploit")
    assert result.confidence == 1.0


def test_confidence_scales_with_hits():
    scorer = KeywordSentimentScorer()
    one_hit = scorer.score("crash")
    two_hits = scorer.score("crash plunge")
    assert one_hit.confidence < two_hits.confidence


def test_matched_keywords_populated():
    scorer = KeywordSentimentScorer()
    result = scorer.score("Bitcoin ETF approval surges institutional adoption")
    assert len(result.matched_keywords) > 0
