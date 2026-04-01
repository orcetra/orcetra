def predict(title, description, market_price, volume):
    """Predict probability based on available signals."""
    p = market_price  # Start from market consensus
    
    # Confidence-based adjustment: extreme markets are usually right
    if market_price > 0.90 or market_price < 0.10:
        # Trust the market on obvious calls
        p = market_price
    elif market_price > 0.70:
        # Slight contrarian: markets slightly overconfident on likely events
        p = market_price - 0.02
    elif market_price < 0.30:
        p = market_price + 0.02
    else:
        # Middle range: look for keywords that signal direction
        keywords = ["yes", "win", "approve", "pass", "up", "increase"]
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in title.lower() or keyword.lower() in description.lower())
        if keyword_count > 0:
            p = market_price + 0.05
        keywords_no = ["no", "lose", "reject", "fail", "down", "decrease"]
        keyword_count_no = sum(1 for keyword in keywords_no if keyword.lower() in title.lower() or keyword.lower() in description.lower())
        if keyword_count_no > 0:
            p = market_price - 0.05
        p = max(0.01, min(0.99, p))
    
    # Volume signal: high volume = more informed market
    if volume > 100_000_000:  # >$100M volume
        # Very liquid market, trust it more
        p = market_price * 0.95 + p * 0.05
    
    c = 0.6
    if market_price > 0.90 or market_price < 0.10:
        c = 0.8
    elif volume > 100_000_000:
        c = 0.7

    return {
        "probability": p,
        "confidence": c,
        "reasoning": f"Market-informed prediction. Market={market_price:.2f}, Volume=${volume/1e6:.0f}M"
    }
