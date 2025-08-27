def get_categories(encoders):
    """Get available categories from encoders for manual selection."""
    categories = {}
    for name, mapping in encoders.items():
        # Filter out padding tokens and get just the category names
        categories[name] = [
            k
            for k in mapping.keys()
            if isinstance(k, str) and not k.endswith("_padding")
        ]
    return categories
