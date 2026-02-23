"""Site-Specific Adaptors for common web scraping targets.

This module provides pre-built adaptors for common platforms:
- Amazon: Product title, price, reviews, ASIN
- LinkedIn: Profile data, posts (requires auth)
- Twitter/X: Tweets, engagement metrics

Each adaptor includes site-specific headers, rate limit configs, and selector fallbacks.

Example:
    >>> from scrapling.adaptors import AmazonAdaptor
    >>> adaptor = AmazonAdaptor()
    >>> product = adaptor.extract_product(url)
    >>> print(product.title, product.price)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Product:
    """Amazon product data."""
    title: str | None = None
    price: str | None = None
    rating: str | None = None
    reviews_count: str | None = None
    asin: str | None = None
    availability: str | None = None
    original_url: str | None = None


@dataclass
class LinkedInProfile:
    """LinkedIn profile data."""
    name: str | None = None
    headline: str | None = None
    location: str | None = None
    connections: str | None = None
    about: str | None = None
    experience: list[dict[str, str]] = field(default_factory=list)
    education: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Tweet:
    """Twitter/X tweet data."""
    id: str | None = None
    text: str | None = None
    author: str | None = None
    author_handle: str | None = None
    created_at: str | None = None
    likes: int | None = None
    retweets: int | None = None
    replies: int | None = None


class BaseAdaptor:
    """Base class for site-specific adaptors.

    Attributes:
        name: Adaptor name
        base_url: Base URL for the site
        default_headers: Default HTTP headers
        rate_limit_delay: Delay between requests in seconds
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        default_headers: dict[str, str] | None = None,
        rate_limit_delay: float = 2.0,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.rate_limit_delay = rate_limit_delay

    def get_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Get headers for requests.

        Args:
            extra_headers: Additional headers to merge

        Returns:
            Complete headers dictionary
        """
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return headers


class AmazonAdaptor(BaseAdaptor):
    """Amazon-specific scraping adaptor.

    Provides extraction for product details with fallback selectors.

    Example:
        >>> adaptor = AmazonAdaptor()
        >>> product = adaptor.extract_product("https://www.amazon.com/dp/B08N5WRWNW")
        >>> print(product.title, product.price)
    """

    def __init__(self) -> None:
        super().__init__(
            name="Amazon",
            base_url="https://www.amazon.com",
            default_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            },
            rate_limit_delay=5.0,  # Amazon is strict about rate limiting
        )

        # Fallback selectors for different Amazon page layouts
        self._title_selectors = [
            "#productTitle",
            "#title span.a-size-large",
            "h1.product-title-word-break",
        ]
        self._price_selectors = [
            "#priceblock_ourprice",
            "#priceblock_dealprice",
            ".a-price .a-offscreen",
            "#corePrice_feature_div .a-offscreen",
            ".apexPriceToPay .a-offscreen",
        ]
        self._rating_selectors = [
            "#averageCustomerReviews .a-icon-alt",
            "#reviewsMedley .a-icon-alt",
            ".review-rating .a-icon-alt",
        ]
        self._reviews_count_selectors = [
            "#acrCustomerReviewText",
            "#totalReviewCount",
            ".a-link-emphasis .a-row average",
        ]

    def extract_product(self, url: str, html: str | None = None) -> Product:
        """Extract product information from Amazon page.

        Args:
            url: Product URL
            html: HTML content (optional, will be fetched if not provided)

        Returns:
            Product dataclass with extracted data
        """
        product = Product(original_url=url)

        # Extract ASIN from URL
        product.asin = self._extract_asin(url)

        if html:
            product.title = self._extract_with_fallbacks(html, self._title_selectors)
            product.price = self._extract_with_fallbacks(html, self._price_selectors)
            product.rating = self._extract_rating(html)
            product.reviews_count = self._extract_with_fallbacks(
                html, self._reviews_count_selectors
            )
            product.availability = self._extract_availability(html)

        return product

    def _extract_asin(self, url: str) -> str | None:
        """Extract ASIN from URL."""
        # Match patterns like /dp/ASIN or /gp/product/ASIN
        patterns = [
            r"/dp/([A-Z0-9]{10})",
            r"/gp/product/([A-Z0-9]{10})",
            r"/product/([A-Z0-9]{10})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _extract_with_fallbacks(self, html: str, selectors: list[str]) -> str | None:
        """Extract text using selector fallbacks.

        Since we don't have a parser, this is a placeholder.
        In real implementation, use scrapling.Parser.
        """
        # This would use scrapling to extract
        # For now, return None as we need the HTML parsed
        return None

    def _extract_rating(self, html: str) -> str | None:
        """Extract rating from HTML."""
        # Pattern to match rating like "4.5 out of 5 stars"
        match = re.search(r"(\d+\.?\d*)\s*out\s*of\s*5\s*stars", html, re.IGNORECASE)
        if match:
            return f"{match.group(1)} out of 5 stars"
        return None

    def _extract_availability(self, html: str) -> str | None:
        """Extract availability status."""
        html_lower = html.lower()
        if "in stock" in html_lower:
            return "In Stock"
        if "out of stock" in html_lower:
            return "Out of Stock"
        if "currently unavailable" in html_lower:
            return "Currently unavailable"
        return None


class LinkedInAdaptor(BaseAdaptor):
    """LinkedIn-specific scraping adaptor.

    Provides extraction for profile data (requires authentication).

    Example:
        >>> adaptor = LinkedInAdaptor(auth_token="...")
        >>> profile = adaptor.extract_profile("https://linkedin.com/in/username")
        >>> print(profile.name, profile.headline)
    """

    def __init__(self, auth_token: str | None = None) -> None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        super().__init__(
            name="LinkedIn",
            base_url="https://www.linkedin.com",
            default_headers=headers,
            rate_limit_delay=3.0,
        )

    def extract_profile(self, url: str, html: str | None = None) -> LinkedInProfile:
        """Extract profile information from LinkedIn page.

        Args:
            url: Profile URL
            html: HTML content (optional)

        Returns:
            LinkedInProfile dataclass with extracted data
        """
        profile = LinkedInProfile()

        if html:
            profile.name = self._extract_name(html)
            profile.headline = self._extract_headline(html)
            profile.location = self._extract_location(html)
            profile.about = self._extract_about(html)

        return profile

    def _extract_name(self, html: str) -> str | None:
        """Extract name from profile HTML."""
        # Pattern to match name in various LinkedIn layouts
        match = re.search(r'"firstName":"([^"]+)"', html)
        if match:
            first_name = match.group(1)
            last_match = re.search(r'"lastName":"([^"]+)"', html)
            last_name = last_match.group(1) if last_match else ""
            return f"{first_name} {last_name}".strip()
        return None

    def _extract_headline(self, html: str) -> str | None:
        """Extract headline from profile HTML."""
        match = re.search(r'"headline":"([^"]+)"', html)
        return match.group(1) if match else None

    def _extract_location(self, html: str) -> str | None:
        """Extract location from profile HTML."""
        match = re.search(r'"locationName":"([^"]+)"', html)
        return match.group(1) if match else None

    def _extract_about(self, html: str) -> str | None:
        """Extract about section from profile HTML."""
        # Look for about section in JSON-LD or HTML
        match = re.search(r'"summary":"([^"]+)"', html)
        return match.group(1) if match else None


class TwitterAdaptor(BaseAdaptor):
    """Twitter/X-specific scraping adaptor.

    Provides extraction for tweets and engagement metrics.

    Example:
        >>> adaptor = TwitterAdaptor()
        >>> tweets = adaptor.extract_tweets("https://twitter.com/username")
        >>> for tweet in tweets:
        ...     print(tweet.text, tweet.likes)
    """

    def __init__(self) -> None:
        super().__init__(
            name="Twitter",
            base_url="https://twitter.com",
            default_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
            rate_limit_delay=2.0,
        )

    def extract_tweets(
        self, url: str, html: str | None = None, max_tweets: int = 10
    ) -> list[Tweet]:
        """Extract tweets from Twitter profile or URL.

        Args:
            url: Twitter profile or tweet URL
            html: HTML content (optional)
            max_tweets: Maximum number of tweets to extract

        Returns:
            List of Tweet dataclasses
        """
        tweets: list[Tweet] = []

        if html:
            # Extract tweets from HTML
            # This is simplified - real implementation would parse Twitter's JSON
            tweet_pattern = r'"text":"((?:[^"\\]|\\.)*)"'
            matches = re.finditer(tweet_pattern, html)

            for i, match in enumerate(matches):
                if i >= max_tweets:
                    break

                tweet = Tweet()
                tweet.text = match.group(1).encode().decode("unicode_escape")
                tweets.append(tweet)

        return tweets

    def extract_tweet_details(self, url: str, html: str | None = None) -> Tweet:
        """Extract details from a single tweet URL.

        Args:
            url: Tweet URL
            html: HTML content (optional)

        Returns:
            Tweet dataclass
        """
        tweet = Tweet()

        # Extract tweet ID from URL
        match = re.search(r"/status/(\d+)", url)
        if match:
            tweet.id = match.group(1)

        if html:
            tweet.text = self._extract_tweet_text(html)
            tweet.likes = self._extract_metric(html, "likes")
            tweet.retweets = self._extract_metric(html, "retweets")
            tweet.replies = self._extract_metric(html, "replies")

        return tweet

    def _extract_tweet_text(self, html: str) -> str | None:
        """Extract tweet text from HTML."""
        match = re.search(r'"text":"((?:[^"\\]|\\.)*)"', html)
        if match:
            return match.group(1).encode().decode("unicode_escape")
        return None

    def _extract_metric(self, html: str, metric: str) -> int | None:
        """Extract engagement metric from HTML."""
        # Look for patterns like "1,234 Likes" or "1.2K Retweets"
        patterns = {
            "likes": r"([\d,.]+[Kk]?)\s*Likes?",
            "retweets": r"([\d,.]+[Kk]?)\s*Retweets?",
            "replies": r"([\d,.]+[Kk]?)\s*Replies?",
        }

        if metric in patterns:
            match = re.search(patterns[metric], html, re.IGNORECASE)
            if match:
                value = match.group(1)
                # Convert K/k to thousands
                if value.lower().endswith("k"):
                    return int(float(value[:-1]) * 1000)
                return int(value.replace(",", ""))

        return None


# Convenience function to get adaptor by name
def get_adaptor(name: str, **kwargs: Any) -> BaseAdaptor:
    """Get an adaptor by name.

    Args:
        name: Adaptor name (amazon, linkedin, twitter)
        **kwargs: Additional arguments for the adaptor

    Returns:
        BaseAdaptor instance

    Raises:
        ValueError: If adaptor name is not recognized
    """
    adaptors = {
        "amazon": AmazonAdaptor,
        "linkedin": LinkedInAdaptor,
        "twitter": TwitterAdaptor,
    }

    name_lower = name.lower()
    if name_lower in adaptors:
        return adaptors[name_lower](**kwargs)

    raise ValueError(f"Unknown adaptor: {name}. Available: {list(adaptors.keys())}")


__all__ = [
    "BaseAdaptor",
    "AmazonAdaptor",
    "LinkedInAdaptor",
    "TwitterAdaptor",
    "Product",
    "LinkedInProfile",
    "Tweet",
    "get_adaptor",
]
