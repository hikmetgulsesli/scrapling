"""Tests for Site-Specific Adaptors (US-008).

Tests cover:
1. BaseAdaptor - base class functionality
2. AmazonAdaptor - product extraction with ASIN
3. LinkedInAdaptor - profile extraction
4. TwitterAdaptor - tweet extraction
5. get_adaptor - factory function
"""

import pytest

from scrapling.adaptors import (
    BaseAdaptor,
    AmazonAdaptor,
    LinkedInAdaptor,
    TwitterAdaptor,
    Product,
    LinkedInProfile,
    Tweet,
    get_adaptor,
)


class TestBaseAdaptor:
    """Tests for BaseAdaptor."""

    def test_initialization(self):
        """Test adaptor initialization."""
        adaptor = BaseAdaptor(
            name="Test",
            base_url="https://example.com",
            rate_limit_delay=1.5,
        )
        assert adaptor.name == "Test"
        assert adaptor.base_url == "https://example.com"
        assert adaptor.rate_limit_delay == 1.5

    def test_default_headers(self):
        """Test default headers are set."""
        adaptor = BaseAdaptor(
            name="Test",
            base_url="https://example.com",
            default_headers={"User-Agent": "test"},
        )
        assert "User-Agent" in adaptor.default_headers
        assert adaptor.default_headers["User-Agent"] == "test"

    def test_get_headers_no_extras(self):
        """Test get_headers without extra headers."""
        adaptor = BaseAdaptor(
            name="Test",
            base_url="https://example.com",
            default_headers={"User-Agent": "test"},
        )
        headers = adaptor.get_headers()
        assert headers["User-Agent"] == "test"

    def test_get_headers_with_extras(self):
        """Test get_headers with extra headers."""
        adaptor = BaseAdaptor(
            name="Test",
            base_url="https://example.com",
            default_headers={"User-Agent": "test"},
        )
        headers = adaptor.get_headers({"Accept": "text/html"})
        assert headers["User-Agent"] == "test"
        assert headers["Accept"] == "text/html"


class TestAmazonAdaptor:
    """Tests for AmazonAdaptor."""

    def test_initialization(self):
        """Test Amazon adaptor initialization."""
        adaptor = AmazonAdaptor()
        assert adaptor.name == "Amazon"
        assert adaptor.base_url == "https://www.amazon.com"
        assert adaptor.rate_limit_delay == 5.0
        assert "User-Agent" in adaptor.default_headers

    def test_extract_asin_dp(self):
        """Test ASIN extraction from /dp/ URL."""
        adaptor = AmazonAdaptor()
        asin = adaptor._extract_asin("https://www.amazon.com/dp/B08N5WRWNW")
        assert asin == "B08N5WRWNW"

    def test_extract_asin_gp_product(self):
        """Test ASIN extraction from /gp/product/ URL."""
        adaptor = AmazonAdaptor()
        asin = adaptor._extract_asin("https://www.amazon.com/gp/product/B08N5WRWNW")
        assert asin == "B08N5WRWNW"

    def test_extract_asin_not_found(self):
        """Test ASIN extraction when not in URL."""
        adaptor = AmazonAdaptor()
        asin = adaptor._extract_asin("https://www.amazon.com/")
        assert asin is None

    def test_extract_product_initialization(self):
        """Test product extraction initializes correctly."""
        adaptor = AmazonAdaptor()
        product = adaptor.extract_product("https://www.amazon.com/dp/B08N5WRWNW")
        
        assert isinstance(product, Product)
        assert product.original_url == "https://www.amazon.com/dp/B08N5WRWNW"
        assert product.asin == "B08N5WRWNW"

    def test_extract_rating(self):
        """Test rating extraction."""
        adaptor = AmazonAdaptor()
        html = '<span class="a-icon-alt">4.5 out of 5 stars</span>'
        rating = adaptor._extract_rating(html)
        assert rating == "4.5 out of 5 stars"

    def test_extract_rating_not_found(self):
        """Test rating extraction when not present."""
        adaptor = AmazonAdaptor()
        html = '<div class="no-rating">No rating</div>'
        rating = adaptor._extract_rating(html)
        assert rating is None

    def test_extract_availability_in_stock(self):
        """Test availability extraction for in-stock."""
        adaptor = AmazonAdaptor()
        html = '<div class="availability">In Stock</div>'
        availability = adaptor._extract_availability(html)
        assert availability == "In Stock"

    def test_extract_availability_out_of_stock(self):
        """Test availability extraction for out-of-stock."""
        adaptor = AmazonAdaptor()
        html = '<div class="availability">Out of Stock</div>'
        availability = adaptor._extract_availability(html)
        assert availability == "Out of Stock"


class TestLinkedInAdaptor:
    """Tests for LinkedInAdaptor."""

    def test_initialization(self):
        """Test LinkedIn adaptor initialization."""
        adaptor = LinkedInAdaptor()
        assert adaptor.name == "LinkedIn"
        assert adaptor.base_url == "https://www.linkedin.com"
        assert "User-Agent" in adaptor.default_headers

    def test_initialization_with_auth(self):
        """Test LinkedIn adaptor with auth token."""
        adaptor = LinkedInAdaptor(auth_token="test_token")
        assert "Authorization" in adaptor.default_headers
        assert adaptor.default_headers["Authorization"] == "Bearer test_token"

    def test_extract_profile_initialization(self):
        """Test profile extraction initializes correctly."""
        adaptor = LinkedInAdaptor()
        profile = adaptor.extract_profile("https://linkedin.com/in/testuser")
        
        assert isinstance(profile, LinkedInProfile)

    def test_extract_name(self):
        """Test name extraction from JSON."""
        adaptor = LinkedInAdaptor()
        html = '{"firstName":"John","lastName":"Doe"}'
        name = adaptor._extract_name(html)
        assert name == "John Doe"

    def test_extract_name_single(self):
        """Test name extraction with only first name."""
        adaptor = LinkedInAdaptor()
        html = '{"firstName":"John"}'
        name = adaptor._extract_name(html)
        assert name == "John"

    def test_extract_headline(self):
        """Test headline extraction."""
        adaptor = LinkedInAdaptor()
        html = '{"headline":"Software Engineer at Company"}'
        headline = adaptor._extract_headline(html)
        assert headline == "Software Engineer at Company"

    def test_extract_location(self):
        """Test location extraction."""
        adaptor = LinkedInAdaptor()
        html = '{"locationName":"San Francisco Bay Area"}'
        location = adaptor._extract_location(html)
        assert location == "San Francisco Bay Area"


class TestTwitterAdaptor:
    """Tests for TwitterAdaptor."""

    def test_initialization(self):
        """Test Twitter adaptor initialization."""
        adaptor = TwitterAdaptor()
        assert adaptor.name == "Twitter"
        assert adaptor.base_url == "https://twitter.com"
        assert "User-Agent" in adaptor.default_headers

    def test_extract_tweets_initialization(self):
        """Test tweets extraction initializes correctly."""
        adaptor = TwitterAdaptor()
        tweets = adaptor.extract_tweets("https://twitter.com/testuser")
        
        assert isinstance(tweets, list)

    def test_extract_tweet_details_id_extraction(self):
        """Test tweet ID extraction from URL."""
        adaptor = TwitterAdaptor()
        tweet = adaptor.extract_tweet_details(
            "https://twitter.com/user/status/1234567890"
        )
        
        assert tweet is not None
        assert tweet.id == "1234567890"

    def test_extract_metric_likes(self):
        """Test likes metric extraction."""
        adaptor = TwitterAdaptor()
        html = '123 Likes'
        likes = adaptor._extract_metric(html, "likes")
        assert likes == 123

    def test_extract_metric_k_format(self):
        """Test metric extraction with K suffix."""
        adaptor = TwitterAdaptor()
        html = '1.5K Retweets'
        retweets = adaptor._extract_metric(html, "retweets")
        assert retweets == 1500

    def test_extract_metric_not_found(self):
        """Test metric extraction when not present."""
        adaptor = TwitterAdaptor()
        html = '<div>No metrics</div>'
        likes = adaptor._extract_metric(html, "likes")
        assert likes is None


class TestGetAdaptor:
    """Tests for get_adaptor factory function."""

    def test_get_amazon_adaptor(self):
        """Test getting Amazon adaptor."""
        adaptor = get_adaptor("amazon")
        assert isinstance(adaptor, AmazonAdaptor)

    def test_get_amazon_adaptor_case_insensitive(self):
        """Test getting Amazon adaptor case insensitive."""
        adaptor = get_adaptor("AMAZON")
        assert isinstance(adaptor, AmazonAdaptor)

    def test_get_linkedin_adaptor(self):
        """Test getting LinkedIn adaptor."""
        adaptor = get_adaptor("linkedin")
        assert isinstance(adaptor, LinkedInAdaptor)

    def test_get_twitter_adaptor(self):
        """Test getting Twitter adaptor."""
        adaptor = get_adaptor("twitter")
        assert isinstance(adaptor, TwitterAdaptor)

    def test_get_adaptor_with_kwargs(self):
        """Test getting adaptor with kwargs."""
        adaptor = get_adaptor("linkedin", auth_token="test")
        assert isinstance(adaptor, LinkedInAdaptor)
        assert "Authorization" in adaptor.default_headers

    def test_get_adaptor_invalid(self):
        """Test getting invalid adaptor raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_adaptor("invalid")
        
        assert "Unknown adaptor" in str(exc_info.value)


# Mark tests
pytestmark = pytest.mark.adaptors
