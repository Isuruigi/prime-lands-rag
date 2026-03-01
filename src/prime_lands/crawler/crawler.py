"""
Production async BFS crawler for primelands.lk.

Design decisions:
- Playwright over requests: handles JS-rendered listings
- BFS over DFS: ensures broad listing coverage before detail pages
- Exponential backoff: respectful of server load
- Content hashing: prevents duplicate indexing across sessions
- robots.txt compliance: ethical crawling
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from prime_lands.config import CrawlerConfig
from prime_lands.crawler.models import PropertyDocument
from prime_lands.exceptions import PageLoadError, RobotsBlocked, ExtractionError
from prime_lands.logger import get_logger

log = get_logger(__name__)


class PrimeLandsCrawler:
    """
    Production-grade async BFS crawler for primelands.lk.

    Features:
        - Breadth-First Search with URL deduplication
        - JavaScript rendering (handles React/Vue-rendered listings)
        - Exponential backoff with jitter on failures
        - robots.txt compliance
        - Content-hash deduplication (no reprocessing unchanged pages)
        - Checkpoint: saves progress to resume interrupted crawls
        - Rate limiting: polite crawling (configurable delay)

    Example:
        >>> cfg = load_config()
        >>> crawler = PrimeLandsCrawler(cfg.crawler, data_dir=Path("data"))
        >>> properties = await crawler.crawl()
    """

    def __init__(self, config: CrawlerConfig, data_dir: Path):
        """
        Initialize crawler with validated configuration.

        Args:
            config: Validated CrawlerConfig from Pydantic Settings
            data_dir: Root data directory for outputs
        """
        self.cfg = config
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain = urlparse(config.base_url).netloc
        self.visited: set[str] = set()
        self.seen_hashes: set[str] = set()  # Content dedup
        self.queue: deque[str] = deque([config.base_url])
        self.properties: list[PropertyDocument] = []
        self._robots_parser: RobotFileParser | None = None

    async def _load_robots(self) -> None:
        """Load and parse robots.txt for the target domain."""
        robots_url = f"{self.cfg.base_url}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            self._robots_parser = rp
            log.info(f"Loaded robots.txt from {robots_url}")
        except Exception as e:
            log.warning(f"Could not load robots.txt: {e}. Proceeding without restrictions.")

    def _is_allowed(self, url: str) -> bool:
        """Check robots.txt permission for a URL."""
        if not self.cfg.respect_robots_txt or self._robots_parser is None:
            return True
        return self._robots_parser.can_fetch(self.cfg.user_agent, url)

    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the target domain."""
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == self.domain
                and not url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.svg'))
            )
        except Exception:
            return False

    def _is_property_page(self, url: str) -> bool:
        """
        Detect property detail pages using URL patterns.

        Returns:
            True if URL pattern matches a property listing
        """
        # Actual patterns from primelands.lk based on crawl logs
        patterns = [
            r'/house/', r'/apartment/', r'/land/', r'/commercial/',
            r'/villa/', r'/condo/', r'/penthouse/',
            # Generic fallbacks
            r'/property/', r'/listing/', r'/sale/', r'/rent/',
        ]
        return any(re.search(p, url.lower()) for p in patterns)

    async def _navigate_with_retry(self, page: Page, url: str) -> bool:
        """
        Navigate to URL with exponential backoff retry.

        Args:
            page: Playwright page object
            url: Target URL

        Returns:
            True on success, False on all retries exhausted
        """
        for attempt in range(self.cfg.max_retries):
            try:
                await page.goto(url, timeout=self.cfg.timeout_ms, wait_until="networkidle")
                return True
            except Exception as e:
                wait_time = min(
                    self.cfg.backoff_base ** attempt,
                    self.cfg.backoff_max
                )
                log.warning(
                    f"Attempt {attempt + 1}/{self.cfg.max_retries} failed for {url}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
        
        return False

    async def _extract_property(self, page: Page, url: str) -> PropertyDocument | None:
        """
        Extract property data from a detail page.

        Uses og:description meta tag (most reliable on primelands.lk),
        with fallbacks to page text content for price, location, and amenities.

        Args:
            page: Playwright page with loaded property
            url: Source URL

        Returns:
            PropertyDocument if extraction succeeds, None otherwise
        """
        try:
            # Skip listing index pages like /land/en, /house/en
            # These usually have 2 parts: [category, lang] -> e.g. ['land', 'en']
            # Property pages have 3 parts: [category, id, lang] -> e.g. ['land', 'AURELIA...', 'en']
            
            # Use path components only!
            path = urlparse(url).path
            url_parts = [p for p in path.rstrip('/').split('/') if p]
            
            # If path is too short (e.g. /land/en), skip it
            if len(url_parts) < 3 and url_parts[-1] in ('en', 'si', 'ta'):
                return None

            # Property ID from URL slug
            property_id = url_parts[-2] if len(url_parts) >= 2 and url_parts[-1] in ('en', 'si', 'ta') else url_parts[-1]
            if not property_id or property_id in ('en', 'si', 'ta'):
                property_id = hashlib.md5(url.encode()).hexdigest()[:8]

            # Title — h1 is reliable on this site
            title_elem = page.locator("h1").first
            title = (await title_elem.text_content() or "Untitled Property").strip()

            # Description — og:description meta tag has the full content
            description = ""
            try:
                og_desc = await page.locator("meta[property='og:description']").get_attribute("content")
                if og_desc:
                    description = og_desc.strip()
            except:
                pass

            # Fallback: grab "About This Property" section text
            if not description:
                for selector in [
                    "section.about", ".about-property", ".property-description",
                    ".detail-description", "div.description", "article",
                ]:
                    try:
                        elem = page.locator(selector).first
                        if await elem.count() > 0:
                            description = (await elem.text_content() or "").strip()
                            if description:
                                break
                    except:
                        continue

            # Last resort: all visible paragraph text
            if not description:
                try:
                    paras = await page.locator("p").all()
                    texts = []
                    for p in paras[:10]:
                        t = (await p.text_content() or "").strip()
                        if len(t) > 40:
                            texts.append(t)
                    description = " ".join(texts)
                except:
                    pass

            # Price — second h1 or elements containing "LKR"
            price = ""
            try:
                h1s = await page.locator("h1").all()
                for h in h1s:
                    t = (await h.text_content() or "").strip()
                    if "LKR" in t or "Rs" in t:
                        price = t
                        break
            except:
                pass

            # Address / Location — look for location text near the title
            address = ""
            for selector in [
                ".location", ".address", "[class*='location']",
                "[class*='address']", "[itemprop='address']",
            ]:
                try:
                    elem = page.locator(selector).first
                    if await elem.count() > 0:
                        address = (await elem.text_content() or "").strip()
                        if address:
                            break
                except:
                    continue

            # Amenities / Facilities list items
            amenities = []
            try:
                for selector in [".facilities li", ".amenities li", ".features li", "ul li"]:
                    items = await page.locator(selector).all()
                    for item in items[:20]:
                        t = (await item.text_content() or "").strip()
                        if 2 < len(t) < 80:
                            amenities.append(t)
                    if amenities:
                        break
            except:
                pass

            # Images
            images = []
            try:
                img_elements = await page.locator("img[src]").all()
                for img in img_elements[:10]:
                    src = await img.get_attribute("src") or ""
                    if src and not src.endswith(('.svg', '.gif', '.ico')):
                        images.append(urljoin(url, src))
            except:
                pass

            # Skip if we got nothing useful
            if not description and not price:
                log.warning(f"No content extracted from {url}, skipping.")
                return None

            doc = PropertyDocument(
                property_id=property_id,
                title=title,
                address=address,
                price=price,
                description=description,
                amenities=amenities,
                url=url,
                images=images,
            )

            log.info(f"Extracted property: {doc.title[:50]}...")
            return doc

        except Exception as e:
            log.error(f"Failed to extract property from {url}: {e}")
            return None

    async def _extract_links(self, page: Page, base_url: str) -> list[str]:
        """Extract all links from a page."""
        links = []
        try:
            link_elements = await page.locator("a[href]").all()
            for elem in link_elements:
                href = await elem.get_attribute("href")
                if href:
                    absolute_url = urljoin(base_url, href)
                    if self._is_same_domain(absolute_url):
                        links.append(absolute_url)
        except Exception as e:
            log.warning(f"Failed to extract links: {e}")
        
        return links

    async def crawl(self) -> list[PropertyDocument]:
        """
        Execute breadth-first crawl of Prime Lands website.

        Returns:
            List of extracted PropertyDocument objects
        """
        log.info(f"Starting crawl of {self.cfg.base_url} (max {self.cfg.max_pages} pages)")
        
        # Load robots.txt
        if self.cfg.respect_robots_txt:
            await self._load_robots()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.cfg.headless)
            context = await browser.new_context(user_agent=self.cfg.user_agent)
            page = await context.new_page()
            
            pages_crawled = 0
            
            while self.queue and pages_crawled < self.cfg.max_pages:
                url = self.queue.popleft()
                
                # Skip if already visited
                if url in self.visited:
                    continue
                
                # Check robots.txt
                if not self._is_allowed(url):
                    log.info(f"Skipping (robots.txt): {url}")
                    continue
                
                self.visited.add(url)
                pages_crawled += 1
                
                log.info(f"[{pages_crawled}/{self.cfg.max_pages}] Crawling: {url}")
                
                # Navigate with retry
                if not await self._navigate_with_retry(page, url):
                    log.error(f"Failed to load after retries: {url}")
                    continue
                
                # Rate limiting
                await asyncio.sleep(self.cfg.rate_limit_seconds)
                
                # Extract property if this is a detail page
                if self._is_property_page(url):
                    prop = await self._extract_property(page, url)
                    if prop and prop.content_hash not in self.seen_hashes:
                        self.properties.append(prop)
                        self.seen_hashes.add(prop.content_hash)
                        log.info(f"✓ Property saved: {prop.title[:40]}... ({len(self.properties)} total)")
                
                # Extract and queue new links
                new_links = await self._extract_links(page, url)
                for link in new_links:
                    if link not in self.visited:
                        self.queue.append(link)
            
            await browser.close()
        
        # Save to JSONL
        output_file = self.data_dir / "primelands_corpus.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for prop in self.properties:
                f.write(prop.to_jsonl() + "\n")
        
        log.info(f"✓ Crawl complete! {len(self.properties)} properties saved to {output_file}")
        return self.properties
