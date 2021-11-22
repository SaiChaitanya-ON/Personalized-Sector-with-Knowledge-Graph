from urllib.parse import urlparse

import html2text
import scrapy
from scrapy.linkextractors import IGNORED_EXTENSIONS


def extract_text_from_html(html, upper_bound=10000):
    if len(html.split("\n")) > upper_bound:
        return ""
    parser = html2text.HTML2Text()
    parser.wrap_links = False
    parser.skip_internal_links = True
    parser.inline_links = True
    parser.ignore_anchors = True
    parser.ignore_images = True
    parser.ignore_emphasis = True
    parser.ignore_links = True
    return parser.handle(html)


class MySpider(scrapy.Spider):
    name = "my_spider"

    # crawl in BFS
    custom_settings = {
        "DEPTH_LIMIT": 2,
        "DEPTH_PRIORITY": 1,
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
        "SCHEDULER_PRIORITY_QUEUE": "scrapy.pqueues.DownloaderAwarePriorityQueue",
        "DNS_TIMEOUT": 20,
        "DOWNLOAD_TIMEOUT": 30,
        "LOG_LEVEL": "ERROR",
        "CONCURRENT_REQUESTS": 100,
        "REACTOR_THREADPOOL_MAXSIZE": 20,
        "REDIRECT_ENABLED": False,
        "RETRY_ENABLED": False,
    }

    def __init__(self, *args, **kwargs):
        start_urls = kwargs.get("start_urls")
        parsed_urls = [urlparse(start_url) for start_url in start_urls]
        url_domains = [parsed_url.netloc for parsed_url in parsed_urls]
        self.allowed_domains = url_domains

        self.output = kwargs.get("output")

        self.allowed_extensions = (".pdf", ".doc", ".xls", ".docx", ".xlsx")
        self.denied_extensions = tuple(
            set(["." + ext for ext in IGNORED_EXTENSIONS])
            - set(self.allowed_extensions)
        )

        super(MySpider, self).__init__(*args, **kwargs)

    def save_file(self, response, upper_limit=2000000):  # ~ 2 mbs tops
        page_url = response.meta.get("page_url")
        root_page = response.meta.get("root_page")

        path = response.url.split("/")[-1]
        filename = path[path.rfind("/") + 1 :]
        if len(response.body) < upper_limit:
            self.output.append(
                (
                    root_page,
                    description,
                    page_url,
                    filename,
                    "file",
                    bytearray(response.body),
                )
            )

    def parse(self, response):
        title_xpath = response.selector.xpath("//title/text()").extract()
        page_title = title_xpath[0] if title_xpath else ""

        root_page = response.meta.get("root_page")
        if root_page is None:
            root_page = response.url

        description = response.meta.get("root_description")
        if description is None:
            description_xpath = response.selector.xpath(
                "//meta[@name='description']/@content"
            ).extract()
            description = description_xpath[0] if description_xpath else ""

        page_content = extract_text_from_html(response.text)
        self.log("Page: %s (%s)" % (response.url, page_title))

        extracted_links = LinkExtractor().extract_links(response)
        for link in extracted_links:
            url = link.url
            parsed_url = urlparse(url)
            url_domain = parsed_url.netloc

            follow_link = False
            for marker in ["who we are", "overview", "about", "mission"]:
                if marker in link.url.lower():
                    follow_link = True

            for marker in ["who we are", "overview", "about", "mission"]:
                if marker in link.text.lower():
                    follow_link = True

            if not follow_link:
                continue

            if url_domain in self.allowed_domains:
                if link.url.endswith(self.allowed_extensions):
                    yield scrapy.Request(
                        url,
                        callback=self.save_file,
                        meta={
                            "page_url": response.url,
                            "root_page": root_page,
                            "root_description": description,
                        },
                    )
                elif link.url.endswith(self.denied_extensions):
                    continue
                else:
                    yield scrapy.Request(
                        url,
                        callback=self.parse,
                        meta={"root_page": root_page, "root_description": description},
                    )

        self.output.append(
            (root_page, description, response.url, page_title, "html", page_content)
        )
