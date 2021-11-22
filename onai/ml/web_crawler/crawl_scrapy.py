from urllib.parse import ParseResult, urlparse

from scrapy.crawler import CrawlerProcess

from onai.ml.web_crawler.spider import MySpider


def convert_url(url):
    p = urlparse(url, "http")
    netloc = p.netloc or p.path
    path = p.path if p.netloc else ""
    if not netloc.startswith("www."):
        netloc = "www." + netloc

    p = ParseResult("http", netloc, path, *p[3:])
    return p.geturl()


process = CrawlerProcess(
    {
        "USER_AGENT": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "CLOSESPIDER_PAGECOUNT": 100,
    }
)

data = []
process.crawl(
    MySpider, start_url=convert_url("www.beyondbasicsphysicaltherapy.com"), output=data
)

process.start()
