import argparse
import concurrent.futures
from time import time

import html2text
import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer
from tqdm import tqdm

count_total = 0
count_200 = 0
count_has_about_url = 0
count_has_description = 0

weird_char = "\xfe"

TIMEOUT = 15
STATUS_CODE_OK = 200


def send_request(url):
    try:
        try:
            response = requests.get(url, timeout=TIMEOUT)
        except requests.exceptions.MissingSchema or requests.exceptions.InvalidSchema:
            url = "http://" + url
            response = requests.get(url, timeout=TIMEOUT)

        if response.status_code != STATUS_CODE_OK:
            return response.status_code, ""
        else:
            return response.status_code, response
    except:
        return -1, ""


def pre_process_text(text):
    """
        Replace line break with a special character to keep the formatting
    """
    text = text.replace("\n", weird_char).replace("\r", " ").replace("\t", " ")
    return text


def parse_all_links_and_meta_fields(html):
    links = set()
    metas = []
    for line in BeautifulSoup(
        html, "html.parser", parse_only=SoupStrainer(["a", "meta"])
    ):
        if line.name == "a" and line.has_attr("href"):
            link = line["href"]
            links.add(link)
        if line.name == "meta":
            metas.append(line)
    return links, metas


def extract_about_us_urls(base_url, links):
    about_urls = set()
    for link in links:
        link = link.lower().replace("\n", "")
        if "about" in link:
            if link.startswith("http"):
                about_urls.add(link)
            else:
                if link.startswith("/"):
                    about_urls.add(base_url + link)
                else:
                    about_urls.add(base_url + "/" + link)

    return about_urls


def extract_description_from_meta(metas):
    description = ""
    for meta in metas:
        if (
            meta.has_attr("name")
            and meta["name"].lower() == "description"
            and meta.has_attr("content")
        ):
            description = pre_process_text(meta["content"])
        elif (
            meta.has_attr("name")
            and meta["name"].lower() == "og: description"
            and meta.has_attr("content")
        ):
            description = pre_process_text(meta["content"])
    return description


def extract_text_from_about_us_urls(about_urls):
    about_text = ""
    about_url = ""
    if len(about_urls) > 0:
        about_url = min(list(about_urls), key=len)

        about_response_code, about_response = send_request(about_url)
        if about_response_code == STATUS_CODE_OK:
            try:
                about_text = extract_text_from_html(about_response.text)
            except:
                about_text = ""
    return about_text, about_url


def extract_text_from_html(html):
    parser = html2text.HTML2Text()
    parser.wrap_links = False
    parser.skip_internal_links = True
    parser.inline_links = True
    parser.ignore_anchors = True
    parser.ignore_images = True
    parser.ignore_emphasis = True
    parser.ignore_links = True
    return pre_process_text(parser.handle(html))


def extract_text_from_url(input_data):
    global count_total, count_200, count_has_about_url, count_has_description
    count_total += 1

    response_code, response = send_request(input_data["url"])

    about_us_text = ""
    about_us_url = ""
    text = ""
    meta_description = ""

    if response_code == STATUS_CODE_OK:
        count_200 += 1
        links, metas = parse_all_links_and_meta_fields(response.text)
        meta_description = extract_description_from_meta(metas)

        if meta_description:
            count_has_description += 1

        text = extract_text_from_html(response.text)

        about_us_urls = extract_about_us_urls(input_data["url"], links)
        about_us_text, about_us_url = extract_text_from_about_us_urls(about_us_urls)
        if about_us_text:
            count_has_about_url += 1

    to_write = [
        input_data["name"],
        input_data["url"],
        response_code,
        about_us_url,
        text,
        about_us_text,
        meta_description,
    ]

    return "\t".join([str(x) for x in to_write])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from URLs")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, header=0, sep="\t")
    count_written = 0

    start_time = time()
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        f_out.write(
            "\t".join(
                [
                    "name",
                    "url",
                    "response_code",
                    "about_us_url",
                    "text",
                    "about_us_text",
                    "description",
                ]
            )
            + "\n"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            future_to_url = (
                executor.submit(extract_text_from_url, row)
                for index, row in df.iterrows()
            )

            for future in tqdm(concurrent.futures.as_completed(future_to_url)):
                try:
                    result = future.result()
                    f_out.write(result + "\n")
                    count_written += 1
                except Exception as exc:
                    print(exc)

    print("Number of rows in total: %d" % count_total)
    print("Number of URLs that return 200 status code: %d" % count_200)
    print('Number of URLs that have an "about" page: %d' % count_has_about_url)
    print("Number of URLs that have a description: %d" % count_has_description)

    print("Execution time: %fs" % (time() - start_time))

    print("Number of lines written: %d" % count_written)
