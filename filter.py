import json
import time

import requests
from bs4 import BeautifulSoup
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# consts
URL_PREFIX = "https://www.jiqizhixin.com"

# input
text_prompt_template = """以下是一段 HTML 文本，包含了一些文章的标题和链接，请你完成以下任务，并按格式返回结果：
1. 提取文章的标题
2. 提取文章的链接，如果链接不完整，默认前缀是 {url_prefix}
3. 提前文章的封面图
注意：只需要返回结果，不需要对过程进行解释
    
返回格式：
{output_format}

HTML 文本：
```html
{html_article_list}
```
"""


# output
class ArticleListItem(BaseModel):
    title: str = Field(..., description="该文章的标题")
    url: str = Field(..., description="该文章的链接")
    image_url: str = Field(..., description="该文章的封面图")


# chain
prompt_template = PromptTemplate.from_template(text_prompt_template)
output_format = PydanticOutputParser(
    pydantic_object=ArticleListItem
).get_format_instructions()
prompt = prompt_template.partial(url_prefix=URL_PREFIX, output_format=output_format)

callback = OpenAICallbackHandler()
llm = ChatOpenAI(
    base_url="https://console.siflow.cn/model-api",
    model="simaas-qwen2-5-72b-instruct-v1",
    temperature=0,
    callbacks=[callback],
)

output_parser = JsonOutputParser()
chain = prompt | llm | output_parser

# prepare data
resp = requests.get(
    URL_PREFIX,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    },
)
if not resp.ok:
    print(resp.text)
    exit(-1)
soup = BeautifulSoup(resp.text, "lxml")
div_article_list = soup.find("div", class_="js-article-container")
html_article_list = str(div_article_list)

# get summaries
start = time.perf_counter()
article_list = chain.invoke({"html_article_list": html_article_list})
end = time.perf_counter()
print(f"time cost: {end - start:.2f}s")
print(callback.prompt_tokens)
print(callback.completion_tokens)

# avoid repetition when write
with open("articles.json", encoding="utf-8") as fh:
    current_articles = json.loads(fh.read())
    url_articles_dict = dict()
    for article in current_articles:
        url_articles_dict[article["url"]] = article
    for article in article_list:
        url_articles_dict[article["url"]] = article
with open("articles.json", "w", encoding="utf-8") as fh:
    fh.write(json.dumps(list(url_articles_dict.values())))
