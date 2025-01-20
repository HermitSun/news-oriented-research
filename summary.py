import time
from pprint import pprint

import requests
from bs4 import BeautifulSoup
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# input
text_prompt_template = """以下是一段 HTML 文本，包含了一篇文章的标题和内容，请你完成以下任务，并按格式返回结果：
1. 提取文章的标题
2. 用不少于 200 字且不超过 300 字总结文章的内容
3. 提取文章内容中的前三张图片（如果不足三张，提取所有的图片）
4. 用不超过五个关键词概括文章的内容
5. 判断是否向用户推荐这篇文章，一般来说，用户会对这些内容感兴趣：
    - 科技大厂和知名创企的动向，如 Google, Meta, OpenAI, Anthropic, Qwen, DeepSeek, MiniMax, StepFun 等，但：
        - 用户对没有产品发布的新闻不感兴趣
        - 用户对荣誉奖项不感兴趣
    - 重大学术突破，如 Scaling Law, KV Cache，但：
        - 用户对游戏、视频生成、自动驾驶这些方向不感兴趣
        - 用户对 AAAI、IJCAI 这些刊物不感兴趣
    - 开源产品或开源模型发布
    - 大额的融投资
    - 面向 AI 的法律法规
注意：只需要返回
    
返回格式：
{output_format}

HTML 文本：
```html
{html_article}
```
"""


# output
class Article(BaseModel):
    title: str = Field(..., description="该文章的标题")
    summary: str = Field(..., description="该文章的总结")
    image_urls: list[str] = Field(..., description="该文章中的图片链接")
    keywords: list[str] = Field(..., description="该文章的关键词")
    recommend: bool = Field(..., description="是否推荐该文章")


# chain
prompt_template = PromptTemplate.from_template(text_prompt_template)
output_format = PydanticOutputParser(pydantic_object=Article).get_format_instructions()
prompt = prompt_template.partial(output_format=output_format)

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
html_articles = []
with open("filter.md", encoding="utf-8") as fh:
    for line in fh:
        link = line.strip().split("(")[1][:-1]
        resp = requests.get(
            link,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            },
        )
        if resp.ok:
            soup = BeautifulSoup(resp.text, "lxml")
            div_article = soup.find("div", class_="article")
            # html_article = div_article.text
            html_article = str(div_article)
            html_articles.append(html_article)

# get summaries
start = time.perf_counter()
summaries = chain.batch([{"html_article": article} for article in html_articles])
end = time.perf_counter()
print(f"time cost: {end - start:.2f}s")
print(callback.prompt_tokens)
print(callback.completion_tokens)

pprint(summaries)
