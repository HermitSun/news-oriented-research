import time

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

## init
model_api_client = OpenAI(
    base_url="https://console.siflow.cn/model-api",
    api_key="",
)

## get summary
with open("filter.md", encoding="utf-8") as fh:
    for line in fh:
        link = line.strip().split("(")[1][:-1]
        resp = requests.get(
            link,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            },
        )
        if resp.ok:
            soup = BeautifulSoup(resp.text, "lxml")
            article = soup.find("div", class_="article")
            html = article.text
            # html = str(article)

            start = time.perf_counter()
            resp = model_api_client.chat.completions.create(
                model="simaas-qwen2-5-72b-instruct-v1",
                messages=[
                    {
                        "role": "user",
                        "content": "以下是一段 HTML 文本，包含了一篇文章的标题和内容，请你完成以下任务，并以 markdown 格式返回结果："
                        "1. 提取文章的标题"
                        "2. 用不超过 300 字总结文章的内容"
                        "3. 提取文章内容中的前三张图片（如果不足三张，提取所有的图片）"
                        "4. 用不超过五个**英文**关键词概括文章的内容"
                        "5. 判断用户是否会对这篇文章感兴趣，一般来说，用户会对科技大厂的动向、新发布的开源产品、大额的融投资、取得突破的学术成果感兴趣"
                        ""
                        "返回格式如下："
                        "```"
                        "## {文章标题}"
                        "{总结内容}"
                        "### 图片"
                        "1. ![文中第一张图片]({第一张图片的链接})"
                        "2. ![文中第二张图片]({第二张图片的链接})"
                        "3. ![文中第三张图片]({第三张图片的链接})"
                        "### 关键词"
                        "{关键词}"
                        "### 是否推荐"
                        "{是否推荐}"
                        "```"
                        ""
                        "HTML 文本："
                        "```html"
                        f"{html}"
                        "```",
                    }
                ],
                temperature=0,
            )
            end = time.perf_counter()
            print(f"time cost: {end - start:.2f}s")
            summary = resp.choices[0].message.content
            print(summary)
            print(resp.usage)

            # post process
            if summary.startswith("```markdown"):
                summary = summary[len("```markdown") :].split("```")[0].strip()

            ## get result
            with open("summary.md", "a+", encoding="utf-8") as fh:
                fh.write(summary + "\n\n")
                # generate image
                # if "### 关键词" in summary:
                #     keywords = summary.split("### 关键词")[1]
                #     image_url = generate_image([keywords])
                # fh.write(f"### 生成图片\n![生成的图片]({image_url})")
