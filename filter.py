from openai import OpenAI

with open("filter.html", encoding="utf-8") as fh:
    html = fh.read()

client = OpenAI(
    base_url="https://console.siflow.cn/model-api",
    api_key="",
)

resp = client.chat.completions.create(
    model="simaas-qwen2-5-72b-instruct-v1",
    messages=[
        {
            "role": "user",
            "content": "以下是一段 HTML 文本，包含了一些文章的标题和链接，请你完成以下任务，并以 markdown 列表格式返回结果："
            "1. 提取文章的标题"
            "2. 提取文章的链接，如果链接不完整，默认前缀是 https://www.jiqizhixin.com"
            ""
            "返回格式类似于："
            "```"
            "1. [{标题1}]({链接1})"
            "2. [{标题2}]({链接2})"
            "3. [{标题3}]({链接3})"
            "```"
            ""
            "```"
            "HTML 文本："
            "```html"
            f"{html}"
            "```",
        },
    ],
    temperature=0,
)
summary = resp.choices[0].message.content
print(summary)
print(resp.usage)

if summary.startswith("```markdown"):
    summary = summary[len("```markdown") :].split("```")[0]

with open("filter.md", "w", encoding="utf-8") as fh:
    fh.write(summary)
