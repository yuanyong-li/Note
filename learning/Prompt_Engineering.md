[TOC]



# Prompt_Engineering

## 常用

```python
# 中文
prompt = f"""
判断主题列表中的每一项是否是给定文本中的一个话题，

以列表的形式给出答案，每个主题用 0 或 1。

主题列表：美国航空航天局、当地政府、工程、员工满意度、联邦政府

给定文本: ```{story_zh}```
"""
response = get_completion(prompt)
print(response)
```

```python
prompt = f"""
针对以下三个反引号之间的英文评论文本，
首先进行拼写及语法纠错，
然后将其转化成中文，
再将其转化成优质淘宝评论的风格，从各种角度出发，分别说明产品的优点与缺点，并进行总结。
润色一下描述，使评论更具有吸引力。
输出结果格式为：
【优点】xxx
【缺点】xxx
【总结】xxx
注意，只需填写xxx部分，并分段输出。
将结果输出成Markdown格式。
```{text}```
"""
response = get_completion(prompt)
display(Markdown(response))
```

## 环境配置

`pip install openai`

```python
import openai
import os

openai.api_key='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'
def get_completion(prompt, model="gpt-3.5-turbo"):
    '''
    prompt: 对应的提示
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
    '''
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # 模型输出的温度系数，控制输出的随机程度
    )
    # print(response)
    # 调用 OpenAI 的 ChatCompletion 接口
    return response.choices[0].message["content"]
```

## 两个提示原则

### 清晰、具体的指令

#### 分隔符

```python
text = f"""
你应该提供尽可能清晰、具体的指示，以表达你希望模型执行的任务。\
这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
不要将写清晰的提示与写简短的提示混淆。\
在许多情况下，更长的提示可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
"""
# 需要总结的文本内容
prompt = f"""
把用三个反引号括起来的文本总结成一句话。
```{text}```
"""
# 指令内容，使用 ``` 来分隔指令和待总结的内容
response = get_completion(prompt)
print(response)
```

#### 结构化输出

```python
prompt = f"""
请生成包括书名、作者和类别的三本虚构书籍清单，\
并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。
"""
response = get_completion(prompt)
print(response)
```

#### 检查是否满足条件

```python
# 有步骤的文本
text_1 = f"""
泡一杯茶很容易。首先，需要把水烧开。\
在等待期间，拿一个杯子并把茶包放进去。\
一旦水足够热，就把它倒在茶包上。\
等待一会儿，让茶叶浸泡。几分钟后，取出茶包。\
如果你愿意，可以加一些糖或牛奶调味。\
就这样，你可以享受一杯美味的茶了。
"""
prompt = f"""
您将获得由三个引号括起来的文本。\
如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

第一步 - ...
第二步 - …
…
第N步 - …

如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Text 1 的总结:")
print(response)
```

#### 提供少量样例

```python
prompt = f"""
你的任务是以一致的风格回答问题。

<孩子>: 教我耐心。

<祖父母>: 挖出最深峡谷的河流源于一处不起眼的泉眼；最宏伟的交响乐从单一的音符开始；最复杂的挂毯以一根孤独的线开始编织。

<孩子>: 教我韧性。
"""
response = get_completion(prompt)
print(response)
```

### 给模型时间去思考

```python
prompt_2 = f"""
1-用一句话概括下面用<>括起来的文本。
2-将摘要翻译成英语。
3-在英语摘要中列出每个名称。
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。

请使用以下格式：
文本：<要总结的文本>
摘要：<摘要>
翻译：<摘要的翻译>
名称：<英语摘要中的名称列表>
输出 JSON：<带有 English_summary 和 num_names 的 JSON>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nprompt 2:")
print(response)
```

## 迭代优化

```python
# 示例：产品说明书
fact_sheet_chair = """
概述
    美丽的中世纪风格办公家具系列的一部分，包括文件柜、办公桌、书柜、会议桌等。
    多种外壳颜色和底座涂层可选。

结构
    五个轮子的塑料涂层铝底座。
    气动椅子调节，方便升降。

尺寸
    宽度53厘米|20.87英寸
    深度51厘米|20.08英寸
    高度80厘米|31.50英寸
    座椅高度44厘米|17.32英寸
    座椅深度41厘米|16.14英寸

选项
    软地板或硬地板滚轮选项。
    两种座椅泡沫密度可选：中等（1.8磅/立方英尺）或高（2.8磅/立方英尺）。
    无扶手或8个位置PU扶手。

材料
外壳底座滑动件
    改性尼龙PA6/PA66涂层的铸铝。
    外壳厚度：10毫米。
    座椅
    HD36泡沫

原产国
    意大利
"""
```

```python
# 提示：基于说明书创建营销描述
prompt = f"""
你的任务是帮助营销团队基于技术说明书创建一个产品的营销描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

技术说明: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```

```python
# 优化后的 Prompt，要求生成描述不多于 50 词
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的零售网站描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

使用最多50个词。

技术规格：```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```

```python
# 优化后的 Prompt，说明面向对象，应具有什么性质且侧重于什么方面
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的零售网站描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

该描述面向家具零售商，因此应具有技术性质，并侧重于产品的材料构造。

使用最多50个单词。

技术规格： ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
```

```python
# 要求它抽取信息并组织成表格，并指定表格的列、表名和格式
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的零售网站描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

该描述面向家具零售商，因此应具有技术性质，并侧重于产品的材料构造。

在描述末尾，包括技术规格中每个7个字符的产品ID。

在描述之后，包括一个表格，提供产品的尺寸。表格应该有两列。第一列包括尺寸的名称。第二列只包括英寸的测量值。

给表格命名为“产品尺寸”。

将所有内容格式化为可用于网站的HTML格式。将描述放在<div>元素中。

技术规格：```{fact_sheet_chair}```
"""

response = get_completion(prompt)
print(response)

# 表格是以 HTML 格式呈现的，加载出来
from IPython.display import display, HTML

display(HTML(response))
```

## 文本总结

```python
# 规定字数，和聚焦方向
prompt = f"""
你的任务是从电子商务网站上生成一个产品评论的简短摘要。

请对三个反引号之间的评论文本进行概括，最多30个词汇，并且聚焦在产品价格和质量上。

评论: ```{prod_review_zh}```
"""

response = get_completion(prompt)
print(response)
```

```python
# 文本提取不同于文本摘要，更简短凝练
prompt = f"""
你的任务是从电子商务网站上的产品评论中提取相关信息。

请从以下三个反引号之间的评论文本中提取产品运输相关的信息，最多30个词汇。

评论: ```{prod_review_zh}```
"""

response = get_completion(prompt)
print(response)
```

```python
# for循环
for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary of a product \ 
    review from an ecommerce site. 

    Summarize the review below, delimited by triple \
    backticks in at most 20 words. 

    Review: ```{reviews[i]}```
    """
    response = get_completion(prompt)
    print(i, response, "\n")
```

## 推断

### 情感推断

```python
prompt = f"""
以下用三个反引号分隔的产品评论的情感是什么？

用一个单词回答：「正面」或「负面」。

评论文本: ```{lamp_review_zh}```
"""
response = get_completion(prompt)
print(response)
```

### 识别情感
```python
prompt = f"""
识别以下评论的作者表达的情感。包含不超过五个项目(items)。将答案格式化为以逗号分隔的单词列表。

评论文本: ```{lamp_review_zh}```
"""
response = get_completion(prompt)
print(response)
```

## 文本转换

### 翻译

```python
prompt = f"""
请将以下文本翻译成中文，分别展示成正式与非正式两种语气: 
```Would you like to order a pillow?```
"""
response = get_completion(prompt)
print(response)
```

```python
for issue in user_messages:
    time.sleep(20)
    prompt = f"告诉我以下文本是什么语种，直接输出语种，如法语，无需输出标点符号: ```{issue}```"
    lang = get_completion(prompt)
    print(f"原始消息 ({lang}): {issue}\n")

    prompt = f"""
    将以下消息分别翻译成英文和中文，并写成
    中文翻译：xxx
    英文翻译：yyy
    的格式：
    ```{issue}```
    """
    response = get_completion(prompt)
    print(response, "\n=========================================")
```

### 语气风格调整

```python
prompt = f"""
将以下文本翻译成商务信函的格式: 
```小老弟，我小羊，上回你说咱部门要采购的显示器是多少寸来着？```
"""
response = get_completion(prompt)
print(response)
```

### 综合

```python
prompt = f"""
针对以下三个反引号之间的英文评论文本，
首先进行拼写及语法纠错，
然后将其转化成中文，
再将其转化成优质淘宝评论的风格，从各种角度出发，分别说明产品的优点与缺点，并进行总结。
润色一下描述，使评论更具有吸引力。
输出结果格式为：
【优点】xxx
【缺点】xxx
【总结】xxx
注意，只需填写xxx部分，并分段输出。
将结果输出成Markdown格式。
```{text}```
"""
response = get_completion(prompt)
display(Markdown(response))
```

## 文本扩展

```python
prompt = f"""
你是一位客户服务的AI助手。
你的任务是给一位重要客户发送邮件回复。
根据客户通过“```”分隔的评价，生成回复以感谢客户的评价。提醒模型使用评价中的具体细节
用简明而专业的语气写信。
作为“AI客户代理”签署电子邮件。
客户评论：
```{review}```
评论情感：{sentiment}
"""
response = get_completion(prompt)
print(response)
```

### temperature

在温度为零时，模型将总是选择最有可能的下一个词，而在较高的温度下，它还将选择其中一个不太可能的词，在更高的温度下，它甚至可能选择塔可，而这种可能性仅为五分之一。

一般设置temperature为0
