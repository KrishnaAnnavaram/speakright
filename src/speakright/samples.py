"""Built-in practice sentences. Each English word is in the built-in lexicon, and each Chinese
character is in the built-in pinyin table, so the samples work without any extra package."""

ENGLISH = [
    "she sells sea shells by the sea shore",
    "three thin thieves thought a thousand thoughts",
    "red lorry yellow lorry",
    "the weather is very nice today",
    "i would like a cup of coffee",
    "please call stella",
    "this ship is not a sheep",
    "we will visit the village with vincent",
]

MANDARIN = [
    "你好",
    "谢谢",
    "我很好",
    "妈麻马骂",
    "不客气",
    "一起去",
    "中国",
    "四是四十是十",
    "今天天气很好",
    "吃饭",
]


def samples(language: str) -> list[str]:
    if language == "en":
        return list(ENGLISH)
    if language == "zh":
        return list(MANDARIN)
    raise ValueError("language must be en or zh")
