# 负责把业务知识文档切分成适合向量检索的小文本块。

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

#: 切分配置参与 knowledge_version 计算：改这里等于换了一个索引口径。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def split_documents(
    documents: list[Document]
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(
        documents
    )