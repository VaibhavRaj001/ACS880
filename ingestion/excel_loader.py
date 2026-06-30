import pandas as pd
from langchain_core.documents import Document


def load_excel(path: str):

    df = pd.read_excel(path)

    docs = []

    for _, row in df.iterrows():

        text = "\n".join(
            [
                f"{col}: {row[col]}"
                for col in df.columns
                if pd.notna(row[col])
            ]
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "excel",
                    "row": int(_)
                }
            )
        )

    return docs


if __name__ == "__main__":

    docs = load_excel("../data/Drive Failure Data ABB 880.xlsx")

    print(len(docs))
    print(docs[0].page_content)