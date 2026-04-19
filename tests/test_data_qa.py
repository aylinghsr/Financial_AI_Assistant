# test_data_qa.py
from src.data_qa import DataQA

qa = DataQA()

questions = [
    "What is the total amount by entity in gl_balances?",
    "Show the top 5 GL accounts by average amount in gl_balances",
    "What is the average amount for gl_number 4100?",
    "How many transactions are there for Solaris SE?",
    "Show monthly balances for gl_number 1300",
    "List all income accounts",
    "Which entity has the higher total balance amount?"
]

for q in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", q)

    result = qa.ask(q)

    print("\nSQL:")
    print(result["sql"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nROWS:")
    if result["rows"] is not None:
        print(result["rows"].head(10).to_string(index=False))
    else:
        print("No rows returned.")