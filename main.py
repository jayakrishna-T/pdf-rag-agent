from graph import graph


while True:

    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        break

    result = graph.invoke(
        {
            "question": question,
            "documents": [],
            "context": "",
            "recommendation": "",
            "final_response": ""
        }
    )

    print("\nAnswer:")
    print(result["final_response"])