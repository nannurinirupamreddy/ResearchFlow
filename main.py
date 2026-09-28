from dotenv import load_dotenv

from agent import agent

load_dotenv()


def main():
    print("\nResearchFlow")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Research question: ").strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            continue

        print("\nResearching...\n")

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "research_session_1"
                }
            }
        )

        final_message = result["messages"][-1]

        print("\n--- Research Report ---\n")
        print(final_message.content[0]["text"])
        print("\n-----------------------\n")


if __name__ == "__main__":
    main()