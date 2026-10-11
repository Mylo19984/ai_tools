import json
import os
import anthropic

from tools import claude_tavily_tool, tavily_search_tool

def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
    return anthropic.Anthropic(api_key=api_key)


def run_agent(prompt: str, max_turns: int = 2):
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    print("Initial messages:", messages)

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1000,
            tools=[claude_tavily_tool],
            messages=messages,
        )

        content_blocks = list(response.content)
        tool_calls = [block for block in content_blocks if getattr(block, "type", None) == "tool_use"]

        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                arguments = tool_call.input or {}
                print(f"Tool call: {tool_call.name} with {arguments}")
                result = tavily_search_tool(
                    query=arguments.get("query", ""),
                    max_results=arguments.get("max_results", 5),
                    include_images=arguments.get("include_images", False),
                    topic=arguments.get("topic"),
                    time_range=arguments.get("time_range"),
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

            messages.append({"role": "assistant", "content": content_blocks})
            messages.append({"role": "user", "content": tool_results})
            continue

        assistant_message = "\n".join(
            block.text for block in content_blocks if getattr(block, "type", None) == "text"
        )
        print(f"Assistant (Turn {turn + 1}):", assistant_message)
        messages.append({"role": "assistant", "content": content_blocks})

        if "goodbye" in assistant_message.lower():
            print("Conversation ended by the assistant.")
            break


if __name__ == "__main__":
    run_agent("Search web for latest premier league results and provide a summary of the top 5 players by ratings and performance.")