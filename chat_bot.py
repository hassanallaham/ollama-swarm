from swarm import Swarm, Agent

client = Swarm()

my_agent = Agent(
    name="Agent",
    model="llama3.2-1B-Instruct-Q8_0Function:latest",
    temperature=0.1,
    response_format={"type": "text"},
    instructions="You are a helpful agent.",
    # functions=[],
    # tool_choice=None,
    # parallel_tool_calls=True,
    # max_tokens=None
)


def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['sender']}: {message['content']}")


messages = []
agent = my_agent
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    messages = response.messages
    agent = response.agent
    pretty_print_messages(messages)
