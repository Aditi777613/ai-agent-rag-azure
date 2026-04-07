import traceback
from agent import build_agent

try:
    print("Building agent...")
    agent = build_agent('test1')
    print("Invoking agent...")
    result = agent.invoke({'input': 'What is the leave policy?'})
    print(result)
except Exception as e:
    traceback.print_exc()
