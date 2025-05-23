# Reproduce of AgentSquare

## Intro

This is a reproduction of the paper [*AgentSquare: Automatic LLM Agent Search in Modular Design Space*](https://arxiv.org/abs/2410.06153) on [**ALFWorld**](https://github.com/alfworld/alfworld), using the same dependencies as the official [repo](https://github.com/tsinghua-fib-lab/AgentSquare).

🥺 We’re still learning, so please bear with us if you encounter any issues. Feedback and suggestions from the community are always welcome!

## Quick Start

```bash
export OPENAI_API_KEY=your_openai_api_key
python search_agent_alf.py
```

## More Info

### ⚠️ Please Read Before Using!

1. We couldn’t find a way to explicitly disconnect the database in `langchain_chroma`, and the connection does not seem to close when the corresponding Python object is destroyed. This causes subsequent agents using the memory module to lose write permissions (as the database is named after the class by default).
   - Our workaround is to assign a unique UID to the database name each time an instance is created. However, this may result in many open connections in the background.
   - A more robust solution could be to launch each agent in a separate process. If you have a better idea, we’d love to hear it!
2. We did not strictly follow the original experimental setup. Instead, we formally test all reorganization results to assess the LLM's ability to predict agent performance.
3. Each evaluation corresponds to a full agent execution. Running tests on the entire dataset can be quite expensive. To manage this, we provide parameters in the main function to control the number of test cases and the size of the search space.
   - Population size only affects the number of recombinations.
   - The number of evolutions is fixed at 3, since the ALFWorld task uses three modules.
4. We try to reuse the prompts from the original repo whenever possible. However, this can cause issues—such as **only generating one combination even after multiple recombinations**.

## 🙏 Acknowledgements

- The **ALFWorld** team – for the incredible simulation environment.
- The authors of **AgentSquare** – for their brilliant ideas and open-source repo, which made this reproduction possible.
- - **OpenAI GPT** – for being our always-available teammate (and therapist 🧠).
