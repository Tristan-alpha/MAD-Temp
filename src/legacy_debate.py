def build_debate_round_messages(
    sample,
    responses,
    *,
    agent_names=None,
    personas=None,
    suffix=None,
    sparse=False,
    centralized=False,
):
    new_message = {}

    agents = agent_names or list(responses.keys())
    if len(agents) > 1:  # MULTI-AGENT DEBATE
        if not centralized:  # DECENTRALIZED MAD
            for i, agent in enumerate(agents):
                msg = "These are the recent opinions from other agents: "
                if sparse:
                    peers = [agents[(i - 1) % len(agents)], agents[(i + 1) % len(agents)]]
                else:
                    peers = agents[:i] + agents[i + 1 :]
                for other_agent in peers:
                    msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}"

                if suffix is not None:
                    msg += suffix

                if personas is not None:
                    new_message[agent] = [
                        {'role': 'system', 'content': personas[agent.split("__")[-2]]},
                        {'role': 'user', 'content': msg},
                    ]
                else:
                    new_message[agent] = {'role': 'user', 'content': msg}

        else:  # CENTRALIZED MAD
            for i, agent in enumerate(agents):
                if i == 0:
                    msg = "These are the recent opinions from other agents: "
                    peers = agents[:i] + agents[i + 1 :]
                    for other_agent in peers:
                        msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}"
                else:
                    msg = f"This is the recent opinion from another agent: \n{responses[agents[0]]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}"

                if suffix is not None:
                    msg += suffix

                if personas is not None:
                    new_message[agent] = [
                        {'role': 'system', 'content': personas[agent.split("__")[-2]]},
                        {'role': 'user', 'content': msg},
                    ]
                else:
                    new_message[agent] = {'role': 'user', 'content': msg}

    else:  # SINGLE AGENT SELF REFINEMENT
        for i, agent in enumerate(agents):
            msg = f"This was your most recent opinion:\n{responses[agents[i]]}\n"
            msg += f"\n\nRevise your recent opinion to give your updated final answer to the question:\n{sample}"

            if suffix is not None:
                msg += suffix

            if personas is not None:
                new_message[agent] = [
                    {'role': 'system', 'content': personas[agent.split("__")[-2]]},
                    {'role': 'user', 'content': msg},
                ]
            else:
                new_message[agent] = {'role': 'user', 'content': msg}

    return new_message
