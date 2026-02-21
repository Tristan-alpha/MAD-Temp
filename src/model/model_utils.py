
import gc
import torch

model_dirs = {
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct',
    # Qwen3 models
    'qwen3-8b': 'Qwen/Qwen3-8B',
    'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen3-4b-base': 'Qwen/Qwen3-4B',
    'qwen3-4b-thinking': 'Qwen/Qwen3-4B-Thinking-2507',
    'qwen3-30b-a3b': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
    'qwen3-30b-a3b-base': 'Qwen/Qwen3-30B-A3B',
    'qwen3-235b-a22b': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
}



def engine(messages, agent, num_agents=1, temperatures=1.0, stop_sequences=None, max_new_tokens=512):
    if isinstance(temperatures, (float, int)):
        temperatures = [temperatures] * num_agents

    # Group messages by temperature
    temp_groups = {}
    for i, (msg, temp) in enumerate(zip(messages, temperatures)):
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append((i, msg))

    responses = [None] * len(messages)

    for temp, group in temp_groups.items():
        indices, msgs = zip(*group)

        if type(msgs[0]) == list :
            prompts = [agent.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        else :
            prompts = [msg['content'] for msg in msgs]  # we find that NOT using chat template is better in MAD
        
        inputs = agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids'].to(agent.huggingface_model.device)
        attention_mask = inputs['attention_mask'].to(agent.huggingface_model.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pad_token_id": agent.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": False,
            "do_sample": (temp > 0),
            "temperature": temp if temp > 0 else 1.0,
            "num_return_sequences": 1,
        }
        if temp > 0:
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        outputs = agent.huggingface_model.generate(**gen_kwargs)

        generated_sequences = outputs.sequences  # shape: (batch_size * num_agents, seq_len)

        for idx, input_id, sequence in zip(indices, input_ids, generated_sequences):
            gen_only = sequence[len(input_id):]
            decoded = agent.tokenizer.decode(gen_only, skip_special_tokens=True)
            responses[idx] = decoded

        del outputs, generated_sequences, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

    return responses 




def get_agents(args, peft_path=None):

    if args.model in ['llama3.1-8b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
        from model.llama import LlamaWrapper
        lversion = 3
        agent = LlamaWrapper(args, model_dirs[args.model], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path, llama_version=lversion)
    
    elif args.model in ['qwen2.5-7b','qwen2.5-32b'] :
        from model.qwen import QwenWrapper
        agent = QwenWrapper(args, model_dirs[args.model], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    
    elif args.model in ['qwen3-8b', 'qwen3-4b-base', 'qwen3-4b', 'qwen3-30b-a3b', 'qwen3-30b-a3b-base', 'qwen3-235b-a22b', 'qwen3-4b-thinking']:
        from model.qwen3 import Qwen3Wrapper
        agent = Qwen3Wrapper(args, model_dirs[args.model], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    
    else:
        raise ValueError("invalid model!")

    # update pad token
    if agent.tokenizer.pad_token is None :
        agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        agent.huggingface_model.resize_token_embeddings(len(agent.tokenizer))

    # Personas: taken from DyLAN: https://arxiv.org/pdf/2310.02170
    if args.multi_persona :
        personas = {
            "None": "",
            "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
            "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
            "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
            "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
            "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations.",
            "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
            "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
            "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).", # from https://github.com/composable-models/llm_multiagent_debate.git
            "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature)."
        }
        if args.data in ['arithmetics','gsm8k']:
            personas = {
                "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
                "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
                "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
                "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
                "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware."
            }
        elif args.data in ['pro_medicine']:
            personas = {
                "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
                "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
                "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
                "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
                "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations."
            }

    else:
        personas = {"None": ""}

            
    return agent, personas
